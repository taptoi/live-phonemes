"""Phoneme recognition utilities powered by Wav2Vec2.

This script provides two modes of operation:
1. Offline transcription for a given audio file.
2. Real-time transcription from the microphone.

It downloads the facebook/wav2vec2-lv-60-espeak-cv-ft model on first use
and caches it locally via Hugging Face transformers.
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import sys
from dataclasses import dataclass
from typing import Iterable, Optional
from datasets import load_dataset

import numpy as np
import torch

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover - dependency not installed
    raise SystemExit(
        "soundfile is required. Install it with `pip install soundfile`."
    ) from exc

MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"
TARGET_SAMPLE_RATE = 16_000
DEFAULT_MIC_CHUNK_SECONDS = 0.6
DEFAULT_SILENCE_THRESHOLD = 8e-4
DEFAULT_FILE_CHUNK_SECONDS = 0.5  # seconds
DEFAULT_FILE_CHUNK_OVERLAP_SECONDS = 0.0  # seconds


def pick_device(preferred: Optional[str] = None) -> torch.device:
    """Return the most appropriate torch device for inference."""
    if preferred:
        device = torch.device(preferred)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this system.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available on this system.")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # macOS Metal backend
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class PhonemeRecognizer:
    """Wraps a wav2vec2 model and processor for phoneme inference."""

    device: torch.device
    model_name: str = MODEL_NAME

    def __post_init__(self) -> None:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _ensure_mono(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio
        if audio.ndim == 2:
            return np.mean(audio, axis=1)
        raise ValueError("Audio must be 1-D or 2-D array")

    def _resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == TARGET_SAMPLE_RATE:
            return audio
        if orig_sr <= 0:
            raise ValueError("Invalid sampling rate")
        duration = audio.shape[0] / float(orig_sr)
        target_length = int(round(duration * TARGET_SAMPLE_RATE))
        if target_length <= 0:
            raise ValueError("Audio is too short after resampling")
        # Linear interpolation resampling
        source_times = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
        target_times = np.linspace(0.0, duration, num=target_length, endpoint=False)
        return np.interp(target_times, source_times, audio).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def transcribe_array(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe the provided audio array into phoneme sequence."""
        audio = self._ensure_mono(audio)
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains NaNs or infinite values")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if sample_rate != TARGET_SAMPLE_RATE:
            audio = self._resample(audio, sample_rate)
        inputs = self.processor(
            audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = None
        if inputs.get("attention_mask") is not None:
            attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values, attention_mask=attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        return transcription[0]

    def transcribe_file(self, path: str) -> str:
        # ...existing code...
        raise NotImplementedError("Use transcribe_file_chunked with chunking support.")

    def transcribe_file_chunked(
        self,
        path: str,
        chunk_seconds: float = DEFAULT_FILE_CHUNK_SECONDS,
        overlap_seconds: float = DEFAULT_FILE_CHUNK_OVERLAP_SECONDS,
        joiner: str = " ",
    ) -> str:
        """Transcribe a file by splitting it into (possibly overlapping) chunks.

        Parameters
        ----------
        path: str
            Path to the audio file.
        chunk_seconds: float
            Target size of each chunk in seconds. Default 0.2s.
        overlap_seconds: float
            Overlap between consecutive chunks in seconds. Default 0.0s.
        joiner: str
            String used to join per-chunk phoneme sequences.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be > 0")
        if overlap_seconds < 0:
            raise ValueError("overlap_seconds must be >= 0")
        if overlap_seconds >= chunk_seconds:
            raise ValueError("overlap_seconds must be smaller than chunk_seconds")

        audio, sample_rate = sf.read(path, dtype="float32")
        # Ensure mono and resample once for efficiency
        audio = self._ensure_mono(audio)
        if sample_rate != TARGET_SAMPLE_RATE:
            audio = self._resample(audio, sample_rate)
            sample_rate = TARGET_SAMPLE_RATE

        chunk_frames = int(round(chunk_seconds * sample_rate))
        overlap_frames = int(round(overlap_seconds * sample_rate))
        if chunk_frames <= 0:
            raise ValueError("Computed chunk_frames <= 0; check chunk_seconds")
        step = chunk_frames - overlap_frames if chunk_frames > overlap_frames else chunk_frames

        results: list[str] = []
        n = audio.shape[0]
        for start in range(0, n, step):
            end = min(start + chunk_frames, n)
            chunk = audio[start:end]
            if chunk.size == 0:
                continue
            # Skip near-silent chunks quickly
            rms = math.sqrt(float(np.mean(chunk ** 2)))
            if rms < 1e-5:  # conservative silence threshold for files
                continue
            text = self.transcribe_array(chunk, sample_rate).strip()
            if text:
                results.append(text)
            if end == n:
                break
        return joiner.join(results)

    # ------------------------------------------------------------------
    # Demo helper
    # ------------------------------------------------------------------
    def run_demo(self) -> str:
        """Run the demo provided by Hugging Face for quick verification."""
        from datasets import load_dataset

        dataset = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
        audio = np.asarray(dataset[0]["audio"]["array"], dtype=np.float32)
        return self.transcribe_array(audio, TARGET_SAMPLE_RATE)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    def stream_microphone(
        self,
        chunk_seconds: float = DEFAULT_MIC_CHUNK_SECONDS,
        silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
    ) -> None:
        """Stream audio from the default microphone and print phonemes."""
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - dependency not installed
            raise SystemExit(
                "sounddevice is required for streaming. Install it with `pip install sounddevice`."
            ) from exc

        chunk_frames = max(1, int(TARGET_SAMPLE_RATE * chunk_seconds))
        audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

        def callback(indata, frames, _time, status):
            if status:
                print(status, file=sys.stderr)
            audio_queue.put(indata.copy())

        print(
            "Starting microphone stream. Press Ctrl+C to stop.",
            file=sys.stderr,
        )
        with sd.InputStream(
            samplerate=TARGET_SAMPLE_RATE,
            channels=1,
            blocksize=chunk_frames,
            dtype="float32",
            callback=callback,
        ):
            try:
                while True:
                    chunk = audio_queue.get()
                    if chunk.size == 0:
                        continue
                    chunk = chunk[:, 0]
                    rms = math.sqrt(float(np.mean(chunk ** 2)))
                    if rms < silence_threshold:
                        continue
                    transcription = self.transcribe_array(chunk, TARGET_SAMPLE_RATE)
                    cleaned = transcription.strip()
                    if cleaned:
                        print(cleaned, flush=True)
            except KeyboardInterrupt:
                print("\nMicrophone streaming stopped.", file=sys.stderr)


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("file", "stream", "demo"),
        help="Whether to transcribe a file, use the microphone stream, or run the demo.",
    )
    parser.add_argument(
        "--path",
        "-p",
        help="Path to the audio file when using file mode.",
    )
    parser.add_argument(
        "--device",
        help="Torch device to use (e.g. cpu, cuda, mps). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_MIC_CHUNK_SECONDS,
        help="Chunk size in seconds when streaming from the microphone.",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=DEFAULT_SILENCE_THRESHOLD,
        help="RMS threshold for silence detection during streaming.",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Name of the Hugging Face model to download and run.",
    )
    parser.add_argument(
        "--file-chunk-seconds",
        type=float,
        default=DEFAULT_FILE_CHUNK_SECONDS,
        help="Chunk size (seconds) for file transcription (default: 0.2).",
    )
    parser.add_argument(
        "--file-chunk-overlap",
        type=float,
        default=DEFAULT_FILE_CHUNK_OVERLAP_SECONDS,
        help="Overlap (seconds) between file chunks (default: 0.0).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        device = pick_device(args.device)
    except Exception as exc:  # pragma: no cover - user input error
        print(f"Failed to select device: {exc}", file=sys.stderr)
        return 2

    recognizer = PhonemeRecognizer(device=device, model_name=args.model_name)

    if args.mode == "file":
        if not args.path:
            print("--path is required in file mode", file=sys.stderr)
            return 2
        try:
            transcription = recognizer.transcribe_file_chunked(
                args.path,
                chunk_seconds=args.file_chunk_seconds,
                overlap_seconds=args.file_chunk_overlap,
            )
        except Exception as exc:  # pragma: no cover - runtime error
            print(f"Failed to transcribe file: {exc}", file=sys.stderr)
            return 1
        print(transcription)
        return 0

    if args.mode == "demo":
        try:
            transcription = recognizer.run_demo()
        except Exception as exc:  # pragma: no cover - runtime error
            print(f"Failed to run demo: {exc}", file=sys.stderr)
            return 1
        print(transcription)
        return 0

    if args.mode == "stream":
        try:
            recognizer.stream_microphone(
                chunk_seconds=args.chunk_seconds,
                silence_threshold=args.silence_threshold,
            )
        except Exception as exc:  # pragma: no cover - runtime error
            print(f"Streaming failed: {exc}", file=sys.stderr)
            return 1
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
