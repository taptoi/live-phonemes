"""Command line utility for phoneme recognition with wav2vec 2.0.

This script downloads and caches the ``facebook/wav2vec2-lv-60-espeak-cv-ft``
checkpoint on first run.  It provides two entry points:

``file``
    Transcribe the phonetic sequence of an audio file.  The file will be
    resampled to 16 kHz automatically and the resulting phoneme sequence is
    printed to stdout.

``stream``
    Capture audio from the system microphone in real time, run phoneme
    recognition on voiced segments and print the decoded phoneme strings to
    stdout as they are produced.  Silence is ignored.

The script has been designed to run on a Mac M1 (CPU-only) environment.  The
model is loaded only once and reused across inference calls to keep latency
low.

Example usage::

    python phoneme_recognizer.py example
    python phoneme_recognizer.py file --path path/to/audio.wav
    python phoneme_recognizer.py stream

Dependencies (install via pip if missing)::

    pip install torch transformers datasets soundfile sounddevice numpy

``sounddevice`` is only required for the real-time streaming mode.  On macOS it
is recommended to run the script from a Python virtual environment with access
to the system microphone.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

try:  # Optional dependency used for the streaming mode.
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime.
    sd = None

try:  # Preferred audio loader.
    import soundfile as sf  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime.
    sf = None

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
TARGET_SAMPLE_RATE = 16_000


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Convert a multi-channel signal to mono by averaging channels."""

    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=1)
    raise ValueError(f"Unsupported audio shape {audio.shape!r}")


def _resample_linear(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    """Resample using linear interpolation without external dependencies."""

    if original_sr == target_sr:
        return audio.astype(np.float32, copy=False)

    duration = len(audio) / float(original_sr)
    if duration == 0:
        return np.array([], dtype=np.float32)

    target_length = int(round(duration * target_sr))
    if target_length <= 1:
        return np.array([], dtype=np.float32)

    source_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_times = np.linspace(0.0, duration, num=target_length, endpoint=False)
    resampled = np.interp(target_times, source_times, audio)
    return resampled.astype(np.float32)


def _load_audio_wave(path: str) -> tuple[np.ndarray, int]:
    """Fallback WAV loader that uses the standard library."""

    import wave

    with wave.open(path, "rb") as wf:
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        num_frames = wf.getnframes()
        raw_data = wf.readframes(num_frames)

    if sample_width == 1:
        data = np.frombuffer(raw_data, dtype=np.uint8)
        data = (data.astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        data /= 32768.0
    elif sample_width == 4:
        data = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32)
        data /= 2147483648.0
    else:  # pragma: no cover - uncommon formats.
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    return data, sample_rate


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file as floating point mono data and return with sample rate."""

    if sf is not None:
        audio, sample_rate = sf.read(path, always_2d=False)
        audio = _ensure_mono(np.asarray(audio, dtype=np.float32))
        if sample_rate is None:
            raise ValueError(f"Unable to determine sample rate for {path}")
        return audio, int(sample_rate)

    # Fallback to the WAV-only loader when soundfile is not available.
    try:
        return _load_audio_wave(path)
    except Exception as exc:  # pragma: no cover - fallback path.
        raise RuntimeError(
            "Failed to load audio. Install the 'soundfile' package for wider format support."
        ) from exc


@dataclass
class PhonemeRecognizer:
    """Convenience wrapper around the wav2vec2 phoneme recognition model."""

    model_id: str = MODEL_ID
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

    def _prepare_input(self, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        audio = _ensure_mono(audio.astype(np.float32))
        audio = _resample_linear(audio, sample_rate, TARGET_SAMPLE_RATE)
        if audio.size == 0:
            raise ValueError("Audio segment is empty after preprocessing")

        inputs = self.processor(
            audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )
        return inputs.input_values.to(self.device)

    def decode(self, audio: np.ndarray, sample_rate: int) -> str:
        """Return the phoneme transcription for the given audio array."""

        input_values = self._prepare_input(audio, sample_rate)
        with torch.no_grad():
            logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription.strip()

    def transcribe_file(self, path: str) -> str:
        audio, sample_rate = load_audio(path)
        return self.decode(audio, sample_rate)

    # --- Real-time streaming ---
    def stream_microphone(
        self,
        block_size: int = 2_000,
        energy_threshold: float = 0.01,
        max_silence: float = 0.5,
    ) -> None:
        """Stream microphone audio and print detected phonemes."""

        if sd is None:  # pragma: no cover - runtime guard.
            raise RuntimeError(
                "sounddevice is required for streaming mode. Install it with 'pip install sounddevice'."
            )

        stream = sd.InputStream(
            samplerate=TARGET_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=block_size,
        )

        buffer: List[float] = []
        silence_blocks = int(max_silence * TARGET_SAMPLE_RATE / block_size)
        silence_counter = 0

        def flush_buffer() -> None:
            nonlocal buffer
            if not buffer:
                return
            segment = np.asarray(buffer, dtype=np.float32)
            buffer = []
            try:
                transcript = self.decode(segment, TARGET_SAMPLE_RATE)
            except ValueError:
                return
            if transcript:
                print(transcript)

        try:
            with stream:
                print("Listening for phonemes. Press Ctrl+C to stop.")
                while True:
                    chunk, _ = stream.read(block_size)
                    chunk = np.squeeze(chunk)
                    if chunk.size == 0:
                        continue
                    energy = float(np.sqrt(np.mean(chunk ** 2)))
                    if energy > energy_threshold:
                        buffer.extend(chunk.tolist())
                        silence_counter = 0
                    else:
                        if buffer:
                            silence_counter += 1
                            if silence_counter >= max(1, silence_blocks):
                                flush_buffer()
                                silence_counter = 0
        except KeyboardInterrupt:
            flush_buffer()
            print("\nStreaming stopped.")


def run_example(recognizer: PhonemeRecognizer) -> None:
    """Run the minimal example published by the model authors."""

    from datasets import load_dataset

    print("Running built-in example transcription...")
    dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    sample = dataset[0]["audio"]
    audio = np.array(sample["array"], dtype=np.float32)
    sample_rate = int(sample["sampling_rate"])
    transcript = recognizer.decode(audio, sample_rate)
    print("Example transcription:")
    print(transcript)


def parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="wav2vec2 phoneme recognizer")
    subparsers = parser.add_subparsers(dest="command")

    example_parser = subparsers.add_parser("example", help="Run the official example transcription")
    example_parser.set_defaults(command="example")

    file_parser = subparsers.add_parser("file", help="Transcribe a local audio file")
    file_parser.add_argument("--path", required=True, help="Path to the audio file")

    stream_parser = subparsers.add_parser("stream", help="Real-time transcription from microphone input")
    stream_parser.add_argument(
        "--block-size",
        type=int,
        default=2_000,
        help="Number of samples per audio chunk (default: 2000, i.e. 125 ms)",
    )
    stream_parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.01,
        help="RMS energy threshold for voice activity detection",
    )
    stream_parser.add_argument(
        "--max-silence",
        type=float,
        default=0.5,
        help="Maximum silence duration (seconds) before decoding a segment",
    )

    parsed = parser.parse_args(args=args)
    if parsed.command is None:
        parsed.command = "example"
    return parsed


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    try:
        recognizer = PhonemeRecognizer()
    except Exception as exc:  # pragma: no cover - runtime dependent.
        print(f"Failed to initialize phoneme recognizer: {exc}", file=sys.stderr)
        return 1

    if args.command == "example":
        run_example(recognizer)
        return 0
    if args.command == "file":
        try:
            transcript = recognizer.transcribe_file(args.path)
        except Exception as exc:
            print(f"Error transcribing {args.path}: {exc}", file=sys.stderr)
            return 1
        print(transcript)
        return 0
    if args.command == "stream":
        try:
            recognizer.stream_microphone(
                block_size=args.block_size,
                energy_threshold=args.energy_threshold,
                max_silence=args.max_silence,
            )
        except Exception as exc:
            print(f"Streaming error: {exc}", file=sys.stderr)
            return 1
        return 0

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point.
    raise SystemExit(main())
