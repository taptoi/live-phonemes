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
import socket
from dataclasses import dataclass
from typing import Iterable, Optional
from datasets import load_dataset
from pythonosc.udp_client import SimpleUDPClient

import numpy as np
import torch

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover - dependency not installed
    raise SystemExit(
        "soundfile is required. Install it with `pip install soundfile`."
    ) from exc

MODEL_NAME = "./wav2vec2_model"
TARGET_SAMPLE_RATE = 16_000
DEFAULT_MIC_CHUNK_SECONDS = 0.6
DEFAULT_SILENCE_THRESHOLD = 8e-4
DEFAULT_FILE_CHUNK_SECONDS = 0.5  # seconds
DEFAULT_FILE_CHUNK_OVERLAP_SECONDS = 0.0  # seconds

espeak_lib_path = r"C:\Users\franc\scoop\apps\espeak-ng\current\espeak NG\libespeak-ng.dll"
espeak_data_path = r"C:\Users\franc\scoop\apps\espeak-ng\current\espeak NG\espeak-ng-data"

def configure_espeak_windows():
    if not os.path.exists(espeak_lib_path):
        print(f"Warning: eSpeak NG library not found at {espeak_lib_path}")
    else:
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_lib_path

    if not os.path.exists(espeak_data_path):
        print(f"Warning: eSpeak NG data folder not found at {espeak_data_path}")
    else:
        os.environ["ESPEAK_DATA_PATH"] = espeak_data_path

if sys.platform == "win32":
    configure_espeak_windows()

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
        input_device: int = 4
    ) -> None:
        """Stream audio from the requested input and print phonemes."""
        import sounddevice as sd

        try:
            device_info = sd.query_devices(input_device)  # Use the selected device index
            device_name = device_info['name']

        except ImportError as exc:
            raise SystemExit("sounddevice is required for streaming. Install it with `pip install sounddevice`.") from exc

        #print(sd.query_devices())

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
            device=4,
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
    # ------------------------------------------------------------------
    # OSC Streaming
    # ------------------------------------------------------------------
    def stream_microphone_osc(
            self,
            chunk_seconds: float = DEFAULT_MIC_CHUNK_SECONDS,
            silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
            osc_host: str = "127.0.0.1",
            osc_port: int = 8000,
            osc_address: str = "/phonemes",
            input_device: int = 4
        ) -> None:
            """Stream audio from the requested input and send phonemes via OSC."""
            import sounddevice as sd

            try:
                device_info = sd.query_devices(input_device)  # Use the selected device index
                device_name = device_info['name']
    
            except ImportError as exc:
                raise SystemExit("sounddevice is required for streaming. Install it with `pip install sounddevice`.") from exc

            #print(sd.query_devices())

            client = SimpleUDPClient(osc_host, osc_port)

            chunk_frames = max(1, int(TARGET_SAMPLE_RATE * chunk_seconds))
            audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

            def callback(indata, frames, _time, status):
                if status:
                    print(status, file=sys.stderr)
                audio_queue.put(indata.copy())

            print(f"Starting microphone stream using device '{device_name}'. Press Ctrl+C to stop.", file=sys.stderr)

            with sd.InputStream(
                device=4,
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
                        transcription = self.transcribe_array(chunk, TARGET_SAMPLE_RATE).strip()
                        if transcription:
                            client.send_message(osc_address, transcription)
                except KeyboardInterrupt:
                    print("\nMicrophone streaming stopped.", file=sys.stderr)
    # ------------------------------------------------------------------
    # Socket Streaming
    # ------------------------------------------------------------------
    def stream_microphone_socket(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        chunk_seconds: float = DEFAULT_MIC_CHUNK_SECONDS,
        silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
        reuse_addr: bool = True,
        accept_timeout: float = 0.0,
    ) -> None:
        """Stream microphone phoneme output to a single TCP client.

        Protocol: newline-delimited UTF-8 strings (one phoneme sequence per chunk).

        Parameters
        ----------
        host, port: Where to bind the server (defaults: 127.0.0.1:8765)
        chunk_seconds: Microphone chunk length in seconds.
        silence_threshold: RMS threshold; chunks below are skipped.
        reuse_addr: Whether to set SO_REUSEADDR.
        accept_timeout: If > 0, socket accept will timeout allowing graceful Ctrl+C.
        """
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "sounddevice is required for streaming. Install it with `pip install sounddevice`."
            ) from exc

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if reuse_addr:
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if accept_timeout > 0:
                srv.settimeout(accept_timeout)
            srv.bind((host, port))
            srv.listen(1)
            print(f"Phoneme server listening on {host}:{port}", file=sys.stderr)

            # Accept a single client
            while True:
                try:
                    conn, addr = srv.accept()
                    break
                except socket.timeout:  # pragma: no cover - only if timeout used
                    continue
            with conn:
                print(f"Client connected: {addr}", file=sys.stderr)
                try:
                    # Prepare audio stream
                    chunk_frames = max(1, int(TARGET_SAMPLE_RATE * chunk_seconds))
                    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

                    def callback(indata, frames, _time, status):
                        if status:
                            print(status, file=sys.stderr)
                        audio_queue.put(indata.copy())

                    with sd.InputStream(
                        samplerate=TARGET_SAMPLE_RATE,
                        channels=1,
                        blocksize=chunk_frames,
                        dtype="float32",
                        callback=callback,
                    ):
                        print(
                            "Streaming microphone phonemes over TCP. Ctrl+C to stop.",
                            file=sys.stderr,
                        )
                        while True:
                            chunk = audio_queue.get()
                            if chunk.size == 0:
                                continue
                            mono = chunk[:, 0]
                            rms = math.sqrt(float(np.mean(mono ** 2)))
                            if rms < silence_threshold:
                                continue
                            text = self.transcribe_array(mono, TARGET_SAMPLE_RATE).strip()
                            if not text:
                                continue
                            try:
                                conn.sendall((text + "\n").encode("utf-8"))
                                print(f"{text}", file=sys.stderr)
                            except (BrokenPipeError, ConnectionResetError):
                                print("Client disconnected.", file=sys.stderr)
                                return
                except KeyboardInterrupt:  # pragma: no cover - interactive
                    print("\nSocket streaming stopped.", file=sys.stderr)
                finally:
                    try:  # attempt to signal end to client
                        conn.shutdown(socket.SHUT_RDWR)
                    except Exception:
                        pass
        finally:
            srv.close()

    # ------------------------------------------------------------------
    # File real-time streaming (audible replay + phoneme output)
    # ------------------------------------------------------------------
    def stream_file(
        self,
        path: str,
        chunk_seconds: float = DEFAULT_MIC_CHUNK_SECONDS,
        silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
        play_audio: bool = True,
    ) -> None:
        """Replay an audio file in (approximately) real-time and stream phoneme output.

        Similar to microphone streaming but pulls data from a file. Audio is optionally
        played back so the user can hear it while phonemes are printed.

        Parameters
        ----------
        path: str
            Audio file path.
        chunk_seconds: float
            Duration of each processing chunk (defaults to microphone chunk size).
        silence_threshold: float
            RMS threshold below which a chunk is skipped.
        play_audio: bool
            If True, audibly play the audio as it's processed.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be > 0")

        audio, sample_rate = sf.read(path, dtype="float32")
        audio = self._ensure_mono(audio)
        if sample_rate != TARGET_SAMPLE_RATE:
            audio = self._resample(audio, sample_rate)
            sample_rate = TARGET_SAMPLE_RATE

        chunk_frames = max(1, int(round(chunk_seconds * sample_rate)))
        n = audio.shape[0]

        try:
            import sounddevice as sd  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover
            if play_audio:
                raise SystemExit(
                    "sounddevice is required for playback. Install it with `pip install sounddevice`."
                ) from exc
            sd = None  # type: ignore

        stream = None
        if play_audio and sd is not None:
            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                blocksize=chunk_frames,
                dtype="float32",
            )
            stream.start()
            print("Replaying file with real-time phoneme streaming...", file=sys.stderr)
        else:
            print("Streaming file (silent mode)...", file=sys.stderr)

        try:
            start = 0
            while start < n:
                end = min(start + chunk_frames, n)
                chunk = audio[start:end]
                # Pad last chunk to preserve timing if playing
                if play_audio and stream is not None and chunk.shape[0] < chunk_frames:
                    pad_len = chunk_frames - chunk.shape[0]
                    chunk_for_play = np.concatenate(
                        [chunk, np.zeros(pad_len, dtype=chunk.dtype)]
                    )
                else:
                    chunk_for_play = chunk
                # Playback (blocking write ensures approximate real-time pacing)
                if play_audio and stream is not None:
                    stream.write(chunk_for_play.reshape(-1, 1))

                # Phoneme inference on original (non-padded) chunk
                rms = math.sqrt(float(np.mean(chunk ** 2))) if chunk.size else 0.0
                if rms >= silence_threshold and chunk.size:
                    text = self.transcribe_array(chunk, sample_rate).strip()
                    if text:
                        print(text, flush=True)
                start = end
        except KeyboardInterrupt:  # pragma: no cover - interactive
            print("\nFile streaming stopped.", file=sys.stderr)
        finally:
            if stream is not None:
                stream.stop(); stream.close()


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("file", "stream", "demo", "osc", "socket", "file-stream"),
        help=(
            "Modes: file (offline full), stream (mic), demo (HF sample), osc (UDP mic), socket (TCP mic), "
            "file-stream (real-time file replay)."
        ),
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
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind for socket or OSC mode (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind for socket mode or OSC mode (default: 8765).",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=4,
        help="Audio input device (default: 4).",
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

    if args.mode == "osc":
            try:
                recognizer.stream_microphone_osc(
                    osc_host=args.host,
                    osc_port=args.port,
                    chunk_seconds=args.chunk_seconds,
                    silence_threshold=args.silence_threshold,
                    input_device=args.input_device
                )
            except Exception as exc:  # pragma: no cover - runtime error
                print(f"OSC streaming failed: {exc}", file=sys.stderr)
                return 1
            return 0
    
    if args.mode == "socket":
        try:
            recognizer.stream_microphone_socket(
                host=args.host,
                port=args.port,
                chunk_seconds=args.chunk_seconds,
                silence_threshold=args.silence_threshold,
            )
        except Exception as exc:  # pragma: no cover - runtime error
            print(f"Socket streaming failed: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.mode == "file-stream":
        if not args.path:
            print("--path is required in file-stream mode", file=sys.stderr)
            return 2
        try:
            recognizer.stream_file(
                args.path,
                chunk_seconds=args.chunk_seconds,
                silence_threshold=args.silence_threshold,
                play_audio=True,
            )
        except Exception as exc:  # pragma: no cover - runtime error
            print(f"File streaming failed: {exc}", file=sys.stderr)
            return 1
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
