# live-phonemes

Real-time and offline phoneme recognition utilities powered by
[`facebook/wav2vec2-lv-60-espeak-cv-ft`](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft).

## Requirements

- Python 3.9+
- [PyTorch](https://pytorch.org/) with CPU, CUDA, or MPS support (for Apple Silicon).
- Python packages: `transformers`, `datasets`, `soundfile`, `sounddevice`, `numpy`.

You can install the Python dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio
pip install transformers datasets soundfile sounddevice numpy
```

Alternatively, install everything via the bundled requirements file:

```bash
pip install -r requirements.txt
```

> ℹ️  On Apple Silicon you can install the Metal backend with
> `pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cpu`.

The script automatically downloads the pretrained model on first use and caches
it locally using the Hugging Face transformers cache.

## Usage

All functionality lives in `phoneme_recognizer.py`. The script exposes three
modes that load the model once and re-use it for inference to minimise latency.

### Run the demo transcription

Downloads a tiny Librispeech example (from the official model docs) and
transcribes it:

```bash
python phoneme_recognizer.py demo
```

### Transcribe an audio file

Pass the path to an audio file. The audio is converted to mono, resampled to
16kHz if necessary, and the phoneme string is printed to stdout.

```bash
python phoneme_recognizer.py file --path /path/to/audio.wav
```

### Real-time transcription from the microphone

Starts a microphone stream sampled at 16kHz, ignoring silence by default. Each
chunk of detected speech is decoded immediately and the phoneme sequence is
printed to stdout.

```bash
python phoneme_recognizer.py stream
```

Optional tuning parameters:

- `--chunk-seconds`: length of microphone chunks (seconds, default `0.75`).
- `--silence-threshold`: RMS energy threshold to decide whether a chunk is
  silent (default `8e-4`).
- `--device`: explicitly set the torch device (`cpu`, `cuda`, `mps`, ...). By
  default, the script picks CUDA, then MPS, otherwise CPU.

Press <kbd>Ctrl</kbd> + <kbd>C</kbd> to stop streaming.

## Notes

- The model outputs phoneme strings (no word decoding) as described in the
  original research. You can post-process the phoneme output using your own
  pronunciation dictionary if desired.
- Ensure your microphone or input audio is sampled at 16kHz for the best
  results. The script resamples automatically but original 16kHz recordings
  minimise artifacts.

## Debug
- This was required:
export PHONEMIZER_ESPEAK_LIBRARY=$(brew --prefix espeak-ng)/lib/libespeak-ng.dylib