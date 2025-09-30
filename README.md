# live-phonemes

Utilities for phoneme recognition using the
[`facebook/wav2vec2-lv-60-espeak-cv-ft`](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft)
checkpoint.

## Requirements

* Python 3.9+
* [PyTorch](https://pytorch.org/) (CPU build is sufficient on Apple Silicon)
* `transformers`
* `datasets`
* `numpy`
* `soundfile` (file transcription)
* `sounddevice` (real-time microphone transcription)

Install the dependencies with::

    pip install torch transformers datasets numpy soundfile sounddevice

## Usage

The `phoneme_recognizer.py` script supports multiple modes:

### Built-in example

Runs the official example published with the model. This downloads a small
sample from Hugging Face Datasets and prints the detected phoneme sequence::

    python phoneme_recognizer.py example

### Audio file transcription

Processes a local audio file (any format supported by `soundfile`) and prints
the recognized phonemes::

    python phoneme_recognizer.py file --path path/to/audio.wav

### Real-time streaming

Opens the system microphone and prints phoneme sequences as soon as they are
detected. Silence is ignored. Press `Ctrl+C` to stop::

    python phoneme_recognizer.py stream

Optional arguments let you adjust the block size, energy threshold and silence
duration used by the simple voice activity detector. Run `-h` for the full
command reference.
