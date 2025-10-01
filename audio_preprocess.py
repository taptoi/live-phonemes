#!/usr/bin/env python3
"""
Convert stereo WAV → mono WAV at 16 kHz
- Input:  stereo .wav file
- Output: mono .wav file (16kHz, same dtype as input)
"""

import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

def stereo_to_mono_16k(input_path: str, output_path: str):
    # Load audio
    sr, data = wavfile.read(input_path)  # sr = sample rate, data = np.array

    if data.ndim == 1:
        print("Input is already mono.")
        mono = data.astype(np.float32)
    elif data.ndim == 2 and data.shape[1] == 2:
        # Average left and right channels
        mono = data.mean(axis=1).astype(np.float32)
    else:
        raise ValueError("Unsupported audio shape: {}".format(data.shape))

    # Target sample rate
    target_sr = 16000
    if sr != target_sr:
        # Use rational resampling to preserve quality
        gcd = np.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        mono = resample_poly(mono, up, down)

    # Normalize back to original dtype range
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        min_val = np.iinfo(data.dtype).min
        mono = np.clip(mono, min_val, max_val).astype(data.dtype)
    else:
        mono = mono.astype(np.float32)

    # Save output
    wavfile.write(output_path, target_sr, mono)
    print(f"✅ Saved mono 16kHz wav to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python stereo_to_mono_16k.py input.wav output.wav")
        sys.exit(1)

    stereo_to_mono_16k(sys.argv[1], sys.argv[2])
