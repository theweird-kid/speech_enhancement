"""
predict.py
----------
Load a trained U-Net model and denoise an audio file.

Run directly:
    python predict.py --input my_noisy.wav --output denoised.wav

Or call `denoise_audio()` from a notebook.
"""

import os
import argparse
import numpy as np
import librosa
import soundfile as sf

import tensorflow as tf
import config as C
from model import build_unet
from data_tools import (
    audio_to_frames,
    frames_to_spectrograms,
    spectrograms_to_frames,
    scale_input,
    inv_scale_output,
    save_wav,
)


def denoise_audio(
    input_path,
    output_path,
    weights_path  = None,
    sample_rate   = C.SAMPLE_RATE,
    frame_length  = C.FRAME_LENGTH,
    hop_frame     = C.FRAME_LENGTH,      # non-overlapping for inference
    n_fft         = C.N_FFT,
    hop_fft       = C.HOP_LENGTH_FFT,
    dim           = C.DIM_SQUARE_SPEC,
):
    """Denoise a single WAV file and save the result.

    Args:
        input_path:   Path to the noisy input WAV.
        output_path:  Path to save the denoised WAV.
        weights_path: Path to .h5 weights (defaults to PRETRAINED_WEIGHTS in config).
    """
    if weights_path is None:
        weights_path = C.PRETRAINED_WEIGHTS

    # ---- Load model --------------------------------------------------------
    print(f"Loading model weights: {weights_path}")
    model = build_unet(pretrained_weights=weights_path)

    # ---- Load audio --------------------------------------------------------
    print(f"Loading audio: {input_path}")
    audio, _ = librosa.load(input_path, sr=sample_rate)

    # ---- Split into frames -------------------------------------------------
    frames = audio_to_frames(audio, frame_length, hop_frame)
    print(f"Audio split into {frames.shape[0]} frames of length {frame_length}.")

    # ---- Compute spectrograms ----------------------------------------------
    m_noisy_db, m_phase = frames_to_spectrograms(frames, dim, n_fft, hop_fft)

    # ---- Scale & predict ---------------------------------------------------
    X_in   = scale_input(m_noisy_db).astype(np.float32)[..., np.newaxis]
    X_pred = model.predict(X_in, verbose=1)

    # ---- Subtract predicted noise model ------------------------------------
    noise_model  = inv_scale_output(X_pred[..., 0])
    X_denoised   = m_noisy_db - noise_model          # spectral subtraction

    # ---- Reconstruct audio -------------------------------------------------
    frames_denoised = spectrograms_to_frames(X_denoised, m_phase, hop_fft, frame_length)
    audio_out       = frames_denoised.reshape(-1) * 10.0   # amplitude correction

    # ---- Save --------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    save_wav(output_path, audio_out, sample_rate)
    print(f"Denoised audio saved: {output_path}")

    return audio_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Denoise a WAV file')
    parser.add_argument('--input',   required=True, help='Noisy input WAV path')
    parser.add_argument('--output',  required=True, help='Denoised output WAV path')
    parser.add_argument('--weights', default=None,  help='Model weights .h5 path')
    args = parser.parse_args()

    denoise_audio(
        input_path   = args.input,
        output_path  = args.output,
        weights_path = args.weights,
    )
