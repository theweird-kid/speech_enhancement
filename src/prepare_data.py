"""
prepare_data.py
---------------
Data-preparation pipeline.

Run directly:
    python prepare_data.py

Or import the `create_data()` function from a notebook.
"""

import os
import numpy as np
from data_tools import (
    load_audio_dir_to_frames,
    blend_noise,
    frames_to_spectrograms,
    save_wav,
)
import config as C


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def create_data(
    noise_dir      = C.NOISE_DIR,
    voice_dir      = C.VOICE_DIR,
    spec_dir       = C.SPEC_DIR,
    sound_dir      = C.SOUND_DIR,
    sample_rate    = C.SAMPLE_RATE,
    min_duration   = C.MIN_DURATION,
    frame_length   = C.FRAME_LENGTH,
    hop_frame      = C.HOP_LENGTH_FRAME,
    hop_frame_noise= C.HOP_LENGTH_FRAME_NOISE,
    nb_samples     = C.NB_SAMPLES,
    n_fft          = C.N_FFT,
    hop_fft        = C.HOP_LENGTH_FFT,
    dim            = C.DIM_SQUARE_SPEC,
):
    """
    1. Load voice & noise audio files from Drive.
    2. Randomly blend them (with random noise level 20-80 %).
    3. Compute magnitude-dB spectrograms + phase for every frame.
    4. Save numpy arrays to spec_dir for training.
    5. Save QC WAV files to sound_dir.
    """
    ensure_dirs(spec_dir, sound_dir)

    # --- 1. Load audio files -----------------------------------------------
    noise_files = sorted([f for f in os.listdir(noise_dir) if not f.startswith('.')])
    voice_files = sorted([f for f in os.listdir(voice_dir) if not f.startswith('.')])

    print(f"Found {len(voice_files)} voice files, {len(noise_files)} noise files.")

    print("Loading voice files …")
    voice_frames = load_audio_dir_to_frames(
        voice_dir, voice_files, sample_rate, frame_length, hop_frame, min_duration,
        label='voice files')

    print("Loading noise files …")
    noise_frames = load_audio_dir_to_frames(
        noise_dir, noise_files, sample_rate, frame_length, hop_frame_noise, min_duration,
        label='noise files')

    print(f"Voice frames: {voice_frames.shape}, Noise frames: {noise_frames.shape}")

    # --- 2. Blend -----------------------------------------------------------
    print(f"Blending {nb_samples} samples …")
    clean, noise, noisy = blend_noise(voice_frames, noise_frames, nb_samples, frame_length)

    # --- 3. QC WAV output ---------------------------------------------------
    print("Saving QC audio …")
    save_wav(os.path.join(sound_dir, 'noisy_voice_long.wav'),
             noisy.reshape(-1), sample_rate)
    save_wav(os.path.join(sound_dir, 'clean_voice_long.wav'),
             clean.reshape(-1), sample_rate)
    save_wav(os.path.join(sound_dir, 'noise_long.wav'),
             noise.reshape(-1), sample_rate)

    # --- 4. Spectrograms ----------------------------------------------------
    print("Computing spectrograms …")
    m_clean, p_clean  = frames_to_spectrograms(clean, dim, n_fft, hop_fft, label='clean')
    m_noise, _        = frames_to_spectrograms(noise, dim, n_fft, hop_fft, label='noise')
    m_noisy, p_noisy  = frames_to_spectrograms(noisy, dim, n_fft, hop_fft, label='noisy')

    # --- 5. Save to disk ----------------------------------------------------
    print("Saving spectrograms …")
    np.save(os.path.join(spec_dir, 'voice_amp_db.npy'),       m_clean)
    np.save(os.path.join(spec_dir, 'noise_amp_db.npy'),       m_noise)
    np.save(os.path.join(spec_dir, 'noisy_voice_amp_db.npy'), m_noisy)
    np.save(os.path.join(spec_dir, 'voice_phase.npy'),        p_clean)
    np.save(os.path.join(spec_dir, 'noisy_voice_phase.npy'),  p_noisy)

    print("Done! Spectrograms saved to:", spec_dir)
    print(f"  voice_amp_db   : {m_clean.shape}")
    print(f"  noisy_voice_amp_db: {m_noisy.shape}")


if __name__ == '__main__':
    create_data()
