"""
data_tools.py
-------------
Core audio processing utilities for speech enhancement.
Handles audio loading, STFT, spectrogram creation, and reconstruction.
"""

import os
import numpy as np
import librosa
import soundfile as sf


# ---------------------------------------------------------------------------
# Audio frame utilities
# ---------------------------------------------------------------------------

def audio_to_frames(audio, frame_length, hop_length_frame):
    """Split an audio array into overlapping frames.

    Returns a 2D array of shape (n_frames, frame_length).
    """
    n = audio.shape[0]
    frames = [
        audio[start: start + frame_length]
        for start in range(0, n - frame_length + 1, hop_length_frame)
    ]
    return np.vstack(frames)


def load_audio_dir_to_frames(audio_dir, file_list, sample_rate,
                              frame_length, hop_length_frame, min_duration=0.5):
    """Load all audio files in a directory and return stacked frames."""
    all_frames = []
    for fname in file_list:
        if fname.startswith('.'):   # skip hidden / .DS_Store
            continue
        path = os.path.join(audio_dir, fname)
        try:
            y, _ = librosa.load(path, sr=sample_rate)
        except Exception as e:
            print(f"[SKIP] {path}: {e}")
            continue

        duration = len(y) / sample_rate
        if duration < min_duration:
            print(f"[SKIP] {path} too short ({duration:.2f}s)")
            continue

        all_frames.append(audio_to_frames(y, frame_length, hop_length_frame))

    return np.vstack(all_frames)


# ---------------------------------------------------------------------------
# Noise blending
# ---------------------------------------------------------------------------

def blend_noise(voice_frames, noise_frames, nb_samples, frame_length):
    """Randomly blend voice frames with noise frames.

    Returns arrays: (clean_voice, noise, noisy_voice), each of shape
    (nb_samples, frame_length).
    """
    clean  = np.zeros((nb_samples, frame_length), dtype=np.float32)
    noise  = np.zeros((nb_samples, frame_length), dtype=np.float32)
    noisy  = np.zeros((nb_samples, frame_length), dtype=np.float32)

    n_voice = voice_frames.shape[0]
    n_noise = noise_frames.shape[0]

    for i in range(nb_samples):
        v_idx  = np.random.randint(n_voice)
        n_idx  = np.random.randint(n_noise)
        level  = np.random.uniform(0.2, 0.8)          # random noise level

        clean[i] = voice_frames[v_idx]
        noise[i] = level * noise_frames[n_idx]
        noisy[i] = clean[i] + noise[i]

    return clean, noise, noisy


# ---------------------------------------------------------------------------
# STFT / spectrogram utilities
# ---------------------------------------------------------------------------

def audio_to_mag_phase(audio, n_fft, hop_length_fft):
    """Convert an audio array to dB-magnitude and phase spectrograms."""
    stft        = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)
    magnitude, phase = librosa.magphase(stft)
    mag_db      = librosa.amplitude_to_db(magnitude, ref=np.max)
    return mag_db, phase                          # both: (n_fft/2+1, time)


def frames_to_spectrograms(frames, dim, n_fft, hop_length_fft):
    """Convert a (N, frame_length) array into (N, dim, dim) mag & phase arrays.

    The STFT can produce (dim, dim+1) due to rounding in librosa's centering.
    Both axes are cropped to `dim` so the output is always square and compatible
    with the U-Net which expects (dim, dim) inputs.
    """
    N = frames.shape[0]

    mag_db = np.zeros((N, dim, dim), dtype=np.float32)
    phase  = np.zeros((N, dim, dim), dtype=complex)

    for i in range(N):
        m, p = audio_to_mag_phase(frames[i], n_fft, hop_length_fft)
        # Crop to (dim, dim) — STFT may produce (dim, dim+1) due to centering
        mag_db[i] = m[:dim, :dim]
        phase[i]  = p[:dim, :dim]

    return mag_db, phase


def mag_phase_to_audio(mag_db, phase, hop_length_fft, frame_length):
    """Reconstruct audio from dB-magnitude and phase spectrograms."""
    magnitude = librosa.db_to_amplitude(mag_db, ref=1.0)
    stft_rec  = magnitude * phase
    audio     = librosa.istft(stft_rec, hop_length=hop_length_fft,
                               length=frame_length)
    return audio


def spectrograms_to_frames(mag_db_arr, phase_arr, hop_length_fft, frame_length):
    """Reconstruct (N, frame_length) audio from spectrogram arrays."""
    audios = [
        mag_phase_to_audio(mag_db_arr[i], phase_arr[i], hop_length_fft,
                           frame_length)
        for i in range(mag_db_arr.shape[0])
    ]
    return np.vstack(audios)


# ---------------------------------------------------------------------------
# Scaling helpers  (maps distributions to [-1, 1])
# ---------------------------------------------------------------------------

def scale_input(spec):
    """Scale noisy-voice spectrograms to [-1, 1]."""
    return (spec + 46) / 50


def scale_output(spec):
    """Scale noise-model spectrograms to [-1, 1]."""
    return (spec - 6) / 82


def inv_scale_input(spec):
    """Inverse of scale_input."""
    return spec * 50 - 46


def inv_scale_output(spec):
    """Inverse of scale_output."""
    return spec * 82 + 6


# ---------------------------------------------------------------------------
# Audio I/O  (use soundfile – librosa.output was removed in librosa ≥ 0.10)
# ---------------------------------------------------------------------------

def save_wav(path, audio, sample_rate):
    """Save a 1-D float array as a WAV file."""
    sf.write(path, audio, sample_rate)
