"""
data_tools.py
-------------
Core audio processing utilities for speech enhancement.
Handles audio loading, STFT, spectrogram creation, and reconstruction.
"""

import os
import time
import numpy as np
import librosa
import soundfile as sf

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def _progress(iterable, **kwargs):
    """Wrap iterable with tqdm if available, else plain iteration."""
    if HAS_TQDM:
        return _tqdm(iterable, **kwargs)
    return iterable


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
                              frame_length, hop_length_frame, min_duration=0.5,
                              label='files'):
    """Load all audio files in a directory and return stacked frames."""
    all_frames = []
    skipped = 0
    t0 = time.time()

    valid_files = [f for f in file_list if not f.startswith('.')]
    bar = _progress(valid_files, desc=f'  Loading {label}', unit='file')

    for fname in bar:
        path = os.path.join(audio_dir, fname)
        try:
            y, _ = librosa.load(path, sr=sample_rate)
        except Exception as e:
            print(f"\n[SKIP] {path}: {e}")
            skipped += 1
            continue

        duration = len(y) / sample_rate
        if duration < min_duration:
            skipped += 1
            continue

        all_frames.append(audio_to_frames(y, frame_length, hop_length_frame))

        if HAS_TQDM:
            bar.set_postfix(frames=sum(f.shape[0] for f in all_frames),
                            skipped=skipped)

    result = np.vstack(all_frames)
    elapsed = time.time() - t0
    print(f"  ✔ {label}: {result.shape[0]} frames from "
          f"{len(valid_files) - skipped}/{len(valid_files)} files "
          f"in {elapsed:.1f}s")
    return result


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
    t0 = time.time()

    for i in _progress(range(nb_samples), desc='  Blending', unit='sample'):
        v_idx  = np.random.randint(n_voice)
        n_idx  = np.random.randint(n_noise)
        level  = np.random.uniform(0.2, 0.8)          # random noise level

        clean[i] = voice_frames[v_idx]
        noise[i] = level * noise_frames[n_idx]
        noisy[i] = clean[i] + noise[i]

    print(f"  ✔ Blending done in {time.time() - t0:.1f}s")
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


def _compute_one_frame(args):
    """Worker function for parallel spectrogram computation."""
    frame, dim, n_fft, hop_length_fft = args
    m, p = audio_to_mag_phase(frame, n_fft, hop_length_fft)
    # Cast phase to complex64 (5.25 GB per 40k array vs 10.5 GB for complex128)
    return m[:dim, :dim].astype(np.float32), p[:dim, :dim].astype(np.complex64)


def frames_to_spectrograms(frames, dim, n_fft, hop_length_fft, n_jobs=-1, label=''):
    """Convert a (N, frame_length) array into (N, dim, dim) mag & phase arrays.

    The STFT can produce (dim, dim+1) due to rounding in librosa's centering.
    Both axes are cropped to `dim` so the output is always square and compatible
    with the U-Net which expects (dim, dim) inputs.

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers. -1 = use all CPU cores (default).
        Set to 1 to disable parallelism.
    label : str
        Short name shown in the progress bar (e.g. 'clean', 'noise', 'noisy').
    """
    N = frames.shape[0]
    args = [(frames[i], dim, n_fft, hop_length_fft) for i in range(N)]
    desc = f'  Spectrograms [{label}]' if label else '  Spectrograms'
    t0 = time.time()

    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_one_frame)(a)
            for a in _progress(args, desc=desc, unit='frame')
        )
    except ImportError:
        # Fallback: serial loop
        results = []
        for i, a in enumerate(_progress(args, desc=desc, unit='frame')):
            results.append(_compute_one_frame(a))
            if not HAS_TQDM and (i + 1) % 1000 == 0:
                pct = 100 * (i + 1) / N
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (N - i - 1)
                print(f"    [{label}] {i+1}/{N}  ({pct:.0f}%)  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    mag_db = np.stack([r[0] for r in results]).astype(np.float32)
    phase  = np.stack([r[1] for r in results]).astype(np.complex64)  # half the RAM vs complex128
    elapsed = time.time() - t0
    print(f"  ✔ Spectrograms [{label}]: {N} frames in {elapsed:.1f}s "
          f"({N/elapsed:.0f} frames/s)")
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
