"""
add_noise.py
------------
Mix a clean voice recording with a noise file from the dataset.

Usage examples:
    # Basic — pick a random noise file, mix at 5 dB SNR:
    python add_noise.py --voice my_voice.wav --noise_dir data/noise/

    # Specific noise file, specific SNR, specific output:
    python add_noise.py --voice my_voice.wav \
                        --noise_dir data/noise/ \
                        --snr 0 \
                        --output noisy_output.wav

    # Try several SNR levels at once:
    python add_noise.py --voice my_voice.wav \
                        --noise_dir data/noise/ \
                        --snr 10 5 0 -5 \
                        --output_dir noisy_samples/
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 8_000   # must match the model's expected sample rate


def load_wav(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a WAV/FLAC/OGG file and resample to *target_sr* if needed."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:                       # stereo → mono
        audio = np.mean(audio, axis=1)

    if sr != target_sr:
        try:
            import soxr
            audio = soxr.resample(audio, sr, target_sr)
        except ImportError:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(target_sr, sr)
            audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)

    return audio.astype(np.float32)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)) + 1e-9)


def mix_at_snr(voice: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Mix voice + noise so that the resulting SNR equals *snr_db*.

    SNR (dB) = 10 * log10(rms_voice² / rms_noise²)
    → scale noise so its rms = rms_voice / 10^(snr_db/20)
    """
    # Trim / loop noise to match voice length
    if len(noise) < len(voice):
        repeats = int(np.ceil(len(voice) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[: len(voice)]

    target_noise_rms = rms(voice) / (10 ** (snr_db / 20.0))
    noise_scaled = noise * (target_noise_rms / rms(noise))

    mixed = voice + noise_scaled

    # Prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak

    return mixed.astype(np.float32)


def pick_noise_file(noise_dir: str) -> str:
    """Return a random WAV/FLAC file from *noise_dir* (recursive)."""
    exts = {".wav", ".flac", ".ogg", ".mp3"}
    files = [
        str(p)
        for p in Path(noise_dir).rglob("*")
        if p.suffix.lower() in exts
    ]
    if not files:
        sys.exit(f"[ERROR] No audio files found in: {noise_dir}")
    chosen = random.choice(files)
    print(f"  Noise file : {chosen}")
    return chosen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mix a clean voice recording with dataset noise at a given SNR."
    )
    p.add_argument("--voice", required=True,
                   help="Path to your clean recorded voice WAV.")
    p.add_argument("--noise_dir", required=True,
                   help="Directory containing noise WAV files (e.g. data/noise/).")
    p.add_argument("--noise_file", default=None,
                   help="Use a specific noise file instead of picking randomly.")
    p.add_argument("--snr", type=float, nargs="+", default=[5.0],
                   help="Target SNR in dB. Pass multiple values to generate several "
                        "files (e.g. --snr 10 5 0 -5). Default: 5.")
    p.add_argument("--output", default=None,
                   help="Output path for single-SNR mode. Ignored when --snr has "
                        "multiple values.")
    p.add_argument("--output_dir", default="noisy_samples",
                   help="Output directory when generating multiple SNR files. "
                        "Default: noisy_samples/")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducible noise selection.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load voice
    print(f"\n  Voice file : {args.voice}")
    voice = load_wav(args.voice)
    print(f"  Duration   : {len(voice) / SAMPLE_RATE:.2f}s  ({len(voice)} samples @ {SAMPLE_RATE} Hz)")

    # Pick noise
    noise_path = args.noise_file or pick_noise_file(args.noise_dir)
    noise = load_wav(noise_path)

    # Single-output mode
    if len(args.snr) == 1:
        snr_db = args.snr[0]
        mixed = mix_at_snr(voice, noise, snr_db)

        if args.output:
            out_path = args.output
        else:
            stem = Path(args.voice).stem
            out_path = f"{stem}_noisy_snr{int(snr_db):+d}dB.wav"

        sf.write(out_path, mixed, SAMPLE_RATE)
        print(f"\n  ✔ Saved  → {out_path}  (SNR={snr_db:+.1f} dB)")

    # Multi-SNR mode
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        stem = Path(args.voice).stem
        print(f"\n  Generating {len(args.snr)} files in: {args.output_dir}/")

        for snr_db in args.snr:
            mixed = mix_at_snr(voice, noise, snr_db)
            fname = f"{stem}_snr{int(snr_db):+d}dB.wav"
            out_path = os.path.join(args.output_dir, fname)
            sf.write(out_path, mixed, SAMPLE_RATE)
            print(f"    ✔ {fname}")

        print(f"\n  Done — {len(args.snr)} files written to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
