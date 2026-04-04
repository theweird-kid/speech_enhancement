"""
config.py
---------
All hyper-parameters and paths in one place.
Edit the DRIVE_ROOT to match your Google Drive folder name.
"""

import os

# ============================================================
# ⚙️  Google Drive root  ← CHANGE THIS to match your Drive layout
#    (Notebooks will override this via their User Settings cell)
# ============================================================
DRIVE_ROOT = '/content/drive/MyDrive/speech_enhancement'

# ============================================================
# Data directories  (will be created automatically)
# ============================================================
NOISE_DIR     = os.path.join(DRIVE_ROOT, 'data', 'noise')      # raw noise .wav files
VOICE_DIR     = os.path.join(DRIVE_ROOT, 'data', 'voice')      # raw clean voice .wav files
SPEC_DIR      = os.path.join(DRIVE_ROOT, 'data', 'spectrograms')  # saved .npy spectrograms
SOUND_DIR     = os.path.join(DRIVE_ROOT, 'data', 'sounds')     # QC .wav outputs
WEIGHTS_DIR   = os.path.join(DRIVE_ROOT, 'weights')            # model checkpoints
PRED_DIR      = os.path.join(DRIVE_ROOT, 'predictions')        # denoised outputs

# ============================================================
# Audio settings
# ============================================================
SAMPLE_RATE       = 8_000    # Hz
MIN_DURATION      = 1.0      # minimum clip duration to accept (seconds)

# Frame settings
FRAME_LENGTH            = 8_064   # ~1 second at 8 kHz  (must match STFT window)
HOP_LENGTH_FRAME        = 8_064   # non-overlapping frames (for voices)
HOP_LENGTH_FRAME_NOISE  = 4_000   # overlapping for noise augmentation

# ============================================================
# STFT settings
# ============================================================
N_FFT          = 254          # FFT size  → spectrogram height = 128
HOP_LENGTH_FFT = 63           # STFT hop  → spectrogram width  = 128

# Spectrogram size (both dims must equal n_fft/2 + 1 = 128)
DIM_SQUARE_SPEC = int(N_FFT / 2) + 1   # 128

# ============================================================
# Data-creation settings
# ============================================================
NB_SAMPLES = 500   # number of (voice+noise) training pairs to generate
                   # Use 40 000+ for production; 500 is fine for a smoke-test

# ============================================================
# Training settings
# ============================================================
EPOCHS              = 10
BATCH_SIZE          = 32
TRAINING_FROM_SCRATCH = True    # False = load PRETRAINED_WEIGHTS and fine-tune
MODEL_NAME          = 'model_unet'
PRETRAINED_WEIGHTS  = os.path.join(WEIGHTS_DIR, MODEL_NAME + '.weights.h5')

# ============================================================
# Prediction / inference settings
# ============================================================
AUDIO_INPUT_DIR   = os.path.join(DRIVE_ROOT, 'data', 'test')   # noisy .wav to denoise
