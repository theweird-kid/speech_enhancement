"""
train.py
--------
Train the U-Net speech-enhancement model.

Run directly:
    python train.py

Or call `train_model()` from a notebook.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import config as C
from model import build_unet
from data_tools import scale_input, scale_output


def train_model(
    spec_dir              = C.SPEC_DIR,
    weights_dir           = C.WEIGHTS_DIR,
    model_name            = C.MODEL_NAME,
    epochs                = C.EPOCHS,
    batch_size            = C.BATCH_SIZE,
    training_from_scratch = C.TRAINING_FROM_SCRATCH,
    pretrained_weights    = C.PRETRAINED_WEIGHTS,
):
    """Load saved spectrograms, train U-Net, save best weights."""
    os.makedirs(weights_dir, exist_ok=True)

    # ---- Load data ---------------------------------------------------------
    print("Loading spectrograms …")
    X_noisy = np.load(os.path.join(spec_dir, 'noisy_voice_amp_db.npy'))
    X_clean = np.load(os.path.join(spec_dir, 'voice_amp_db.npy'))

    # Model predicts the NOISE = noisy - clean  (spectral subtraction target)
    X_noise_model = X_noisy - X_clean

    print(f"Data shape: {X_noisy.shape}")

    # ---- Scale to [-1, 1] --------------------------------------------------
    X_in = scale_input(X_noisy).astype(np.float32)
    X_ou = scale_output(X_noise_model).astype(np.float32)

    # ---- Reshape for Keras (N, H, W, 1) ------------------------------------
    X_in = X_in[..., np.newaxis]
    X_ou = X_ou[..., np.newaxis]

    # ---- Train / val split -------------------------------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_in, X_ou, test_size=0.10, random_state=42)
    print(f"Train: {X_tr.shape}  Val: {X_val.shape}")

    # ---- Build model -------------------------------------------------------
    if training_from_scratch or not os.path.exists(pretrained_weights):
        print("Training from scratch …")
        model = build_unet()
    else:
        print(f"Loading weights from {pretrained_weights} …")
        model = build_unet(pretrained_weights=pretrained_weights)

    model.summary()

    # ---- Callbacks ---------------------------------------------------------
    best_path = os.path.join(weights_dir, 'model_best.weights.h5')
    callbacks = [
        ModelCheckpoint(best_path, monitor='val_loss',
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, verbose=1, min_lr=1e-6),
    ]

    # ---- Train -------------------------------------------------------------
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    # ---- Save final weights ------------------------------------------------
    # Keras 3 expects the explicit .weights.h5 suffix when using save_weights.
    final_path = os.path.join(weights_dir, model_name + '.weights.h5')
    model.save_weights(final_path)
    print(f"Final weights saved: {final_path}")

    # Optional convenience artifact: full model in legacy .h5 format.
    # Keep this separate from weight checkpoints so predict.py stays unchanged.
    full_model_path = os.path.join(weights_dir, model_name + '.h5')
    try:
        model.save(full_model_path)
        print(f"Full model saved: {full_model_path}")
    except Exception as e:
        print(f"[WARN] Could not save full model .h5: {e}")

    # ---- Plot loss ---------------------------------------------------------
    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    ep_range = range(1, len(loss) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(ep_range, loss,     label='Training loss')
    plt.plot(ep_range, val_loss, label='Validation loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(weights_dir, 'loss_curve.png'), dpi=120)
    plt.show()

    return model, history


if __name__ == '__main__':
    train_model()
