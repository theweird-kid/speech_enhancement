"""
model.py
--------
U-Net architecture for speech enhancement.
Input:  magnitude spectrogram of noisy voice  (128 x 128 x 1)
Output: magnitude spectrogram of the noise model (128 x 128 x 1)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, LeakyReLU, MaxPooling2D,
    Dropout, concatenate, UpSampling2D,
)
from tensorflow.keras.models import load_model


def build_unet(input_shape=(128, 128, 1), pretrained_weights=None):
    """Build and compile the U-Net model.

    Args:
        input_shape:        Spectrogram shape (height, width, channels).
        pretrained_weights: Optional path to a saved .h5 model or legacy weights file.

    Returns:
        Compiled Keras Model.
    """
    f       = 16            # base number of filters
    init    = 'he_normal'
    act     = None          # activation handled by separate LeakyReLU layers

    def conv_block(x, filters):
        x = Conv2D(filters, 3, activation=act, padding='same',
                   kernel_initializer=init)(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters, 3, activation=act, padding='same',
                   kernel_initializer=init)(x)
        x = LeakyReLU()(x)
        return x

    # ---- Encoder ----
    inputs = Input(input_shape)

    c1 = conv_block(inputs, f)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, f * 2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, f * 4)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, f * 8)
    d4 = Dropout(0.5)(c4)
    p4 = MaxPooling2D((2, 2))(d4)

    # ---- Bottleneck ----
    c5 = conv_block(p4, f * 16)
    d5 = Dropout(0.5)(c5)

    # ---- Decoder ----
    def up_block(x, skip, filters):
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filters, 2, activation=act, padding='same',
                   kernel_initializer=init)(x)
        x = LeakyReLU()(x)
        x = concatenate([skip, x], axis=-1)
        x = conv_block(x, filters)
        return x

    u6 = up_block(d5, d4, f * 8)
    u7 = up_block(u6, c3, f * 4)
    u8 = up_block(u7, c2, f * 2)
    u9 = up_block(u8, c1, f)

    # Extra conv before final 1x1
    u9 = Conv2D(2, 3, activation=act, padding='same',
                kernel_initializer=init)(u9)
    u9 = LeakyReLU()(u9)

    outputs = Conv2D(1, 1, activation='tanh')(u9)

    model = Model(inputs, outputs, name='SpeechEnhancement_UNet')
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.Huber(),
        metrics=['mae'],
    )

    if pretrained_weights:
        try:
            loaded_model = load_model(pretrained_weights, compile=False)
            loaded_model.compile(
                optimizer='adam',
                loss=tf.keras.losses.Huber(),
                metrics=['mae'],
            )
            print(f"Loaded model from: {pretrained_weights}")
            return loaded_model
        except Exception:
            model.load_weights(pretrained_weights)
            print(f"Loaded weights from: {pretrained_weights}")

    return model


if __name__ == '__main__':
    m = build_unet()
    m.summary()
