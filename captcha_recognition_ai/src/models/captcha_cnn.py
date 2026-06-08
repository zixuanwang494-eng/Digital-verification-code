from tensorflow import keras
from tensorflow.keras import layers

from src.config import CAPTCHA_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES


def build_captcha_cnn() -> keras.Model:
    """Build a stable four-output CNN for fixed-length digit captchas."""
    inputs = keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name="captcha_image")

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(192, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.35)(x)

    outputs = [
        layers.Dense(NUM_CLASSES, activation="softmax", name=f"digit_{idx + 1}")(x)
        for idx in range(CAPTCHA_LENGTH)
    ]

    model = keras.Model(inputs=inputs, outputs=outputs, name="captcha_four_output_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={f"digit_{idx + 1}": "sparse_categorical_crossentropy" for idx in range(CAPTCHA_LENGTH)},
        metrics={f"digit_{idx + 1}": ["accuracy"] for idx in range(CAPTCHA_LENGTH)},
    )
    return model
