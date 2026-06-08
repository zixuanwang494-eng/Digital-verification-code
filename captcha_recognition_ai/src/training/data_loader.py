from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tensorflow import keras

from src.config import CAPTCHA_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, PROJECT_ROOT


def read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"label": str})


def load_images_and_labels(manifest_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    metadata = read_manifest(manifest_path)
    images = []
    labels = {f"digit_{idx + 1}": [] for idx in range(CAPTCHA_LENGTH)}

    for _, row in metadata.iterrows():
        image_path = PROJECT_ROOT / str(row["path"])
        image = Image.open(image_path).convert("L")
        image = ImageOps.grayscale(image).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        images.append(arr[..., None])

        label = str(row["label"]).zfill(CAPTCHA_LENGTH)
        for idx, char in enumerate(label):
            labels[f"digit_{idx + 1}"].append(int(char))

    x = np.stack(images, axis=0)
    y = {name: np.asarray(values, dtype=np.int64) for name, values in labels.items()}
    return x, y, metadata


def make_dataset(manifest_path: Path, batch_size: int = 64, shuffle: bool = False):
    x, y, metadata = load_images_and_labels(manifest_path)
    dataset = keras.utils.timeseries_dataset_from_array
    # Keras has no native multi-output image manifest helper; arrays are small
    # enough for the first project version, so fit/evaluate can use x/y directly.
    return x, y, metadata
