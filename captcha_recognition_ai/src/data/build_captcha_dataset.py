import argparse
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from src.config import (
    ALL_CAPTCHA_CSV,
    CAPTCHA_DIR,
    CAPTCHA_LENGTH,
    DATA_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RAW_DIGITS_CSV,
    PROJECT_ROOT,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
    ensure_project_dirs,
)
from src.data.augment import augment_captcha, params_to_json, random_params


def _load_digit(path: Path, target_height: int, rng: random.Random) -> Image.Image:
    image = Image.open(path).convert("L")
    scale = rng.uniform(0.78, 1.02)
    height = int(target_height * scale)
    width = max(16, int(image.width * height / image.height))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    angle = rng.uniform(-8, 8)
    image = image.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=0)
    return image


def _compose_digits(digit_paths: list[Path], rng: random.Random) -> Image.Image:
    canvas = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), 255)
    digit_height = int(IMAGE_HEIGHT * 0.62)
    slot_width = IMAGE_WIDTH // CAPTCHA_LENGTH

    for index, path in enumerate(digit_paths):
        digit = _load_digit(path, digit_height, rng)
        digit_black = ImageOps.invert(digit)
        max_x_offset = max(1, slot_width - digit.width - 2)
        x = index * slot_width + rng.randint(0, max_x_offset)
        y = rng.randint(4, max(5, IMAGE_HEIGHT - digit.height - 4))
        canvas.paste(digit_black, (x, y), digit)
    return canvas


def _split_metadata(metadata: pd.DataFrame, seed: int) -> None:
    rng = random.Random(seed)
    rows = metadata.to_dict("records")
    rng.shuffle(rows)
    total = len(rows)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    splits = {
        TRAIN_CSV: rows[:train_end],
        VAL_CSV: rows[train_end:val_end],
        TEST_CSV: rows[val_end:],
    }
    for path, split_rows in splits.items():
        pd.DataFrame(split_rows).to_csv(path, index=False, encoding="utf-8")


def build_dataset(count: int, seed: int = 42) -> pd.DataFrame:
    ensure_project_dirs()
    if not RAW_DIGITS_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_DIGITS_CSV} does not exist. Run: python -m src.data.prepare_digits --source sklearn"
        )

    rng = random.Random(seed)
    raw = pd.read_csv(RAW_DIGITS_CSV)
    by_label = {label: raw[raw["label"] == label].to_dict("records") for label in range(10)}
    rows = []

    for idx in range(count):
        labels = [rng.randint(0, 9) for _ in range(CAPTCHA_LENGTH)]
        selected = [rng.choice(by_label[label]) for label in labels]
        source_paths = [PROJECT_ROOT / row["path"] for row in selected]
        label_text = "".join(str(label) for label in labels)
        params = random_params(rng)

        captcha = _compose_digits(source_paths, rng)
        captcha = augment_captcha(captcha, params, rng)

        file_name = f"captcha_{idx:06d}_{label_text}.png"
        relative_path = Path("data") / "captcha_images" / file_name
        output_path = CAPTCHA_DIR / file_name
        captcha.save(output_path)

        rows.append(
            {
                "captcha_id": f"captcha_{idx:06d}",
                "label": label_text,
                "path": str(relative_path).replace("\\", "/"),
                "digit_sources": "|".join(str(row["digit_id"]) for row in selected),
                "augment_params": params_to_json(params),
                "quality_status": "unreviewed",
            }
        )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(ALL_CAPTCHA_CSV, index=False, encoding="utf-8")
    _split_metadata(metadata, seed)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build four-digit captcha dataset from real digit images.")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata = build_dataset(args.count, args.seed)
    print(f"Saved {len(metadata)} captcha images to {CAPTCHA_DIR}")
    print(f"All labels: {ALL_CAPTCHA_CSV}")
    print(f"Splits: {TRAIN_CSV}, {VAL_CSV}, {TEST_CSV}")


if __name__ == "__main__":
    main()
