import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import load_digits

from src.config import RAW_DIGITS_CSV, RAW_DIGITS_DIR, ensure_project_dirs


def _save_digit_image(array: np.ndarray, output_path: Path) -> None:
    image = Image.fromarray(array.astype(np.uint8), mode="L")
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def prepare_sklearn_digits(limit_per_class: int | None = None) -> pd.DataFrame:
    """Export sklearn's real handwritten digit images into a file dataset.

    The sklearn digits dataset contains real handwritten digits collected from
    scanned forms. Images are 8x8 grayscale arrays; this function upscales them
    to 28x28 PNG files without creating synthetic digit shapes.
    """
    ensure_project_dirs()
    dataset = load_digits()
    rows = []
    per_class_count: dict[int, int] = {i: 0 for i in range(10)}

    for idx, (image, label) in enumerate(zip(dataset.images, dataset.target)):
        label = int(label)
        if limit_per_class is not None and per_class_count[label] >= limit_per_class:
            continue

        scaled = (image / image.max() * 255.0).astype(np.uint8)
        file_name = f"sklearn_{idx:05d}_{label}.png"
        relative_path = Path("data") / "raw_digits" / str(label) / file_name
        output_path = RAW_DIGITS_DIR / str(label) / file_name
        _save_digit_image(scaled, output_path)
        per_class_count[label] += 1

        rows.append(
            {
                "digit_id": f"sklearn_{idx:05d}",
                "label": label,
                "path": str(relative_path).replace("\\", "/"),
                "source": "sklearn.load_digits",
                "source_note": "Real handwritten digit dataset bundled with scikit-learn",
            }
        )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(RAW_DIGITS_CSV, index=False, encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real digit image dataset.")
    parser.add_argument("--source", choices=["sklearn"], default="sklearn")
    parser.add_argument("--limit-per-class", type=int, default=None)
    args = parser.parse_args()

    if args.source == "sklearn":
        metadata = prepare_sklearn_digits(args.limit_per_class)
    else:
        raise ValueError(f"Unsupported source: {args.source}")

    print(f"Saved {len(metadata)} real digit images to {RAW_DIGITS_DIR}")
    print(f"Metadata: {RAW_DIGITS_CSV}")


if __name__ == "__main__":
    main()
