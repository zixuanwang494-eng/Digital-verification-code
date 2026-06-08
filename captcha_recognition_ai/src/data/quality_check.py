import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from src.config import ALL_CAPTCHA_CSV, PROJECT_ROOT, REPORTS_DIR, ensure_project_dirs


def _sharpness_score(image: Image.Image) -> float:
    gray = image.convert("L")
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=1.5))
    arr = np.asarray(gray, dtype=np.float32)
    arr_blur = np.asarray(blurred, dtype=np.float32)
    return float(np.mean(np.abs(arr - arr_blur)))


def _dark_pixel_ratio(image: Image.Image) -> float:
    arr = np.asarray(image.convert("L"), dtype=np.uint8)
    return float(np.mean(arr < 180))


def run_quality_check() -> pd.DataFrame:
    ensure_project_dirs()
    metadata = pd.read_csv(ALL_CAPTCHA_CSV, dtype={"label": str})
    rows = []
    for row in metadata.to_dict("records"):
        image = Image.open(PROJECT_ROOT / row["path"])
        sharpness = _sharpness_score(image)
        dark_ratio = _dark_pixel_ratio(image)
        local_status = "normal"
        if sharpness < 2.0:
            local_status = "possibly_too_blurry"
        if dark_ratio < 0.015:
            local_status = "possibly_too_faint"
        if dark_ratio > 0.55:
            local_status = "possibly_too_noisy"
        rows.append(
            {
                "captcha_id": row["captcha_id"],
                "label": str(row["label"]).zfill(4),
                "path": row["path"],
                "sharpness_score": sharpness,
                "dark_pixel_ratio": dark_ratio,
                "local_quality_status": local_status,
            }
        )

    result = pd.DataFrame(rows)
    output_csv = REPORTS_DIR / "local_quality_check.csv"
    output_json = REPORTS_DIR / "local_quality_summary.json"
    result.to_csv(output_csv, index=False, encoding="utf-8")
    summary = result["local_quality_status"].value_counts().to_dict()
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local image quality checks.")
    parser.parse_args()
    result = run_quality_check()
    print(result["local_quality_status"].value_counts().to_string())
    print(f"Saved: {REPORTS_DIR / 'local_quality_check.csv'}")


if __name__ == "__main__":
    main()
