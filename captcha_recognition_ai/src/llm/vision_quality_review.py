import argparse
import json
import random

import pandas as pd

from src.config import ALL_CAPTCHA_CSV, PROJECT_ROOT, QUALITY_CSV, REPORTS_DIR, ensure_project_dirs
from src.llm.client import call_vision_json


QUALITY_PROMPT = """你是验证码数据质量审核助手。
请观察这张四位数字验证码图片，并结合给定标签判断样本质量。
你只能做审核和建议，不能生成训练图片，不能替代真实标签。

请严格输出 JSON，字段包括：
human_readable: boolean
estimated_text: string
blur_level: "low" | "medium" | "high"
distortion_level: "low" | "medium" | "high"
occlusion_level: "low" | "medium" | "high"
label_maybe_wrong: boolean
suggestion: "keep" | "review" | "exclude"
reason: string
"""


def review_samples(sample_size: int = 20, seed: int = 42) -> pd.DataFrame:
    ensure_project_dirs()
    metadata = pd.read_csv(ALL_CAPTCHA_CSV, dtype={"label": str})
    rows = metadata.to_dict("records")
    random.Random(seed).shuffle(rows)
    selected = rows[:sample_size]
    results = []

    for row in selected:
        image_path = PROJECT_ROOT / row["path"]
        prompt = (
            QUALITY_PROMPT
            + f"\n当前真实标签：{str(row['label']).zfill(4)}\n"
            + f"增强参数：{row.get('augment_params', '')}\n"
        )
        result = call_vision_json(image_path, prompt)
        result.update(
            {
                "captcha_id": row["captcha_id"],
                "label": str(row["label"]).zfill(4),
                "path": row["path"],
            }
        )
        results.append(result)

    reviews = pd.DataFrame(results)
    reviews.to_csv(QUALITY_CSV, index=False, encoding="utf-8")
    (REPORTS_DIR / "llm_quality_review.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return reviews


def main() -> None:
    parser = argparse.ArgumentParser(description="Review captcha image quality with optional multimodal LLM.")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reviews = review_samples(args.sample_size, args.seed)
    print(reviews.head().to_string())
    print(f"Saved reviews: {QUALITY_CSV}")


if __name__ == "__main__":
    main()
