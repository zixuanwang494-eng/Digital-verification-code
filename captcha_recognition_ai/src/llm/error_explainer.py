import argparse
import json

import pandas as pd

from src.config import PROJECT_ROOT, REPORTS_DIR, ensure_project_dirs
from src.llm.client import call_vision_json


ERROR_PROMPT = """你是验证码识别错误分析助手。
请根据图片、真实标签、模型预测和增强参数，分析模型为什么可能识别错误。
你只能解释错误和提出改进建议，不能改写真实标签，不能生成训练图片。

请严格输出 JSON，字段包括：
likely_reason: string
confusable_digits: array
blur_issue: boolean
distortion_issue: boolean
occlusion_issue: boolean
label_maybe_wrong: boolean
data_suggestion: string
model_suggestion: string
"""


def explain_errors(limit: int = 20) -> list[dict]:
    ensure_project_dirs()
    wrong_path = REPORTS_DIR / "wrong_predictions.csv"
    if not wrong_path.exists():
        raise FileNotFoundError("Run evaluation first to generate reports/wrong_predictions.csv")

    wrong = pd.read_csv(wrong_path, dtype={"label": str, "true_label": str, "predicted_label": str})
    results = []

    for row in wrong.head(limit).to_dict("records"):
        image_path = PROJECT_ROOT / row["path"]
        prompt = (
            ERROR_PROMPT
            + f"\n真实标签：{str(row.get('true_label', row.get('label', ''))).zfill(4)}\n"
            + f"模型预测：{str(row.get('predicted_label', '')).zfill(4)}\n"
            + f"增强参数：{row.get('augment_params', '')}\n"
        )
        result = call_vision_json(image_path, prompt)
        result.update(
            {
                "captcha_id": row.get("captcha_id"),
                "true_label": str(row.get("true_label", row.get("label", ""))).zfill(4),
                "predicted_label": str(row.get("predicted_label", "")).zfill(4),
                "path": row.get("path"),
            }
        )
        results.append(result)

    json_path = REPORTS_DIR / "llm_error_explanations.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = REPORTS_DIR / "error_analysis.md"
    lines = ["# 错误样本分析报告", ""]
    for item in results:
        lines.append(f"## {item.get('captcha_id')}")
        lines.append("")
        lines.append(f"- 真实标签：`{item.get('true_label')}`")
        lines.append(f"- 模型预测：`{item.get('predicted_label')}`")
        lines.append(f"- 图片路径：`{item.get('path')}`")
        lines.append(f"- 可能原因：{item.get('likely_reason', item.get('note', '未启用大模型或无解释'))}")
        lines.append(f"- 数据建议：{item.get('data_suggestion', '人工复核')}")
        lines.append(f"- 模型建议：{item.get('model_suggestion', '继续收集错误样本并增强训练')}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain wrong predictions with optional multimodal LLM.")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    results = explain_errors(args.limit)
    print(f"Explained {len(results)} wrong samples.")
    print(f"Saved: {REPORTS_DIR / 'error_analysis.md'}")


if __name__ == "__main__":
    main()
