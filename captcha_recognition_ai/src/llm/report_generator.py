import argparse
import json
from pathlib import Path

from src.config import REPORTS_DIR, ensure_project_dirs
from src.llm.client import call_text_markdown


def _read_text(path: Path, fallback: str = "") -> str:
    return path.read_text(encoding="utf-8") if path.exists() else fallback


def generate_report() -> Path:
    ensure_project_dirs()
    metrics = _read_text(REPORTS_DIR / "metrics.json", "{}")
    history = _read_text(REPORTS_DIR / "training_history.json", "{}")
    error_analysis = _read_text(REPORTS_DIR / "error_analysis.md", "暂无错误样本解释。")
    quality_review = _read_text(REPORTS_DIR / "llm_quality_review.json", "[]")

    prompt = f"""请根据以下材料生成一份中文 Markdown 训练报告。
报告需要包含：训练目标、数据来源、数据增强、模型结构、训练参数、评估指标、错误分析、数据质量审核结果、下一步优化建议。
注意：大模型只负责生成报告和分析，不得声称生成了训练图片。

训练历史 JSON:
{history}

评估指标 JSON:
{metrics}

错误分析:
{error_analysis}

质量审核 JSON:
{quality_review}
"""
    report = call_text_markdown(prompt)
    if not report.strip().startswith("#"):
        report = "# 四位验证码识别训练报告\n\n" + report

    output = REPORTS_DIR / "training_report.md"
    output.write_text(report, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Markdown training report with optional LLM.")
    parser.parse_args()
    output = generate_report()
    print(f"Saved report: {output}")


if __name__ == "__main__":
    main()
