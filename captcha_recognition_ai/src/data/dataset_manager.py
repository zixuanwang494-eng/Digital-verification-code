import argparse
from pathlib import Path

import pandas as pd

from src.config import ALL_CAPTCHA_CSV, QUALITY_CSV, RAW_DIGITS_CSV, REPORTS_DIR, ensure_project_dirs


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, dtype={"label": str})


def stats() -> dict:
    ensure_project_dirs()
    captcha = _load(ALL_CAPTCHA_CSV)
    raw = _load(RAW_DIGITS_CSV)

    digit_counts = {str(i): 0 for i in range(10)}
    for label in captcha["label"].astype(str).str.zfill(4):
        for char in label:
            digit_counts[char] += 1

    summary = {
        "raw_digit_count": int(len(raw)),
        "captcha_count": int(len(captcha)),
        "raw_digits_by_label": raw["label"].value_counts().sort_index().to_dict(),
        "captcha_digit_occurrences": digit_counts,
        "quality_status": captcha["quality_status"].value_counts().to_dict()
        if "quality_status" in captcha.columns
        else {},
    }
    return summary


def write_stats_report() -> Path:
    ensure_project_dirs()
    summary = stats()
    path = REPORTS_DIR / "dataset_quality_report.md"
    lines = [
        "# 数据集质量统计报告",
        "",
        f"- 原始单数字图片数量：{summary['raw_digit_count']}",
        f"- 四位验证码图片数量：{summary['captcha_count']}",
        "",
        "## 原始数字类别分布",
        "",
    ]
    for label, count in summary["raw_digits_by_label"].items():
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## 验证码数字出现次数", ""])
    for label, count in summary["captcha_digit_occurrences"].items():
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## 样本质量状态", ""])
    for status, count in summary["quality_status"].items():
        lines.append(f"- {status}: {count}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def mark_quality(captcha_id: str, status: str, note: str = "") -> None:
    ensure_project_dirs()
    captcha = _load(ALL_CAPTCHA_CSV)
    if captcha_id not in set(captcha["captcha_id"]):
        raise ValueError(f"Unknown captcha_id: {captcha_id}")

    captcha.loc[captcha["captcha_id"] == captcha_id, "quality_status"] = status
    captcha.to_csv(ALL_CAPTCHA_CSV, index=False, encoding="utf-8")

    review_row = pd.DataFrame(
        [{"captcha_id": captcha_id, "quality_status": status, "note": note}]
    )
    if QUALITY_CSV.exists():
        reviews = pd.concat([pd.read_csv(QUALITY_CSV), review_row], ignore_index=True)
    else:
        reviews = review_row
    reviews.to_csv(QUALITY_CSV, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage captcha dataset metadata.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("stats")
    subparsers.add_parser("report")

    mark_parser = subparsers.add_parser("mark")
    mark_parser.add_argument("captcha_id")
    mark_parser.add_argument("status", choices=["normal", "suspicious", "too_blurry", "too_distorted", "label_suspect", "excluded"])
    mark_parser.add_argument("--note", default="")

    args = parser.parse_args()
    if args.command == "stats":
        print(stats())
    elif args.command == "report":
        print(write_stats_report())
    elif args.command == "mark":
        mark_quality(args.captcha_id, args.status, args.note)
        print(f"Marked {args.captcha_id} as {args.status}")


if __name__ == "__main__":
    main()
