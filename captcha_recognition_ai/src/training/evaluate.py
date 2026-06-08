import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow import keras

from src.config import CAPTCHA_LENGTH, MODELS_DIR, REPORTS_DIR, TEST_CSV, ensure_project_dirs
from src.training.data_loader import load_images_and_labels


def _decode_predictions(predictions: list[np.ndarray]) -> list[str]:
    digits = [np.argmax(output, axis=1).astype(str) for output in predictions]
    return ["".join(chars) for chars in zip(*digits)]


def _true_labels(metadata: pd.DataFrame) -> list[str]:
    return [str(label).zfill(CAPTCHA_LENGTH) for label in metadata["label"].astype(str)]


def _save_confusion_matrix(true_labels: list[str], pred_labels: list[str]) -> None:
    y_true = [int(ch) for label in true_labels for ch in label]
    y_pred = [int(ch) for label in pred_labels for ch in label]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title("Digit Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def evaluate(model_path: Path | None = None, manifest_path: Path = TEST_CSV) -> dict:
    ensure_project_dirs()
    model_path = model_path or MODELS_DIR / "captcha_cnn.keras"
    model = keras.models.load_model(model_path)

    x_test, _, metadata = load_images_and_labels(manifest_path)
    start = time.perf_counter()
    predictions = model.predict(x_test)
    elapsed = time.perf_counter() - start

    pred_labels = _decode_predictions(predictions)
    true_labels = _true_labels(metadata)

    exact = [truth == pred for truth, pred in zip(true_labels, pred_labels)]
    position_acc = []
    for pos in range(CAPTCHA_LENGTH):
        correct = sum(t[pos] == p[pos] for t, p in zip(true_labels, pred_labels))
        position_acc.append(correct / len(true_labels))

    char_total = len(true_labels) * CAPTCHA_LENGTH
    char_correct = sum(
        t[pos] == p[pos]
        for t, p in zip(true_labels, pred_labels)
        for pos in range(CAPTCHA_LENGTH)
    )

    metrics = {
        "sample_count": len(true_labels),
        "captcha_exact_accuracy": float(np.mean(exact)),
        "character_accuracy": char_correct / char_total,
        "position_accuracy": {f"digit_{idx + 1}": acc for idx, acc in enumerate(position_acc)},
        "inference_seconds_total": elapsed,
        "inference_seconds_per_sample": elapsed / max(1, len(true_labels)),
    }

    wrong_rows = []
    for row, truth, pred, is_exact in zip(metadata.to_dict("records"), true_labels, pred_labels, exact):
        if not is_exact:
            row["true_label"] = truth
            row["predicted_label"] = pred
            wrong_rows.append(row)

    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(wrong_rows).to_csv(REPORTS_DIR / "wrong_predictions.csv", index=False, encoding="utf-8")
    _save_confusion_matrix(true_labels, pred_labels)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate captcha model.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=TEST_CSV)
    args = parser.parse_args()

    metrics = evaluate(args.model, args.manifest)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
