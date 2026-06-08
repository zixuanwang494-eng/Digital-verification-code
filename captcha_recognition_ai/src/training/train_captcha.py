import argparse
import json
from pathlib import Path

from tensorflow import keras

from src.config import MODELS_DIR, REPORTS_DIR, TRAIN_CSV, VAL_CSV, ensure_project_dirs
from src.models.captcha_cnn import build_captcha_cnn
from src.training.data_loader import load_images_and_labels


def train(epochs: int = 10, batch_size: int = 64, output: Path | None = None) -> Path:
    ensure_project_dirs()
    output = output or MODELS_DIR / "captcha_cnn.keras"

    x_train, y_train, _ = load_images_and_labels(TRAIN_CSV)
    x_val, y_val, _ = load_images_and_labels(VAL_CSV)

    model = build_captcha_cnn()
    callbacks = [
        keras.callbacks.ModelCheckpoint(output, monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )
    model.save(output)

    history_path = REPORTS_DIR / "training_history.json"
    history_path.write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train four-output captcha CNN.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model_path = train(args.epochs, args.batch_size, args.output)
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
