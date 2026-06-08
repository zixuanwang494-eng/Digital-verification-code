"""Single digit training entry point placeholder.

The upgraded project focuses on four-digit captcha recognition first. This
module exists to keep the original project's single-digit CNN as an explicit
extension point for segmentation-based workflows.
"""

from src.models.single_digit_cnn import build_single_digit_cnn


def main() -> None:
    model = build_single_digit_cnn()
    model.summary()
    print("Single-digit model definition is ready. Dataset-specific training can be added here.")


if __name__ == "__main__":
    main()
