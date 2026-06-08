from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIGITS_DIR = DATA_DIR / "raw_digits"
CAPTCHA_DIR = DATA_DIR / "captcha_images"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"
ALL_CAPTCHA_CSV = DATA_DIR / "captcha_labels.csv"
RAW_DIGITS_CSV = DATA_DIR / "raw_digits.csv"
QUALITY_CSV = DATA_DIR / "quality_reviews.csv"

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 160
CAPTCHA_LENGTH = 4
NUM_CLASSES = 10


def ensure_project_dirs() -> None:
    for path in [
        DATA_DIR,
        RAW_DIGITS_DIR,
        CAPTCHA_DIR,
        REPORTS_DIR,
        MODELS_DIR,
        DOCS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
