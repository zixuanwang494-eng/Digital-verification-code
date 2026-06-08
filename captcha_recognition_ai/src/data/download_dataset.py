"""Dataset acquisition entry point.

The first project version uses scikit-learn's bundled real handwritten digits
dataset, so no network download is required. This module keeps the command name
explicit for future real dataset sources.
"""

from src.data.prepare_digits import main


if __name__ == "__main__":
    main()
