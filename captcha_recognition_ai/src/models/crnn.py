"""Placeholder module for a future CNN + sequence decoder model.

The initial implementation uses a four-output CNN because captchas have a fixed
length of four digits. This file documents the planned extension point for CRNN
or CTC-based sequence recognition when variable-length captchas are introduced.
"""


def build_crnn():
    raise NotImplementedError("CRNN support is planned after the fixed-length CNN baseline is validated.")
