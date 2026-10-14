"""NumPy-only H&E tissue-mask calculations."""

from __future__ import annotations

import numpy as np


MASK_METHOD_VERSION = "color_darkness_chroma_otsu_v4"
DARK_NEUTRAL_VALUE_MAX = 32
DARK_NEUTRAL_CHROMA_MAX = 16
BRIGHT_NEUTRAL_VALUE_MIN = 220
BRIGHT_NEUTRAL_CHROMA_MAX = 16


def otsu_gray_threshold(image: np.ndarray) -> int:
    """Return an Otsu threshold for a uint8 scalar image."""
    array = np.clip(np.asarray(image), 0, 255).astype(np.uint8, copy=False)
    histogram = np.bincount(array.ravel(), minlength=256).astype(float)
    total = histogram.sum()
    if total <= 0:
        raise ValueError("Cannot estimate an Otsu threshold from an empty image")
    probability = histogram / total
    cumulative_probability = np.cumsum(probability)
    cumulative_mean = np.cumsum(probability * np.arange(256, dtype=float))
    global_mean = cumulative_mean[-1]
    denominator = cumulative_probability * (1.0 - cumulative_probability)
    between_class_variance = np.zeros(256, dtype=float)
    valid = denominator > 0
    between_class_variance[valid] = (
        global_mean * cumulative_probability[valid] - cumulative_mean[valid]
    ) ** 2 / denominator[valid]
    return int(np.argmax(between_class_variance))


def color_tissue_mask_rgb(
    image: np.ndarray,
    *,
    dark_neutral_value_max: int = DARK_NEUTRAL_VALUE_MAX,
    dark_neutral_chroma_max: int = DARK_NEUTRAL_CHROMA_MAX,
    bright_neutral_value_min: int = BRIGHT_NEUTRAL_VALUE_MIN,
    bright_neutral_chroma_max: int = BRIGHT_NEUTRAL_CHROMA_MAX,
) -> tuple[np.ndarray, int, int]:
    """Classify uint8 RGB pixels using adaptive darkness and chroma scores."""
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("color_tissue_mask_rgb expects an H x W x 3 uint8 array")

    rgb_int = rgb.astype(np.int16)
    channel_min = rgb_int.min(axis=2)
    channel_max = rgb_int.max(axis=2)
    chroma = channel_max - channel_min
    darkness = (255 - channel_min).astype(np.uint8)

    darkness_threshold = otsu_gray_threshold(darkness)
    chroma_threshold = otsu_gray_threshold(chroma)
    dark_neutral = (
        (channel_max <= int(dark_neutral_value_max))
        & (chroma <= int(dark_neutral_chroma_max))
    )
    bright_neutral = (
        (channel_min >= int(bright_neutral_value_min))
        & (chroma <= int(bright_neutral_chroma_max))
    )
    tissue = (
        (
            (darkness >= max(int(darkness_threshold), 1))
            | (chroma >= max(int(chroma_threshold), 1))
        )
        & ~dark_neutral
        & ~bright_neutral
    )
    return tissue, darkness_threshold, chroma_threshold
