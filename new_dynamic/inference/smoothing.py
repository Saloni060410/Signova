"""
inference/smoothing.py - Prediction smoothing for stable real-time output.

Uses a deque-based rolling window with majority voting to suppress
flickering predictions and enforce a confidence threshold.
"""

import collections
from typing import Optional, Tuple

import numpy as np


class PredictionSmoother:
    """
    Smooth predictions over a rolling window using majority voting.

    A prediction is considered "stable" when:
      1. The same class wins the majority vote (> half the window)
      2. The average confidence over the window is >= confidence_threshold

    Args:
        window_size:          Number of recent frames to consider (default 10).
        confidence_threshold: Minimum mean confidence to emit a result (default 0.7).
    """

    def __init__(self, window_size: int = 10, confidence_threshold: float = 0.7):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self._labels: collections.deque = collections.deque(maxlen=window_size)
        self._confidences: collections.deque = collections.deque(maxlen=window_size)

    # ------------------------------------------------------------------
    def update(self, label: int, confidence: float):
        """Push a new raw prediction into the buffer."""
        self._labels.append(label)
        self._confidences.append(confidence)

    # ------------------------------------------------------------------
    def get_stable_prediction(self) -> Tuple[Optional[int], float]:
        """
        Return the dominant label and its mean confidence if stable,
        otherwise (None, 0.0).
        """
        if not self._labels:
            return None, 0.0

        label_counts = collections.Counter(self._labels)
        most_common_label, count = label_counts.most_common(1)[0]
        avg_confidence = float(np.mean(self._confidences))

        # Majority vote + confidence gate
        if count > self.window_size // 2 and avg_confidence >= self.confidence_threshold:
            return most_common_label, avg_confidence

        return None, 0.0

    # ------------------------------------------------------------------
    def reset(self):
        """Clear both queues."""
        self._labels.clear()
        self._confidences.clear()

    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        """True once the window is fully populated."""
        return len(self._labels) == self.window_size
