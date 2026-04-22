"""
inference/utils.py - Utilities for real-time inference.

Handles model loading, prediction smoothing, and FPS tracking.
"""

import os
import json
import time
import collections
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


# ─── Label Loading ────────────────────────────────────────────────────────────

def load_labels(labels_json_path: str) -> Tuple[List[str], dict]:
    """
    Load class names and label map from the exported JSON file.

    Args:
        labels_json_path: Path to labels.json.

    Returns:
        classes: List of class name strings.
        label_map: Dict mapping class name → index.
    """
    with open(labels_json_path, "r") as f:
        data = json.load(f)
    return data["classes"], data["label_map"]


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model(model_path: str, config_path: str, device: torch.device) -> nn.Module:
    """
    Load a trained SignLanguageLSTM from disk.

    Args:
        model_path: Path to .pth weights file.
        config_path: Path to training_config.json.
        device: torch.device to load model onto.

    Returns:
        Loaded model in eval mode.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import SignLanguageLSTM

    with open(config_path, "r") as f:
        config = json.load(f)

    model = SignLanguageLSTM(
        input_size=config.get("input_size", 63),
        hidden_size=config.get("hidden_size", 128),
        num_layers=config.get("num_layers", 2),
        num_classes=config["num_classes"],
        dropout=0.0,  # Disable dropout during inference
    ).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[Model] Loaded from {model_path}")
    print(f"[Model] Classes: {config['classes']}")
    return model


# ─── Prediction Smoothing ─────────────────────────────────────────────────────

class PredictionSmoother:
    """
    Smooth predictions over a rolling window to reduce flickering.

    Maintains a deque of recent predictions and returns the most
    common class with its average confidence.

    Args:
        window_size: Number of recent predictions to consider.
        confidence_threshold: Minimum confidence to report a prediction.
    """

    def __init__(self, window_size: int = 5, confidence_threshold: float = 0.8):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self._labels = collections.deque(maxlen=window_size)
        self._confidences = collections.deque(maxlen=window_size)

    def update(self, label: int, confidence: float):
        """Add a new prediction to the buffer."""
        self._labels.append(label)
        self._confidences.append(confidence)

    def get_stable_prediction(self) -> Tuple[Optional[int], float]:
        """
        Return the most frequent label in the window and its mean confidence,
        if above the threshold. Otherwise return (None, 0.0).

        Returns:
            (label_index, confidence) or (None, 0.0)
        """
        if not self._labels:
            return None, 0.0

        # Count occurrences of each label
        label_counts = collections.Counter(self._labels)
        most_common_label, count = label_counts.most_common(1)[0]
        avg_confidence = float(np.mean(self._confidences))

        # Require majority vote AND minimum confidence
        if count >= (self.window_size // 2 + 1) and avg_confidence >= self.confidence_threshold:
            return most_common_label, avg_confidence
        return None, 0.0

    def reset(self):
        """Clear the prediction buffer."""
        self._labels.clear()
        self._confidences.clear()


# ─── FPS Counter ──────────────────────────────────────────────────────────────

class FPSCounter:
    """
    Track frames per second using a rolling time window.

    Args:
        window: Number of recent frame timestamps to average over.
    """

    def __init__(self, window: int = 30):
        self._timestamps = collections.deque(maxlen=window)

    def tick(self):
        """Register a new frame."""
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        """Current rolling FPS."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# ─── Sequence Buffer ──────────────────────────────────────────────────────────

class SequenceBuffer:
    """
    Rolling buffer that accumulates keypoint frames until a full sequence
    is available for prediction.

    Args:
        sequence_length: Number of frames per sequence.
        feature_size: Number of features per frame.
    """

    def __init__(self, sequence_length: int = 30, feature_size: int = 225):
        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self._buffer = collections.deque(maxlen=sequence_length)

    def add_frame(self, keypoints: np.ndarray):
        """Add a single frame of keypoints to the buffer."""
        assert keypoints.shape == (self.feature_size,), \
            f"Expected ({self.feature_size},), got {keypoints.shape}"
        self._buffer.append(keypoints)

    @property
    def is_full(self) -> bool:
        """True when the buffer holds a complete sequence."""
        return len(self._buffer) == self.sequence_length

    def get_sequence(self) -> np.ndarray:
        """
        Return the buffered sequence as a numpy array.

        Returns:
            np.ndarray of shape (sequence_length, feature_size).
        """
        return np.array(self._buffer, dtype=np.float32)

    def reset(self):
        """Clear the buffer."""
        self._buffer.clear()


# ─── Inference Function ───────────────────────────────────────────────────────

@torch.no_grad()
def predict_sequence(
    model: nn.Module,
    sequence: np.ndarray,
    device: torch.device,
) -> Tuple[int, float, np.ndarray]:
    """
    Run inference on a single sequence.

    Args:
        model: Trained SignLanguageLSTM.
        sequence: np.ndarray of shape (seq_len, 63).
        device: Target device.

    Returns:
        predicted_class: Integer class index.
        confidence: Softmax confidence for the predicted class.
        probabilities: Full probability array of shape (num_classes,).
    """
    # Add batch dimension: (1, seq_len, 225)
    x = torch.tensor(sequence[np.newaxis], dtype=torch.float32, device=device)
    
    # model(x) now returns (class_logits, cat_logits)
    class_logits, _ = model(x)
    probs = torch.softmax(class_logits, dim=-1).squeeze(0).cpu().numpy()  # (num_classes,)

    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    return predicted_class, confidence, probs
