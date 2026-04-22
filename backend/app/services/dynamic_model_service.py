from __future__ import annotations

from collections import defaultdict, deque
from io import BytesIO
import json
from pathlib import Path
from typing import Any
import urllib.request

import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision



BASE_DIR = Path(__file__).resolve().parents[3]
DYNAMIC_MODEL_PATH = BASE_DIR / "new_dynamic" / "models" / "best_model.pth"
DYNAMIC_LABELS_PATH = BASE_DIR / "new_dynamic" / "labels.json"
DYNAMIC_CONFIG_PATH = BASE_DIR / "new_dynamic" / "models" / "training_config.json"
HAND_LANDMARKER_PATH = BASE_DIR / "MPR_STATIC_M" / "hand_landmarker.task"
POSE_LANDMARKER_PATH = BASE_DIR / "backend" / ".model_cache" / "pose_landmarker_lite.task"
POSE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

_DEFAULT_DYNAMIC_CLASSES = [
    "bye",
    "come",
    "drink",
    "eat",
    "give",
    "go",
    "hello",
    "help",
    "home",
    "how",
    "i",
    "ily",
    "need",
    "no",
    "please",
    "school",
    "sorry",
    "take",
    "thank_you",
    "they",
    "today",
    "want",
    "water",
    "we",
    "what",
    "where",
    "why",
    "work",
    "yes",
    "you",
]


class DynamicInvalidImageError(ValueError):
    pass


class DynamicModelInitializationError(RuntimeError):
    pass


class _SignLanguageLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        num_categories: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.class_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )
        self.category_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_categories),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        last_timestep = self.norm(out[:, -1, :])
        return self.class_head(last_timestep), self.category_head(last_timestep)


class _DynamicInferenceService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: _SignLanguageLSTM | None = None
        self.hands_detector: Any = None
        self.pose_detector: Any = None
        self.classes: list[str] = []
        self.sequence_length = 30
        self.feature_size = 225
        # Match new_dynamic/realtime.py defaults for smoother, stricter output.
        self.confidence_threshold = 0.70
        self._checkpoint_source: str | None = None
        self._is_trained_weights = False
        self._sequence_buffers: dict[str, deque[np.ndarray]] = {}
        self._prediction_buffers: dict[str, deque[tuple[int, float]]] = {}
        self._prediction_window = 5
        self._model_config: dict[str, Any] = {}
        self._label_map: dict[str, int] = {}
        self._session_timestamps_ms: dict[str, int] = {}
        self._no_hand_streak: dict[str, int] = {}
        self._max_no_hand_gap = 8

    def initialize(self) -> None:
        if self.model is not None and self.hands_detector is not None:
            return

        if not DYNAMIC_MODEL_PATH.exists():
            raise DynamicModelInitializationError(f"Dynamic model checkpoint not found: {DYNAMIC_MODEL_PATH}")

        try:
            self._model_config = self._load_model_config()
            self.classes, self._label_map = self._load_labels()
            self.sequence_length = int(self._model_config.get("sequence_length", self.sequence_length))
            self.feature_size = int(self._model_config.get("input_size", self.feature_size))

            state_dict = self._load_state_dict(DYNAMIC_MODEL_PATH)
            self.model = self._build_model_from_state_dict(state_dict)
            self.model.eval()
            self.hands_detector, self.pose_detector = self._create_detectors()
            self._checkpoint_source = str(DYNAMIC_MODEL_PATH)
            self._is_trained_weights = True
        except Exception as exc:
            self.shutdown()
            raise DynamicModelInitializationError(str(exc)) from exc

    def shutdown(self) -> None:
        if self.hands_detector is not None:
            try:
                self.hands_detector.close()
            except Exception:
                pass
        if self.pose_detector is not None:
            try:
                self.pose_detector.close()
            except Exception:
                pass
        self.hands_detector = None
        self.pose_detector = None
        self.model = None
        self.classes = []
        self._checkpoint_source = None
        self._is_trained_weights = False
        self._sequence_buffers.clear()
        self._prediction_buffers.clear()
        self._session_timestamps_ms.clear()
        self._no_hand_streak.clear()
        self._model_config = {}
        self._label_map = {}

    def status(self) -> dict[str, object]:
        source = Path(self._checkpoint_source).name if self._checkpoint_source else None
        return {
            "model_loaded": self.model is not None and self.hands_detector is not None,
            "is_trained_weights": self._is_trained_weights,
            "checkpoint_source": source,
            "num_classes": len(self.classes),
            "classes": list(self.classes),
            "device": str(self.device),
        }

    def reset_session(self, session_id: str) -> None:
        self._sequence_buffers.pop(session_id, None)
        self._prediction_buffers.pop(session_id, None)
        self._session_timestamps_ms.pop(session_id, None)
        self._no_hand_streak.pop(session_id, None)

    def predict(self, image_bytes: bytes, session_id: str) -> dict[str, object]:
        if self.model is None or self.hands_detector is None:
            raise DynamicModelInitializationError("Dynamic model service is not initialized.")

        image_rgb = self._decode_image(image_bytes)
        keypoints, hand_detected = self._extract_landmarks(image_rgb, session_id=session_id)

        seq_buffer = self._sequence_buffers.setdefault(session_id, deque(maxlen=self.sequence_length))
        pred_buffer = self._prediction_buffers.setdefault(session_id, deque(maxlen=self._prediction_window))

        # Ignore no-hand frames and clear stale context after a gap so
        # old gesture fragments do not leak into the next prediction.
        if hand_detected:
            self._no_hand_streak[session_id] = 0
            seq_buffer.append(keypoints)
        else:
            no_hand_streak = self._no_hand_streak.get(session_id, 0) + 1
            self._no_hand_streak[session_id] = no_hand_streak
            if no_hand_streak >= self._max_no_hand_gap:
                seq_buffer.clear()
                pred_buffer.clear()

        frames_collected = len(seq_buffer)
        if frames_collected < self.sequence_length:
            return {
                "ready": False,
                "frames_collected": frames_collected,
                "frames_required": self.sequence_length,
                "prediction": None,
                "confidence": None,
                "hand_detected": hand_detected,
            }

        with torch.no_grad():
            seq = np.stack(seq_buffer, axis=0)
            x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            class_logits, _ = self.model(x)
            probs = F.softmax(class_logits, dim=1).squeeze(0)
            conf, idx = torch.max(probs, dim=0)

        pred_idx = int(idx.item())
        confidence = float(conf.item())
        pred_buffer.append((pred_idx, confidence))
        stable_idx, smoothed_conf = self._smooth_prediction(pred_buffer)

        prediction = self.classes[stable_idx] if stable_idx is not None and stable_idx < len(self.classes) else None
        output_confidence = smoothed_conf if stable_idx is not None else None

        return {
            "ready": True,
            "frames_collected": self.sequence_length,
            "frames_required": self.sequence_length,
            "prediction": prediction,
            "confidence": output_confidence,
            "hand_detected": hand_detected,
        }

    def _smooth_prediction(self, buffer: deque[tuple[int, float]]) -> tuple[int | None, float]:
        if not buffer:
            return None, 0.0

        label_counts: dict[int, int] = defaultdict(int)
        confidences: list[float] = []
        for label_idx, conf in buffer:
            label_counts[label_idx] += 1
            confidences.append(conf)

        most_common_idx = max(label_counts, key=label_counts.get)
        count = label_counts[most_common_idx]
        avg_confidence = float(np.mean(confidences))

        # Match new_dynamic/inference/smoothing.py majority behavior.
        if count > self._prediction_window // 2 and avg_confidence >= self.confidence_threshold:
            return most_common_idx, avg_confidence
        return None, 0.0

    def _extract_landmarks(self, image_rgb: np.ndarray, session_id: str) -> tuple[np.ndarray, bool]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_ms = self._next_timestamp_ms(session_id)
        hand_results = self.hands_detector.detect_for_video(mp_image, timestamp_ms)
        pose_results = self.pose_detector.detect_for_video(mp_image, timestamp_ms) if self.pose_detector is not None else None

        both_hands = self._extract_keypoints_both_hands(hand_results)
        hand1 = self._normalize_keypoints(both_hands[:63])
        hand2 = self._normalize_keypoints(both_hands[63:])
        hands_normalized = np.concatenate([hand1, hand2], axis=0)

        pose_kp = self._extract_pose_keypoints(pose_results)
        all_keypoints = np.concatenate([hands_normalized, pose_kp], axis=0).astype(np.float32)
        if all_keypoints.shape[0] != self.feature_size:
            raise DynamicModelInitializationError(
                f"Feature size mismatch. Expected {self.feature_size}, got {all_keypoints.shape[0]}."
            )

        hand_detected = bool(getattr(hand_results, "hand_landmarks", None))
        return all_keypoints, hand_detected

    @staticmethod
    def _extract_pose_keypoints(pose_results: Any) -> np.ndarray:
        if pose_results and getattr(pose_results, "pose_landmarks", None):
            first_pose = pose_results.pose_landmarks[0] if pose_results.pose_landmarks else None
            if not first_pose:
                return np.zeros(99, dtype=np.float32)
            return np.array(
                [[lm.x, lm.y, lm.z] for lm in first_pose],
                dtype=np.float32,
            ).flatten()
        return np.zeros(99, dtype=np.float32)

    @staticmethod
    def _extract_keypoints_both_hands(hand_results: Any) -> np.ndarray:
        right_hand = np.zeros(63, dtype=np.float32)
        left_hand = np.zeros(63, dtype=np.float32)

        result_landmarks = getattr(hand_results, "hand_landmarks", None)
        result_handedness = getattr(hand_results, "handedness", None)
        if hand_results and result_landmarks and result_handedness:
            for hand_landmarks, handedness in zip(result_landmarks, result_handedness):
                label = handedness[0].category_name if handedness else "Left"
                kp = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32).flatten()
                if label == "Right":
                    right_hand = kp
                else:
                    left_hand = kp

        return np.concatenate([right_hand, left_hand], axis=0)

    @staticmethod
    def _normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
        # Mirror new_dynamic/utils.py normalize_keypoints.
        if np.all(keypoints == 0):
            return keypoints

        kp = keypoints.reshape(21, 3)
        wrist = kp[0].copy()
        kp -= wrist
        max_val = np.max(np.abs(kp))
        if max_val > 0:
            kp /= max_val
        return kp.flatten()

    @staticmethod
    def _create_detectors() -> tuple[Any, Any]:
        if not HAND_LANDMARKER_PATH.exists():
            raise DynamicModelInitializationError(f"Hand Landmarker asset not found: {HAND_LANDMARKER_PATH}")
        _ensure_pose_landmarker_asset()

        hands_options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH)),
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        pose_options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(POSE_LANDMARKER_PATH)),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        hands = mp_vision.HandLandmarker.create_from_options(hands_options)
        pose = mp_vision.PoseLandmarker.create_from_options(pose_options)
        return hands, pose

    def _next_timestamp_ms(self, session_id: str) -> int:
        current = self._session_timestamps_ms.get(session_id)
        if current is None:
            current = 0
        else:
            current += 33
        self._session_timestamps_ms[session_id] = current
        return current

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image = image.convert("RGB")
                rgb_array = np.array(image, dtype=np.uint8)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise DynamicInvalidImageError("Invalid image file.") from exc
        return rgb_array

    @staticmethod
    def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
        loaded = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

        def normalize_keys(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            # Handle checkpoints saved via DataParallel: "module.<name>".
            normalized: dict[str, torch.Tensor] = {}
            for key, value in raw.items():
                new_key = key[7:] if key.startswith("module.") else key
                normalized[new_key] = value
            return normalized

        if isinstance(loaded, dict):
            for key in ("model_state", "state_dict", "model_state_dict", "net", "weights"):
                nested = loaded.get(key)
                if isinstance(nested, dict) and nested:
                    tensors = {k: v for k, v in nested.items() if isinstance(v, torch.Tensor)}
                    if tensors:
                        return normalize_keys(tensors)
            tensors = {k: v for k, v in loaded.items() if isinstance(v, torch.Tensor)}
            if tensors:
                return normalize_keys(tensors)
        raise DynamicModelInitializationError("Invalid dynamic checkpoint format.")

    def _load_model_config(self) -> dict[str, Any]:
        if DYNAMIC_CONFIG_PATH.exists():
            with DYNAMIC_CONFIG_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        return {}

    def _load_labels(self) -> tuple[list[str], dict[str, int]]:
        if DYNAMIC_LABELS_PATH.exists():
            with DYNAMIC_LABELS_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                classes = data.get("classes")
                label_map = data.get("label_map")
                if isinstance(classes, list) and isinstance(label_map, dict):
                    classes_str = [str(c) for c in classes]
                    label_map_int = {str(k): int(v) for k, v in label_map.items()}
                    return classes_str, label_map_int

        classes = self._model_config.get("classes")
        if isinstance(classes, list) and classes:
            classes_str = [str(c) for c in classes]
            label_map = {cls: idx for idx, cls in enumerate(classes_str)}
            return classes_str, label_map

        return self._resolve_classes(int(self._model_config.get("num_classes", 0) or 0)), {}

    def _build_model_from_state_dict(self, state_dict: dict[str, torch.Tensor]) -> _SignLanguageLSTM:
        input_size = int(self._model_config.get("input_size", state_dict["lstm.weight_ih_l0"].shape[1]))
        hidden_size = int(self._model_config.get("hidden_size", state_dict["lstm.weight_hh_l0"].shape[1]))
        num_layers = int(self._model_config.get("num_layers", len([k for k in state_dict if k.startswith("lstm.weight_ih_l")])))

        class_head_weight = state_dict.get("class_head.3.weight")
        if class_head_weight is None:
            class_head_weight = state_dict.get("classifier.4.weight")
        if class_head_weight is None:
            raise DynamicModelInitializationError("Could not infer dynamic class head size from checkpoint.")
        num_classes = int(self._model_config.get("num_classes", class_head_weight.shape[0]))

        category_head_weight = state_dict.get("category_head.3.weight")
        inferred_categories = category_head_weight.shape[0] if category_head_weight is not None else 5
        num_categories = int(self._model_config.get("num_categories", inferred_categories))
        dropout = 0.0

        if not self.classes:
            self.classes = self._resolve_classes(num_classes)
        model = _SignLanguageLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            num_categories=num_categories,
            dropout=dropout,
        ).to(self.device)
        model.load_state_dict(state_dict, strict=True)
        return model

    @staticmethod
    def _resolve_classes(num_classes: int) -> list[str]:
        if num_classes <= len(_DEFAULT_DYNAMIC_CLASSES):
            return _DEFAULT_DYNAMIC_CLASSES[:num_classes]
        extras = [f"class_{i}" for i in range(len(_DEFAULT_DYNAMIC_CLASSES), num_classes)]
        return _DEFAULT_DYNAMIC_CLASSES + extras


dynamic_inference_service = _DynamicInferenceService()


def initialize_dynamic_model_service() -> None:
    dynamic_inference_service.initialize()


def shutdown_dynamic_model_service() -> None:
    dynamic_inference_service.shutdown()


def get_dynamic_model_status() -> dict[str, object]:
    if dynamic_inference_service.model is None or dynamic_inference_service.hands_detector is None:
        try:
            dynamic_inference_service.initialize()
        except Exception:
            pass
    return dynamic_inference_service.status()


def predict_dynamic_from_image(image_bytes: bytes, session_id: str = "default") -> dict[str, object]:
    if dynamic_inference_service.model is None or dynamic_inference_service.hands_detector is None:
        dynamic_inference_service.initialize()
    return dynamic_inference_service.predict(image_bytes, session_id=session_id)


def reset_dynamic_session(session_id: str) -> None:
    dynamic_inference_service.reset_session(session_id)


def _ensure_pose_landmarker_asset() -> None:
    if POSE_LANDMARKER_PATH.exists():
        return
    POSE_LANDMARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(POSE_LANDMARKER_URL, str(POSE_LANDMARKER_PATH))
    except Exception as exc:
        raise DynamicModelInitializationError(
            f"Pose Landmarker asset not found and download failed: {POSE_LANDMARKER_PATH}"
        ) from exc
