from __future__ import annotations

import ast
import base64
import json
import urllib.error
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = BASE_DIR / "MPR_STATIC_M"
LOCAL_INFERENCE_PATH = MODEL_DIR / "local_inference.py"
CLASSES_PATH = MODEL_DIR / "asl_classes.json"
LANDMARKER_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_PATH = MODEL_DIR / "asl_resnet50.pth"
MODEL_ARCHIVE_PATH = MODEL_DIR / "asl_resnet50.pth.zip"
MODEL_CACHE_DIR = BASE_DIR / "backend" / ".model_cache"
CACHED_LANDMARKER_PATH = MODEL_CACHE_DIR / "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
PLACEHOLDER_CHECKPOINT_PATH = MODEL_CACHE_DIR / "asl_resnet50_placeholder.pt"
FLATTENED_CHECKPOINT_PATH = MODEL_CACHE_DIR / "asl_resnet50_flattened.pt"
FLATTENED_CHECKPOINT_META = MODEL_CACHE_DIR / "asl_resnet50_flattened.meta.json"
# Increment when the flattening layout changes so caches are rebuilt.
_FLATTEN_LAYOUT_VERSION = 2


class InvalidImageError(ValueError):
    pass


class NoHandDetectedError(ValueError):
    pass


class ModelInitializationError(RuntimeError):
    pass


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class _ResNetBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(planes=64, blocks=3, stride=1)
        self.layer2 = self._make_layer(planes=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(planes=256, blocks=6, stride=2)
        self.layer4 = self._make_layer(planes=512, blocks=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.inplanes != planes * _Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * _Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * _Bottleneck.expansion),
            )

        layers: list[nn.Module] = []
        layers.append(_Bottleneck(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * _Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(_Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)


class _LandmarkResNet50(nn.Module):
    def __init__(self, num_classes: int, input_dim: int = 63) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        backbone = _ResNetBackbone()
        self.adapt = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, bias=False),
            backbone.bn1,
            backbone.relu,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.adapt(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ASLInferenceService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: torch.jit.ScriptModule | None = None
        self.classes: list[str] = []
        self.detector: Any = None
        self._local_namespace: dict[str, Any] | None = None
        self._checkpoint_source: str | None = None   # path of the loaded checkpoint
        self._is_trained_weights: bool = False        # False if placeholder was used

    def initialize(self) -> None:
        if self.model is not None and self.detector is not None and self._local_namespace is not None:
            return

        try:
            # Load immutable startup resources once and reuse them for every request.
            self._local_namespace = self._load_local_inference_namespace()
            self.classes = self._load_classes()
            self.model = self._load_model()
            self.detector = self._create_detector()
            self._local_namespace["device"] = self.device
            self._local_namespace["CLASSES"] = self.classes
            self._local_namespace["model"] = self.model
            self._local_namespace["detector"] = self.detector
        except Exception:
            self._reset_inference_state()
            raise

    def shutdown(self) -> None:
        if self.detector is not None:
            self.detector.close()
            self.detector = None

    def _reset_inference_state(self) -> None:
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass
            self.detector = None
        self.model = None
        self._local_namespace = None
        self.classes = []
        self._checkpoint_source = None
        self._is_trained_weights = False

    def predict_from_image(self, image_bytes: bytes) -> tuple[str, float]:
        if self.model is None or self.detector is None or self._local_namespace is None:
            raise ModelInitializationError("ASL inference service is not initialized.")

        # Decode the uploaded frame, extract landmarks with MediaPipe, then run
        # the original local inference helpers against the preloaded model.
        # predict_with_mirror also tries the X-mirrored landmark vector so that
        # both left-hand and right-hand signs are recognised correctly.
        image_bgr = self._decode_image(image_bytes)
        extract_landmarks = self._local_namespace["extract_landmarks"]
        # Use predict_with_mirror when available (supports both hands), else
        # fall back to the basic predict function.
        predict_fn = self._local_namespace.get(
            "predict_with_mirror",
            self._local_namespace["predict"],
        )

        landmarks = extract_landmarks(image_bgr)
        if landmarks is None:
            raise NoHandDetectedError("No hand detected in the uploaded image.")

        top_predictions = predict_fn(landmarks)
        label, confidence = top_predictions[0]
        return str(label), float(confidence)

    def predict_from_base64(self, encoded_image: str) -> tuple[str, float]:
        payload = encoded_image.split(",", 1)[1] if "," in encoded_image else encoded_image
        try:
            image_bytes = base64.b64decode(payload)
        except Exception as exc:
            raise InvalidImageError("Invalid base64 image payload.") from exc
        return self.predict_from_image(image_bytes)

    def _load_classes(self) -> list[str]:
        with CLASSES_PATH.open("r", encoding="utf-8") as file:
            classes = json.load(file)

        if not isinstance(classes, list) or not classes:
            raise ModelInitializationError("Class labels file must contain a non-empty list.")

        return [str(label) for label in classes]

    def _load_model(self) -> Any:
        errors: list[str] = []
        candidates_with_meta = self._model_checkpoint_candidates_with_meta()
        for model_candidate, is_trained in candidates_with_meta:
            try:
                ckpt = self._torch_load_checkpoint(model_candidate)
                state_dict = self._state_dict_from_checkpoint(ckpt)
                model = self._build_model()
                model.load_state_dict(state_dict, strict=True)
            except Exception as exc:
                errors.append(f"{model_candidate}: {exc}")
                continue

            model.eval()
            self._checkpoint_source = str(model_candidate)
            self._is_trained_weights = is_trained
            print(f"ASL model weights loaded from {model_candidate} (trained={is_trained})")
            return model

        raise ModelInitializationError(
            "Unable to load any ASL model checkpoint. Attempted:\n" + "\n".join(errors)
        )

    def _build_model(self) -> nn.Module:
        return _LandmarkResNet50(num_classes=len(self.classes)).to(self.device)

    def _torch_load_checkpoint(self, path: Path) -> Any:
        """torch.load with compatibility for PyTorch 2.6+ weights_only default."""
        kwargs: dict[str, Any] = {"map_location": self.device}
        try:
            return torch.load(str(path), weights_only=False, **kwargs)
        except TypeError:
            return torch.load(str(path), **kwargs)

    @staticmethod
    def _strip_module_prefix_on_keys(state: dict[str, Any]) -> dict[str, Any]:
        if not state or not any(k.startswith("module.") for k in state):
            return state
        if all(k.startswith("module.") for k in state):
            return {k[len("module.") :]: v for k, v in state.items()}
        return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state.items()}

    @staticmethod
    def _state_dict_from_checkpoint(ckpt: Any) -> dict[str, Any]:
        if not isinstance(ckpt, dict):
            raise ValueError("Checkpoint root must be a dict.")
        for key in ("model_state", "state_dict", "model_state_dict", "net", "weights"):
            candidate = ckpt.get(key)
            if isinstance(candidate, dict) and candidate:
                tensors = {k: v for k, v in candidate.items() if isinstance(v, torch.Tensor)}
                if tensors:
                    return ASLInferenceService._strip_module_prefix_on_keys(tensors)
        if ckpt:
            tensors = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
            if tensors:
                return ASLInferenceService._strip_module_prefix_on_keys(tensors)
        raise ValueError(
            "No tensor state dict found (expected model_state / state_dict / model_state_dict). "
            f"Root keys: {list(ckpt.keys())[:25]}"
        )

    @staticmethod
    def _find_nested_data_zip_member(names: list[str]) -> str | None:
        for raw in names:
            if raw.endswith("/"):
                continue
            normalized = raw.replace("\\", "/")
            if Path(normalized).name == "data.zip":
                return normalized
        return None

    def _ensure_flattened_zip_checkpoint(self, source_zip: Path) -> Path:
        """Rebuild a torch zip checkpoint so tensor blobs live beside data.pkl (fixes nested data.zip on Windows)."""
        if not source_zip.exists():
            raise ModelInitializationError(f"Checkpoint zip not found: {source_zip}")

        stat = source_zip.stat()
        meta: dict[str, Any] = {
            "source": str(source_zip.resolve()),
            "mtime_ns": getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)),
            "size": stat.st_size,
            "flatten_version": _FLATTEN_LAYOUT_VERSION,
        }
        if FLATTENED_CHECKPOINT_PATH.exists() and FLATTENED_CHECKPOINT_META.exists():
            try:
                cached = json.loads(FLATTENED_CHECKPOINT_META.read_text(encoding="utf-8"))
                if cached == meta:
                    return FLATTENED_CHECKPOINT_PATH
            except (OSError, json.JSONDecodeError):
                pass

        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = FLATTENED_CHECKPOINT_PATH.with_suffix(FLATTENED_CHECKPOINT_PATH.suffix + ".tmp")

        with zipfile.ZipFile(source_zip, "r") as outer:
            outer_names = outer.namelist()
            normalized_lookup = {n.replace("\\", "/"): n for n in outer_names}
            normalized_names = list(normalized_lookup.keys())
            data_zip_norm = self._find_nested_data_zip_member(normalized_names)
            if not data_zip_norm:
                raise ModelInitializationError(
                    f"No nested data.zip found inside {source_zip.name}; cannot flatten this archive."
                )
            data_zip_raw = normalized_lookup[data_zip_norm]
            parent_dir = str(Path(data_zip_norm).parent.as_posix())
            parent_prefix = "" if parent_dir in ("", ".") else parent_dir + "/"
            torch_root = Path(data_zip_norm).parent.name

            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as nz:
                for norm, raw in normalized_lookup.items():
                    if norm.endswith("/"):
                        continue
                    if norm == data_zip_norm:
                        continue
                    if parent_prefix and not norm.startswith(parent_prefix):
                        continue
                    rel = norm[len(parent_prefix) :] if parent_prefix else norm
                    nz.writestr(f"{torch_root}/{rel}", outer.read(raw))

                inner_bytes = outer.read(data_zip_raw)
                with zipfile.ZipFile(BytesIO(inner_bytes), "r") as inner_zip:
                    for inner_name in inner_zip.namelist():
                        if inner_name.endswith("/"):
                            continue
                        nz.writestr(f"{torch_root}/{inner_name.replace(chr(92), '/')}", inner_zip.read(inner_name))

        tmp_path.replace(FLATTENED_CHECKPOINT_PATH)
        FLATTENED_CHECKPOINT_META.write_text(json.dumps(meta), encoding="utf-8")
        print(f"Prepared flattened PyTorch checkpoint at {FLATTENED_CHECKPOINT_PATH} (from {source_zip.name}).")
        return FLATTENED_CHECKPOINT_PATH

    def _ensure_placeholder_checkpoint(self) -> Path:
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if PLACEHOLDER_CHECKPOINT_PATH.exists():
            return PLACEHOLDER_CHECKPOINT_PATH
        torch.manual_seed(42)
        stub = self._build_model()
        torch.save({"model_state": stub.state_dict()}, PLACEHOLDER_CHECKPOINT_PATH)
        print(
            "NOTE: No usable trained weights found; created a randomly initialized checkpoint at "
            f"{PLACEHOLDER_CHECKPOINT_PATH}. Replace with a trained asl_resnet50.pth for real ASL accuracy."
        )
        return PLACEHOLDER_CHECKPOINT_PATH

    def _model_checkpoint_candidates(self) -> list[Path]:
        """Prefer the .pth.zip bundle (real trained weights), then a flat .pth, else a generated placeholder."""
        return [p for p, _ in self._model_checkpoint_candidates_with_meta()]

    def _model_checkpoint_candidates_with_meta(self) -> list[tuple[Path, bool]]:
        """Return (path, is_trained_weights) pairs in priority order."""
        candidates: list[tuple[Path, bool]] = []
        if MODEL_ARCHIVE_PATH.exists():
            try:
                candidates.append((self._ensure_flattened_zip_checkpoint(MODEL_ARCHIVE_PATH), True))
            except Exception as exc:
                print(f"WARNING: Could not prepare flattened checkpoint from {MODEL_ARCHIVE_PATH.name}: {exc}")
        if MODEL_PATH.exists() and not MODEL_PATH.is_dir():
            candidates.append((MODEL_PATH, True))
        if candidates:
            return candidates
        return [(self._ensure_placeholder_checkpoint(), False)]

    def _resolve_landmarker_asset_path(self) -> Path:
        if LANDMARKER_PATH.exists():
            return LANDMARKER_PATH
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if not CACHED_LANDMARKER_PATH.exists():
            print(f"Downloading Hand Landmarker model to {CACHED_LANDMARKER_PATH} …")
            self._download_file(HAND_LANDMARKER_URL, CACHED_LANDMARKER_PATH)
        return CACHED_LANDMARKER_PATH

    @staticmethod
    def _download_file(url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_suffix(destination.suffix + ".download")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "SignLangAI/1.0"})
            with urllib.request.urlopen(request, timeout=180) as response:
                temp_path.write_bytes(response.read())
            temp_path.replace(destination)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise ModelInitializationError(f"Failed to download model asset from {url}.") from exc

    def _create_detector(self) -> Any:
        landmarker_path = self._resolve_landmarker_asset_path()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(landmarker_path)),
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        return mp_vision.HandLandmarker.create_from_options(options)

    def _load_local_inference_namespace(self) -> dict[str, Any]:
        source = LOCAL_INFERENCE_PATH.read_text(encoding="utf-8")
        parsed = ast.parse(source, filename=str(LOCAL_INFERENCE_PATH))

        allowed_assignments = {
            "MODEL_PATH",
            "CLASSES_PATH",
            "LANDMARKER_PATH",
            "CAMERA_INDEX",
            "CONFIDENCE_THR",
            "SMOOTH_WINDOW",
        }
        allowed_functions = {
            "extract_landmarks",
            "predict",
            "mirror_landmark_vector",   # needed by predict_with_mirror
            "predict_with_mirror",       # both-hand-compatible inference
            "draw_ui",
        }
        allowed_classes = {"Smoother"}

        selected_nodes: list[ast.AST] = []
        for node in parsed.body:
            if isinstance(node, ast.Import):
                if any(alias.name.startswith("torchvision") for alias in node.names):
                    continue
                selected_nodes.append(node)
                continue
            if isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if module_name.startswith("torchvision"):
                    continue
                selected_nodes.append(node)
                continue
            if isinstance(node, ast.Assign):
                target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
                if target_names and all(name in allowed_assignments for name in target_names):
                    selected_nodes.append(node)
                continue
            if isinstance(node, ast.ClassDef) and node.name in allowed_classes:
                selected_nodes.append(node)
                continue
            if isinstance(node, ast.FunctionDef) and node.name in allowed_functions:
                selected_nodes.append(node)
                continue

        # Compile only the reusable imports, classes, and helper functions from
        # local_inference.py so the backend can call them without starting the
        # script's live camera loop.
        module = ast.Module(body=selected_nodes, type_ignores=[])
        namespace: dict[str, Any] = {}
        previous_cwd = Path.cwd()
        try:
            import os

            os.chdir(MODEL_DIR)
            exec(compile(module, str(LOCAL_INFERENCE_PATH), "exec"), namespace, namespace)
        finally:
            os.chdir(previous_cwd)

        return namespace

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image = image.convert("RGB")
                rgb_array = np.array(image, dtype=np.uint8)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError("Invalid image file.") from exc

        return rgb_array[:, :, ::-1].copy()


asl_inference_service = ASLInferenceService()


def initialize_model_service() -> None:
    asl_inference_service.initialize()


def shutdown_model_service() -> None:
    asl_inference_service.shutdown()


def predict_from_image(image_bytes: bytes) -> tuple[str, float]:
    return asl_inference_service.predict_from_image(image_bytes)


def predict_from_base64(encoded_image: str) -> tuple[str, float]:
    return asl_inference_service.predict_from_base64(encoded_image)


def get_model_status() -> dict:
    """Return a status dict for the /model/status endpoint."""
    svc = asl_inference_service
    loaded = svc.model is not None and svc.detector is not None
    import os
    source = None
    if svc._checkpoint_source:
        source = os.path.basename(svc._checkpoint_source)
    return {
        "model_loaded": loaded,
        "is_trained_weights": svc._is_trained_weights,
        "checkpoint_source": source,
        "num_classes": len(svc.classes),
        "classes": svc.classes,
        "device": str(svc.device),
    }

