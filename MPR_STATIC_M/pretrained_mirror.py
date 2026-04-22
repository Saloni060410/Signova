import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
import os

MODEL_PATH      = "asl_resnet50.pth"
CLASSES_PATH    = "asl_classes.json"
LANDMARKER_PATH = "hand_landmarker.task"
CAMERA_INDEX    = 0
CONFIDENCE_THR  = 0.6
SMOOTH_WINDOW   = 10

class LandmarkResNet50(nn.Module):
    def __init__(self, num_classes: int, input_dim: int = 63):
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
        backbone = models.resnet50(weights=None)
        self.adapt = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, bias=False),
            backbone.bn1,
            backbone.relu,
        )
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

with open(CLASSES_PATH) as f:
    CLASSES = json.load(f)
print(f"Classes: {len(CLASSES)} → {CLASSES}")

ckpt  = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model = LandmarkResNet50(num_classes=len(CLASSES)).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("✅ Model loaded")

options  = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
    running_mode=mp_vision.RunningMode.IMAGE
)
detector = mp_vision.HandLandmarker.create_from_options(options)
print("✅ MediaPipe detector ready")


def extract_landmarks(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result    = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None
    lm     = result.hand_landmarks[0]
    coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    origin  = coords[0].copy()
    coords -= origin
    scale   = np.linalg.norm(coords[9]) + 1e-6
    coords /= scale
    return coords.flatten()


def mirror_landmark_vector(vec):
    coords = vec.reshape(21, 3).copy()
    coords[:, 0] *= -1.0
    return coords.flatten()


@torch.no_grad()
def predict(vec):
    x     = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1).squeeze(0)
    top_v, top_i = probs.topk(3)
    top3  = [(CLASSES[i], v.item()) for i, v in zip(top_i, top_v)]
    return top3


@torch.no_grad()
def predict_with_mirror(vec):
    top3_normal = predict(vec)
    top3_mirror = predict(mirror_landmark_vector(vec))
    if top3_mirror[0][1] > top3_normal[0][1]:
        return top3_mirror
    return top3_normal


class Smoother:
    def __init__(self, window=10):
        self.buf = deque(maxlen=window)
    def update(self, label, conf):
        self.buf.append((label, conf))
    def get(self):
        if not self.buf:
            return None, 0.0
        votes = {}
        for lbl, c in self.buf:
            votes[lbl] = votes.get(lbl, 0.0) + c
        best = max(votes, key=votes.get)
        return best, votes[best] / len(self.buf)
    def clear(self):
        self.buf.clear()


def draw_ui(frame, label, conf, top3, fps, hand_detected, saved_text):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
    if not hand_detected:
        cv2.putText(frame, "No hand", (10, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (80, 80, 255), 2)
        cv2.putText(frame, "Show hand to camera", (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
    else:
        color = (0, 220, 80) if conf >= CONFIDENCE_THR else (0, 140, 255)
        cv2.putText(frame, str(label), (10, 90),
                    cv2.FONT_HERSHEY_DUPLEX, 2.8, color, 3)
        cv2.putText(frame, f"{conf*100:.1f}%", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        cv2.putText(frame, "Top 3:", (10, 158),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        for i, (lbl, c) in enumerate(top3):
            y0  = 168 + i * 32
            bw  = int(c * 220)
            bc  = (0, 180, 70) if i == 0 else (50, 100, 180)
            cv2.rectangle(frame, (10, y0), (10 + bw, y0 + 20), bc, -1)
            cv2.rectangle(frame, (10, y0), (230,     y0 + 20), (70, 70, 70), 1)
            cv2.putText(frame, f"{lbl}  {c*100:.1f}%",
                        (14, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1)

    # --- Saved text banner at the bottom ---
    bar_h = 50
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "Saved:", (10, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
    # Draw each saved letter spaced out
    x_offset = 80
    for ch in saved_text:
        cv2.putText(frame, ch, (x_offset, h - 12),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 230, 255), 2)
        x_offset += 35

    cv2.putText(frame, "Enter=save | Backspace=del | C=clear | Q=quit | S=screenshot",
                (10, h - bar_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
    return frame


smoother = Smoother(SMOOTH_WINDOW)
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

os.makedirs("screenshots", exist_ok=True)
print("✅ Camera open — starting live inference")
print("   Enter = save letter | Backspace = delete | C = clear")
print("   Q = quit  |  S = save screenshot\n")

prev_time     = time.time()
label         = "---"
conf          = 0.0
top3          = []
hand_detected = False
saved_text    = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    vec   = extract_landmarks(frame)

    if vec is not None:
        hand_detected = True
        top3          = predict_with_mirror(vec)
        smoother.update(top3[0][0], top3[0][1])
        label, conf   = smoother.get()
    else:
        hand_detected = False
        smoother.clear()
        label, conf, top3 = "---", 0.0, []

    now       = time.time()
    fps       = 1.0 / (now - prev_time + 1e-9)
    prev_time = now

    frame = draw_ui(frame, label, conf, top3, fps, hand_detected, saved_text)
    cv2.imshow("ASL Sign Language — ResNet50  |  Q=quit  S=screenshot", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        path = f"screenshots/capture_{int(time.time())}.jpg"
        cv2.imwrite(path, frame)
        print(f"Screenshot saved → {path}")
    elif key == 13:  # Enter key — save current letter
        if hand_detected and conf >= CONFIDENCE_THR and label not in ("---", "del", "space"):
            saved_text += label
            print(f"   ✅ Saved '{label}' → {saved_text}")
        elif hand_detected and label == "space":
            saved_text += " "
            print(f"   ✅ Added space → '{saved_text}'")
        elif hand_detected and label == "del" and saved_text:
            saved_text = saved_text[:-1]
            print(f"   🗑️  Deleted last → '{saved_text}'")
    elif key == 8 or key == 127:  # Backspace — delete last letter
        if saved_text:
            removed = saved_text[-1]
            saved_text = saved_text[:-1]
            print(f"   🗑️  Deleted '{removed}' → '{saved_text}'")
    elif key == ord('c'):  # C — clear all
        saved_text = ""
        print("   🧹 Cleared all saved text")

cap.release()
cv2.destroyAllWindows()
detector.close()
print("Closed")
