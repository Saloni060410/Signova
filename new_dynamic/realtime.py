"""
realtime.py - Upgraded real-time sign language recognition with:
    • Deque-based prediction smoothing (window=10, majority vote)
    • Confidence threshold (ignore < 0.7)
    • Sentence builder (up to 8 words, cooldown, no consecutive duplicates)
    • Text-to-speech via pyttsx3
    • Special commands: 'clear' -> clears sentence, 'speak' -> speaks sentence
    • Live display: predicted word, confidence, current sentence, FPS

Usage:
    python realtime.py [--model models/best_model.pth]
                       [--config models/training_config.json]
                       [--labels labels.json]
                       [--confidence 0.7]
                       [--smooth 10]
                       [--camera 0]
                       [--seq_len 30]

Keyboard controls:
    q  -> quit
    r  -> reset sequence buffer
    c  -> clear sentence
    s  -> speak current sentence
"""

import os
import sys
import argparse
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    get_mediapipe_hands,
    get_mediapipe_pose,
    extract_all_keypoints,
    normalize_keypoints,
    draw_all_landmarks,
    put_text_with_background,
)
from inference_utils import (
    load_labels,
    load_model,
    FPSCounter,
    SequenceBuffer,
    predict_sequence,
)
from inference.smoothing import PredictionSmoother
from inference.sentence_builder import SentenceBuilder
from inference.tts import TTSEngine


# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Sign Language Recognition v2")
    parser.add_argument("--model",      type=str,   default="models/best_model.pth")
    parser.add_argument("--config",     type=str,   default="models/training_config.json")
    parser.add_argument("--labels",     type=str,   default="labels.json")
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--smooth",     type=int,   default=10)
    parser.add_argument("--camera",     type=int,   default=0)
    parser.add_argument("--seq_len",    type=int,   default=30)
    parser.add_argument("--cooldown",   type=float, default=1.5)
    parser.add_argument("--tts_gap",    type=float, default=3.0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
def draw_sequence_bar(frame, current, total):
    h, w = frame.shape[:2]
    y = h - 12
    x0, x1 = 10, w - 10
    bw = x1 - x0
    filled = int(bw * current / max(total, 1))
    cv2.rectangle(frame, (x0, y - 6), (x1, y + 6), (50, 50, 50), cv2.FILLED)
    cv2.rectangle(frame, (x0, y - 6), (x0 + filled, y + 6), (0, 200, 100), cv2.FILLED)
    cv2.putText(frame, f"{current}/{total}",
                (x0 + bw // 2 - 18, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def draw_prob_bars(frame, probabilities, classes):
    """Draw all class probabilities in a compact 2-column grid on the right side."""
    if len(probabilities) == 0:
        return
    h, w = frame.shape[:2]

    top_idx = int(np.argmax(probabilities))
    sorted_idx = np.argsort(probabilities)[::-1]  # highest first

    # --- Top-3 highlight panel (top-right corner) ---
    panel_x = w - 280
    cv2.rectangle(frame, (panel_x - 5, 0), (w, 120), (20, 20, 20), cv2.FILLED)
    cv2.putText(frame, "TOP PREDICTIONS", (panel_x, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    medal_colors = [(0, 215, 255), (200, 200, 200), (0, 165, 255)]  # gold, silver, bronze
    for rank, idx in enumerate(sorted_idx[:3]):
        yy = 38 + rank * 28
        pct = probabilities[idx] * 100
        col = medal_colors[rank]
        bar_w = int(250 * probabilities[idx])
        cv2.rectangle(frame, (panel_x, yy), (panel_x + 250, yy + 20), (40, 40, 40), cv2.FILLED)
        cv2.rectangle(frame, (panel_x, yy), (panel_x + bar_w, yy + 20), col, cv2.FILLED)
        
        # ADDED: Keyboard shortcut hint [1], [2], [3]
        label = f"[{rank+1}] {classes[idx].upper():<12} {pct:5.1f}%"
        cv2.putText(frame, label, (panel_x + 4, yy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (10, 10, 10) if bar_w > 40 else (220, 220, 220),
                    1, cv2.LINE_AA)

    # --- Full class grid (2 columns, compact rows, bottom half) ---
    n_classes = len(classes)
    cols = 2
    col_w = w // cols
    bar_h = 13
    gap = 3
    y_start = 130
    rows_available = (h - y_start - 90) // (bar_h + gap)
    n_show = min(n_classes, rows_available * cols)

    # Sort by probability descending for the grid too
    for pos, idx in enumerate(sorted_idx[:n_show]):
        col_idx = pos % cols
        row_idx = pos // cols
        x0 = col_idx * col_w + (col_w - 260) // 2
        y0 = y_start + row_idx * (bar_h + gap)
        prob = probabilities[idx]
        filled = int(240 * prob)
        is_top = (idx == top_idx)
        bg = (50, 50, 50)
        fg = (0, 200, 60) if is_top else (80, 120, 200)
        cv2.rectangle(frame, (x0, y0), (x0 + 240, y0 + bar_h), bg, cv2.FILLED)
        cv2.rectangle(frame, (x0, y0), (x0 + filled, y0 + bar_h), fg, cv2.FILLED)
        label = f"{classes[idx][:10]}: {prob:.2f}"
        font_col = (255, 255, 0) if is_top else (210, 210, 210)
        cv2.putText(frame, label, (x0 + 2, y0 + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, font_col, 1, cv2.LINE_AA)


def draw_sentence_panel(frame, sentence, word_count, max_words):
    h, w = frame.shape[:2]
    py = h - 60
    cv2.rectangle(frame, (0, py - 12), (w, py + 36), (20, 20, 20), cv2.FILLED)
    text = f"Sentence ({word_count}/{max_words}): {sentence if sentence else '_'}"
    put_text_with_background(
        frame, text, (10, py + 22),
        font_scale=0.72, color=(255, 230, 50), bg_color=(20, 20, 20), thickness=2,
    )


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    classes, label_map = load_labels(args.labels)
    model = load_model(args.model, args.config, device)
    print(f"[Classes] {classes}")

    seq_buffer  = SequenceBuffer(sequence_length=args.seq_len, feature_size=225)
    smoother    = PredictionSmoother(window_size=args.smooth,
                                     confidence_threshold=args.confidence)
    fps_counter = FPSCounter(window=30)
    builder     = SentenceBuilder(max_words=8, cooldown_sec=args.cooldown)
    tts         = TTSEngine(min_interval_sec=args.tts_gap)

    hands = get_mediapipe_hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.7, min_tracking_confidence=0.5,
    )
    pose = get_mediapipe_pose(
        static_image_mode=False,
        min_detection_confidence=0.7, min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    current_word   = None
    current_conf   = 0.0
    last_probs     = np.zeros(len(classes))
    status_msg     = ""
    status_expire  = 0.0

    def set_status(msg, dur=1.5):
        nonlocal status_msg, status_expire
        status_msg    = msg
        status_expire = time.time() + dur

    print("\n[v2] Running. q=quit | r=reset | c=clear sentence | s=speak\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            fps_counter.tick()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hand_results = hands.process(rgb)
            pose_results = pose.process(rgb)
            rgb.flags.writeable = True

            frame = draw_all_landmarks(frame, hand_results, pose_results)
            kp = extract_all_keypoints(hand_results, pose_results)
            seq_buffer.add_frame(kp)

            # Keep track of top 3 for manual selection
            top_3_indices = []

            # ── Inference ─────────────────────────────────────────────────
            if seq_buffer.is_full:
                seq = seq_buffer.get_sequence()
                pred_idx, confidence, probs = predict_sequence(model, seq, device)
                last_probs = probs
                top_3_indices = np.argsort(probs)[::-1][:3].tolist()

                smoother.update(pred_idx, confidence)
                stable_idx, stable_conf = smoother.get_stable_prediction()

                if stable_idx is not None:
                    word         = classes[stable_idx]
                    current_word = word
                    current_conf = stable_conf

                    result = builder.try_add_word(word)

                    if result == "added":
                        set_status(f"+ {word}")
                        print(f"[Sentence] {builder.sentence}")

                    elif result == "command":
                        cmd = builder.pop_command()
                        if cmd == "clear":
                            builder.clear()
                            set_status("Sentence cleared")
                            print("[Command] clear")
                        elif cmd == "speak":
                            s = builder.sentence
                            if s:
                                ok = tts.speak(s, force=True)
                                set_status("Speaking..." if ok else "TTS unavailable")
                                print(f'[Command] speak -> "{s}"')
                            else:
                                set_status("Nothing to speak")

            # ── Draw UI ───────────────────────────────────────────────────
            h, w = frame.shape[:2]

            put_text_with_background(
                frame, f"FPS: {fps_counter.fps:.1f}", (10, 30),
                font_scale=0.65, color=(200, 200, 200), bg_color=(0, 0, 0),
            )

            if current_word:
                put_text_with_background(
                    frame,
                    f"{current_word.upper()}  {current_conf*100:.1f}%",
                    (10, h // 2 - 10),
                    font_scale=1.5, color=(0, 255, 100),
                    bg_color=(0, 0, 0), thickness=3,
                )
            else:
                put_text_with_background(
                    frame, "Waiting for gesture...", (10, h // 2 - 10),
                    font_scale=0.9, color=(150, 150, 150), bg_color=(0, 0, 0),
                )

            if status_msg and time.time() < status_expire:
                put_text_with_background(
                    frame, status_msg, (10, h // 2 + 40),
                    font_scale=0.75, color=(255, 200, 50), bg_color=(0, 0, 0),
                )

            draw_sentence_panel(frame, builder.sentence, builder.word_count, 8)
            draw_prob_bars(frame, last_probs, classes)
            draw_sequence_bar(frame, len(seq_buffer._buffer), args.seq_len)

            put_text_with_background(
                frame, "q:quit | r:reset | c:clear | s:speak",
                (10, h - 75),
                font_scale=0.48, color=(160, 160, 160), bg_color=(0, 0, 0),
            )

            cv2.imshow("Sign Language Recognition v2", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in [ord("1"), ord("2"), ord("3")]:
                # Manual word selection
                rank = int(chr(key)) - 1
                if len(top_3_indices) > rank:
                    word = classes[top_3_indices[rank]]
                    builder.add_word_force(word)  # We'll need to define this helper
                    set_status(f"Selected: {word}")
                    print(f"[Manual] {word}")
            elif key == ord("r"):
                seq_buffer.reset(); smoother.reset()
                current_word = None
                set_status("Buffer reset")
            elif key == ord("c"):
                builder.clear()
                set_status("Sentence cleared")
                print("[Clear] Sentence cleared.")
            elif key == ord("s"):
                s = builder.sentence
                if s:
                    tts.speak(s, force=True)
                    set_status(f"Speaking: {s}")
                    print(f'[Speak] "{s}"')
                else:
                    set_status("Nothing to speak")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        pose.close()
        print("[Exit] Done.")


if __name__ == "__main__":
    main()
