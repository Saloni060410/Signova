"""
utils.py - Utility functions for data collection
Handles MediaPipe initialization, keypoint extraction, and visualization.
"""

import cv2
import numpy as np
import mediapipe as mp

# MediaPipe solutions
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def get_mediapipe_hands(
    static_image_mode: bool = False,
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.7,
    min_tracking_confidence: float = 0.5,
):
    """
    Initialize and return a MediaPipe Hands instance.

    Args:
        static_image_mode: Whether to treat input as static images (slower but more accurate).
        max_num_hands: Maximum number of hands to detect.
        min_detection_confidence: Confidence threshold for hand detection.
        min_tracking_confidence: Confidence threshold for hand tracking.

    Returns:
        MediaPipe Hands instance.
    """
    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def get_mediapipe_pose(
    static_image_mode: bool = False,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """Initialize and return a MediaPipe Pose instance."""
    return mp_pose.Pose(
        static_image_mode=static_image_mode,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def extract_keypoints(results) -> np.ndarray:
    """
    Extract hand keypoints from MediaPipe results.

    Each hand has 21 landmarks (x, y, z) → 63 features.
    If no hand is detected, returns a zero array of length 63.

    Args:
        results: MediaPipe Hands process() output.

    Returns:
        np.ndarray of shape (63,) — flattened x, y, z for 21 landmarks.
    """
    if results.multi_hand_landmarks:
        # Use the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()  # Shape: (63,)
    else:
        # Pad with zeros if no hand detected
        keypoints = np.zeros(63, dtype=np.float32)

    return keypoints


def extract_pose_keypoints(results) -> np.ndarray:
    """
    Extract pose keypoints from MediaPipe results.
    33 landmarks (x, y, z) → 99 features.
    """
    if results.pose_landmarks:
        keypoints = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        keypoints = np.zeros(99, dtype=np.float32)
    return keypoints


def extract_all_keypoints(hand_results, pose_results) -> np.ndarray:
    """
    Combine both hands (126) and pose (99) into 225 total features.
    Applies wrist-normalization to each hand for translation invariance.
    """
    # 1. Extract raw keypoints for both hands (126 features)
    # This gives us [hand1_x,y,z... (63), hand2_x,y,z... (63)]
    both_hands_kp = extract_keypoints_both_hands(hand_results)  # (126,)

    # 2. Normalize each hand individually relative to its own wrist
    hand1 = normalize_keypoints(both_hands_kp[:63])
    hand2 = normalize_keypoints(both_hands_kp[63:])
    hands_normalized = np.concatenate([hand1, hand2])

    # 3. Extract raw pose keypoints (99 features)
    pose_kp = extract_pose_keypoints(pose_results)

    return np.concatenate([hands_normalized, pose_kp])  # Total 225 features


def extract_keypoints_both_hands(results) -> np.ndarray:
    """
    Extract keypoints for up to 2 hands (126 features total).
    Strictly slots Right hand into 0-62 and Left hand into 63-125.
    
    Args:
        results: MediaPipe Hands process() output.

    Returns:
        np.ndarray of shape (126,).
    """
    # Initialize with zeros (2 hands, 63 features each)
    hand1_data = np.zeros(63, dtype=np.float32) # Slot 0: Right Hand
    hand2_data = np.zeros(63, dtype=np.float32) # Slot 1: Left Hand

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Get hand label (MediaPipe's 'Right'/'Left' labels are based on the person's real hands)
            label = handedness.classification[0].label # "Left" or "Right"
            
            kp = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32,
            ).flatten()

            if label == "Right":
                hand1_data = kp
            else:
                hand2_data = kp

    return np.concatenate([hand1_data, hand2_data])  # Shape: (126,)


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize keypoints relative to the wrist (landmark 0).
    This makes the representation translation-invariant.

    Args:
        keypoints: np.ndarray of shape (63,).

    Returns:
        Normalized keypoints of shape (63,).
    """
    if np.all(keypoints == 0):
        return keypoints  # Return zeros if no hand detected

    # Reshape to (21, 3)
    kp = keypoints.reshape(21, 3)

    # Translate so wrist is at origin
    wrist = kp[0].copy()
    kp -= wrist

    # Scale by the max absolute value to bring into [-1, 1]
    max_val = np.max(np.abs(kp))
    if max_val > 0:
        kp /= max_val

    return kp.flatten()


def draw_landmarks(frame: np.ndarray, results) -> np.ndarray:
    """
    Draw hand landmarks and connections on the frame.

    Args:
        frame: BGR image as np.ndarray.
        results: MediaPipe Hands results.

    Returns:
        Frame with landmarks drawn.
    """
    # 1. Draw Hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

    # 2. Draw Pose (if available)
    # Check if results has pose_landmarks (for combined calls) or if it's a pose results object
    pose_lms = getattr(results, 'pose_landmarks', None)
    if pose_lms:
        mp_drawing.draw_landmarks(
            frame,
            pose_lms,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    return frame


def draw_all_landmarks(frame, hand_results, pose_results):
    """Draw both hand and pose skeletons."""
    frame = draw_landmarks(frame, hand_results)
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    return frame


def put_text_with_background(
    frame: np.ndarray,
    text: str,
    position: tuple,
    font_scale: float = 1.0,
    color: tuple = (255, 255, 255),
    bg_color: tuple = (0, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw text on frame with a solid background rectangle for readability.

    Args:
        frame: BGR image.
        text: Text string to draw.
        position: (x, y) bottom-left corner of text.
        font_scale: Font size.
        color: Text color in BGR.
        bg_color: Background rectangle color in BGR.
        thickness: Text thickness.

    Returns:
        Frame with text drawn.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position

    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x - 4, y - text_h - 4),
        (x + text_w + 4, y + baseline + 4),
        bg_color,
        cv2.FILLED,
    )
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame
