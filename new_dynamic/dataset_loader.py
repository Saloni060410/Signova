"""
dataset_loader.py - Dataset loading and augmentation for sign language sequences.

Loads .npy files from the dataset/ folder, applies optional augmentation,
and returns PyTorch DataLoaders for training and validation.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional

# ─── Category Mapping ────────────────────────────────────────────────────────
CATEGORIES = ["Pronouns", "Actions", "Social", "Questions", "Context"]
CAT_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}

CATEGORY_MAP = {
    "i": "Pronouns", "you": "Pronouns", "we": "Pronouns", "they": "Pronouns",
    "come": "Actions", "go": "Actions", "eat": "Actions", "drink": "Actions",
    "give": "Actions", "take": "Actions", "help": "Actions", "work": "Actions",
    "want": "Actions", "need": "Actions",
    "hello": "Social", "bye": "Social", "thank_you": "Social", "sorry": "Social",
    "please": "Social", "ily": "Social", "yes": "Social", "no": "Social",
    "what": "Questions", "where": "Questions", "why": "Questions", "how": "Questions",
    "home": "Context", "school": "Context", "water": "Context", "today": "Context"
}



# ─── Label Management ─────────────────────────────────────────────────────────

def load_labels(dataset_path: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Discover class labels from dataset folder structure and export to JSON.

    Args:
        dataset_path: Path to the dataset root directory.

    Returns:
        classes: Sorted list of class names.
        label_map: Dictionary mapping class name → integer index.
    """
    classes = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    label_map = {cls: idx for idx, cls in enumerate(classes)}

    # Export labels to JSON for use in inference
    labels_file = os.path.join(dataset_path, "..", "labels.json")
    with open(labels_file, "w") as f:
        json.dump({"classes": classes, "label_map": label_map}, f, indent=2)
    print(f"[Labels] Saved to {labels_file}: {label_map}")

    return classes, label_map


# ─── Data Augmentation ────────────────────────────────────────────────────────

def augment_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Symmetric Hand-Agnostic Augmentation:
    1. Temporal Warping: Speed/Slow variations.
    2. Mirroring (Swap+Flip): Correctly simulates Left-handed signing.
    3. 3D Rotation & Noise: General spatial robustness.
    """
    seq = sequence.copy()
    seq_len = seq.shape[0]
    n_features = seq.shape[1]
    
    # 1. Temporal Warping (±15% speed variation)
    if np.random.random() < 0.4:
        speed_factor = np.random.uniform(0.85, 1.15)
        new_indices = np.linspace(0, seq_len-1, num=max(10, int(seq_len * speed_factor)))
        warped = np.zeros((len(new_indices), n_features), dtype=np.float32)
        for i in range(n_features):
            warped[:, i] = np.interp(new_indices, np.arange(seq_len), seq[:, i])
        final_indices = np.linspace(0, len(warped)-1, num=seq_len)
        for i in range(n_features):
            seq[:, i] = np.interp(final_indices, np.arange(len(warped)), warped[:, i])

    # 2. Symmetric Mirroring (Swap Slots + Flip X)
    # This teaches the model that signs are the same regardless of which hand is used.
    if np.random.random() < 0.5:
        # Swap Hand 1 (0:63) and Hand 2 (63:126)
        h1 = seq[:, :63].copy()
        h2 = seq[:, 63:126].copy()
        seq[:, :63] = h2
        seq[:, 63:126] = h1
        # Flip X coordinates for everything (Hands + Pose)
        seq_reshaped = seq.reshape(seq_len, -1, 3)
        seq_reshaped[:, :, 0] *= -1
        seq = seq_reshaped.reshape(seq_len, -1)

    # 3. 3D Rotation (Small tilts ±5 degrees)
    if np.random.random() < 0.4:
        angle_x = np.radians(np.random.uniform(-5, 5))
        angle_y = np.radians(np.random.uniform(-5, 5))
        Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
        Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
        R = np.dot(Ry, Rx)
        seq_reshaped = seq.reshape(-1, 3)
        seq_rotated = np.dot(seq_reshaped, R.T)
        seq = seq_rotated.reshape(seq_len, -1)

    # 4. Add Gaussian noise
    if np.random.random() < 0.4:
        noise = np.random.normal(0, 0.005, seq.shape).astype(np.float32)
        seq += noise

    return np.clip(seq, -2.0, 2.0).astype(np.float32)


# ─── Dataset Class ────────────────────────────────────────────────────────────

class SignLanguageDataset(Dataset):
    """
    PyTorch Dataset for sign language gesture sequences.

    Args:
        sequences: List of np.ndarray sequences, each of shape (seq_len, 63).
        labels: List of integer labels.
        augment: Whether to apply data augmentation.
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        labels: List[int],
        classes: List[str],  # Needed for category lookup
        augment: bool = False,
    ):
        self.sequences = sequences
        self.labels = labels
        self.classes = classes
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx].copy()

        # Apply augmentation during training
        if self.augment:
            seq = augment_sequence(seq)

        # Convert to tensors
        x = torch.tensor(seq, dtype=torch.float32)       # (seq_len, 225)
        
        class_idx = self.labels[idx]
        class_name = self.classes[class_idx]
        cat_name = CATEGORY_MAP.get(class_name, "Social")
        cat_idx = CAT_IDX[cat_name]
        
        y = torch.tensor(class_idx, dtype=torch.long)
        cat = torch.tensor(cat_idx, dtype=torch.long)
        return x, y, cat


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_dataset(
    dataset_path: str,
    sequence_length: int = 30,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Load all .npy sequences from the dataset directory.

    Args:
        dataset_path: Path to dataset root.
        sequence_length: Expected sequence length (for validation).

    Returns:
        sequences: List of arrays, each (sequence_length, 225).
        labels: List of integer class labels.
        classes: List of class names.
    """
    classes, label_map = load_labels(dataset_path)
    sequences, labels = [], []
    skipped = 0

    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        npy_files = sorted([f for f in os.listdir(cls_path) if f.endswith(".npy")])

        for npy_file in npy_files:
            file_path = os.path.join(cls_path, npy_file)
            try:
                seq = np.load(file_path).astype(np.float32)

                # Pad old data to reach 225 features
                # 63-feature (hand only) → pad 162 zeros (missing hand2 + pose)
                if seq.shape[1] == 63:
                    seq = np.pad(seq, ((0, 0), (0, 162)), mode='constant')
                # 132-feature (both hands + shoulders) → pad to 225
                elif seq.shape[1] == 132:
                    hands = seq[:, :126]
                    shoulders = seq[:, 126:]
                    pose = np.zeros((seq.shape[0], 99), dtype=np.float32)
                    # Put shoulders at landmark 11 and 12 positions (indices 33-38)
                    pose[:, 33:39] = shoulders
                    seq = np.concatenate([hands, pose], axis=1)
                # 162-feature (hand1 + pose) → insert 63 zeros for missing hand2
                elif seq.shape[1] == 162:
                    hand1 = seq[:, :63]    # first 63 = hand 1
                    pose  = seq[:, 63:]    # last 99 = pose
                    hand2 = np.zeros((seq.shape[0], 63), dtype=np.float32)
                    seq = np.concatenate([hand1, hand2, pose], axis=1)  # (seq_len, 225)

                # Validate shape
                if seq.shape == (sequence_length, 225) or (seq.shape[1] == 225 and seq.shape[0] != sequence_length):
                    if seq.shape[0] != sequence_length:
                        seq = _pad_or_truncate(seq, sequence_length)
                    
                    sequences.append(seq)
                    labels.append(label_map[cls])  # Back to simple integer
                else:
                    print(f"  [SKIP] Unexpected shape {seq.shape} in {file_path}")
                    skipped += 1

            except Exception as e:
                print(f"  [ERROR] Loading {file_path}: {e}")
                skipped += 1

        print(f"  [Loaded] {cls}: {len(npy_files) - skipped} sequences")

    print(f"\n[Dataset] Total: {len(sequences)} sequences | Skipped: {skipped}")
    print(f"[Dataset] Classes: {classes}")
    return sequences, labels, classes


def _pad_or_truncate(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with zeros or truncate sequence to target length."""
    current_len = seq.shape[0]
    if current_len >= target_len:
        return seq[:target_len]
    else:
        pad = np.zeros((target_len - current_len, seq.shape[1]), dtype=np.float32)
        return np.vstack([seq, pad])


# ─── DataLoader Factory ───────────────────────────────────────────────────────

def get_dataloaders(
    dataset_path: str,
    sequence_length: int = 30,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Build train and validation DataLoaders.

    Args:
        dataset_path: Path to dataset root directory.
        sequence_length: Frames per sequence.
        batch_size: Mini-batch size.
        val_split: Fraction of data for validation.
        num_workers: DataLoader worker processes.
        seed: Random seed for reproducibility.

    Returns:
        train_loader, val_loader, classes
    """
    sequences, labels, classes = load_dataset(dataset_path, sequence_length)

    # Stratified split to preserve class balance
    X_train, X_val, y_train, y_val = train_test_split(
        sequences,
        labels,
        test_size=val_split,
        stratify=labels,
        random_state=seed,
    )

    # Calculate class distribution for perfect balancing
    # Every class will be sampled "equally" in every epoch.
    counts = np.bincount(y_train)
    weights = 1.0 / counts
    sample_weights = np.array([weights[t] for t in y_train])
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_dataset = SignLanguageDataset(X_train, y_train, classes, augment=True)
    val_dataset = SignLanguageDataset(X_val, y_val, classes, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,               # Replaces shuffle=True for perfect balance
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"\n[DataLoader] Train: {len(train_dataset)} (Balanced) | Val: {len(val_dataset)}")
    print(f"[DataLoader] Batch size: {batch_size} | Sampler: WeightedRandomSampler")

    return train_loader, val_loader, classes


if __name__ == "__main__":
    # Quick test
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "../dataset"
    train_loader, val_loader, classes = get_dataloaders(dataset_path, batch_size=16)
    x, y = next(iter(train_loader))
    print(f"\n[Test] Batch x: {x.shape} | y: {y.shape}")
