"""
model.py - LSTM model for sign language gesture recognition.

Architecture:
    Input (batch, seq_len=30, features=63)
        → LSTM (hidden=128, layers=2, dropout=0.3)
        → Fully Connected (128 → 64)
        → ReLU + Dropout
        → Fully Connected (64 → num_classes)
        → Output logits
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class SignLanguageLSTM(nn.Module):
    """
    LSTM-based classifier for sign language gesture sequences.

    Args:
        input_size: Number of features per frame (default 63 for 21 landmarks × 3 coords).
        hidden_size: Number of LSTM hidden units (default 128).
        num_layers: Number of stacked LSTM layers (default 2).
        num_classes: Number of gesture classes.
        dropout: Dropout probability between LSTM layers (default 0.3).
    """

    def __init__(
        self,
        input_size: int = 225,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 30,
        num_categories: int = 5,
        dropout: float = 0.5,
    ):
        super(SignLanguageLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ── LSTM layers ───────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,          # input: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # ── Shared feature processing ─────────────────────────────────────────
        self.norm = nn.LayerNorm(hidden_size)

        # ── Head 1: Class prediction (the specific word) ──────────────────────
        self.class_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        # ── Head 2: Category prediction (pattern guidance) ───────────────────
        self.category_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_categories),
        )

        # ── Weight initialization ──────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Xavier uniform for linear layers
        and orthogonal initialization for LSTM weights.
        """
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        for layer in self.class_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        for layer in self.category_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, seq_len, input_size).

        Returns:
            class_logits: Shape (batch_size, num_classes)
            category_logits: Shape (batch_size, num_categories)
        """
        batch_size = x.size(0)

        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        # LSTM forward pass -> (batch, seq_len, hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Use the last time step's output
        last_out = self.norm(out[:, -1, :])  # (batch, hidden_size)

        # Predict specific class and broad category
        class_logits = self.class_head(last_out)
        cat_logits   = self.category_head(last_out)

        return class_logits, cat_logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return softmax probabilities for classes only.
        """
        with torch.no_grad():
            class_logits, _ = self.forward(x)
            return torch.softmax(class_logits, dim=-1)


def get_model(num_classes: int, device: torch.device = None) -> SignLanguageLSTM:
    """
    Convenience function to instantiate and move model to device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SignLanguageLSTM(
        input_size=225,
        hidden_size=256,
        num_layers=2,
        num_classes=num_classes,
        num_categories=5,
        dropout=0.5,
    ).to(device)

    return model


if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cpu")
    model = get_model(num_classes=8, device=device)
    dummy = torch.randn(4, 30, 225)  # batch=4, seq=30, features=225
    out = model(dummy)
    proba = model.predict_proba(dummy)
    print(f"\n[Test] Input: {dummy.shape} → Output: {out.shape}")
    print(f"[Test] Probabilities sum: {proba.sum(dim=-1)}")  # Should be ~1.0
