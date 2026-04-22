"""
inference/sentence_builder.py - Build sentences from a stream of predicted words.

Features:
  • Avoids consecutive duplicates
  • Enforces a max sentence length (5–8 words)
  • Handles special commands: "clear", "speak"
  • Capitalises the finished sentence
  • Cooldown between word additions (1–2 s) to prevent spam
"""

import time
from typing import List, Optional


# Words that trigger commands instead of being added to the sentence
COMMAND_WORDS = {"clear", "speak"}


class SentenceBuilder:
    """
    Accumulates predicted gesture words into a running sentence.

    Args:
        max_words:    Maximum words in a sentence before auto-clearing (default 8).
        min_words:    Words to keep after auto-clear (soft lower bound for display).
        cooldown_sec: Minimum seconds between consecutive word additions (default 1.5).
    """

    def __init__(
        self,
        max_words: int = 8,
        cooldown_sec: float = 1.5,
    ):
        self.max_words = max_words
        self.cooldown_sec = cooldown_sec

        self._words: List[str] = []
        self._last_added_time: float = 0.0
        self._last_added_word: Optional[str] = None
        self._pending_command: Optional[str] = None  # "clear" or "speak"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def try_add_word(self, word: str) -> Optional[str]:
        """
        Attempt to add *word* to the sentence.

        Returns:
            "added"   – word was appended
            "command" – a command word was detected (check pending_command)
            "dup"     – duplicate consecutive word, skipped
            "cooldown"– too soon after last addition, skipped
            None      – empty / whitespace word
        """
        word = word.strip().lower()
        if not word:
            return None

        # ── Special commands ──────────────────────────────────────────
        if word in COMMAND_WORDS:
            self._pending_command = word
            return "command"

        # ── Cooldown gate ─────────────────────────────────────────────
        now = time.time()
        if now - self._last_added_time < self.cooldown_sec:
            return "cooldown"

        # ── Duplicate suppression ─────────────────────────────────────
        if word == self._last_added_word:
            return "dup"

        # ── Max length: auto-clear if full ────────────────────────────
        if len(self._words) >= self.max_words:
            self._words.clear()

        self._words.append(word)
        self._last_added_word = word
        self._last_added_time = now
        return "added"

    def add_word_force(self, word: str):
        """Force-add a word (skips cooldown/duplicate check). Manual override."""
        word = word.strip().lower()
        if not word: return

        if word in COMMAND_WORDS:
            self._pending_command = word
            return

        if len(self._words) >= self.max_words:
            self._words.clear()

        self._words.append(word)
        self._last_added_word = word
        self._last_added_time = time.time()

    # ------------------------------------------------------------------
    def pop_command(self) -> Optional[str]:
        """Retrieve and clear any pending command."""
        cmd = self._pending_command
        self._pending_command = None
        return cmd

    # ------------------------------------------------------------------
    def clear(self):
        """Reset the sentence buffer."""
        self._words.clear()
        self._last_added_word = None
        # Keep cooldown timestamp so rapid clears don't fire immediately

    # ------------------------------------------------------------------
    @property
    def sentence(self) -> str:
        """Return the current sentence, capitalised."""
        if not self._words:
            return ""
        raw = " ".join(self._words)
        return raw[0].upper() + raw[1:]

    # ------------------------------------------------------------------
    @property
    def words(self) -> List[str]:
        """Return the raw word list (copy)."""
        return list(self._words)

    # ------------------------------------------------------------------
    @property
    def word_count(self) -> int:
        return len(self._words)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"SentenceBuilder(words={self._words!r})"
