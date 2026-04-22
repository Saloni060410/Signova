"""
inference/tts.py - Text-to-speech output using pyttsx3.

Runs TTS in a background thread so it never blocks the main inference loop.
Includes a minimum interval between consecutive speeches to avoid spam.
"""

import threading
import time
from typing import Optional


class TTSEngine:
    """
    Thin wrapper around pyttsx3 that:
      • Runs speech in a daemon thread (non-blocking)
      • Enforces a minimum gap between utterances
      • Silently degrades if pyttsx3 is not installed

    Args:
        min_interval_sec: Minimum seconds between two speak() calls (default 3.0).
        rate:             Words per minute (default 150).
        volume:           Volume 0.0–1.0 (default 0.9).
    """

    def __init__(
        self,
        min_interval_sec: float = 3.0,
        rate: int = 150,
        volume: float = 0.9,
    ):
        self.min_interval_sec = min_interval_sec
        self._last_spoken_time: float = 0.0
        self._lock = threading.Lock()
        self._engine = None
        self._available = False

        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", rate)
            self._engine.setProperty("volume", volume)
            self._available = True
            print("[TTS] pyttsx3 initialised successfully.")
        except Exception as e:
            print(f"[TTS] WARNING: pyttsx3 not available — TTS disabled. ({e})")

    # ------------------------------------------------------------------
    def speak(self, text: str, force: bool = False) -> bool:
        """
        Speak *text* in a background thread.

        Args:
            text:  The string to speak.
            force: If True, ignore the minimum interval.

        Returns:
            True  if speech was dispatched,
            False if skipped (cooldown or unavailable).
        """
        if not self._available or not text.strip():
            return False

        now = time.time()
        if not force and now - self._last_spoken_time < self.min_interval_sec:
            return False

        # Update timestamp before thread starts to prevent races
        with self._lock:
            self._last_spoken_time = time.time()

        thread = threading.Thread(target=self._run, args=(text,), daemon=True)
        thread.start()
        return True

    # ------------------------------------------------------------------
    def _run(self, text: str):
        """Internal: run pyttsx3 in the calling thread."""
        try:
            import pyttsx3
            # Each thread needs its own engine instance to avoid mutex issues
            engine = pyttsx3.init()
            engine.setProperty("rate", self._engine.getProperty("rate"))
            engine.setProperty("volume", self._engine.getProperty("volume"))
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[TTS] Error during speech: {e}")

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        """True if pyttsx3 loaded successfully."""
        return self._available

    # ------------------------------------------------------------------
    def set_rate(self, rate: int):
        """Change speech rate (words per minute)."""
        if self._available:
            self._engine.setProperty("rate", rate)

    # ------------------------------------------------------------------
    def set_volume(self, volume: float):
        """Change volume (0.0 – 1.0)."""
        if self._available:
            self._engine.setProperty("volume", max(0.0, min(1.0, volume)))
