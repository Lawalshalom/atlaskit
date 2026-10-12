"""Speech recognition with the official N-ATLaS ASR models (Whisper-small fine-tunes).

Models are loaded lazily, one per language, on first use, so a gateway that is
only used for text never pays the memory cost.
"""
from __future__ import annotations

import io
import threading
from typing import Protocol

from .core import LANGUAGES, AtlasError

TARGET_SR = 16_000


class Transcriber(Protocol):
    def transcribe(self, audio_bytes: bytes, language: str) -> dict: ...


def decode_audio(audio_bytes: bytes) -> "tuple[object, int]":
    """Decode any common audio container (wav, mp3, ogg/opus WhatsApp notes, m4a) to 16 kHz mono."""
    import numpy as np

    try:
        import librosa  # handles mp3/ogg/m4a via soundfile/audioread + ffmpeg
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SR, mono=True)
        return audio.astype(np.float32), sr
    except Exception as exc:  # pragma: no cover - depends on system codecs
        raise AtlasError(f"Could not decode audio: {exc}", 400, "invalid_audio") from exc


class WhisperTranscriber:
    """Routes audio to NCAIR1/<Language>-ASR using the transformers pipeline."""

    def __init__(self, device: str = "auto", max_seconds: int = 300):
        self.device = device
        self.max_seconds = max_seconds
        self._pipes: dict[str, object] = {}
        self._lock = threading.Lock()

    def _pipe(self, language: str):
        with self._lock:
            if language not in self._pipes:
                import torch
                from transformers import pipeline

                device = self.device
                if device == "auto":
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                self._pipes[language] = pipeline(
                    "automatic-speech-recognition",
                    model=LANGUAGES[language]["asr_model"],
                    device=device,
                    torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
                    chunk_length_s=30,
                )
            return self._pipes[language]

    def loaded(self) -> list[str]:
        return sorted(self._pipes)

    def transcribe(self, audio_bytes: bytes, language: str) -> dict:
        audio, sr = decode_audio(audio_bytes)
        duration = len(audio) / sr
        if duration > self.max_seconds:
            raise AtlasError(f"Audio is {duration:.0f}s; the limit is {self.max_seconds}s", 413, "audio_too_long")
        result = self._pipe(language)({"raw": audio, "sampling_rate": sr})
        return {
            "text": result["text"].strip(),
            "language": language,
            "duration": round(duration, 2),
            "model": LANGUAGES[language]["asr_model"],
        }
