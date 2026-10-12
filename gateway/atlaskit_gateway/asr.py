"""Speech recognition with the official N-ATLaS ASR models (Whisper-small fine-tunes).

Models are loaded lazily, one per language, on first use, so a gateway that is
only used for text never pays the memory cost.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import threading
from typing import Protocol

from .core import LANGUAGES, AtlasError

TARGET_SR = 16_000


class Transcriber(Protocol):
    def transcribe(self, audio_bytes: bytes, language: str) -> dict: ...


def decode_audio(audio_bytes: bytes) -> "tuple[object, int]":
    """Decode any common audio container to 16 kHz mono float32.

    libsndfile reads wav/flac/ogg-vorbis/mp3 from memory. Browser recordings
    (webm/opus), WhatsApp notes (ogg/opus on older libsndfile) and phone
    recordings (m4a/aac) need ffmpeg.
    """
    import numpy as np

    try:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        return np.ascontiguousarray(audio, dtype=np.float32), TARGET_SR
    except Exception:
        return _ffmpeg_decode(audio_bytes), TARGET_SR


def _ffmpeg_decode(audio_bytes: bytes):
    import numpy as np

    if not shutil.which("ffmpeg"):
        raise AtlasError("Could not decode audio: this format needs ffmpeg on the server", 400, "invalid_audio")
    # A temporary file, not a pipe: m4a files often keep their index at the end, so ffmpeg must seek.
    with tempfile.NamedTemporaryFile() as f:
        f.write(audio_bytes)
        f.flush()
        proc = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", f.name,
                               "-f", "f32le", "-ac", "1", "-ar", str(TARGET_SR), "pipe:1"],
                              capture_output=True, timeout=120)
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["no audio stream"]
        raise AtlasError(f"Could not decode audio: {detail[0]}", 400, "invalid_audio")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


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
