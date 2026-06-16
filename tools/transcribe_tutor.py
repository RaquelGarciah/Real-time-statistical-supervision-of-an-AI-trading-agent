"""Transcribe el audio de una reunión con el tutor a Markdown con timestamps.

Uso: python tools/transcribe_tutor.py <audio.m4a> [salida.md]
Modelo medium, español, int8 en CPU. faster-whisper (CTranslate2).
"""
import sys
from pathlib import Path

from faster_whisper import WhisperModel

audio = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else audio.with_suffix(".md")

model = WhisperModel("medium", device="cpu", compute_type="int8")
segments, info = model.transcribe(str(audio), language="es", vad_filter=True)


def ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


lines = [
    f"# Transcripción — {audio.stem}",
    "",
    f"*Audio: {audio.name} · duración {info.duration/60:.1f} min · "
    f"transcrito con faster-whisper (medium, es). Calidad de transcripción automática, revisar.*",
    "",
]
for seg in segments:
    lines.append(f"**[{ts(seg.start)}]** {seg.text.strip()}")
    print(f"[{ts(seg.start)}] {seg.text.strip()}", flush=True)

out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\n>>> escrito: {out}", flush=True)
