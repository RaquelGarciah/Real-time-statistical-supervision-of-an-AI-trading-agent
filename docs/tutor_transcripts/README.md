# Transcripciones del tutor

Este directorio contiene las transcripciones (Whisper-large-v3 vía Groq) de los audios reales del tutor del TFG. Es **ground truth de tono y de exigencias**: el subagente `harvard-professor` lo lee en cada invocación y prioriza la voz literal del tutor sobre cualquier intuición.

## Cómo se generan

```bash
# 1. Mete el audio (.m4a, .mp3, .wav, ...) en ../tutor_audios/
# 2. Asegura GROQ_API_KEY en .env (https://console.groq.com/keys)
# 3. Ejecuta desde la raíz del repo
.venv/bin/python tools/transcribe_audio.py
```

El script cachea por nombre de fichero: si ya existe la transcripción, la salta (`--force` para reescribir).

## Política de privacidad

- Los **audios crudos** viven en `docs/tutor_audios/` y NO se versionan (gitignored).
- Las **transcripciones .md** sí se versionan; son parte del expediente del TFG.

## Índice
- [Calle de Hilarión Eslava, 46 2](Calle de Hilarión Eslava, 46 2.md) — añadido 2026-06-08
