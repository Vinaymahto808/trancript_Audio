---
title: Audio Transcriber
emoji: 🎧
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
short_description: Batch audio/video transcription to CSV and DOCX
---

# Audio Transcriber

Upload many audio/video files at once and transcribe them in batches, then download every
transcript as one CSV, one DOCX, or one file per recording.

Two engines:

- **Cloud (Groq + auto-fallback, fastest)** — Whisper `large-v3-turbo` on Groq. Audio is
  converted to 16 kHz mono FLAC and split into 10-minute chunks. When a provider answers
  HTTP 429 (or runs out of quota) the next key is used automatically:
  Groq → OpenAI → Deepgram → Gemini → HuggingFace.
- **Local (faster-whisper, offline)** — CTranslate2 int8 on CPU, batched inference with
  voice-activity detection. Works without any API key; slower than the cloud engine.

## Configuration

Keys are read from the environment (or the `.env` file next to `app.py`). This Space uses
**Settings → Variables and secrets**. Add the ones you have; providers without a key are
skipped:

| Secret | Provider |
|---|---|
| `GROQ_API_KEY` | Groq (primary, fastest) |
| `OPENAI_API_KEY` | OpenAI `whisper-1` |
| `DEEPGRAM_API_KEY` | Deepgram `nova-3` |
| `GEMINI_API_KEY` | Gemini 2.5 Flash |
| `HF_TOKEN` | HuggingFace `whisper-large-v3` |

You can also paste a Groq key into the app's key box to override the Space secret for one
session.

## Notes

- FFmpeg is provided by the Space image and is required by both engines.
- The free CPU tier (2 vCPU, 16 GB) can run the local `small`/`base` models; larger local
  models need more RAM and a download on first use.
- Nothing is persisted: uploads and transcripts live in the container's temporary
  directory and disappear when the Space restarts.
