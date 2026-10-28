# call-recording-transcriber

Fast, parallel call-recording transcription using [Deepgram](https://deepgram.com/).

Transcribes every audio file in a folder using multiple workers at once
instead of one file at a time — a batch of a few thousand recordings that
would take hours sequentially finishes in minutes. Handles mixed-language
audio well (e.g. Hindi/English code-switching) using Deepgram's
`nova-3` model with `language=multi`.

## Setup

```bash
pip install -r requirements.txt
```

Get a Deepgram API key from https://console.deepgram.com/ and set it:

```bash
cp .env.example .env
# edit .env and put your key in, then:
export DEEPGRAM_API_KEY=your_key_here      # macOS/Linux
set DEEPGRAM_API_KEY=your_key_here         # Windows cmd
$env:DEEPGRAM_API_KEY="your_key_here"      # Windows PowerShell
```

(Or just pass `--api-key your_key_here` on the command line instead.)

## Usage

```bash
python transcribe.py --folder "/path/to/recordings" --out results.jsonl
```

This writes one JSON line per audio file to `results.jsonl`, e.g.:

```json
{"fileName": "call123.mp3", "fileSize": 48213, "duration": 32.5, "transcript": "...", "error": null}
```

### Also generate a formatted Excel report

```bash
python transcribe.py --folder "/path/to/recordings" --out results.jsonl --excel report.xlsx
```

Produces `report.xlsx` with a **Transcripts** sheet (one row per call) and a
**Summary** sheet (totals, success/fail counts, total duration).

### Options

| Flag            | Default   | Description |
|-----------------|-----------|-------------|
| `--folder`      | —         | Folder of audio files (`.mp3`, `.wav`, `.m4a`, `.ogg`) |
| `--out`         | —         | Output JSONL file |
| `--excel`       | (none)    | Also write an `.xlsx` report |
| `--api-key`     | `$DEEPGRAM_API_KEY` | Deepgram API key |
| `--model`       | `nova-3`  | Deepgram model |
| `--language`    | `multi`   | Use `multi` for mixed-language audio, or a specific code like `hi`/`en` |
| `--workers`     | `10`      | Number of parallel requests |
| `--timeout`     | `300`     | Per-file timeout (seconds) — long calls need a generous timeout |
| `--retry-failed`| off       | Drop failed records from `--out` before running, so they're retried |

## Resuming / retrying

The script is safe to stop (Ctrl+C) and re-run — it skips any file already
recorded in the `--out` JSONL file. If a run finishes with some failures
(timeouts, network blips), just re-run with `--retry-failed`:

```bash
python transcribe.py --folder "/path/to/recordings" --out results.jsonl --retry-failed
```

## Why this is faster than transcribing one file at a time

Each Deepgram request spends most of its time waiting on the network, not
using CPU, so running many requests concurrently (`--workers`, default 10)
gets you roughly an N-times speedup for N workers, up to Deepgram's rate
limits. For a few thousand short call recordings this is the difference
between minutes and hours. If you still hit timeouts/rate limits, lower
`--workers`; if it's going too slow and your plan allows it, raise it.
