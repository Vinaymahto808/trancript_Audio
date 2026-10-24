"""Hugging Face Space entrypoint, generated from video_transcription_gradio.ipynb.

Rebuild with:  python build_space.py
"""
import csv
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import gradio as gr
import httpx
import numpy as np
from docx import Document
from faster_whisper import BatchedInferencePipeline, WhisperModel
from gradio.context import LocalContext

# ZeroGPU Spaces require at least one @spaces.GPU function, so the local engine
# runs on the GPU there. Everywhere else (notebook, CPU Space, GPU failure) the
# CPU path below is used unchanged. The package only exists in a ZeroGPU Space.
try:
    import spaces as _hf_spaces
except ImportError:
    _hf_spaces = None
_ZERO_GPU_FAILED = False

import sys

_PROJECT_DIR = Path(globals().get("__file__", Path.cwd() / "notebook.ipynb")).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))
from cloud_providers import (
    PROVIDERS,
    new_client,
    provider_status,
    set_groq_key_override,
    transcribe_chunk_with_fallback,
)

LOCAL_MODELS = ["small", "base", "tiny", "medium", "large-v3-turbo", "large-v3"]
CLOUD_MODELS = ["whisper-large-v3-turbo", "whisper-large-v3"]
ENGINES = ["Local (faster-whisper, offline)", "Cloud (Groq + auto-fallback, fastest)"]
# The cloud engine is ~15x faster than the CPU model (2 min of audio in ~3 s instead of
# ~70 s), so it is the default whenever a provider key is configured. Without a key the
# app still starts in the offline Local mode.
DEFAULT_ENGINE = ENGINES[1] if any(provider.has_key() for provider in PROVIDERS) else ENGINES[0]
# Parallelism for the cloud engine: files in a batch run together, and a shared slot
# counter keeps the total number of provider requests bounded either way.
CLOUD_FILE_WORKERS = 3
CLOUD_SLOTS = threading.BoundedSemaphore(CLOUD_FILE_WORKERS)
# Dropdown label -> Whisper language code (None = auto-detect). A hint matters for
# short or noisy clips such as call recordings: without it Whisper may detect
# English and translate Hindi speech into garbled English.
LANGUAGES = {
    "Auto-detect": None,
    "Hindi": "hi",
    "English": "en",
    "Punjabi": "pa",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
}
LOGPROB_THRESHOLD = -1.0  # Whisper default; lower-confidence segments are hallucinations on noise
RELAXED_VAD = {"threshold": 0.3, "min_speech_duration_ms": 100, "min_silence_duration_ms": 300}

# CTranslate2 runs best on physical cores; more threads mostly add memory use.
CPU_THREADS = max(1, min(4, (os.cpu_count() or 4)))
CHUNK_SECONDS = 600  # 10-minute FLAC chunks stay well under Groq's 25 MB limit

FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    raise RuntimeError(
        "FFmpeg was not found on PATH. Install FFmpeg, restart the notebook kernel, and run this cell again."
    )

print(f"Local engine: faster-whisper on CPU, int8, {CPU_THREADS} threads")


@lru_cache(maxsize=1)
def load_local_model(model_name, device="cpu", compute_type="int8"):
    """Load and cache one CTranslate2 Whisper model.

    Only one model is kept in memory; switching models frees the previous one.
    ``device`` is "cpu"/int8 normally and "cuda"/float16 on a ZeroGPU Space.
    """
    if model_name not in LOCAL_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=CPU_THREADS,
    )
    return BatchedInferencePipeline(model=model)


def run_ffmpeg(args):
    try:
        subprocess.run([FFMPEG, "-nostdin", "-y", *args], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        details = error.stderr.strip().splitlines()[-1] if error.stderr.strip() else "FFmpeg failed."
        raise gr.Error(f"Could not read audio from the uploaded file: {details}") from error


def load_pcm(audio_path, work_dir):
    """Decode any audio/video file to 16 kHz mono float32 samples via FFmpeg.

    faster-whisper's own decoder breaks with the PyAV version installed here, so the
    raw PCM is read with NumPy instead and the temporary file is removed right away.
    """
    pcm_path = work_dir / "audio.s16le"
    run_ffmpeg(["-i", str(audio_path), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-c:a", "pcm_s16le", str(pcm_path)])
    samples = np.fromfile(pcm_path, dtype=np.int16).astype(np.float32) / 32768.0
    pcm_path.unlink(missing_ok=True)
    if samples.size == 0:
        raise gr.Error("FFmpeg produced no audio from this file.")
    return samples


def _transcribe_samples(samples, model_name, device, compute_type, language=None, progress=None):
    """Run the local model over decoded PCM.

    ``progress`` is optional: a ZeroGPU call may run outside the request process,
    so it gets no callback and simply runs without reporting progress.
    """
    def report(fraction, desc=""):
        if progress is not None:
            progress(fraction, desc=desc)

    report(0.05, f"Loading {model_name} model")
    pipeline = load_local_model(model_name, device=device, compute_type=compute_type)
    total = max(samples.size / 16000.0, 1.0)

    def run_pass(batch_size, vad_parameters=None):
        kwargs = {"vad_parameters": vad_parameters} if vad_parameters else {}
        segments_iter, _info = pipeline.transcribe(
            samples,
            batch_size=batch_size,
            vad_filter=True,
            beam_size=1,
            word_timestamps=False,
            language=language,
            **kwargs,
        )
        segments = []
        for segment in segments_iter:
            if getattr(segment, "avg_logprob", 0.0) < LOGPROB_THRESHOLD:
                continue  # hallucinated text on ring tone / noise
            segments.append({"start": segment.start, "end": segment.end, "text": segment.text})
            report(
                min(0.05 + 0.9 * segment.end / total, 0.95),
                f"Transcribing {int(segment.end // 60)} min (batch {batch_size})",
            )
        return segments

    # Bigger batches are faster but need more RAM. On a low-memory machine the
    # encoder can fail with "mkl_malloc: failed to allocate memory"; when that
    # happens the whole pass is retried with a smaller batch.
    last_error = None
    for batch_size in (8, 4, 2, 1):
        try:
            segments = run_pass(batch_size)
            if not any(s["text"].strip() for s in segments):
                # Quiet telephony audio can slip under the default VAD; try once more
                # with a more sensitive VAD before giving up on the file.
                report(0.5, "No speech found, retrying with sensitive VAD")
                segments = run_pass(batch_size, RELAXED_VAD)
            return segments
        except (RuntimeError, MemoryError) as error:
            if "no clip timestamps" in str(error).lower():
                return []  # VAD found no speech at all
            if not any(word in str(error).lower() for word in ("allocate", "memory", "alloc")):
                raise
            last_error = error
            print(f"Low memory at batch size {batch_size}; retrying with a smaller batch.")
    raise gr.Error(
        "Not enough free RAM to run the local model. Close other programs (or stray Python servers) "
        f"and try again, or use the Cloud engine. Last error: {last_error}"
    )


def transcribe_local(audio_path, model_name, work_dir, progress, language=None):
    """Transcribe one file with the local engine; returns (segments, device label).

    On a ZeroGPU Space the model runs on the GPU (float16). Any GPU failure is
    remembered so later files go straight to the CPU path instead of retrying.
    """
    global _ZERO_GPU_FAILED
    samples = load_pcm(audio_path, work_dir)

    if _hf_spaces is not None and not _ZERO_GPU_FAILED:
        progress(0.05, desc=f"Loading {model_name} on the ZeroGPU device")
        try:
            segments = _transcribe_on_gpu(samples, model_name, language)
            return segments, f"faster-whisper {model_name} (ZeroGPU)"
        except Exception as error:  # noqa: BLE001 - a GPU problem must not break the app
            _ZERO_GPU_FAILED = True
            print(f"ZeroGPU run failed ({type(error).__name__}: {error}); using the CPU engine instead.")

    segments = _transcribe_samples(samples, model_name, "cpu", "int8", language, progress)
    return segments, f"faster-whisper {model_name} (CPU int8)"


if _hf_spaces is not None:
    @_hf_spaces.GPU(duration=300)
    def _transcribe_on_gpu(samples, model_name, language):
        """Local engine on the ZeroGPU device (float16, no Gradio progress)."""
        return _transcribe_samples(samples, model_name, "cuda", "float16", language)


def chunk_duration(path):
    """Length of an audio file in seconds, via ffprobe (same folder as ffmpeg)."""
    ffprobe = shutil.which("ffprobe") or str(Path(FFMPEG).with_name("ffprobe.exe"))
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return float(out)
    except (subprocess.CalledProcessError, ValueError, OSError):
        return float(CHUNK_SECONDS)


# ---------------------------------------------------------------------------
# Parallel helpers for the cloud engine.
#
# Gradio keeps the request context (what gr.Progress needs to report progress) in
# contextvars, and plain worker threads do not inherit them. Every parallel helper
# takes a snapshot of the caller's context and reinstalls it inside the worker, so
# the progress bar keeps moving while files or chunks are transcribed together.
# ---------------------------------------------------------------------------

_GRADIO_CONTEXT_VARS = (
    LocalContext.blocks,
    LocalContext.blocks_config,
    LocalContext.in_event_listener,
    LocalContext.event_id,
    LocalContext.request,
    LocalContext.progress,
)


def _capture_gradio_context():
    return tuple(var.get(None) for var in _GRADIO_CONTEXT_VARS)


def _call_in_context(context, fn, *args, **kwargs):
    for var, value in zip(_GRADIO_CONTEXT_VARS, context):
        var.set(value)
    return fn(*args, **kwargs)


class _MonotonicProgress:
    """Progress callback for parallel workers: never lets the bar jump backwards."""

    def __init__(self, progress):
        self._progress = progress
        self._lock = threading.Lock()
        self._last = 0.0

    def __call__(self, fraction, desc=""):
        with self._lock:
            value = max(float(fraction), self._last)
            self._last = value
        try:
            self._progress(value, desc=desc)
        except Exception:  # noqa: BLE001 - a progress hiccup must not stop a batch
            pass


def transcribe_cloud(audio_path, model_name, api_key, work_dir, progress, language=None):
    """Send 10-minute chunks to the cloud; switch provider automatically on rate limits.

    Keys come from the .env file (GROQ_API_KEY, OPENAI_API_KEY, DEEPGRAM_API_KEY,
    GEMINI_API_KEY, HF_TOKEN). A key typed into the app overrides the Groq key.
    Returns (segments, list of provider names that were used).
    """
    set_groq_key_override(api_key)
    if not any(p.has_key() for p in PROVIDERS):
        raise gr.Error(
            "No cloud API key found. Add GROQ_API_KEY (or OPENAI_API_KEY, DEEPGRAM_API_KEY, "
            "GEMINI_API_KEY, HF_TOKEN) to the .env file, or paste a Groq key in the key box."
        )

    chunk_pattern = work_dir / "chunk_%04d.flac"
    run_ffmpeg([
        "-i", str(audio_path), "-vn", "-ac", "1", "-ar", "16000",
        "-f", "segment", "-segment_time", str(CHUNK_SECONDS), "-c:a", "flac", str(chunk_pattern),
    ])
    chunks = sorted(work_dir.glob("chunk_*.flac"))
    if not chunks:
        raise gr.Error("FFmpeg produced no audio from this file.")

    # Durations are probed up front (cheap) so every chunk knows its offset, then the
    # chunks are fetched in parallel: a long recording becomes several requests that
    # overlap instead of one slow queue. CLOUD_SLOTS caps the total provider requests.
    durations = [chunk_duration(chunk) for chunk in chunks]
    starts = []
    offset = 0.0
    for duration in durations:
        starts.append(offset)
        offset += duration

    context = _capture_gradio_context()
    results = {}
    try:
        with new_client() as client:

            def fetch(index):
                with CLOUD_SLOTS:
                    provider_name, chunk_segments = transcribe_chunk_with_fallback(
                        client, chunks[index], durations[index], language
                    )
                return index, provider_name, chunk_segments

            if len(chunks) > 1:
                with ThreadPoolExecutor(max_workers=min(CLOUD_FILE_WORKERS, len(chunks))) as pool:
                    futures = {
                        pool.submit(_call_in_context, context, fetch, index): index
                        for index in range(len(chunks))
                    }
                    for future in as_completed(futures):
                        index, provider_name, chunk_segments = future.result()
                        results[index] = (provider_name, chunk_segments)
                        progress(
                            0.05 + 0.9 * len(results) / len(chunks),
                            desc=f"Cloud chunk {len(results)}/{len(chunks)}",
                        )
            else:
                _index, provider_name, chunk_segments = fetch(0)
                results[0] = (provider_name, chunk_segments)
                progress(0.95, desc="Cloud chunk 1/1")
    except RuntimeError as error:
        raise gr.Error(str(error)) from error
    finally:
        for chunk in chunks:
            chunk.unlink(missing_ok=True)

    segments = []
    used = []
    for index in range(len(chunks)):
        provider_name, chunk_segments = results[index]
        if provider_name not in used:
            used.append(provider_name)
        for segment in chunk_segments:
            segments.append({
                "start": starts[index] + float(segment["start"]),
                "end": starts[index] + float(segment["end"]),
                "text": segment.get("text", ""),
            })
    return segments, used


def format_timestamp(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def write_exports(segments, output_dir, stem="transcript"):
    csv_path = output_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["start_seconds", "end_seconds", "start", "end", "text"])
        for segment in segments:
            writer.writerow([
                round(segment["start"], 2),
                round(segment["end"], 2),
                format_timestamp(segment["start"]),
                format_timestamp(segment["end"]),
                segment["text"].strip(),
            ])

    docx_path = output_dir / f"{stem}.docx"
    document = Document()
    document.add_heading(f"Transcript: {stem}", level=1)
    for segment in segments:
        stamp = f"[{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}]"
        document.add_paragraph(f"{stamp} {segment['text'].strip()}")
    document.save(docx_path)
    return csv_path, docx_path


NO_SPEECH_HINT = (
    "no speech detected. The audio seems to contain only ring tone, hold music, silence or a "
    "very short announcement (for example an unanswered call). If the file does have a "
    "conversation, pick the right Language and try the Cloud engine."
)


def write_combined_exports(results, output_dir):
    """One CSV and one DOCX holding every file's transcript.

    ``results`` is a list of (file_name, segments). The CSV adds a ``file`` column;
    the DOCX gets a heading per file.
    """
    csv_path = output_dir / "all_transcripts.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file", "start_seconds", "end_seconds", "start", "end", "text"])
        for file_name, segments in results:
            for segment in segments:
                writer.writerow([
                    file_name,
                    round(segment["start"], 2),
                    round(segment["end"], 2),
                    format_timestamp(segment["start"]),
                    format_timestamp(segment["end"]),
                    segment["text"].strip(),
                ])

    docx_path = output_dir / "all_transcripts.docx"
    document = Document()
    document.add_heading("Audio Transcripts", level=1)
    document.add_paragraph(f"{len(results)} file(s)")
    for file_name, segments in results:
        document.add_heading(file_name, level=2)
        for segment in segments:
            stamp = f"[{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])}]"
            document.add_paragraph(f"{stamp} {segment['text'].strip()}")
    document.save(docx_path)
    return csv_path, docx_path


def transcribe_one(audio_path, engine, local_model, cloud_model, api_key, progress, language=None):
    """Transcribe one file.

    Returns (display_name, transcript, note, csv_path, docx_path, segments). When
    no speech is found, transcript is "" and the paths are None instead of
    raising, so a batch keeps going.
    """
    source = Path(audio_path)
    if not source.is_file():
        raise gr.Error(f"The uploaded file could not be found: {source.name}")

    output_dir = Path(tempfile.gettempdir()) / f"audio_transcript_{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    progress(0.0, desc="Converting audio")

    if engine.startswith("Cloud"):
        segments, providers_used = transcribe_cloud(source, cloud_model, api_key, output_dir, progress, language)
        used = "Cloud: " + " -> ".join(providers_used)
    else:
        segments, used = transcribe_local(source, local_model, output_dir, progress, language)

    segments = [s for s in segments if s["text"].strip()]
    elapsed = time.perf_counter() - started
    if not segments:
        return source.name, "", f"{used}: {NO_SPEECH_HINT} ({elapsed:.1f} s)", None, None, []

    transcript = " ".join(s["text"].strip() for s in segments)
    csv_path, docx_path = write_exports(segments, output_dir, source.stem)

    audio_len = segments[-1]["end"]
    note = f"{used}: {format_timestamp(audio_len)} of audio in {elapsed:.1f} s ({audio_len / max(elapsed, 0.01):.0f}x real time)"
    return source.name, transcript, note, str(csv_path), str(docx_path), segments


def transcribe_batch(paths, engine, local_model, cloud_model, api_key, language, progress, offset=0, total=None):
    """Transcribe a slice of files; return one record per file.

    Each record: {"name", "ok", "transcript", "note", "csv", "docx", "segments"}.
    A file with no speech (or a failure) does not stop the batch. ``offset`` and
    ``total`` only shape the progress label, e.g. "[105/350] name".
    """
    total = total or (offset + len(paths))

    # Parallel workers all report progress into the same bar, so keep it monotonic.
    if engine.startswith("Cloud"):
        progress = _MonotonicProgress(progress)

    def transcribe_index(index, path):
        name = Path(path).name
        position = offset + index + 1

        def file_progress(fraction, desc="", _position=position, _name=name):
            progress((_position - 1 + fraction) / max(total, 1), desc=f"[{_position}/{total}] {_name}: {desc}")

        try:
            display_name, transcript, note, csv_path, docx_path, segments = transcribe_one(
                path, engine, local_model, cloud_model, api_key, file_progress, language
            )
        except gr.Error as error:
            return {"name": name, "ok": False, "transcript": "", "note": f"FAILED, {error}",
                    "csv": None, "docx": None, "segments": []}
        return {"name": display_name, "ok": bool(transcript), "transcript": transcript, "note": note,
                "csv": csv_path, "docx": docx_path, "segments": segments}

    # The cloud engine mostly waits on the API, so files overlap for free; the local
    # engine already keeps the CPU busy with one file and stays serial.
    if engine.startswith("Cloud") and len(paths) > 1:
        context = _capture_gradio_context()
        with ThreadPoolExecutor(max_workers=min(CLOUD_FILE_WORKERS, len(paths))) as pool:
            futures = [
                pool.submit(_call_in_context, context, transcribe_index, index, path)
                for index, path in enumerate(paths)
            ]
            return [future.result() for future in futures]

    return [transcribe_index(index, path) for index, path in enumerate(paths)]


def build_outputs(records, uploaded_total=None):
    """Turn accumulated records into the app's outputs (transcript, summary, files)."""
    ok = [r for r in records if r["ok"]]
    transcripts = []
    for r in records:
        body = r["transcript"] if r["ok"] else ("(failed)" if r["note"].startswith("FAILED") else "(no speech detected)")
        transcripts.append(f"=== {r['name']} ===" + chr(10) + body)
    notes = [f"- **{r['name']}**: {r['note']}" for r in records]
    uploaded_total = uploaded_total or len(records)
    summary = (f"Transcribed {len(ok)} of {len(records)} processed file{'s' if len(records) != 1 else ''} "
               f"({uploaded_total} uploaded):" + chr(10) + chr(10).join(notes))

    all_csv = all_docx = None
    if ok:
        combined_dir = Path(tempfile.gettempdir()) / f"audio_transcript_all_{uuid.uuid4().hex}"
        combined_dir.mkdir(parents=True, exist_ok=False)
        all_csv, all_docx = write_combined_exports([(r["name"], r["segments"]) for r in ok], combined_dir)
        all_csv, all_docx = str(all_csv), str(all_docx)
    csv_paths = [r["csv"] for r in ok] or None
    docx_paths = [r["docx"] for r in ok] or None
    return (chr(10) * 2).join(transcripts), summary, csv_paths, docx_paths, all_csv, all_docx


def transcribe_files(files, engine, local_model, cloud_model, api_key, language_label="Auto-detect", progress=gr.Progress()):
    """Transcribe every uploaded file in one go (single-call API)."""
    if not files:
        raise gr.Error("Upload at least one audio file before starting transcription.")
    if isinstance(files, (str, Path)):
        files = [files]
    paths = [f if isinstance(f, (str, Path)) else getattr(f, "name", f) for f in files]
    language = LANGUAGES.get(language_label)
    records = transcribe_batch(paths, engine, local_model, cloud_model, api_key, language, progress)
    if not any(r["ok"] for r in records):
        raise gr.Error("; ".join(f"{r['name']}: {r['note']}" for r in records))
    progress(1.0, desc="Done")
    return build_outputs(records)


# Kept for backwards compatibility with the old single-file API.
def transcribe_audio(audio_path, engine, local_model, cloud_model, api_key, progress=gr.Progress()):
    _name, transcript, note, csv_path, docx_path, _segments = transcribe_one(
        audio_path, engine, local_model, cloud_model, api_key, progress
    )
    return transcript, note, csv_path, docx_path

required_setup = ("gr", "ENGINES", "transcribe_batch", "build_outputs")
missing_setup = [name for name in required_setup if name not in globals()]
if missing_setup:
    raise RuntimeError(
        "Setup is incomplete. Run the install cell and the helpers cell before launching the app. "
        f"Missing: {', '.join(missing_setup)}"
    )

DEFAULT_BATCH_SIZE = 1000


def new_state(paths=None):
    return {"paths": list(paths or []), "done": 0, "records": []}


def batch_plan(state, batch_size):
    """Return (button label, status text) for the current position in the upload list."""
    n, done = len(state["paths"]), state["done"]
    batch_size = max(1, int(batch_size or DEFAULT_BATCH_SIZE))
    if n == 0:
        return "Transcribe", "Upload audio files to begin."
    if done >= n:
        return "All files done", f"All {n} uploaded files are transcribed. Upload new files or click Reset to start again."
    end = min(done + batch_size, n)
    status = (f"{n} files uploaded, {done} done. Next batch: files {done + 1} to {end}."
              if done else f"{n} files uploaded. First batch: files 1 to {end}.")
    return f"Transcribe files {done + 1}-{end} of {n}", status


def on_upload(files, batch_size):
    paths = [f if isinstance(f, str) else getattr(f, "name", f) for f in (files or [])]
    state = new_state(paths)
    label, status = batch_plan(state, batch_size)
    return state, status, gr.update(value=label), "", "", None, None, None, None


def on_batch_size(state, batch_size):
    label, status = batch_plan(state, batch_size)
    return status, gr.update(value=label)


def on_reset():
    state = new_state()
    label, status = batch_plan(state, DEFAULT_BATCH_SIZE)
    return state, None, status, gr.update(value=label), "", "", None, None, None, None


def run_next_batch(state, batch_size, engine, local_model, cloud_model, api_key, language_label, progress=gr.Progress()):
    """Transcribe the next ``batch_size`` files; results accumulate across batches."""
    state = state or new_state()
    paths = state["paths"]
    if not paths:
        raise gr.Error("Upload audio files first.")
    n, done = len(paths), state["done"]
    if done >= n:
        raise gr.Error(f"All {n} files are already transcribed. Upload new files or click Reset.")
    batch_size = max(1, int(batch_size or DEFAULT_BATCH_SIZE))
    end = min(done + batch_size, n)

    records = transcribe_batch(
        paths[done:end], engine, local_model, cloud_model, api_key, LANGUAGES.get(language_label),
        progress, offset=done, total=n,
    )
    state["records"].extend(records)
    state["done"] = end

    progress(0.99, desc="Writing combined CSV and DOCX")
    transcript, summary, csv_paths, docx_paths, all_csv, all_docx = build_outputs(state["records"], uploaded_total=n)
    progress(1.0, desc="Done")
    label, status = batch_plan(state, batch_size)
    status = f"Batch {done + 1}-{end} finished. " + status
    return transcript, summary, csv_paths, docx_paths, all_csv, all_docx, status, gr.update(value=label), state


with gr.Blocks(title="Audio Transcriber") as demo:
    gr.Markdown(
        "# Audio Transcriber" + chr(10) +
        "Upload up to 1,000 audio/video files at once. They are transcribed in batches (default 1,000 files per batch): "
        "click the button for the first batch, then again for the next one. Results and downloads accumulate. "
        "**Cloud** is fastest (Groq with automatic fallback to other keys in .env); **Local** works offline."
    )
    state = gr.State(new_state())
    with gr.Row():
        audio_input = gr.File(
            label="Audio files (select all of them at once)",
            file_count="multiple",
            file_types=["audio", "video"],
            type="filepath",
        )
        with gr.Column():
            batch_size_input = gr.Number(
                value=DEFAULT_BATCH_SIZE, precision=0, minimum=1, label="Files per batch",
            )
            language_input = gr.Dropdown(
                choices=list(LANGUAGES),
                value="Auto-detect",
                label="Language",
                info="Pick Hindi for call recordings: short or noisy clips are often mis-detected as English.",
            )
            engine_input = gr.Radio(
                choices=ENGINES,
                value=DEFAULT_ENGINE,
                label="Engine",
                info="Cloud (Groq) is ~15x faster than the local CPU model; Local works offline.",
            )
            local_model_input = gr.Dropdown(
                choices=LOCAL_MODELS,
                value="small",
                label="Local model",
                info="small balances quality and speed; base is ~3.5x faster (about 8x real-time "
                     "on this PC) but less accurate. medium and large must download first.",
            )
            cloud_model_input = gr.Dropdown(
                choices=CLOUD_MODELS,
                value="whisper-large-v3-turbo",
                label="Cloud model",
                visible=False,
            )
            api_key_input = gr.Textbox(
                label="Groq API key (optional, overrides .env)",
                type="password",
                placeholder="gsk_... (leave empty to use .env)",
                visible=False,
            )
            provider_status_output = gr.Markdown(visible=False)

    def status_markdown():
        header = "**Cloud providers (used in this order, auto-switch on rate limit):**"
        return header + chr(10) + chr(10).join(f"- {line}" for line in provider_status())

    def toggle_engine(engine):
        cloud = engine.startswith("Cloud")
        return (
            gr.update(visible=not cloud),
            gr.update(visible=cloud),
            gr.update(visible=cloud),
            gr.update(visible=cloud, value=status_markdown()),
        )

    engine_input.change(
        toggle_engine,
        inputs=engine_input,
        outputs=[local_model_input, cloud_model_input, api_key_input, provider_status_output],
    )

    batch_status = gr.Markdown("Upload audio files to begin.")
    with gr.Row():
        transcribe_button = gr.Button("Transcribe", variant="primary", scale=3)
        reset_button = gr.Button("Reset", variant="secondary", scale=1)
    timing_output = gr.Markdown()
    transcript_output = gr.Textbox(label="Transcript (all files so far)", lines=16)
    with gr.Row():
        all_csv_output = gr.File(label="Download ALL transcripts (one CSV)")
        all_docx_output = gr.File(label="Download ALL transcripts (one DOCX)")
    with gr.Row():
        csv_output = gr.File(label="Download CSV (one per file)", file_count="multiple")
        docx_output = gr.File(label="Download DOCX (one per file)", file_count="multiple")

    result_outputs = [transcript_output, timing_output, csv_output, docx_output, all_csv_output, all_docx_output]

    audio_input.change(
        on_upload,
        inputs=[audio_input, batch_size_input],
        outputs=[state, batch_status, transcribe_button] + result_outputs,
    )
    batch_size_input.change(
        on_batch_size,
        inputs=[state, batch_size_input],
        outputs=[batch_status, transcribe_button],
    )
    reset_button.click(
        on_reset,
        inputs=None,
        outputs=[state, audio_input, batch_status, transcribe_button] + result_outputs,
    )
    transcribe_button.click(
        fn=run_next_batch,
        inputs=[state, batch_size_input, engine_input, local_model_input, cloud_model_input, api_key_input, language_input],
        outputs=result_outputs + [batch_status, transcribe_button, state],
        api_name="run_next_batch",
    ).then(
        fn=lambda engine: gr.update(visible=engine.startswith("Cloud"), value=status_markdown()),
        inputs=engine_input,
        outputs=provider_status_output,
    )

demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
