"""Build the Hugging Face Space payload from the notebook.

Extracts every code cell of video_transcription_gradio.ipynb (minus IPython
magics), rewrites the launch call for the Space runtime, applies the ZeroGPU
patches (see ZERO_GPU_PATCHES) and writes the result plus cloud_providers.py
into ./hf_space ready to push.
"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "video_transcription_gradio.ipynb"
SPACE = ROOT / "hf_space"

HEADER = '''"""Hugging Face Space entrypoint, generated from video_transcription_gradio.ipynb.

Rebuild with:  python build_space.py
"""
'''

OLD_LAUNCH = 'demo.launch(share=False, server_port=8002, show_error=True)'
NEW_LAUNCH = 'demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)'

# --------------------------------------------------------------------------
# ZeroGPU patches.
#
# Gradio Spaces on the free cpu-basic hardware now require a PRO subscription;
# the free path is a ZeroGPU Space, which refuses to boot unless the app defines
# at least one @spaces.GPU function ("No @spaces.GPU function detected during
# startup"). So the local engine gets a real GPU path: faster-whisper on CUDA,
# with the original CPU code kept as the fallback for the notebook and for any
# GPU failure. Each entry is (anchor, replacement) and must match exactly once.
# --------------------------------------------------------------------------

IMPORT_ANCHOR = "from faster_whisper import BatchedInferencePipeline, WhisperModel\n"

IMPORT_REPLACEMENT = '''from faster_whisper import BatchedInferencePipeline, WhisperModel

# ZeroGPU Spaces require at least one @spaces.GPU function, so the local engine
# runs on the GPU there. Everywhere else (notebook, CPU Space, GPU failure) the
# CPU path below is used unchanged. The package only exists in a ZeroGPU Space.
try:
    import spaces as _hf_spaces
except ImportError:
    _hf_spaces = None
_ZERO_GPU_FAILED = False
'''

MODEL_ANCHOR = '''@lru_cache(maxsize=1)
def load_local_model(model_name):
    """Load and cache one CTranslate2 Whisper model (int8 on CPU).

    Only one model is kept in memory; switching models frees the previous one.
    """
    if model_name not in LOCAL_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=CPU_THREADS,
    )
    return BatchedInferencePipeline(model=model)
'''

MODEL_REPLACEMENT = '''@lru_cache(maxsize=1)
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
'''

LOCAL_HEADER_ANCHOR = '''def transcribe_local(audio_path, model_name, work_dir, progress, language=None):
    samples = load_pcm(audio_path, work_dir)

    progress(0.05, desc=f"Loading {model_name} model")
    pipeline = load_local_model(model_name)
    total = max(samples.size / 16000.0, 1.0)
'''

LOCAL_HEADER_REPLACEMENT = '''def _transcribe_samples(samples, model_name, device, compute_type, language=None, progress=None):
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
'''

SEGMENT_PROGRESS_ANCHOR = '''            segments.append({"start": segment.start, "end": segment.end, "text": segment.text})
            progress(
                min(0.05 + 0.9 * segment.end / total, 0.95),
                desc=f"Transcribing {int(segment.end // 60)} min (batch {batch_size})",
            )
'''

SEGMENT_PROGRESS_REPLACEMENT = '''            segments.append({"start": segment.start, "end": segment.end, "text": segment.text})
            report(
                min(0.05 + 0.9 * segment.end / total, 0.95),
                f"Transcribing {int(segment.end // 60)} min (batch {batch_size})",
            )
'''

VAD_PROGRESS_ANCHOR = '                progress(0.5, desc="No speech found, retrying with sensitive VAD")\n'
VAD_PROGRESS_REPLACEMENT = '                report(0.5, "No speech found, retrying with sensitive VAD")\n'

LOCAL_TAIL_ANCHOR = '''    raise gr.Error(
        "Not enough free RAM to run the local model. Close other programs (or stray Python servers) "
        f"and try again, or use the Cloud engine. Last error: {last_error}"
    )


def chunk_duration(path):
'''

LOCAL_TAIL_REPLACEMENT = '''    raise gr.Error(
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
'''

DISPATCH_ANCHOR = '''    else:
        segments = transcribe_local(source, local_model, output_dir, progress, language)
        used = f"faster-whisper {local_model} (CPU int8)"
'''

DISPATCH_REPLACEMENT = '''    else:
        segments, used = transcribe_local(source, local_model, output_dir, progress, language)
'''

ZERO_GPU_PATCHES = [
    (IMPORT_ANCHOR, IMPORT_REPLACEMENT),
    (MODEL_ANCHOR, MODEL_REPLACEMENT),
    (LOCAL_HEADER_ANCHOR, LOCAL_HEADER_REPLACEMENT),
    (SEGMENT_PROGRESS_ANCHOR, SEGMENT_PROGRESS_REPLACEMENT),
    (VAD_PROGRESS_ANCHOR, VAD_PROGRESS_REPLACEMENT),
    (LOCAL_TAIL_ANCHOR, LOCAL_TAIL_REPLACEMENT),
    (DISPATCH_ANCHOR, DISPATCH_REPLACEMENT),
]


def code_cells(notebook_path):
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in data["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        lines = [line for line in source.splitlines() if not line.lstrip().startswith("%")]
        cell_source = "\n".join(lines).strip()
        if cell_source:
            yield cell_source


def apply_zero_gpu_patches(script):
    """Turn the notebook code into a ZeroGPU-capable Space app."""
    for index, (anchor, replacement) in enumerate(ZERO_GPU_PATCHES, start=1):
        matches = script.count(anchor)
        if matches != 1:
            raise SystemExit(
                f"ZeroGPU patch {index} matched {matches} times instead of once; the notebook "
                "code changed, so update ZERO_GPU_PATCHES in build_space.py."
            )
        script = script.replace(anchor, replacement)
    return script


def main():
    script = "\n\n".join(code_cells(NOTEBOOK))
    if OLD_LAUNCH not in script:
        raise SystemExit(f"Launch call not found; expected:\n  {OLD_LAUNCH}")
    script = script.replace(OLD_LAUNCH, NEW_LAUNCH)
    script = apply_zero_gpu_patches(script)

    if "def _transcribe_on_gpu" not in script or "import spaces as _hf_spaces" not in script:
        raise SystemExit("ZeroGPU patch did not apply as expected.")

    SPACE.mkdir(exist_ok=True)
    (SPACE / "app.py").write_text(HEADER + script + "\n", encoding="utf-8")
    shutil.copy2(ROOT / "cloud_providers.py", SPACE / "cloud_providers.py")
    print(f"Wrote {SPACE / 'app.py'} ({script.count(chr(10)) + 1} lines)")


if __name__ == "__main__":
    main()
