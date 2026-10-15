"""Run every code cell of video_transcription_gradio.ipynb in order.

This launches the same Gradio app the notebook does, without needing Jupyter.
It also works around this machine's setup:
  * caps BLAS/OMP threads so numpy/torch avoid the OpenBLAS allocation error
  * puts the winget-installed ffmpeg on PATH (the notebook requires it)
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# When started without a console (hidden window, scheduler, double-click on
# pythonw) sys.stdout/sys.stderr are None and uvicorn's logger crashes on
# sys.stdout.isatty(). Point them at a log file next to this script instead.
if sys.stdout is None or sys.stderr is None:
    _log = open(Path(__file__).with_name("app.log"), "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = _log
    if sys.stderr is None:
        sys.stderr = _log

FFMPEG_BIN = Path(
    r"C:\Users\Galaxy\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-9.0.2-full_build\bin"
)
if FFMPEG_BIN.is_dir():
    os.environ["PATH"] = str(FFMPEG_BIN) + os.pathsep + os.environ["PATH"]

NOTEBOOK = Path(__file__).with_name("video_transcription_gradio.ipynb")


def code_cells(notebook_path):
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in data["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        # Drop IPython magics such as "%pip install"; dependencies are present.
        lines = [line for line in source.splitlines() if not line.lstrip().startswith("%")]
        cell_source = "\n".join(lines).strip()
        if cell_source:
            yield cell_source


def main():
    script = "\n\n".join(code_cells(NOTEBOOK))
    namespace = {"__name__": "__main__", "__file__": str(NOTEBOOK)}
    exec(compile(script, str(NOTEBOOK), "exec"), namespace)


if __name__ == "__main__":
    sys.exit(main())
