"""Run the actual H3 Studio Gradio UI on FTT without loading SGLang or GPU.

Example:
    .venv/bin/python dev/preview_without_gpu.py ref2va 7869
UI preview uses isolated MP4 fixtures under .preview-data.
"""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
import types
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
task = sys.argv[1] if len(sys.argv) >= 2 else "ref2va"
port = int(sys.argv[2]) if len(sys.argv) >= 3 else 7869
if task not in {"ref2va", "fl2va", "t2va"}:
    raise SystemExit(f"unsupported task: {task}")

os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["H3_PREVIEW_MODE"] = "1"
data = ROOT / ".preview-data"
output_dir = data / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
os.environ["H3_OUTPUTS_DIR"] = str(output_dir)
os.environ["H3_HISTORY_DB"] = str(data / "history.sqlite3")

sample = output_dir / "预览演示_镜头场景.mp4"
if not sample.is_file():
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=20",
        "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(sample),
    ]
    subprocess.run(cmd, check=True, timeout=30)
    old = time.time() - 3600
    os.utime(sample, (old, old))

# The project is intentionally UI-only. Install just the helper namespace.
for module in (
    "sglang", "sglang.multimodal_gen", "sglang.multimodal_gen.apps",
    "sglang.multimodal_gen.apps.webui",
):
    if module not in sys.modules:
        package = types.ModuleType(module)
        package.__path__ = []
        sys.modules[module] = package

name = "sglang.multimodal_gen.apps.webui.h3_studio_history"
spec = importlib.util.spec_from_file_location(name, ROOT / "assets/history_store.py")
history_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(history_module)
sys.modules[name] = history_module


class DummyScheduler:
    def initialize(self, *args, **kwargs):
        return None


def fake_generate(*args, **kwargs):
    # Output is a synthetic video fixture, never a real model result.
    return str(sample)


source = (ROOT / "overlay/minimax_h3.py").read_text(encoding="utf-8")
tree = ast.parse(source)
fn = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "run_minimax_h3_webui"
)
fragment = ast.get_source_segment(source, fn)
scope = {
    "os": os,
    "minimax_h3_tasks_for_server": lambda args: (task,),
    "minimax_h3_task_profile": lambda t: SimpleNamespace(
        default_flow_shift=12.0, default_audio_flow_shift=3.0
    ),
    "sync_scheduler_client": DummyScheduler(),
    "MINIMAX_H3_RECOMMENDED_SHORT_EDGE": 768,
    "MINIMAX_H3_FINITE_ASPECT_RATIOS": ["16:9", "9:16", "1:1", "4:3", "3:4"],
    "build_minimax_h3_sampling_params_kwargs": lambda **kwargs: kwargs,
    "_generate_minimax_h3": fake_generate,
}
exec(compile(fragment, str(ROOT / "overlay/minimax_h3.py"), "exec"), scope)
print(f"[FTT UI PREVIEW - no GPU] http://0.0.0.0:{port}", flush=True)
scope["run_minimax_h3_webui"](SimpleNamespace(webui_port=port))
