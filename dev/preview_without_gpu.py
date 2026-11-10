"""Dev-only visual preview of the real Gradio layout without importing SGLang.

Usage: python dev/preview_without_gpu.py [fl2va|ref2va|t2va] [port]
The UI can be browsed; generation is intentionally unavailable.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import sys

HERE = Path(__file__).resolve().parents[1]
code = (HERE / "overlay/minimax_h3.py").read_text(encoding="utf-8")
tree = ast.parse(code)
func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_minimax_h3_webui")
func_src = ast.get_source_segment(code, func)
chosen_task = sys.argv[1] if len(sys.argv) >= 2 else "ref2va"
port = int(sys.argv[2]) if len(sys.argv) >= 3 else 7869

class DummyScheduler:
    def initialize(self, *args, **kwargs):
        return None

def dummy_generate(*args, **kwargs):
    raise RuntimeError("预览模式无法推理；请在实际 SGLang 服务中使用生成按钮。")

scope = dict(
    os=os,
    minimax_h3_tasks_for_server=lambda _server_args: (chosen_task,),
    minimax_h3_task_profile=lambda task: SimpleNamespace(default_flow_shift=5.0, default_audio_flow_shift=2.0),
    sync_scheduler_client=DummyScheduler(),
    MINIMAX_H3_RECOMMENDED_SHORT_EDGE=720,
    MINIMAX_H3_FINITE_ASPECT_RATIOS=["16:9", "9:16", "1:1", "4:3", "3:4"],
    build_minimax_h3_sampling_params_kwargs=lambda **k: k,
    _generate_minimax_h3=dummy_generate,
)
exec(compile(func_src, str(HERE / "overlay/minimax_h3.py"), "exec"), scope)
print(f"[VISUAL PREVIEW ONLY] http://127.0.0.1:{port}", flush=True)
scope["run_minimax_h3_webui"](SimpleNamespace(webui_port=port))
