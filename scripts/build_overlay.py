#!/usr/bin/env python3
"""Generate a deterministic Chinese UI overlay for SGLang v0.5.22 MiniMax H3.

Only visible Gradio UI strings, the top controls, without explanatory Markdown
are changed. The selected model variant is retained internally as Gradio State.
Inference, request construction and generation callbacks are unchanged.
"""
from __future__ import annotations
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "upstream" / "minimax_h3.py"
DST = ROOT / "overlay" / "minimax_h3.py"
PATCH = ROOT / "patches" / "minimax_h3_zh.patch"
EXPECTED_SHA256 = "d140b465872a9f16f658b6dfc92e25e2a98306482b4c9cb6c31c7ebe6a8413a9"

original = SRC.read_text(encoding="utf-8")
source_hash = hashlib.sha256(original.encode("utf-8")).hexdigest()
if source_hash != EXPECTED_SHA256:
    raise SystemExit(f"FAIL: upstream file changed: {source_hash}")

replacements = [
    ('with gr.Blocks() as demo:', 'with gr.Blocks(title="SGLang MiniMax H3") as demo:'),
    ('with gr.Row():\n'
     '            gr.Textbox(label="Model", value=server_args.model_path)\n'
     '            task = gr.Dropdown(choices=list(tasks), value=tasks[0], label="Task")',
     'task = gr.State(value=tasks[0])'),
    ('gr.Textbox(label="Prompt", value="A curious raccoon")',
     'gr.Textbox(label="提示词（可输入中文）", value="一只好奇的浣熊")'),
    ('gr.Image(label="First keyframe", type="filepath")',
     'gr.Image(label="首帧关键帧", type="filepath")'),
    ('gr.Image(label="Last keyframe", type="filepath")',
     'gr.Image(label="尾帧关键帧", type="filepath")'),
    ('gr.Image(label="Reference image", type="filepath")',
     'gr.Image(label="参考图片", type="filepath")'),
    ('gr.Video(label="Reference video")', 'gr.Video(label="参考视频")'),
    ('gr.Audio(label="Reference audio", type="filepath")',
     'gr.Audio(label="参考音频", type="filepath")'),
    ('gr.Video(label="Generated video", include_audio=True)',
     'gr.Video(label="生成结果（含音频）", include_audio=True)'),
    ('gr.Number(label="Seed", precision=0, value=1234)',
     'gr.Number(label="随机种子", precision=0, value=1234)'),
    ('value=50, step=1, label="Steps"',
     'value=50, step=1, label="推理步数"'),
    ('label="Duration (seconds)",', 'label="生成时长（秒）",'),
    ('label="Short edge",', 'label="短边分辨率",'),
    ('label="Aspect ratio",', 'label="画面比例",'),
    ('label="Video flow shift",', 'label="视频流偏移",'),
    ('label="Audio flow shift",', 'label="音频流偏移",'),
    ('gr.Button("Generate", variant="primary")',
     'gr.Button("开始生成", variant="primary")'),
]
modified = original
for old, new in replacements:
    count = modified.count(old)
    if count != 1:
        raise SystemExit(f"FAIL: expected exactly 1 occurrence for {old!r}, found {count}")
    modified = modified.replace(old, new, 1)

DST.parent.mkdir(parents=True, exist_ok=True)
PATCH.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(modified, encoding="utf-8")
PATCH.write_text("".join(difflib.unified_diff(
    original.splitlines(keepends=True),
    modified.splitlines(keepends=True),
    fromfile="a/python/sglang/multimodal_gen/apps/webui/minimax_h3.py",
    tofile="b/python/sglang/multimodal_gen/apps/webui/minimax_h3.py",
)), encoding="utf-8")
manifest = {
    "project": "SGLang MiniMax-H3 Simplified Chinese UI overlay",
    "upstream_repo": "https://github.com/sgl-project/sglang",
    "upstream_tag": "v0.5.22",
    "upstream_commit": "35ce313293052f8362e0829e7512bc9bc3d336c9",
    "source_relative_path": "python/sglang/multimodal_gen/apps/webui/minimax_h3.py",
    "upstream_sha256": source_hash,
    "localized_sha256": hashlib.sha256(modified.encode("utf-8")).hexdigest(),
    "change_scope": "H3 labels, English title, top model/task controls hidden; task via Gradio State",
    "changes": len(replacements),
    "release": "v0.5.22-zh3",
}
(ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
