#!/usr/bin/env python3
"""Build the deterministic Chinese H3 Studio UI for pinned SGLang v0.5.22.

Only the Gradio visual layout, labels and styling are rebuilt. The selected
model remains internal State; inference and request construction are unchanged.
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

# Functional opt-in reference-image switch, applied after localization.
extra_replacements = [
    ("    sync_scheduler_client.initialize(server_args)\n\n    def generate(", "    sync_scheduler_client.initialize(server_args)\n\n    # The checked-in Ref2VA fallback is generated only once when the UI starts.\n    # Keep it as an RGBA PNG; the existing media pipeline owns decoding.\n    transparent_reference = None\n    if \"ref2va\" in tasks:\n        import tempfile\n        from PIL import Image\n\n        placeholder_dir = tempfile.mkdtemp(prefix=\"sglang_h3_ref_\")\n        transparent_reference = os.path.join(placeholder_dir, \"transparent.png\")\n        Image.new(\"RGBA\", (512, 512), (0, 0, 0, 0)).save(\n            transparent_reference, format=\"PNG\"\n        )\n\n    def generate("),
    ("        reference_audio,\n        seed,\n", "        reference_audio,\n        use_uploaded_reference_image,\n        seed,\n"),
    ("    ):\n        kwargs = build_minimax_h3_sampling_params_kwargs(\n", "    ):\n        if task == \"ref2va\":\n            if use_uploaded_reference_image:\n                if not reference_image:\n                    raise gr.Error(\"已开启图片上传，请先选择参考图片\")\n            else:\n                # Ignore any previously uploaded image when the switch is off.\n                reference_image = transparent_reference\n\n        kwargs = build_minimax_h3_sampling_params_kwargs(\n"),
    ("        task = gr.State(value=tasks[0])\n\n        prompt =", "        task = gr.State(value=tasks[0])\n\n        upload_reference_image = gr.Checkbox(\n            label=\"是否选择上传图片\", value=False, visible=\"ref2va\" in tasks\n        )\n\n        prompt ="),
    ("reference_image = gr.Image(label=\"参考图片\", type=\"filepath\")", "reference_image = gr.Image(\n                label=\"参考图片\",\n                type=\"filepath\",\n                visible=\"ref2va\" not in tasks,\n            )"),
    ("        run_btn = gr.Button(\"开始生成\", variant=\"primary\")\n", "        upload_reference_image.change(\n            fn=lambda enabled: gr.update(visible=bool(enabled), value=None),\n            inputs=upload_reference_image,\n            outputs=reference_image,\n        )\n\n        run_btn = gr.Button(\"开始生成\", variant=\"primary\")\n"),
    ("                reference_audio,\n                seed,\n", "                reference_audio,\n                upload_reference_image,\n                seed,\n"),
]
for old, new in extra_replacements:
    count = modified.count(old)
    if count != 1:
        raise SystemExit(f"FAIL: expected exactly 1 occurrence for {old!r}, found {count}")
    modified = modified.replace(old, new, 1)

# The workbench templates are the source of truth.  Keep the generated
# overlay self-contained: Docker only copies overlay/minimax_h3.py.
css = (ROOT / "assets" / "workbench.css").read_text(encoding="utf-8")
js = (ROOT / "assets" / "workbench.js").read_text(encoding="utf-8")
layout = (ROOT / "assets" / "workbench_layout.pyfrag").read_text(encoding="utf-8")
ui_start = '    with gr.Blocks(title="SGLang MiniMax H3") as demo:'
if modified.count(ui_start) != 1:
    raise SystemExit("FAIL: unexpected Gradio UI entrypoint")
start = modified.index(ui_start)
modified = (modified[:start]
            + '    workbench_css = r"""\n' + css.rstrip() + '\n"""\n\n'
            + '    workbench_js = r"""\n' + js.rstrip() + '\n"""\n\n'
            + layout)

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
    "project": "内蒙古民族幼儿师范视频模型 - SGLang MiniMax-H3 workbench",
    "upstream_repo": "https://github.com/sgl-project/sglang",
    "upstream_tag": "v0.5.22",
    "upstream_commit": "35ce313293052f8362e0829e7512bc9bc3d336c9",
    "source_relative_path": "python/sglang/multimodal_gen/apps/webui/minimax_h3.py",
    "upstream_sha256": source_hash,
    "localized_sha256": hashlib.sha256(modified.encode("utf-8")).hexdigest(),
    "history_sha256": hashlib.sha256((ROOT / "assets/history_store.py").read_bytes()).hexdigest(),
    "change_scope": "Studio V4.3: persistent output history (SQLite + MP4 browser) and unchanged SGLang inference core",
    "changes": len(replacements) + len(extra_replacements) + 2,
    "release": "v0.5.22-studio4.3",
}
(ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
