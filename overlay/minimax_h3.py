# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    prepare_request,
    save_outputs,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_RECOMMENDED_SHORT_EDGE,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    MINIMAX_H3_FINITE_ASPECT_RATIOS,
    MINIMAX_H3_TASK_PARTITIONS,
    minimax_h3_task_profile,
)
from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs


def is_minimax_h3(server_args: "ServerArgs") -> bool:
    return isinstance(server_args.pipeline_config, MiniMaxH3PipelineConfig)


def minimax_h3_tasks_for_server(server_args: "ServerArgs") -> tuple[str, ...]:
    variant = server_args.model_variant
    if variant is None and server_args.model_subfolder:
        variant = os.path.basename(os.path.normpath(server_args.model_subfolder))
    partition = str(variant or "fl2va").strip().lower()
    return tuple(
        task
        for task, task_partition in MINIMAX_H3_TASK_PARTITIONS.items()
        if partition == "hybrid" or task_partition == partition
    )


def _material_uri(path: str | os.PathLike | None) -> str | None:
    if not path:
        return None
    value = os.fspath(path)
    if urlparse(value).scheme:
        return value
    return f"file://{os.path.abspath(value)}"


def build_minimax_h3_sampling_params_kwargs(
    *,
    prompt: str,
    task: str,
    first_frame: str | os.PathLike | None,
    last_frame: str | os.PathLike | None,
    reference_image: str | os.PathLike | None,
    reference_video: str | os.PathLike | None,
    reference_audio: str | os.PathLike | None,
    seed: int | float,
    num_inference_steps: int | float,
    short_edge: int | float,
    aspect_ratio: str,
    duration_seconds: int | float,
    flow_shift: int | float,
    audio_flow_shift: int | float,
) -> dict[str, Any]:
    """Build H3's native task/conditions/target sampling request."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("MiniMax H3 prompt cannot be empty")
    task = (task or "").strip().lower()
    if task not in MINIMAX_H3_TASK_PARTITIONS:
        raise ValueError(f"Unsupported MiniMax H3 task: {task!r}")

    keyframes = []
    for path, frame_index in ((first_frame, 0), (last_frame, -1)):
        if uri := _material_uri(path):
            keyframes.append(
                {
                    "type": "image",
                    "uri": uri,
                    "role": "keyframe",
                    "frame_index": frame_index,
                }
            )

    references = []
    for path, condition_type in (
        (reference_image, "image"),
        (reference_video, "video_audio"),
        (reference_audio, "audio"),
    ):
        if uri := _material_uri(path):
            references.append({"type": condition_type, "uri": uri, "role": "reference"})

    if task == "t2va":
        if keyframes or references:
            raise ValueError("t2va does not accept conditioning media")
        conditions = []
    elif task == "fl2va":
        if not keyframes:
            raise ValueError("fl2va requires a first frame, a last frame, or both")
        if references:
            raise ValueError("fl2va accepts keyframes only; use ref2va for references")
        conditions = keyframes
    else:
        if not references:
            raise ValueError("ref2va requires a reference image, video, or audio")
        conditions = [*keyframes, *references]

    return {
        "prompt": prompt,
        "seed": int(seed),
        "num_inference_steps": int(num_inference_steps),
        "task": task,
        "conditions": conditions,
        "target": {
            "short_edge": int(short_edge),
            "aspect_ratio": aspect_ratio,
            "duration_seconds": float(duration_seconds),
        },
        "flow_shift": float(flow_shift),
        "audio_flow_shift": float(audio_flow_shift),
        "return_file_paths_only": False,
    }


def _generate_minimax_h3(
    server_args: "ServerArgs", sampling_params_kwargs: dict[str, Any]
) -> str:
    sampling_params = SamplingParams.from_user_sampling_params_args(
        server_args.model_path,
        server_args=server_args,
        **sampling_params_kwargs,
    )
    request = prepare_request(server_args, sampling_params)
    prepared = False
    try:
        sampling_params.prepare_video_request_for_queue(request)
        prepared = True
        result = sync_scheduler_client.forward([request])
        if result.error:
            raise RuntimeError(result.error)
        if result.output is None:
            raise ValueError("MiniMax H3 WebUI generation returned no output")

        output_paths = save_outputs(
            result.output,
            request.data_type,
            request.fps,
            request.save_output,
            lambda index: request.output_file_path(len(result.output), index),
            audio=result.audio,
            audio_sample_rate=result.audio_sample_rate,
            output_compression=request.output_compression,
            x264_preset=request.x264_preset,
        )
        sampling_params.validate_video_final_outputs(output_paths, request)
        return output_paths[0]
    finally:
        if prepared:
            sampling_params.cleanup_video_request(request)


def run_minimax_h3_webui(server_args: "ServerArgs"):
    import gradio as gr

    tasks = minimax_h3_tasks_for_server(server_args)
    if not tasks:
        raise ValueError("The loaded MiniMax H3 partition does not serve any tasks")
    profile = minimax_h3_task_profile(tasks[0])
    sync_scheduler_client.initialize(server_args)

    # The checked-in Ref2VA fallback is generated only once when the UI starts.
    # Keep it as an RGBA PNG; the existing media pipeline owns decoding.
    transparent_reference = None
    if "ref2va" in tasks:
        import tempfile
        from PIL import Image

        placeholder_dir = tempfile.mkdtemp(prefix="sglang_h3_ref_")
        transparent_reference = os.path.join(placeholder_dir, "transparent.png")
        Image.new("RGBA", (512, 512), (0, 0, 0, 0)).save(
            transparent_reference, format="PNG"
        )

    def generate(
        prompt,
        task,
        first_frame,
        last_frame,
        reference_image,
        reference_video,
        reference_audio,
        use_uploaded_reference_image,
        seed,
        num_inference_steps,
        short_edge,
        aspect_ratio,
        duration_seconds,
        flow_shift,
        audio_flow_shift,
    ):
        if task == "ref2va":
            if use_uploaded_reference_image:
                if not reference_image:
                    raise gr.Error("已开启图片上传，请先选择参考图片")
            else:
                # Ignore any previously uploaded image when the switch is off.
                reference_image = transparent_reference

        kwargs = build_minimax_h3_sampling_params_kwargs(
            prompt=prompt,
            task=task,
            first_frame=first_frame,
            last_frame=last_frame,
            reference_image=reference_image,
            reference_video=reference_video,
            reference_audio=reference_audio,
            seed=seed,
            num_inference_steps=num_inference_steps,
            short_edge=short_edge,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            flow_shift=flow_shift,
            audio_flow_shift=audio_flow_shift,
        )
        return _generate_minimax_h3(server_args, kwargs)

    workbench_css = r"""
/* H3 Studio - workspace redesign (FTT preview branch).
   Only presentation; pipeline and data callbacks remain in Python. */

:root { color-scheme: dark; }
html, body { min-height:100%; margin:0; background:#0b1120 !important; }
.gradio-container {
  --body-background-fill:#0b1120;
  --body-text-color:#edf3ff;
  --body-text-color-subdued:#93a4c1;
  --background-fill-primary:#111b2c;
  --background-fill-secondary:#182337;
  --block-background-fill:#141e30;
  --block-border-color:#29384f;
  --block-label-text-color:#bdc9df;
  --block-title-text-color:#ecf2ff;
  --input-background-fill:#101a2d;
  --input-border-color:#34435c;
  --input-text-color:#f0f4ff;
  --border-color-primary:#2b3951;
  --border-color-accent:#7194ff;
  --button-primary-background-fill:#6485f1;
  --button-primary-background-fill-hover:#7695ff;
  --button-primary-text-color:#fff;
  --color-accent-soft:#252f51;
  --table-even-background-fill:#19243a;
  --table-odd-background-fill:#131e31;
  --table-row-focus:#243956;
  --table-border-color:#334660;
  max-width:none !important;
  padding:0 !important;
  margin:0 !important;
  min-height:100vh !important;
  background:#0b1120 !important;
  color:#edf3ff !important;
  font-family:Inter,"Noto Sans CJK SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif !important;
}
.gradio-container * { box-sizing:border-box; }
#h3-root {
  --h3-bg:#0b1120;
  --h3-panel:#121c2f;
  --h3-raised:#172338;
  --h3-ink:#edf3ff;
  --h3-muted:#94a5c2;
  --h3-border:#293952;
  --h3-input:#0f192b;
  --h3-accent:#7898ff;
  --h3-soft:#253453;
  --h3-shadow:0 14px 42px rgba(0,0,0,.13);
  position:relative;
  min-height:100vh;
  padding:0 28px !important;
  background:var(--h3-bg) !important;
  color:var(--h3-ink);
  gap:0 !important;
}
#h3-root .h3-html, #h3-root .h3-html > div {
  padding:0 !important; border:0 !important;
  background:transparent !important; box-shadow:none !important;
}
#h3-root .h3-header {
  min-height:82px; padding:15px 0;
  display:flex; align-items:center; justify-content:space-between;
  gap:18px; border-bottom:1px solid var(--h3-border);
}
#h3-root .h3-brand { display:flex; align-items:center; gap:13px; min-width:0; }
#h3-root .h3-brand-icon {
  width:49px; height:49px; flex:0 0 49px; border-radius:14px;
  background:linear-gradient(135deg,#263754,#192743);
  border:1px solid rgba(155,176,255,.18);
  display:flex; align-items:center; justify-content:center; overflow:hidden;
}
#h3-root .h3-brand-icon img { width:46px; height:46px; object-fit:contain; }
#h3-root .h3-brand-name {
  color:var(--h3-ink); font-size:15px; letter-spacing:.1px; font-weight:770; line-height:1.45;
}
#h3-root .h3-brand-caption { font-size:10px; letter-spacing:1.6px; margin-top:4px; color:var(--h3-muted); font-weight:640; }
#h3-root .h3-brand-separator { opacity:.4; margin:0 5px; }
#h3-root .h3-header-actions { display:flex; align-items:center; gap:12px; }
#h3-root .h3-engine-chip {
  display:flex; align-items:center; gap:9px;
  font-size:11px; font-weight:720; color:#cbd9ff;
  background:rgba(95,122,199,.08);
  border:1px solid var(--h3-border);
  padding:10px 13px; border-radius:10px; white-space:nowrap;
}
#h3-root .h3-chip-dot { width:7px; height:7px; border-radius:50%; background:#65dbbd; box-shadow:0 0 0 3px rgba(101,219,189,.1); }
#h3-root .h3-engine-mode { color:var(--h3-muted); border-left:1px solid var(--h3-border); padding-left:8px; }
#h3-root .h3-theme-button {
  display:flex; align-items:center; gap:8px;
  background:var(--h3-panel); color:var(--h3-ink); border:1px solid var(--h3-border);
  border-radius:10px; padding:10px 12px; font-size:11px;
  font-weight:650; cursor:pointer; white-space:nowrap;
}
#h3-root .h3-theme-button:hover { border-color:var(--h3-accent); background:var(--h3-soft); }
#h3-root #h3-app-shell {
  display:flex !important; flex-wrap:nowrap !important; align-items:stretch !important;
  gap:26px !important; margin:0 !important; min-height:calc(100vh - 150px);
}
#h3-root #h3-sidebar {
  width:211px !important; max-width:211px !important;
  min-width:211px !important; flex:0 0 211px !important;
  margin:0 !important; padding:30px 0 18px !important;
  border-right:1px solid var(--h3-border);
  gap:8px !important;
}
#h3-root .h3-side-section {
  color:var(--h3-muted); font-size:11px; font-weight:720;
  letter-spacing:1.4px; padding:3px 10px 14px;
}
#h3-root .h3-nav {
  display:block; width:calc(100% - 18px); min-height:45px !important;
  padding:11px 15px !important; margin:0 9px 4px 0 !important;
  border:1px solid transparent !important; border-radius:11px !important;
  background:transparent !important; color:var(--h3-muted) !important;
  text-align:left !important; font-size:13px !important; font-weight:680 !important;
  box-shadow:none !important; transition:background .15s ease,color .15s ease;
}
#h3-root .h3-nav:hover {
  background:var(--h3-soft) !important; color:var(--h3-ink) !important;
}
#h3-root .h3-nav.primary {
  background:linear-gradient(95deg,rgba(93,122,227,.23),rgba(93,122,227,.08)) !important;
  color:var(--h3-ink) !important;
  border-color:rgba(118,147,247,.19) !important;
  box-shadow:inset 3px 0 0 #809cff !important;
}
#h3-root .h3-sidebar-bottom {
  margin:55px 15px 10px 0; padding:20px 15px 15px;
  border:1px solid var(--h3-border); border-radius:13px;
  background:linear-gradient(140deg,rgba(80,100,160,.12),transparent);
}
#h3-root .h3-sidebar-bottom-icon { color:var(--h3-accent); font-size:23px; line-height:1; }
#h3-root .h3-sidebar-bottom-title { font-size:12px; font-weight:740; margin:12px 0 6px; }
#h3-root .h3-sidebar-bottom-copy { color:var(--h3-muted); font-size:11px; line-height:1.8; }
#h3-root .h3-sidebar-build { color:var(--h3-muted); opacity:.55; font-size:9px; letter-spacing:1px; margin-top:15px; }
#h3-root #h3-content {
  min-width:0 !important; flex:1 !important;
  padding:31px 0 40px !important;
  max-width:1430px; margin:0 auto !important;
}
#h3-root .h3-page { gap:17px !important; }
#h3-root .h3-page-heading {
  display:flex; align-items:end; justify-content:space-between; gap:15px; margin:0 0 9px;
}
#h3-root .h3-eyebrow {
  color:var(--h3-accent); font-size:10px;
  text-transform:uppercase; font-weight:790; letter-spacing:2.2px;
}
#h3-root .h3-page-heading h1 {
  color:var(--h3-ink); font-size:clamp(26px,2.3vw,34px);
  line-height:1.2; font-weight:780; letter-spacing:-1.2px; margin:11px 0 8px;
}
#h3-root .h3-page-heading p {
  color:var(--h3-muted); font-size:12px; line-height:1.7; margin:0;
}
#h3-root .h3-title-dot { color:var(--h3-accent); }
#h3-root .h3-page-mode {
  color:var(--h3-muted); font-size:11px; font-weight:650;
  border:1px solid var(--h3-border); background:var(--h3-panel);
  border-radius:8px; padding:9px 12px; white-space:nowrap;
}
#h3-root .h3-create-grid { flex-wrap:wrap !important; gap:20px !important; align-items:start !important; }
#h3-root .h3-editor, #h3-root .h3-preview-column { gap:17px !important; min-width:0 !important; }
#h3-root .h3-card {
  gap:14px !important; background:var(--h3-panel) !important;
  border:1px solid var(--h3-border) !important; border-radius:15px !important;
  padding:21px !important; box-shadow:var(--h3-shadow) !important;
}
#h3-root .h3-section-head {
  display:flex; justify-content:space-between; align-items:center; gap:10px;
  font-size:13px; color:var(--h3-ink);
}
#h3-root .h3-section-head > div { display:flex; align-items:center; gap:10px; }
#h3-root .h3-section-index {
  font-size:10px; font-weight:790; color:var(--h3-accent);
  background:rgba(112,145,249,.1); padding:7px 8px;
  border-radius:7px; letter-spacing:.7px;
}
#h3-root .h3-section-head strong { font-size:13px; font-weight:780; }
#h3-root .h3-section-hint { font-size:10px; color:var(--h3-muted); opacity:.77; letter-spacing:1.3px; font-weight:760; }
#h3-root #h3-prompt textarea {
  min-height:144px !important; padding:15px !important;
  border-radius:10px !important; border:1px solid var(--h3-border) !important;
  background:var(--h3-input) !important; color:var(--h3-ink) !important;
  font-size:13px !important; line-height:1.8 !important; resize:vertical !important;
}
#h3-root #h3-prompt textarea::placeholder { color:var(--h3-muted); opacity:.65; }
#h3-root #h3-prompt textarea:focus { outline:1px solid var(--h3-accent); }
#h3-root .h3-card-footnote { font-size:10px; line-height:1.7; color:var(--h3-muted); }
#h3-root .h3-settings-row { gap:12px !important; }
#h3-root .h3-settings-row > div { min-width:135px !important; }
#h3-root .h3-card label { font-size:11px !important; font-weight:680 !important; color:var(--h3-muted) !important; }
#h3-root .h3-card :is(input, textarea, select) { color:var(--h3-ink) !important; }
#h3-root .h3-card :is(input:not([type=range]), select) {
  border-color:var(--h3-border) !important; background:var(--h3-input) !important;
}
#h3-root .h3-card input[type=range] { accent-color:var(--h3-accent); }
#h3-root #h3-image-toggle {
  border:1px solid var(--h3-border) !important;
  border-radius:9px !important; background:var(--h3-input) !important;
  padding:9px 13px !important;
}
#h3-root #h3-image-toggle label { color:var(--h3-ink) !important; }
#h3-root .h3-media-row { gap:12px !important; }
#h3-root .h3-media-row > div { min-width:156px !important; }
#h3-root .h3-accordion {
  border:1px solid var(--h3-border) !important;
  background:transparent !important; border-radius:10px !important;
  box-shadow:none !important; overflow:hidden;
}
#h3-root .h3-accordion > button, #h3-root .h3-accordion > .label-wrap {
  background:var(--h3-raised) !important; color:var(--h3-ink) !important;
  padding:12px 14px !important; font-size:12px !important;
}
#h3-root .h3-accordion > div:not(.label-wrap) { background:var(--h3-panel) !important; }
#h3-root .h3-generate {
  width:100%; border-radius:11px !important; min-height:53px !important;
  background:linear-gradient(96deg,#527bef,#768cff) !important;
  color:white !important; font-size:14px !important; font-weight:760 !important;
  box-shadow:0 9px 24px rgba(69,105,218,.19) !important;
  border:1px solid rgba(182,201,255,.2) !important;
}
#h3-root .h3-generate:hover { filter:brightness(1.11); }
#h3-root .h3-preview-card { gap:17px !important; }
#h3-root .h3-preview-symbol { color:var(--h3-accent); font-size:14px; }
#h3-root #wb-output-video { border:1px solid var(--h3-border); border-radius:10px; overflow:hidden; background:var(--h3-input); }
#h3-root #wb-output-video label { color:var(--h3-muted) !important; }
#h3-root .h3-preview-hint {
  text-align:center; color:var(--h3-muted); font-size:11px; line-height:1.9;
  background:var(--h3-input); border-radius:10px; padding:13px;
}
#h3-root .h3-preview-note {
  display:flex; gap:13px; padding:17px 19px; border:1px solid var(--h3-border);
  border-radius:13px; color:var(--h3-muted); background:var(--h3-panel);
}
#h3-root .h3-note-star { font-size:22px; line-height:1; color:var(--h3-accent); }
#h3-root .h3-preview-note strong { color:var(--h3-ink); font-size:12px; }
#h3-root .h3-preview-note p { margin:7px 0 0; line-height:1.75; font-size:11px; }
#h3-root .h3-history-toolbar { align-items:end !important; gap:10px !important; }
#h3-root .h3-history-toolbar button { min-width:78px; }
#h3-root .h3-history-pager { gap:10px !important; align-items:center !important; }
#h3-root .h3-history-pager button { min-width:80px; }
#h3-root .h3-history-media { gap:16px !important; align-items:start !important; }
#h3-root #wb-history-table {
  --table-even-background-fill:var(--h3-raised);
  --table-odd-background-fill:var(--h3-input);
  --table-row-focus:var(--h3-soft);
  --table-border-color:var(--h3-border);
  max-height:460px; overflow:auto; border-radius:10px;
  background:var(--h3-input) !important;
}
#h3-root #wb-history-table :is(.table-wrap,.table-container,.virtual-table-viewport,.upload-container) {
  background:var(--h3-input) !important; color:var(--h3-ink) !important;
}
#h3-root #wb-history-table :is(.header-table,thead,th,.header-cell) {
  color:var(--h3-ink) !important; background:var(--h3-raised) !important;
  border-color:var(--h3-border) !important;
}
#h3-root #wb-history-table :is(td,.virtual-cell,.row-number-cell,[role=gridcell]) {
  color:var(--h3-ink) !important; border-color:var(--h3-border) !important;
}
#h3-root #wb-history-table .virtual-row { background:var(--h3-input) !important; }
#h3-root #wb-history-table .virtual-row.row-odd { background:var(--h3-raised) !important; }
#h3-root #wb-history-table .virtual-row:hover { background:var(--h3-soft) !important; }
#h3-root .h3-guide-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; margin-top:4px; }
#h3-root .h3-guide-card { padding:24px; background:var(--h3-panel); border:1px solid var(--h3-border); border-radius:15px; min-height:220px; }
#h3-root .h3-guide-number { color:var(--h3-accent); font-weight:780; font-size:12px; letter-spacing:1px; }
#h3-root .h3-guide-card h3 { margin:26px 0 9px; font-size:17px; font-weight:760; color:var(--h3-ink); }
#h3-root .h3-guide-card p { color:var(--h3-muted); margin:0; line-height:1.9; font-size:12px; }
#h3-root .h3-help-note { background:var(--h3-raised); border:1px solid var(--h3-border); border-radius:13px; padding:22px 24px; margin-top:20px; }
#h3-root .h3-help-note strong { color:var(--h3-ink); font-size:13px; }
#h3-root .h3-help-note p { color:var(--h3-muted); margin:8px 0 0; font-size:12px; line-height:1.8; }
#h3-root .h3-footer {
  display:flex; justify-content:space-between; padding:20px 0 22px 0;
  border-top:1px solid var(--h3-border);
  color:var(--h3-muted); font-size:10px; letter-spacing:.35px;
}
/* Day mode uses the same design tokens; no grey text on a white table. */
html:has(#h3-root[data-h3-theme="light"]) { color-scheme:light; background:#f5f7fc !important; }
html:has(#h3-root[data-h3-theme="light"]) body,
html:has(#h3-root[data-h3-theme="light"]) .gradio-container { background:#f5f7fc !important; }
#h3-root[data-h3-theme="light"] {
  --h3-bg:#f5f7fc;
  --h3-panel:#ffffff;
  --h3-raised:#edf2fb;
  --h3-ink:#233454;
  --h3-muted:#647795;
  --h3-border:#d9e1ef;
  --h3-input:#f9fbff;
  --h3-accent:#4c65df;
  --h3-soft:#e9eeff;
  --h3-shadow:0 12px 35px rgba(50,70,120,.05);
}
html:has(#h3-root[data-h3-theme="light"]) .gradio-container {
  --body-background-fill:#f5f7fc;
  --body-text-color:#233454;
  --body-text-color-subdued:#647795;
  --background-fill-primary:#ffffff;
  --background-fill-secondary:#f4f7fe;
  --block-background-fill:#ffffff;
  --block-border-color:#d9e1ef;
  --block-label-text-color:#556d8e;
  --block-title-text-color:#233454;
  --input-background-fill:#f9fbff;
  --input-border-color:#d9e1ef;
  --input-text-color:#233454;
  --border-color-primary:#d9e1ef;
  --table-even-background-fill:#eef2fa;
  --table-odd-background-fill:#ffffff;
  --table-row-focus:#dfe7fa;
}
#h3-root[data-h3-theme="light"] .h3-brand-icon { background:#eff3fc; }
#h3-root[data-h3-theme="light"] .h3-engine-chip { color:#445d87; background:#eef3ff; }
#h3-root[data-h3-theme="light"] .h3-chip-dot { background:#34b995; }
#h3-root[data-h3-theme="light"] .h3-generate { color:white !important; }
@media (max-width:1150px) {
  #h3-root { padding:0 18px !important; }
  #h3-root #h3-app-shell { gap:17px !important; }
  #h3-root #h3-sidebar { width:182px !important; max-width:182px !important; min-width:182px !important; flex-basis:182px !important; }
  #h3-root .h3-card { padding:17px !important; }
  #h3-root .h3-create-grid { gap:14px !important; }
}
@media (max-width:760px) {
  #h3-root { padding:0 14px !important; }
  #h3-root .h3-header { flex-wrap:wrap; gap:12px; }
  #h3-root .h3-engine-mode { display:none; }
  #h3-root .h3-brand-name { font-size:13px; }
  #h3-root #h3-app-shell { display:block !important; }
  #h3-root #h3-sidebar {
    width:100% !important; max-width:none !important; min-width:0 !important;
    padding:13px 0 12px !important; border-right:0;
    border-bottom:1px solid var(--h3-border);
    display:flex; flex-direction:row; flex-wrap:wrap; gap:7px !important;
  }
  #h3-root .h3-side-section, #h3-root .h3-sidebar-bottom { display:none !important; }
  #h3-root .h3-nav { width:auto !important; flex:1 1 auto; text-align:center !important; margin:0 !important; font-size:12px !important; }
  #h3-root #h3-content { padding:23px 0 !important; }
  #h3-root .h3-page-heading { align-items:start; }
  #h3-root .h3-page-mode { display:none; }
  #h3-root .h3-guide-grid { grid-template-columns:1fr; }
  #h3-root .h3-guide-card { min-height:0; }
  #h3-root .h3-history-media { flex-wrap:wrap !important; }
  #h3-root .h3-history-media > * { min-width:100% !important; }
}

/* VNext refined component language: true Gradio Column card shells.
   The old gr.Group API does not render elem_classes in Gradio 6.
   In particular, padding and radius were silently missing. */
#h3-root {
  --h3-radius-lg: 19px;
  --h3-radius-md: 13px;
  --h3-radius-sm: 10px;
  --h3-card-border: rgba(129,150,192,.18);
  --h3-control-border: rgba(144,163,202,.23);
  --h3-surface-hover: #1d2a42;
  --h3-focus-ring: rgba(108,150,255,.18);
  --h3-label: #b5c4dc;
}
#h3-root #h3-content { padding-top:35px !important; }
#h3-root .h3-page { gap:22px !important; }
#h3-root .h3-page-heading { margin:0 0 11px; }
#h3-root .h3-card {
  background:linear-gradient(152deg,rgba(24,35,55,.98),rgba(16,25,43,.98)) !important;
  border:1px solid var(--h3-card-border) !important;
  border-radius:var(--h3-radius-lg) !important;
  box-shadow:0 12px 38px rgba(2,7,20,.15),inset 0 1px 0 rgba(218,232,255,.028) !important;
  padding:22px 23px 23px !important;
  gap:17px !important;
  min-width:0;
  overflow:visible !important;
}
#h3-root .h3-card > div { min-width:0; }
#h3-root .h3-card .h3-html { border:0 !important; background:transparent !important; box-shadow:none !important; padding:0 !important; }
#h3-root .h3-card .h3-section-head { padding:0 0 13px; border-bottom:1px solid rgba(142,163,204,.13); }
#h3-root .h3-card .h3-section-head strong { font-size:14px; font-weight:760; letter-spacing:.05px; }
#h3-root .h3-card .h3-section-index {
  display:inline-flex; align-items:center; justify-content:center;
  min-width:31px; height:29px; color:#b5c9ff;
  background:rgba(104,138,239,.12);
  border:1px solid rgba(137,160,249,.16);
  border-radius:9px;
}
#h3-root .h3-card .h3-section-hint { color:#879bb8; font-size:10px; letter-spacing:1.2px; }
#h3-root .h3-editor, #h3-root .h3-preview-column { gap:19px !important; }
#h3-root #h3-prompt {
  background:transparent !important;
  border:0 !important;
  border-radius:var(--h3-radius-md) !important;
  padding:0 !important;
  box-shadow:none !important;
  overflow:visible !important;
}
#h3-root #h3-prompt textarea {
  display:block;
  border:1px solid var(--h3-control-border) !important;
  border-radius:var(--h3-radius-md) !important;
  background:rgba(9,17,32,.62) !important;
  min-height:152px !important;
  padding:17px 18px !important;
  font-size:13px !important;
  line-height:1.85 !important;
  box-shadow:inset 0 1px 3px rgba(0,0,0,.1) !important;
  transition:border-color .15s ease,box-shadow .15s ease;
}
#h3-root #h3-prompt textarea:focus {
  border-color:var(--h3-accent) !important;
  box-shadow:0 0 0 3px var(--h3-focus-ring),inset 0 1px 3px rgba(0,0,0,.1) !important;
  outline:0 !important;
}
#h3-root .h3-card-footnote { color:var(--h3-muted); line-height:1.7; padding:0 2px; }
#h3-root .h3-prompt-card { gap:15px !important; }
#h3-root .h3-material-card { gap:15px !important; }
#h3-root #h3-image-toggle {
  padding:12px 15px !important; border-radius:var(--h3-radius-md) !important;
  border:1px solid var(--h3-control-border) !important;
  background:rgba(78,105,168,.08) !important;
}
#h3-root #h3-image-toggle:hover { background:rgba(87,116,186,.13) !important; }
#h3-root #h3-image-toggle label { color:var(--h3-ink) !important; font-size:12px !important; }
#h3-root .h3-card .h3-media-row > div { min-width:0 !important; }
#h3-root :is(#h3-reference-image,#h3-first-frame,#h3-last-frame,#h3-reference-video,#h3-reference-audio,#wb-history-video,#wb-history-download) {
  background:rgba(9,17,32,.46) !important;
  border:1px solid var(--h3-control-border) !important;
  border-radius:var(--h3-radius-md) !important;
  overflow:hidden !important;
  box-shadow:none !important;
}
#h3-root :is(#h3-reference-image,#h3-first-frame,#h3-last-frame,#h3-reference-video,#h3-reference-audio,#wb-history-video,#wb-history-download) .upload-container,
#h3-root :is(#h3-reference-image,#h3-first-frame,#h3-last-frame,#h3-reference-video,#h3-reference-audio,#wb-history-video,#wb-history-download) .empty,
#h3-root :is(#h3-reference-image,#h3-first-frame,#h3-last-frame,#h3-reference-video,#h3-reference-audio,#wb-history-video,#wb-history-download) .file-preview,
#h3-root #wb-output-video .video-container {
  background:rgba(9,17,32,.45) !important;
  color:var(--h3-muted) !important;
}
#h3-root .h3-accordion {
  border:1px solid var(--h3-control-border) !important;
  border-radius:var(--h3-radius-md) !important;
  background:rgba(9,17,32,.38) !important;
  overflow:hidden !important;
}
#h3-root #h3-material-accordion,
#h3-root #h3-settings-accordion {
  border:1px solid var(--h3-control-border) !important;
  border-radius:var(--h3-radius-md) !important;
  background:rgba(9,17,32,.38) !important;
  overflow:hidden !important;
}
#h3-root :is(#h3-material-accordion,#h3-settings-accordion) > button,
#h3-root :is(#h3-material-accordion,#h3-settings-accordion) > .label-wrap,
#h3-root .h3-accordion > button,
#h3-root .h3-accordion > .label-wrap {
  background:rgba(62,82,124,.16) !important;
  border:0 !important; border-radius:0 !important;
  color:var(--h3-ink) !important;
  padding:14px 16px !important;
  font-size:12px !important; font-weight:710 !important;
}
#h3-root :is(#h3-material-accordion,#h3-settings-accordion) > button:hover,
#h3-root .h3-accordion > button:hover { background:rgba(88,111,159,.24) !important; }
#h3-root .h3-settings-card { gap:16px !important; }
#h3-root .h3-settings-row { gap:18px !important; align-items:center !important; }
#h3-root .h3-settings-row label { color:var(--h3-label) !important; font-size:11px !important; font-weight:680 !important; }
#h3-root .h3-settings-card .wrap,
#h3-root .h3-settings-card .wrap-inner,
#h3-root .h3-settings-card input:not([type=range]),
#h3-root .h3-settings-card select {
  background:rgba(9,17,32,.56) !important;
  color:var(--h3-ink) !important;
  border-color:var(--h3-control-border) !important;
  border-radius:var(--h3-radius-sm) !important;
}
#h3-root .h3-settings-card input:focus-visible { outline:2px solid var(--h3-accent); outline-offset:2px; }
#h3-root .h3-generate {
  border-radius:var(--h3-radius-md) !important;
  min-height:57px !important;
  font-size:14px !important;
  background:linear-gradient(110deg,#557be9,#8196ff) !important;
  border:1px solid rgba(172,191,255,.32) !important;
  box-shadow:0 12px 26px rgba(69,99,204,.18),inset 0 1px 0 rgba(255,255,255,.1) !important;
  transition:filter .15s ease,transform .15s ease,box-shadow .15s ease;
}
#h3-root .h3-generate:hover:not(:disabled) {
  filter:brightness(1.09); transform:translateY(-1px);
  box-shadow:0 17px 32px rgba(69,99,204,.26) !important;
}
#h3-root .h3-generate:focus-visible,
#h3-root .h3-nav:focus-visible,
#h3-root .h3-theme-button:focus-visible {
  outline:2px solid var(--h3-accent) !important; outline-offset:3px !important;
}
#h3-root .h3-preview-card { gap:16px !important; }
#h3-root #wb-output-video {
  border:1px solid var(--h3-control-border) !important;
  border-radius:var(--h3-radius-md) !important;
  overflow:hidden !important;
  background:rgba(7,14,28,.55) !important;
}
#h3-root .h3-preview-hint {
  background:rgba(70,93,143,.075) !important;
  border:1px solid rgba(136,157,197,.10);
  border-radius:var(--h3-radius-md); padding:16px 15px;
  color:var(--h3-muted); line-height:1.9;
}
#h3-root .h3-preview-note {
  background:linear-gradient(140deg,rgba(21,35,59,.9),rgba(18,27,46,.9));
  border:1px solid var(--h3-card-border);
  border-radius:var(--h3-radius-lg);
  padding:19px 21px;
}
#h3-root .h3-preview-note p { line-height:1.85; }
#h3-root .h3-page-mode,
#h3-root .h3-engine-chip,
#h3-root .h3-theme-button {
  border-radius:var(--h3-radius-sm);
  border-color:var(--h3-control-border);
}
#h3-root .h3-nav {
  border-radius:var(--h3-radius-md) !important;
  transition:background .17s ease,color .17s ease,border .17s ease;
}
#h3-root .h3-nav.primary {
  background:linear-gradient(105deg,rgba(95,124,223,.24),rgba(90,120,221,.07)) !important;
}
#h3-root .h3-sidebar-bottom {
  border-radius:var(--h3-radius-lg);
  padding:19px;
  background:linear-gradient(145deg,rgba(63,93,151,.14),rgba(26,41,68,.08));
}
#h3-root .h3-history-card { gap:17px !important; }
#h3-root .h3-history-toolbar { align-items:center !important; gap:10px !important; }
#h3-root .h3-history-toolbar > div { min-width:0 !important; }
#h3-root .h3-history-toolbar button,
#h3-root .h3-history-pager button {
  border:1px solid var(--h3-control-border) !important;
  border-radius:var(--h3-radius-sm) !important;
  background:var(--h3-raised) !important;
  color:var(--h3-ink) !important;
  min-height:41px !important;
  box-shadow:none !important;
  font-weight:650 !important;
  padding:9px 14px !important;
}
#h3-root .h3-history-toolbar button:hover,
#h3-root .h3-history-pager button:hover {
  background:var(--h3-surface-hover) !important;
  border-color:var(--h3-accent) !important;
}
#h3-root #wb-history-search {
  background:transparent !important; border:0 !important; padding:0 !important;
}
#h3-root #wb-history-search input,
#h3-root #wb-history-select input,
#h3-root .h3-history-pager input,
#h3-root .h3-history-detail-card textarea {
  background:rgba(8,17,32,.52) !important;
  color:var(--h3-ink) !important;
  border:1px solid var(--h3-control-border) !important;
  border-radius:var(--h3-radius-sm) !important;
  padding:11px 13px !important;
}
#h3-root #wb-history-table {
  border:1px solid var(--h3-control-border) !important;
  border-radius:var(--h3-radius-md) !important;
  overflow:hidden !important;
  background:var(--h3-input) !important;
}
#h3-root #wb-history-table :is(.table-container,.table-wrap,.upload-container,.virtual-table-viewport) {
  border-radius:var(--h3-radius-md) !important;
  border-color:var(--h3-control-border) !important;
}
#h3-root #wb-history-table :is(.header-table,th,.header-cell) {
  color:var(--h3-ink) !important; background:var(--h3-raised) !important;
}
#h3-root #wb-history-table .virtual-row { color:var(--h3-ink) !important; min-height:39px !important; }
#h3-root #wb-history-table :is(.virtual-cell,.cell-wrap,[role=gridcell]) { color:var(--h3-ink) !important; }
#h3-root .h3-history-pager { align-items:center !important; gap:13px !important; }
#h3-root .h3-history-detail-card { gap:16px !important; }
#h3-root #wb-history-download {
  min-height:240px !important;
  border:1px dashed var(--h3-control-border) !important;
  border-radius:var(--h3-radius-md) !important;
  background:rgba(8,17,32,.45) !important;
}
#h3-root #wb-history-video { border-radius:var(--h3-radius-md) !important; }
#h3-root .h3-guide-card {
  border-radius:var(--h3-radius-lg) !important;
  border-color:var(--h3-card-border) !important;
  padding:27px 26px;
  box-shadow:var(--h3-shadow);
}
#h3-root .h3-guide-card:hover { border-color:rgba(125,151,229,.35) !important; }
#h3-root .h3-help-note { border-radius:var(--h3-radius-lg); }
#h3-root .h3-footer { margin-top:24px; }
/* Light mode redefines component surfaces so the exact same radii and spacing apply. */
#h3-root[data-h3-theme="light"] {
  --h3-card-border:#dbe3ef;
  --h3-control-border:#cfdaea;
  --h3-surface-hover:#e5edfa;
  --h3-focus-ring:rgba(84,114,224,.17);
  --h3-label:#607290;
}
#h3-root[data-h3-theme="light"] .h3-card {
  background:linear-gradient(153deg,#fff,#f9fbff) !important;
  box-shadow:0 12px 32px rgba(55,83,138,.055),inset 0 1px 0 #fff !important;
}
#h3-root[data-h3-theme="light"] #h3-prompt textarea,
#h3-root[data-h3-theme="light"] #wb-history-search input,
#h3-root[data-h3-theme="light"] #wb-history-select input,
#h3-root[data-h3-theme="light"] .h3-history-pager input,
#h3-root[data-h3-theme="light"] .h3-history-detail-card textarea,
#h3-root[data-h3-theme="light"] .h3-settings-card input:not([type=range]),
#h3-root[data-h3-theme="light"] .h3-settings-card .wrap,
#h3-root[data-h3-theme="light"] .h3-settings-card .wrap-inner {
  background:#f8faff !important; color:#273757 !important;
}
#h3-root[data-h3-theme="light"] :is(#h3-reference-image,#h3-first-frame,#h3-last-frame,#h3-reference-video,#h3-reference-audio,#wb-output-video,#wb-history-video,#wb-history-download),
#h3-root[data-h3-theme="light"] :is(#h3-material-accordion,#h3-settings-accordion) {
  background:#f7f9ff !important;
}
#h3-root[data-h3-theme="light"] .h3-preview-note { background:#fff; }
#h3-root[data-h3-theme="light"] .h3-history-toolbar button,
#h3-root[data-h3-theme="light"] .h3-history-pager button { background:#eef3fc !important; color:#273757 !important; }
@media (max-width:1150px) {
  #h3-root .h3-card { padding:20px !important; }
}
@media (max-width:760px) {
  #h3-root .h3-card { border-radius:16px !important; padding:18px !important; }
  #h3-root .h3-history-toolbar { flex-wrap:wrap !important; }
  #h3-root .h3-history-toolbar button { flex:1; }
  #h3-root .h3-create-grid { gap:18px !important; }
}

/* Media and file block labels are hardcoded white by Gradio 6.
   Explicitly skin their exact native [data-testid] element. */
#h3-root :is(#wb-output-video,#wb-history-video,#wb-history-download,#h3-reference-image,#h3-first-frame,#h3-last-frame,#h3-reference-video,#h3-reference-audio) > label[data-testid="block-label"] {
  background:#24344f !important;
  color:#d5e1ff !important;
  border:1px solid rgba(142,163,207,.22) !important;
  border-top:0 !important;
  border-left:0 !important;
  border-radius:0 0 10px 0 !important;
  font-size:10.5px !important;
  font-weight:700 !important;
  letter-spacing:.15px;
  padding:6px 12px !important;
  line-height:1.5 !important;
  box-shadow:none !important;
}
#h3-root :is(#wb-output-video,#wb-history-video,#wb-history-download,#h3-reference-image,#h3-first-frame,#h3-last-frame,#h3-reference-video,#h3-reference-audio) > label[data-testid="block-label"] :is(svg,span) {
  color:#afc5f6 !important;
}
#h3-root #wb-history-select {
  padding:0 !important;
  border:0 !important;
  border-radius:var(--h3-radius-md) !important;
  background:transparent !important;
  box-shadow:none !important;
}
#h3-root #wb-history-select .wrap {
  border-radius:var(--h3-radius-md) !important;
  background:var(--h3-input) !important;
  border:1px solid var(--h3-control-border) !important;
  overflow:hidden;
}
#h3-root #wb-history-select label,
#h3-root .h3-history-detail-card label {
  color:var(--h3-label) !important;
  font-weight:700 !important;
  letter-spacing:.05px;
}
#h3-root .h3-card :is(button,input,textarea,select):focus-visible {
  outline:2px solid var(--h3-accent) !important;
  outline-offset:2px !important;
}
#h3-root[data-h3-theme="light"] :is(#wb-output-video,#wb-history-video,#wb-history-download,#h3-reference-image,#h3-first-frame,#h3-last-frame,#h3-reference-video,#h3-reference-audio) > label[data-testid="block-label"] {
  background:#e7edfa !important;
  color:#384d70 !important;
  border-color:#ccd8ea !important;
}
#h3-root[data-h3-theme="light"] :is(#wb-output-video,#wb-history-video,#wb-history-download,#h3-reference-image,#h3-first-frame,#h3-last-frame,#h3-reference-video,#h3-reference-audio) > label[data-testid="block-label"] :is(svg,span) {
  color:#6278a4 !important;
}

/* VNext pass 3: calmer, smaller radii everywhere. */
#h3-root {
  --h3-radius-lg:14px;
  --h3-radius-md:9px;
  --h3-radius-sm:7px;
}
#h3-root .h3-card,
#h3-root .h3-sidebar-bottom,
#h3-root .h3-preview-note,
#h3-root .h3-guide-card,
#h3-root .h3-help-note { border-radius:14px !important; }
#h3-root .h3-brand-icon,
#h3-root .h3-nav,
#h3-root .h3-preview-hint,
#h3-root .h3-accordion,
#h3-root :is(#h3-material-accordion,#h3-settings-accordion),
#h3-root :is(#h3-reference-image,#h3-first-frame,#h3-last-frame,#h3-reference-video,#h3-reference-audio,#wb-output-video,#wb-history-video,#wb-history-download),
#h3-root .h3-generate,
#h3-root #wb-history-select,
#h3-root #wb-history-select .wrap,
#h3-root #wb-history-table,
#h3-root #h3-prompt textarea,
#h3-root #h3-image-toggle,
#h3-root #wb-history-search input,
#h3-root .h3-history-pager input,
#h3-root .h3-history-detail-card textarea { border-radius:9px !important; }
#h3-root .h3-section-index,
#h3-root :is(.h3-page-mode,.h3-engine-chip,.h3-theme-button),
#h3-root .h3-history-toolbar button,
#h3-root .h3-history-pager button,
#h3-root .h3-settings-card :is(input,select,.wrap,.wrap-inner) { border-radius:7px !important; }
#h3-root :is(#wb-output-video,#wb-history-video,#wb-history-download,#h3-reference-image,#h3-first-frame,#h3-last-frame,#h3-reference-video,#h3-reference-audio) > label[data-testid="block-label"] {
  border-radius:0 0 6px 0 !important;
}
#h3-root .h3-prompt-actions { align-items:center; justify-content:space-between; gap:10px !important; }
#h3-root .h3-prompt-actions > div { flex:1 1 auto; min-width:140px; }
#h3-root #h3-open-templates {
  flex:0 0 auto; padding:8px 13px !important; min-height:34px !important;
  border:1px solid var(--h3-control-border) !important;
  background:rgba(95,120,183,.11) !important;
  color:#b5c9ff !important;
  font-size:11px !important; font-weight:730 !important;
  border-radius:7px !important;
  box-shadow:none !important;
}
#h3-root #h3-open-templates:hover {
  color:#edf4ff !important;
  background:rgba(108,137,217,.19) !important;
  border-color:var(--h3-accent) !important;
}

/* Independent prompt template library. */
#h3-root .h3-template-grid {
  align-items:flex-start !important; gap:20px !important;
  flex-wrap:wrap !important;
}
#h3-root .h3-template-browser,
#h3-root .h3-template-composer { min-width:0 !important; gap:16px !important; }
#h3-root .h3-template-explore,
#h3-root .h3-template-preview-card { gap:15px !important; }
#h3-root .h3-template-search-row {
  align-items:center !important; gap:9px !important;
}
#h3-root .h3-template-search-row > div { min-width:0 !important; }
#h3-root #h3-template-category,
#h3-root #h3-template-query,
#h3-root #h3-template-subject,
#h3-root #h3-template-preview {
  background:transparent !important;
  border:0 !important; padding:0 !important;
  box-shadow:none !important;
}
#h3-root :is(#h3-template-category,#h3-template-query,#h3-template-subject,#h3-template-preview) :is(input,textarea,.wrap,.wrap-inner) {
  border:1px solid var(--h3-control-border) !important;
  background:rgba(8,17,32,.5) !important;
  color:var(--h3-ink) !important;
  border-radius:9px !important;
  box-shadow:none !important;
}
#h3-root :is(#h3-template-category,#h3-template-query,#h3-template-subject,#h3-template-preview) :is(input,textarea):focus {
  border-color:var(--h3-accent) !important;
  box-shadow:0 0 0 3px var(--h3-focus-ring) !important;
  outline:0 !important;
}
#h3-root #h3-template-search {
  padding:9px 15px !important;
  border-radius:7px !important;
  background:var(--h3-raised) !important;
  border:1px solid var(--h3-control-border) !important;
  color:var(--h3-ink) !important;
  font-size:12px !important;
  min-height:41px !important;
  box-shadow:none !important;
}
#h3-root #h3-template-search:hover {
  border-color:var(--h3-accent) !important;
  background:var(--h3-surface-hover) !important;
}
#h3-root .h3-template-count { font-size:11px; color:var(--h3-muted); margin:0 !important; }
#h3-root .h3-template-count p { margin:0 !important; color:var(--h3-muted) !important; font-size:11px; }
#h3-root #h3-template-selector {
  background:transparent !important; border:0 !important;
  padding:0 !important; box-shadow:none !important;
}
#h3-root #h3-template-selector .wrap {
  display:flex !important;
  flex-direction:column !important;
  align-items:stretch !important;
  gap:8px !important;
}
#h3-root #h3-template-selector label {
  display:flex !important;
  align-items:center !important;
  background:rgba(8,17,32,.33) !important;
  border:1px solid var(--h3-control-border) !important;
  border-radius:9px !important;
  padding:12px 13px !important;
  color:var(--h3-ink) !important;
  line-height:1.55 !important;
  font-size:11.5px !important;
  cursor:pointer;
  transition:background .15s ease,border-color .15s ease;
}
#h3-root #h3-template-selector label:hover {
  border-color:var(--h3-accent) !important;
  background:var(--h3-surface-hover) !important;
}
#h3-root #h3-template-selector input[type=radio] { accent-color:var(--h3-accent) !important; flex:0 0 auto; }
#h3-root #h3-template-selector label:has(input:checked) {
  background:rgba(97,127,220,.15) !important;
  border-color:rgba(128,156,247,.63) !important;
  color:#e9efff !important;
}
#h3-root .h3-template-info {
  background:rgba(106,137,233,.075) !important;
  border:1px solid rgba(128,157,236,.16) !important;
  padding:13px 15px !important;
  border-radius:9px !important;
  font-size:11.5px !important;
  line-height:1.75 !important;
}
#h3-root .h3-template-info p { margin:0 !important; color:var(--h3-muted) !important; }
#h3-root #h3-template-preview textarea {
  min-height:235px !important;
  font-size:12.5px !important; line-height:1.95 !important;
  padding:16px !important;
  resize:vertical !important;
}
#h3-root .h3-template-tip {
  color:var(--h3-muted); font-size:11px;
  line-height:1.8;
}
#h3-root #h3-template-apply {
  min-height:49px !important;
  border-radius:9px !important;
  background:linear-gradient(110deg,#557be9,#7e93ff) !important;
  color:#fff !important; font-size:13px !important; font-weight:740 !important;
  border:1px solid rgba(162,187,255,.28) !important;
  box-shadow:0 8px 24px rgba(66,95,196,.16) !important;
}
#h3-root #h3-template-apply:hover { filter:brightness(1.09); }
#h3-root[data-h3-theme="light"] #h3-open-templates {
  background:#eef3ff !important; color:#4b62b5 !important;
}
#h3-root[data-h3-theme="light"] :is(#h3-template-category,#h3-template-query,#h3-template-subject,#h3-template-preview) :is(input,textarea,.wrap,.wrap-inner) {
  background:#f7faff !important; color:#263856 !important;
}
#h3-root[data-h3-theme="light"] #h3-template-selector label {
  background:#f9fbff !important;
  color:#293b5d !important;
}
#h3-root[data-h3-theme="light"] #h3-template-selector label:has(input:checked) {
  background:#eaf0ff !important;
  color:#374d9a !important;
}
#h3-root[data-h3-theme="light"] .h3-template-info { background:#f0f4fc !important; }
@media(max-width:760px) {
  #h3-root .h3-card { border-radius:14px !important; }
  #h3-root .h3-template-grid { gap:15px !important; }
  #h3-root .h3-prompt-actions { flex-wrap:wrap !important; }
  #h3-root #h3-open-templates { width:100%; }
}

/* Final template-page density adjustment: no double labels or oversized search button. */
#h3-root .h3-field-label {
  color:var(--h3-label);
  font-size:11.5px;
  font-weight:710;
  letter-spacing:.08px;
  padding:1px 0 0;
  margin:0 0 -7px;
}
#h3-root .h3-template-search-row {
  display:flex !important;
  flex-wrap:nowrap !important;
  gap:8px !important;
  align-items:stretch !important;
}
#h3-root #h3-template-query {
  flex:1 1 0 !important;
  width:auto !important;
  min-width:0 !important;
}
#h3-root #h3-template-search {
  flex:0 0 78px !important;
  width:78px !important;
  min-width:78px !important;
  max-width:78px !important;
}
#h3-root #h3-template-selector .wrap {
  max-height:513px;
  overflow-y:auto !important;
  scrollbar-width:thin;
  scrollbar-color:rgba(113,137,186,.5) transparent;
  padding-right:4px !important;
  padding-bottom:1px !important;
  gap:7px !important;
}
#h3-root #h3-template-selector label {
  border-radius:7px !important;
  padding:10px 11px !important;
  font-size:11.5px !important;
  line-height:1.5 !important;
}
#h3-root .h3-template-info {
  background:transparent !important;
  border:0 !important;
  box-shadow:none !important;
  border-radius:0 !important;
  padding:0 !important;
  color:var(--h3-muted) !important;
  font-size:11.5px !important;
  line-height:1.75 !important;
}
#h3-root .h3-template-info :is(p,strong) {
  color:var(--h3-muted) !important;
  font-size:11.5px !important;
}
#h3-root .h3-template-preview-card { gap:13px !important; }
#h3-root .h3-template-explore { gap:13px !important; }
#h3-root #h3-template-preview textarea {
  min-height:205px !important;
}
#h3-root[data-h3-theme="light"] .h3-template-info {
  background:transparent !important;
  border:0 !important;
}
@media (max-width:760px) {
  #h3-root #h3-template-selector .wrap { max-height:420px; }
}

/* Gradio 6 radio groups wrap into extra columns when given a max-height.
   Keep a single full-width column and scroll vertically instead. */
#h3-root .h3-template-explore .form {
  width:100% !important;
  min-width:0 !important;
  max-width:100% !important;
}
#h3-root #h3-template-selector {
  display:block !important;
  width:100% !important;
  min-width:0 !important;
  max-width:100% !important;
  overflow:hidden !important;
}
#h3-root #h3-template-selector > .wrap {
  display:flex !important;
  flex-direction:column !important;
  flex-wrap:nowrap !important;
  align-items:stretch !important;
  width:100% !important;
  max-width:100% !important;
  min-width:0 !important;
  max-height:510px;
  overflow-x:hidden !important;
  overflow-y:auto !important;
}
#h3-root #h3-template-selector > .wrap > label {
  flex:0 0 auto !important;
  width:100% !important;
  min-width:0 !important;
  max-width:100% !important;
}
#h3-root #h3-template-selector > .wrap > label span {
  min-width:0;
  white-space:normal !important;
  overflow-wrap:anywhere;
}
"""

    workbench_js = r"""
(() => {
  // Only theme and viewport affordances; Gradio handles native page navigation.
  const key = "h3-studio-preview-theme";
  const root = document.documentElement;
  let theme = "dark";
  try {
    const saved = localStorage.getItem(key);
    if (saved === "light" || saved === "dark") theme = saved;
  } catch (_) {}

  function apply(next) {
    root.setAttribute("data-h3-theme", next);
    document.getElementById("h3-root")?.setAttribute("data-h3-theme", next);
    root.style.colorScheme = next;
    const button = document.getElementById("h3-theme-toggle");
    if (button) {
      button.setAttribute("title", next === "dark" ? "切换到日间主题" : "切换到夜间主题");
      button.setAttribute("aria-pressed", next === "light" ? "true" : "false");
    }
  }
  apply(theme);
  const observer = new MutationObserver(() => {
    const app = document.getElementById("h3-root");
    if (app) {
      app.setAttribute("data-h3-theme", theme);
      observer.disconnect();
    }
  });
  observer.observe(document.documentElement, {childList: true, subtree: true});
  document.addEventListener("click", (event) => {
    const button = event.target.closest?.("#h3-theme-toggle");
    if (button) {
      theme = root.getAttribute("data-h3-theme") === "dark" ? "light" : "dark";
      apply(theme);
      try { localStorage.setItem(key, theme); } catch (_) {}
    }
    const nav = event.target.closest?.(".h3-nav");
    if (nav) {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }, true);
})();
"""

    # V4.3: persistent browser for completed, failed, imported and interrupted videos.
    from datetime import datetime
    import logging
    import math
    from sglang.multimodal_gen.apps.webui.h3_studio_history import VideoHistory
    from sglang.multimodal_gen.apps.webui.h3_studio_prompts import (
        CATEGORIES, TEMPLATES, get_template, render_template, template_choices,
        template_details,
    )

    history_logger = logging.getLogger(__name__)
    history = VideoHistory(
        output_dir=os.environ.get("H3_OUTPUTS_DIR", "/data/minimax-h3/outputs"),
        database=os.environ.get("H3_HISTORY_DB", "/data/minimax-h3/history/history.sqlite3"),
    )
    try:
        history.scan(force=True)
    except Exception:
        history_logger.exception("Initial H3 history scan failed")

    def generate_with_history(*args):
        # Keep generate() and the original SGLang inference pipeline unchanged.
        job_id = history.create_job(
            prompt=str(args[0]), task=str(args[1]), seed=int(args[8]),
            steps=int(args[9]), duration_seconds=float(args[12]),
        )
        try:
            path = generate(*args)
        except Exception as exc:
            history.fail_job(job_id, str(exc))
            raise
        try:
            history.complete_job(job_id, path)
        except Exception:
            # A metadata failure must not hide an otherwise successful video.
            history_logger.exception("H3 video generated but history metadata could not be saved")
        return path

    def history_page_data(page_number=0, search_text=""):
        items, page_number, total = history.page(page_number, search_text)
        status_labels = {
            "completed": "\u5df2\u5b8c\u6210", "failed": "\u5931\u8d25",
            "interrupted": "\u5df2\u4e2d\u65ad", "running": "\u751f\u6210\u4e2d",
        }
        rows = []
        choices = []
        for item in items:
            timestamp = datetime.fromtimestamp(item["created_at"]).strftime("%Y-%m-%d %H:%M")
            filename = item["file_name"] or "-"
            size = f"{item['size_bytes'] / 1048576:.1f} MB" if item["size_bytes"] else "-"
            label = status_labels.get(item["status"], item["status"])
            src = "WebUI" if item["source"] == "webui" else "\u81ea\u52a8\u5bfc\u5165"
            rows.append([timestamp, filename[:95], label, size, src, (item["prompt"] or "-")[:60]])
            choices.append((f"{timestamp} | {label} | {filename[-65:]}", item["id"]))
        pages = max(1, math.ceil(total / 10))
        summary = f"\u7b2c {page_number + 1}/{pages} \u9875  |  \u5171 {total} \u6761\u8bb0\u5f55"
        return (rows, gr.update(choices=choices, value=None), summary, page_number,
                None, None, "")

    def history_refresh(page_number, search_text, delta=0):
        try:
            history.scan(force=True)
        except Exception:
            history_logger.exception("H3 history refresh scan failed")
        return history_page_data(max(0, int(page_number) + delta), search_text)

    def history_select(item_id):
        row = history.get(item_id)
        if not row:
            return None, None, ""
        missing_prompt = "(\u5386\u53f2\u6587\u4ef6\u65e0\u5143\u6570\u636e)"
        details = [
            f"ID: {row['id']}",
            f"\u72b6\u6001: {row['status']}",
            f"\u6765\u6e90: {row['source']}",
            f"\u521b\u5efa\u65f6\u95f4: {datetime.fromtimestamp(row['created_at']).strftime('%Y-%m-%d %H:%M:%S')}",
            f"\u4efb\u52a1: {row['task'] or '-'} | Seed: {row['seed'] if row['seed'] is not None else '-'} | Steps: {row['steps'] if row['steps'] is not None else '-'}",
            f"\u65f6\u957f: {row['duration_seconds'] if row['duration_seconds'] is not None else '-'} s",
            f"\u63d0\u793a\u8bcd: {row['prompt'] or missing_prompt}",
        ]
        if row.get("error"):
            details.append("Error: " + row["error"])
        path = history.video_path(item_id)
        if row["status"] == "completed" and not path:
            details.append("\u8b66\u544a: \u5f55\u50cf\u6587\u4ef6\u4e0d\u5b58\u5728\uff0c\u65e0\u6cd5\u64ad\u653e\u6216\u4e0b\u8f7d")
        return path, path, "\n".join(details)

    # Prompt template library is independent of the model and generation handler.
    def templates_filter(category, keyword):
        choices = template_choices(category, keyword)
        selected = choices[0][1] if choices else None
        item = get_template(selected)
        subject = item["sample_subject"] if item else ""
        return (
            gr.update(choices=choices, value=selected),
            subject, render_template(selected, subject),
            template_details(selected), f"共找到 {len(choices)} 个模板",
        )

    def templates_pick(template_id):
        item = get_template(template_id)
        subject = item["sample_subject"] if item else ""
        return subject, render_template(template_id, subject), template_details(template_id)

    # UI workbench vNext: native Gradio pages, no hidden anchor-only navigation.
    import inspect

    workbench_theme = gr.themes.Base(
        primary_hue="blue", secondary_hue="slate", neutral_hue="slate"
    )
    style_kwargs = {"theme": workbench_theme, "css": workbench_css, "head": "<script>" + workbench_js + "</script>"}
    style_in_launch = "css" not in inspect.signature(gr.Blocks).parameters
    active_task = tasks[0]
    task_display = {
        "t2va": "文生视频",
        "fl2va": "首尾帧视频",
        "ref2va": "参考视频生成",
    }.get(active_task, active_task.upper())
    brand_logo = "data:image/png;base64,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"

    with gr.Blocks(
        title="H3 Studio | 内蒙古民族幼儿师范",
        **({} if style_in_launch else style_kwargs),
    ) as demo:
        task = gr.State(value=active_task)

        with gr.Column(elem_id="h3-root"):
            gr.HTML(
                f"""<header class="h3-header">
                    <div class="h3-brand">
                        <div class="h3-brand-icon"><img src="{brand_logo}" alt="" /></div>
                        <div class="h3-brand-copy">
                            <div class="h3-brand-name">内蒙古民族幼儿师范视频模型</div>
                            <div class="h3-brand-caption">H3 STUDIO <span class="h3-brand-separator">/</span> AI 视频创作平台</div>
                        </div>
                    </div>
                    <div class="h3-header-actions">
                        <div class="h3-engine-chip"><span class="h3-chip-dot"></span> MiniMax-H3 <span class="h3-engine-mode">{task_display}</span></div>
                        <button type="button" class="h3-theme-button" id="h3-theme-toggle" aria-label="切换浅色或深色模式">
                            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a9 9 0 1 0 9 9 7.5 7.5 0 0 1-9-9Z"/></svg>
                            <span>切换主题</span>
                        </button>
                    </div>
                </header>""",
                elem_classes=["h3-html"],
            )

            with gr.Row(elem_id="h3-app-shell"):
                with gr.Column(scale=0, min_width=207, elem_id="h3-sidebar"):
                    gr.HTML(
                        """<div class="h3-side-section">工作空间</div>""",
                        elem_classes=["h3-html"],
                    )
                    nav_create = gr.Button(
                        "✦   视频创作", variant="primary",
                        elem_id="h3-nav-create", elem_classes=["h3-nav"],
                    )
                    nav_templates = gr.Button(
                        "▣   提示词模板", variant="secondary",
                        elem_id="h3-nav-templates", elem_classes=["h3-nav"],
                    )
                    nav_history = gr.Button(
                        "▤   生成记录", variant="secondary",
                        elem_id="h3-nav-history", elem_classes=["h3-nav"],
                    )
                    nav_help = gr.Button(
                        "ⓘ   使用帮助", variant="secondary",
                        elem_id="h3-nav-help", elem_classes=["h3-nav"],
                    )
                    gr.HTML(
                        """<div class="h3-sidebar-bottom">
                          <div class="h3-sidebar-bottom-icon">✧</div>
                          <div class="h3-sidebar-bottom-title">从灵感到画面</div>
                          <div class="h3-sidebar-bottom-copy">清晰创作 · 原片保存 · 随时回看</div>
                          <div class="h3-sidebar-build">H3 STUDIO · vNEXT</div>
                        </div>""",
                        elem_classes=["h3-html"],
                    )

                with gr.Column(scale=1, min_width=450, elem_id="h3-content"):

                    # Page 1: creation. Only the essentials are visible by default.
                    with gr.Column(visible=True, elem_id="h3-page-create", elem_classes=["h3-page"]) as page_create:
                        gr.HTML(
                            f"""<div class="h3-page-heading">
                              <div><div class="h3-eyebrow">CREATE / 工作台</div>
                              <h1>视频创作<span class="h3-title-dot">.</span></h1>
                              <p>从一段描述开始，选择素材与生成参数，然后查看作品。</p></div>
                              <span class="h3-page-mode">{task_display}</span>
                            </div>""",
                            elem_classes=["h3-html"],
                        )
                        with gr.Row(elem_classes=["h3-create-grid"]):
                            with gr.Column(scale=7, min_width=430, elem_classes=["h3-editor"]):
                                with gr.Column(elem_classes=["h3-card", "h3-prompt-card"]):
                                    gr.HTML(
                                        """<div class="h3-section-head">
                                          <div><span class="h3-section-index">01</span><strong>画面描述</strong></div>
                                          <span class="h3-section-hint">PROMPT</span>
                                        </div>""", elem_classes=["h3-html"],
                                    )
                                    prompt = gr.Textbox(
                                        label="提示词（可输入中文）",
                                        value="一只好奇的浣熊",
                                        placeholder="描述人物、场景、动作和镜头语言，例如：雨后的城市街道，镜头缓慢向前移动……",
                                        lines=4, max_lines=9, show_label=False,
                                        elem_id="h3-prompt",
                                    )
                                    with gr.Row(elem_classes=["h3-prompt-actions"]):
                                        gr.HTML(
                                            """<div class="h3-card-footnote">支持中文提示词，建议明确主体、运动和环境。</div>""",
                                            elem_classes=["h3-html"],
                                        )
                                        open_templates_btn = gr.Button(
                                            "浏览提示词模板", size="sm", variant="secondary",
                                            elem_id="h3-open-templates", elem_classes=["h3-text-button"],
                                        )

                                with gr.Column(elem_classes=["h3-card", "h3-material-card"]):
                                    gr.HTML(
                                        """<div class="h3-section-head">
                                          <div><span class="h3-section-index">02</span><strong>素材输入</strong></div>
                                          <span class="h3-section-hint">OPTIONAL</span>
                                        </div>""", elem_classes=["h3-html"],
                                    )
                                    upload_reference_image = gr.Checkbox(
                                        label="是否选择上传图片",
                                        value=False, visible="ref2va" in tasks,
                                        elem_id="h3-image-toggle",
                                    )
                                    reference_image = gr.Image(
                                        label="参考图片",
                                        type="filepath",
                                        height=180,
                                        visible="ref2va" not in tasks,
                                        elem_id="h3-reference-image",
                                    )
                                    with gr.Row(visible="fl2va" in tasks, elem_classes=["h3-media-row"]):
                                        first_frame = gr.Image(
                                            label="首帧关键帧", type="filepath", height=156,
                                            elem_id="h3-first-frame",
                                        )
                                        last_frame = gr.Image(
                                            label="尾帧关键帧", type="filepath", height=156,
                                            elem_id="h3-last-frame",
                                        )
                                    with gr.Accordion("更多素材 · 参考视频 / 音频", open=False, elem_classes=["h3-accordion"], elem_id="h3-material-accordion"):
                                        with gr.Row(elem_classes=["h3-media-row"]):
                                            reference_video = gr.Video(
                                                label="参考视频", height=150,
                                                elem_id="h3-reference-video",
                                            )
                                            reference_audio = gr.Audio(
                                                label="参考音频", type="filepath",
                                                elem_id="h3-reference-audio",
                                            )

                                with gr.Column(elem_classes=["h3-card", "h3-settings-card"]):
                                    gr.HTML(
                                        """<div class="h3-section-head">
                                          <div><span class="h3-section-index">03</span><strong>生成设置</strong></div>
                                          <span class="h3-section-hint">OUTPUT</span>
                                        </div>""", elem_classes=["h3-html"],
                                    )
                                    with gr.Row(elem_classes=["h3-settings-row"]):
                                        num_inference_steps = gr.Slider(
                                            minimum=2, maximum=100, value=2, step=1,
                                            label="推理步数",
                                        )
                                        duration_seconds = gr.Slider(
                                            minimum=4.0, maximum=15.0, value=5.0, step=0.5,
                                            label="视频时长（秒）",
                                        )
                                    with gr.Accordion("高级参数", open=False, elem_classes=["h3-accordion"], elem_id="h3-settings-accordion"):
                                        with gr.Row(elem_classes=["h3-settings-row"]):
                                            seed = gr.Number(label="随机种子", precision=0, value=1234)
                                            short_edge = gr.Number(
                                                label="短边分辨率",
                                                value=MINIMAX_H3_RECOMMENDED_SHORT_EDGE,
                                                precision=0,
                                            )
                                            aspect_ratio = gr.Dropdown(
                                                choices=[*MINIMAX_H3_FINITE_ASPECT_RATIOS, "auto"],
                                                value="16:9", label="画面比例",
                                            )
                                        with gr.Row(elem_classes=["h3-settings-row"]):
                                            flow_shift = gr.Number(
                                                label="视频流偏移", value=profile.default_flow_shift,
                                            )
                                            audio_flow_shift = gr.Number(
                                                label="音频流偏移", value=profile.default_audio_flow_shift,
                                            )
                                run_btn = gr.Button(
                                    "✦   开始生成视频", variant="primary",
                                    elem_id="wb-generate", elem_classes=["h3-generate"],
                                )

                            with gr.Column(scale=5, min_width=315, elem_classes=["h3-preview-column"]):
                                with gr.Column(elem_classes=["h3-card", "h3-preview-card"]):
                                    gr.HTML(
                                        """<div class="h3-section-head">
                                          <div><span class="h3-preview-symbol">▶</span><strong>视频预览</strong></div>
                                          <span class="h3-section-hint">RESULT</span>
                                        </div>""", elem_classes=["h3-html"],
                                    )
                                    video_out = gr.Video(
                                        label="生成结果（含音频）", include_audio=True,
                                        height=302, elem_id="wb-output-video",
                                    )
                                    gr.HTML(
                                        """<div class="h3-preview-hint">
                                            生成完成后，作品会在这里显示。<br/>
                                            同时保存至右侧导航的「生成记录」。
                                        </div>""", elem_classes=["h3-html"],
                                    )
                                gr.HTML(
                                    """<div class="h3-preview-note">
                                      <span class="h3-note-star">✧</span>
                                      <div><strong>创作提示</strong><p>让画面动起来：在提示词里写清主体、动作和镜头变化，通常比单纯堆砌风格词更有效。</p></div>
                                    </div>""", elem_classes=["h3-html"],
                                )

                    # Page 2: a dedicated, independently maintained prompt library.
                    with gr.Column(visible=False, elem_id="h3-page-templates", elem_classes=["h3-page"]) as page_templates:
                        gr.HTML(
                            """<div class="h3-page-heading">
                              <div><div class="h3-eyebrow">PROMPT LIBRARY / 灵感中心</div>
                              <h1>提示词模板<span class="h3-title-dot">.</span></h1>
                              <p>找到合适的镜头描述，替换主体并调整细节，然后直接用于视频创作。</p></div>
                              <span class="h3-page-mode">12 个精选模板</span>
                            </div>""",
                            elem_classes=["h3-html"],
                        )
                        with gr.Row(elem_classes=["h3-template-grid"]):
                            with gr.Column(scale=5, min_width=315, elem_classes=["h3-template-browser"]):
                                with gr.Column(elem_classes=["h3-card", "h3-template-explore"]):
                                    gr.HTML(
                                        """<div class="h3-section-head">
                                          <div><span class="h3-section-index">01</span><strong>探索模板</strong></div>
                                          <span class="h3-section-hint">BROWSE</span>
                                        </div>""",
                                        elem_classes=["h3-html"],
                                    )
                                    gr.HTML(
                                        """<div class="h3-field-label">模板分类</div>""",
                                        elem_classes=["h3-html"],
                                    )
                                    template_category = gr.Dropdown(
                                        choices=list(CATEGORIES), value="全部",
                                        label="模板分类", show_label=False,
                                        elem_id="h3-template-category",
                                    )
                                    with gr.Row(elem_classes=["h3-template-search-row"]):
                                        template_query = gr.Textbox(
                                            placeholder="搜索风格、运镜、场景...",
                                            show_label=False, scale=4,
                                            elem_id="h3-template-query",
                                        )
                                        template_search_btn = gr.Button(
                                            "搜索", variant="secondary", size="sm",
                                            elem_id="h3-template-search",
                                        )
                                    template_count = gr.Markdown(
                                        "共找到 12 个模板", elem_classes=["h3-template-count"],
                                    )
                                    template_selector = gr.Radio(
                                        choices=template_choices(),
                                        value=TEMPLATES[0]["id"],
                                        label="选择一个模板", show_label=False,
                                        elem_id="h3-template-selector",
                                    )
                            with gr.Column(scale=7, min_width=345, elem_classes=["h3-template-composer"]):
                                with gr.Column(elem_classes=["h3-card", "h3-template-preview-card"]):
                                    gr.HTML(
                                        """<div class="h3-section-head">
                                          <div><span class="h3-section-index">02</span><strong>编辑并应用</strong></div>
                                          <span class="h3-section-hint">COMPOSE</span>
                                        </div>""",
                                        elem_classes=["h3-html"],
                                    )
                                    template_info = gr.Markdown(
                                        value=template_details(TEMPLATES[0]["id"]),
                                        elem_classes=["h3-template-info"],
                                    )
                                    gr.HTML(
                                        """<div class="h3-field-label">替换主体</div>""",
                                        elem_classes=["h3-html"],
                                    )
                                    template_subject = gr.Textbox(
                                        value=TEMPLATES[0]["sample_subject"],
                                        label="替换主体", show_label=False,
                                        placeholder="例如：一只橘猫 / 一名旅行者 / 一款耳机",
                                        elem_id="h3-template-subject",
                                    )
                                    gr.HTML(
                                        """<div class="h3-field-label">可编辑提示词</div>""",
                                        elem_classes=["h3-html"],
                                    )
                                    template_preview = gr.Textbox(
                                        value=render_template(TEMPLATES[0]["id"], TEMPLATES[0]["sample_subject"]),
                                        label="可编辑提示词", show_label=False,
                                        lines=9, max_lines=15,
                                        elem_id="h3-template-preview",
                                    )
                                    gr.HTML(
                                        """<div class="h3-template-tip">模板仅提供创作参考。可直接修改上面的描述；应用时不会自动开始生成。</div>""",
                                        elem_classes=["h3-html"],
                                    )
                                    template_apply_btn = gr.Button(
                                        "应用到视频创作", variant="primary",
                                        elem_id="h3-template-apply",
                                    )

                    # Page 3: existing history callbacks are unchanged.
                    with gr.Column(visible=False, elem_id="h3-page-history", elem_classes=["h3-page"]) as page_history:
                        gr.HTML(
                            """<div class="h3-page-heading">
                                <div><div class="h3-eyebrow">LIBRARY / 作品管理</div>
                                <h1>生成记录<span class="h3-title-dot">.</span></h1>
                                <p>查看过去生成的视频，预览与下载原始 MP4。</p></div>
                            </div>""", elem_classes=["h3-html"],
                        )
                        with gr.Column(elem_id="wb-history-card", elem_classes=["h3-card", "h3-history-card"]):
                            history_page_state = gr.State(0)
                            with gr.Row(elem_classes=["h3-history-toolbar"]):
                                history_search = gr.Textbox(
                                    label="搜索文件名或提示词",
                                    placeholder="搜索作品...",
                                    show_label=False,
                                    elem_id="wb-history-search",
                                    scale=4,
                                )
                                history_search_btn = gr.Button("搜索", size="sm", variant="secondary")
                                history_refresh_btn = gr.Button("刷新", size="sm", variant="secondary")
                            history_table = gr.Dataframe(
                                headers=["生成时间", "文件名", "状态", "大小", "来源", "提示词"],
                                datatype=["str"] * 6, interactive=False, wrap=True,
                                elem_id="wb-history-table",
                            )
                            with gr.Row(elem_classes=["h3-history-pager"]):
                                history_prev_btn = gr.Button("上一页", size="sm")
                                history_page_text = gr.Textbox(
                                    value="", show_label=False, interactive=False, scale=3,
                                )
                                history_next_btn = gr.Button("下一页", size="sm")
                        with gr.Column(elem_classes=["h3-card", "h3-history-detail-card"]):
                            gr.HTML(
                                """<div class="h3-section-head">
                                    <div><strong>作品详情</strong></div>
                                    <span class="h3-section-hint">PREVIEW / DOWNLOAD</span>
                                </div>""", elem_classes=["h3-html"],
                            )
                            history_selector = gr.Dropdown(
                                label="选择要查看的记录",
                                choices=[], value=None,
                                elem_id="wb-history-select",
                            )
                            with gr.Row(elem_classes=["h3-history-media"]):
                                history_video = gr.Video(
                                    label="历史视频预览", include_audio=True,
                                    height=270, elem_id="wb-history-video",
                                )
                                history_download = gr.File(
                                    label="下载原始 MP4",
                                    elem_id="wb-history-download",
                                )
                            history_detail = gr.Textbox(
                                label="任务详情", interactive=False,
                                lines=4, max_lines=8,
                            )

                    # Page 4: a genuine separate help page.
                    with gr.Column(visible=False, elem_id="h3-page-help", elem_classes=["h3-page"]) as page_help:
                        gr.HTML(
                            """<div class="h3-page-heading">
                                <div><div class="h3-eyebrow">GUIDE / 使用说明</div>
                                <h1>使用帮助<span class="h3-title-dot">.</span></h1>
                                <p>三步完成视频创作，操作与作品管理都在同一个工作台。</p></div>
                            </div>
                            <section class="h3-guide-grid">
                                <article class="h3-guide-card"><div class="h3-guide-number">01</div>
                                  <h3>描述画面</h3>
                                  <p>进入「视频创作」输入完整描述，也可以从「提示词模板」选择风格、替换主体，再一键应用到创作区。</p>
                                </article>
                                <article class="h3-guide-card"><div class="h3-guide-number">02</div>
                                  <h3>准备素材与参数</h3>
                                  <p>参考生成模式可开启「是否选择上传图片」；关闭时使用系统预设的透明参考图。按需调整步数与时长。</p>
                                </article>
                                <article class="h3-guide-card"><div class="h3-guide-number">03</div>
                                  <h3>生成并保存</h3>
                                  <p>生成完成后直接预览；在「生成记录」中查看历史视频，选择记录可下载原始 MP4。</p>
                                </article>
                            </section>
                            <div class="h3-help-note">
                                <strong>使用提示</strong>
                                <p>生成速度取决于时长、分辨率、推理步数和服务器负载。已有生成记录保存在服务端，不依赖当前浏览器页面。</p>
                            </div>""",
                            elem_classes=["h3-html"],
                        )

            gr.HTML(
                """<footer class="h3-footer">
                    <span>内蒙古民族幼儿师范视频模型</span>
                    <span>H3 STUDIO · SGLang</span>
                </footer>""",
                elem_classes=["h3-html"],
            )

        def navigate(page):
            views = ("create", "templates", "history", "help")
            pages = tuple(gr.update(visible=(page == key)) for key in views)
            navs = tuple(
                gr.update(variant="primary" if page == key else "secondary")
                for key in views
            )
            return pages + navs

        navigation_outputs = [
            page_create, page_templates, page_history, page_help,
            nav_create, nav_templates, nav_history, nav_help,
        ]
        nav_create.click(fn=lambda: navigate("create"), outputs=navigation_outputs)
        nav_templates.click(fn=lambda: navigate("templates"), outputs=navigation_outputs)
        open_templates_btn.click(fn=lambda: navigate("templates"), outputs=navigation_outputs)
        nav_history.click(fn=lambda: navigate("history"), outputs=navigation_outputs)
        nav_help.click(fn=lambda: navigate("help"), outputs=navigation_outputs)

        template_filter_outputs = [
            template_selector, template_subject, template_preview,
            template_info, template_count,
        ]
        template_category.change(
            fn=templates_filter, inputs=[template_category, template_query],
            outputs=template_filter_outputs,
        )
        template_search_btn.click(
            fn=templates_filter, inputs=[template_category, template_query],
            outputs=template_filter_outputs,
        )
        template_query.submit(
            fn=templates_filter, inputs=[template_category, template_query],
            outputs=template_filter_outputs,
        )
        template_selector.change(
            fn=templates_pick, inputs=template_selector,
            outputs=[template_subject, template_preview, template_info],
        )
        template_subject.change(
            fn=render_template, inputs=[template_selector, template_subject],
            outputs=template_preview,
        )

        def apply_selected_template(value):
            text = (value or "").strip()
            if not text:
                raise gr.Error("请选择一个提示词模板并填写内容")
            return (text, *navigate("create"))

        template_apply_btn.click(
            fn=apply_selected_template, inputs=template_preview,
            outputs=[prompt, *navigation_outputs],
        )

        upload_reference_image.change(
            fn=lambda enabled: gr.update(visible=bool(enabled), value=None),
            inputs=upload_reference_image,
            outputs=reference_image,
        )

        history_outputs = [
            history_table, history_selector, history_page_text, history_page_state,
            history_video, history_download, history_detail,
        ]
        demo.load(fn=lambda: history_refresh(0, ""), outputs=history_outputs)
        history_refresh_btn.click(
            fn=lambda page, query: history_refresh(page, query),
            inputs=[history_page_state, history_search], outputs=history_outputs,
        )
        history_search_btn.click(
            fn=lambda query: history_refresh(0, query),
            inputs=history_search, outputs=history_outputs,
        )
        history_prev_btn.click(
            fn=lambda page, query: history_refresh(page, query, -1),
            inputs=[history_page_state, history_search], outputs=history_outputs,
        )
        history_next_btn.click(
            fn=lambda page, query: history_refresh(page, query, 1),
            inputs=[history_page_state, history_search], outputs=history_outputs,
        )
        history_selector.change(
            fn=history_select, inputs=history_selector,
            outputs=[history_video, history_download, history_detail],
        )

        generate_event = run_btn.click(
            fn=generate_with_history,
            inputs=[
                prompt, task, first_frame, last_frame,
                reference_image, reference_video, reference_audio,
                upload_reference_image,
                seed, num_inference_steps, short_edge, aspect_ratio,
                duration_seconds, flow_shift, audio_flow_shift,
            ],
            outputs=video_out,
        )
        generate_event.success(
            fn=lambda page, query: history_refresh(page, query),
            inputs=[history_page_state, history_search], outputs=history_outputs,
        )

        _, local_url, _ = demo.launch(
            server_port=server_args.webui_port,
            quiet=True,
            prevent_thread_lock=True,
            show_error=True,
            allowed_paths=[str(history.output_dir)],
            footer_links=[],
            **(style_kwargs if style_in_launch else {}),
        )
        url = local_url or f"http://localhost:{server_args.webui_port}"
        print(f"H3 Studio UI available at: {url}")
        demo.block_thread()
