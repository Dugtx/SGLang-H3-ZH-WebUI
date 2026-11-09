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

    with gr.Blocks(title="SGLang MiniMax H3") as demo:
        gr.Markdown("# SGLang MiniMax H3")
        task = gr.State(value=tasks[0])

        upload_reference_image = gr.Checkbox(
            label="是否选择上传图片", value=False, visible="ref2va" in tasks
        )

        prompt = gr.Textbox(label="提示词（可输入中文）", value="一只好奇的浣熊")
        with gr.Row():
            first_frame = gr.Image(label="首帧关键帧", type="filepath")
            last_frame = gr.Image(label="尾帧关键帧", type="filepath")
            reference_image = gr.Image(
                label="参考图片",
                type="filepath",
                visible="ref2va" not in tasks,
            )
        with gr.Row():
            reference_video = gr.Video(label="参考视频")
            reference_audio = gr.Audio(label="参考音频", type="filepath")
            video_out = gr.Video(label="生成结果（含音频）", include_audio=True)

        with gr.Row():
            seed = gr.Number(label="随机种子", precision=0, value=1234)
            num_inference_steps = gr.Slider(
                minimum=2, maximum=100, value=50, step=1, label="推理步数"
            )
            duration_seconds = gr.Slider(
                minimum=4.0,
                maximum=15.0,
                value=5.0,
                step=0.5,
                label="生成时长（秒）",
            )
        with gr.Row():
            short_edge = gr.Number(
                label="短边分辨率",
                value=MINIMAX_H3_RECOMMENDED_SHORT_EDGE,
                precision=0,
            )
            aspect_ratio = gr.Dropdown(
                choices=[*MINIMAX_H3_FINITE_ASPECT_RATIOS, "auto"],
                value="16:9",
                label="画面比例",
            )
            flow_shift = gr.Number(
                label="视频流偏移", value=profile.default_flow_shift
            )
            audio_flow_shift = gr.Number(
                label="音频流偏移", value=profile.default_audio_flow_shift
            )

        upload_reference_image.change(
            fn=lambda enabled: gr.update(visible=bool(enabled), value=None),
            inputs=upload_reference_image,
            outputs=reference_image,
        )

        run_btn = gr.Button("开始生成", variant="primary")
        run_btn.click(
            fn=generate,
            inputs=[
                prompt,
                task,
                first_frame,
                last_frame,
                reference_image,
                reference_video,
                reference_audio,
                upload_reference_image,
                seed,
                num_inference_steps,
                short_edge,
                aspect_ratio,
                duration_seconds,
                flow_shift,
                audio_flow_shift,
            ],
            outputs=video_out,
        )

        _, local_url, _ = demo.launch(
            server_port=server_args.webui_port,
            quiet=True,
            prevent_thread_lock=True,
            show_error=True,
        )
        url = local_url or f"http://localhost:{server_args.webui_port}"
        print(f"SGLang MiniMax H3 WebUI available at: {url}")
        demo.block_thread()
