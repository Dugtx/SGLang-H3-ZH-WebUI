#!/usr/bin/env python3
"""Offline regression tests. No SGLang, Gradio or GPU imports required."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "upstream/minimax_h3.py"
LOCALIZED = ROOT / "overlay/minimax_h3.py"
MANIFEST = ROOT / "manifest.json"


def function(tree, name):
    return next(x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == name)


class LocalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = ORIGINAL.read_text(encoding="utf-8")
        cls.localized = LOCALIZED.read_text(encoding="utf-8")
        cls.a = ast.parse(cls.original)
        cls.b = ast.parse(cls.localized)
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_sha256_and_source_version(self):
        self.assertEqual(hashlib.sha256(self.original.encode()).hexdigest(), self.manifest["upstream_sha256"])
        self.assertEqual(hashlib.sha256(self.localized.encode()).hexdigest(), self.manifest["localized_sha256"])
        self.assertEqual(self.manifest["upstream_tag"], "v0.5.22")

    def test_python_syntax(self):
        compile(self.original, "upstream.py", "exec")
        compile(self.localized, "zh.py", "exec")

    def test_inference_functions_unchanged(self):
        for name in ["is_minimax_h3", "minimax_h3_tasks_for_server", "_material_uri",
                     "build_minimax_h3_sampling_params_kwargs", "_generate_minimax_h3"]:
            with self.subTest(function=name):
                self.assertEqual(
                    ast.dump(function(self.a, name), include_attributes=False),
                    ast.dump(function(self.b, name), include_attributes=False),
                )

    def test_reference_image_switch_and_fallback(self):
        self.assertIn('label="是否选择上传图片"', self.localized)
        self.assertIn('value=False, visible="ref2va" in tasks', self.localized)
        self.assertIn('if task == "ref2va":', self.localized)
        self.assertIn('if use_uploaded_reference_image:', self.localized)
        self.assertIn('reference_image = transparent_reference', self.localized)
        self.assertIn('raise gr.Error("已开启图片上传，请先选择参考图片")', self.localized)
        self.assertIn('Image.new("RGBA", (512, 512), (0, 0, 0, 0))', self.localized)
        self.assertIn('visible=bool(enabled), value=None', self.localized)

    def test_ui_labels_zh(self):
        labels = []
        for node in ast.walk(function(self.b, "run_minimax_h3_webui")):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "label" and isinstance(keyword.value, ast.Constant):
                        labels.append(str(keyword.value.value))
        expected = [
            "提示词（可输入中文）", "首帧关键帧",
            "尾帧关键帧", "参考图片", "是否选择上传图片", "参考视频", "参考音频",
            "生成结果（含音频）", "随机种子", "推理步数", "生成时长（秒）",
            "短边分辨率", "画面比例", "视频流偏移", "音频流偏移",
        ]
        self.assertCountEqual(labels, expected)
        self.assertIn('gr.Button("开始生成"', self.localized)
        self.assertIn('gr.Markdown("# SGLang MiniMax H3")', self.localized)
        self.assertNotIn("**说明：**", self.localized)
        self.assertNotIn("当前仅为中文界面", self.localized)
        self.assertNotIn('SGLang MiniMax H3 · 中文版', self.localized)
        self.assertNotIn('label="模型路径"', self.localized)
        self.assertNotIn('label="任务模式"', self.localized)

    def test_no_imported_model_or_engine_changes(self):
        def extract_imports(tree):
            return [ast.dump(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertEqual(extract_imports(self.a), extract_imports(self.b))

    def test_critical_ui_bindings_intact(self):
        ui = function(self.b, "run_minimax_h3_webui")
        def clicked(tree):
            return [ast.dump(x) for x in ast.walk(tree) if isinstance(x, ast.Call)
                    and isinstance(x.func, ast.Attribute) and x.func.attr == "click"]
        self.assertEqual(len(clicked(ui)), 1)
        self.assertIn('upload_reference_image.change(', self.localized)
        self.assertIn('upload_reference_image,\n                seed,', self.localized)
        self.assertIn('fn=generate,', self.localized)
        self.assertIn('outputs=video_out,', self.localized)
        self.assertIn('task = gr.State(value=tasks[0])', self.localized)
        self.assertNotIn('task = gr.Dropdown(', self.localized)


if __name__ == "__main__":
    unittest.main(verbosity=2)
