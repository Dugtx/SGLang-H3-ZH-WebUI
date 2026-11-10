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
        self.assertIn('value=False,', self.localized)
        self.assertIn('visible="ref2va" in tasks', self.localized)
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
            "\u641c\u7d22\u6587\u4ef6\u540d\u6216\u63d0\u793a\u8bcd", "\u9009\u62e9\u8981\u67e5\u770b\u7684\u8bb0\u5f55",
            "\u5386\u53f2\u89c6\u9891\u9884\u89c8", "\u4e0b\u8f7d\u539f\u59cb MP4", "\u4efb\u52a1\u8be6\u60c5",
            "生成结果（含音频）", "随机种子", "推理步数", "生成时长（秒）",
            "短边分辨率", "画面比例", "视频流偏移", "音频流偏移",
        ]
        self.assertCountEqual(labels, expected)
        self.assertIn('"开始生成", variant="primary"', self.localized)
        self.assertIn('gr.Markdown("# 内蒙古民族幼儿师范视频模型", elem_classes=', self.localized)
        self.assertNotIn("**说明：**", self.localized)
        self.assertNotIn("当前仅为中文界面", self.localized)
        self.assertNotIn('SGLang MiniMax H3 · 中文版', self.localized)
        self.assertNotIn('label="模型路径"', self.localized)
        self.assertNotIn('label="任务模式"', self.localized)

    def test_no_imported_model_or_engine_changes(self):
        def extract_imports(tree):
            return [ast.dump(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertEqual(extract_imports(self.a), extract_imports(self.b))

    def test_original_generate_callback_contract_unchanged(self):
        ui = function(self.b, "run_minimax_h3_webui")
        generate = next(x for x in ui.body if isinstance(x, ast.FunctionDef) and x.name == "generate")
        # ast.dump() changes its field representation between Python versions.
        # Hash the exact source segment instead, so identical callbacks pass
        # on the openEuler host and on newer development interpreters.
        callback_source = ast.get_source_segment(self.localized, generate)
        self.assertIsNotNone(callback_source)
        self.assertEqual(
            hashlib.sha256(callback_source.encode("utf-8")).hexdigest(),
            "a6e1bfc6b37205b15a4f6c9e9c028288612c6bd856fa7d9c44d0c3dc6adde303",
        )

    def test_workbench_presentation_and_assets(self):
        css = (ROOT / "assets/workbench.css").read_text(encoding="utf-8")
        layout = (ROOT / "assets/workbench_layout.pyfrag").read_text(encoding="utf-8")
        self.assertIn('"css": workbench_css', self.localized)
        self.assertIn('"js": workbench_js', self.localized)
        self.assertIn('style_in_launch = "css" not in inspect.signature(gr.Blocks).parameters', self.localized)
        for ident in ("wb-root", "wb-app-shell", "wb-prompt-card", "wb-assets-card",
                      "wb-settings-card", "wb-result-card", "wb-generate"):
            self.assertIn(ident, self.localized)
            self.assertIn(ident, css + layout)
        self.assertIn(css.rstrip(), self.localized)
        self.assertIn("内蒙古民族幼儿师范视频模型", self.localized)

    def test_day_night_theme_browser_only(self):
        js = (ROOT / "assets/workbench.js").read_text(encoding="utf-8")
        css = (ROOT / "assets/workbench.css").read_text(encoding="utf-8")
        layout = (ROOT / "assets/workbench_layout.pyfrag").read_text(encoding="utf-8")
        self.assertIn('id="wb-theme-toggle"', layout)
        self.assertIn('data-wb-theme', js)
        self.assertIn('window.localStorage.setItem(storageKey, next)', js)
        self.assertIn('document.addEventListener("click"', js)
        self.assertIn('html[data-wb-theme="light"]', css)
        self.assertIn(js.rstrip(), self.localized)
        self.assertIn('title="内蒙古民族幼儿师范视频模型 | 视频创作工作台"', layout)
        self.assertEqual(self.localized.count('run_btn.click('), 1)

    def test_critical_ui_bindings_intact(self):
        ui = function(self.b, "run_minimax_h3_webui")
        def clicked(tree):
            return [ast.dump(x) for x in ast.walk(tree) if isinstance(x, ast.Call)
                    and isinstance(x.func, ast.Attribute) and x.func.attr == "click"]
        run_clicks = [x for x in ast.walk(ui) if isinstance(x, ast.Call)
                      and isinstance(x.func, ast.Attribute)
                      and isinstance(x.func.value, ast.Name)
                      and x.func.value.id == "run_btn" and x.func.attr == "click"]
        self.assertEqual(len(run_clicks), 1)
        self.assertIn('upload_reference_image.change(', self.localized)
        self.assertIn('upload_reference_image,\n                seed,', self.localized)
        self.assertIn('fn=generate_with_history,', self.localized)
        self.assertIn('allowed_paths=[str(history.output_dir)]', self.localized)
        self.assertIn('fn=generate_with_history,', self.localized)
        self.assertIn('outputs=video_out,', self.localized)
        self.assertIn('task = gr.State(value=tasks[0])', self.localized)
        self.assertNotIn('task = gr.Dropdown(', self.localized)


if __name__ == "__main__":
    unittest.main(verbosity=2)
