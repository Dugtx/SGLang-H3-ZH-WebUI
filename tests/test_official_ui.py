import ast
"""Release-specific checks for production UI label cleanup."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class ReleaseConsistencyTests(unittest.TestCase):
    def test_version_consistency_across_build_and_deploy(self):
        import json
        release = json.loads((ROOT / "manifest.json").read_text("utf-8"))["release"]
        self.assertEqual(release, "v0.5.22-studio4.4")
        for relative in ("Dockerfile", "compose.override.yaml", "scripts/deploy_no_git.sh"):
            self.assertIn(release, (ROOT / relative).read_text("utf-8"), relative)
        deploy = (ROOT / "scripts/deploy_no_git.sh").read_text("utf-8")
        self.assertIn(f'RELEASE="{release}"', deploy)
        self.assertIn(f'm["release"] == "{release}"', deploy)
        image_name = (ROOT / "compose.override.yaml").read_text("utf-8")
        self.assertIn(f"local/sglang-h3-zh:{release}", image_name)

class OfficialUITests(unittest.TestCase):
    def test_no_demo_toolbar_in_ui_assets(self):
        for filename in ('assets/workbench.js', 'assets/workbench_layout.pyfrag', 'assets/workbench.css', 'dev/UI_设计预览_静态.html'):
            body=(ROOT/filename).read_text(encoding='utf-8')
            for term in ('体验生成等待动效', '仅演示动画', 'wb-wait-demo', 'wb-demo-toolbar'):
                self.assertNotIn(term, body, (filename, term))
    def test_real_generation_wait_kept(self):
        js=(ROOT/'assets/workbench.js').read_text(encoding='utf-8')
        layout=(ROOT/'assets/workbench_layout.pyfrag').read_text(encoding='utf-8')
        self.assertIn('startWaiting();',js)
        self.assertIn('wb-generate',js)
        self.assertIn('wb-wait-layer',layout)
        self.assertIn('fn=generate_with_history,', layout)
        self.assertIn('outputs=video_out',layout)


class UIChangeTests(unittest.TestCase):
    def test_removed_decoration(self):
        layout = (ROOT / "assets/workbench_layout.pyfrag").read_text(encoding="utf-8")
        html = (ROOT / "dev/UI_设计预览_静态.html").read_text(encoding="utf-8")
        py = (ROOT / "overlay/minimax_h3.py").read_text(encoding="utf-8")
        for content in (layout, html, py):
            self.assertNotIn("UI Motion Online", content)
            self.assertNotIn("System State", content)
        self.assertEqual(layout.count('class="wb-tech-card"'), 2)

    def test_inference_steps_exact_minimum(self):
        src = (ROOT / "overlay/minimax_h3.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        slider = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute) and node.func.attr == "Slider"
                      and any(k.arg=="label" and isinstance(k.value, ast.Constant)
                              and k.value.value == "推理步数" for k in node.keywords))
        kw = {k.arg: k.value for k in slider.keywords}
        self.assertEqual(kw["minimum"].value, 2)
        self.assertEqual(kw["value"].value, 2)
        self.assertEqual(kw["maximum"].value, 100)
        self.assertEqual(kw["step"].value, 1)

    def test_generation_pipeline_unmodified(self):
        before = ast.parse((ROOT / "upstream/minimax_h3.py").read_text(encoding="utf-8"))
        after = ast.parse((ROOT / "overlay/minimax_h3.py").read_text(encoding="utf-8"))
        for name in ("is_minimax_h3", "minimax_h3_tasks_for_server", "_material_uri",
                     "build_minimax_h3_sampling_params_kwargs", "_generate_minimax_h3"):
            a=next(x for x in before.body if isinstance(x,ast.FunctionDef) and x.name==name)
            b=next(x for x in after.body if isinstance(x,ast.FunctionDef) and x.name==name)
            self.assertEqual(ast.dump(a, include_attributes=False),
                             ast.dump(b, include_attributes=False))

