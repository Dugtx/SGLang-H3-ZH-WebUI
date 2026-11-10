"""Assert v4 waiting HUD is presentation-only and stays self-contained."""
from pathlib import Path
import ast
import unittest

ROOT = Path(__file__).resolve().parents[1]

class WaitingUITests(unittest.TestCase):
    def test_overlay_is_visual_only(self):
        js = (ROOT / 'assets/workbench.js').read_text('utf-8')
        html = (ROOT / 'assets/waiting_visual.html').read_text('utf-8')
        css = (ROOT / 'assets/workbench.css').read_text('utf-8')
        py = (ROOT / 'overlay/minimax_h3.py').read_text('utf-8')
        for name in ('wb-wait-layer','wb-wait-clock','wb-effect-toggle','wb-wait-dismiss'):
            self.assertIn(name, html)
            self.assertIn(name, js)
        self.assertIn('document.addEventListener(\'click\'', js)
        self.assertIn('window.setInterval(inspectWait', js)
        self.assertIn('currentMediaSignature()', js)
        self.assertIn('wb-wait-html', py)
        self.assertIn('wb-wait-core', css)
        self.assertIn('prefers-reduced-motion: reduce', css)
        self.assertEqual(py.count('run_btn.click('), 1)
        for forbidden in ('fetch(', 'XMLHttpRequest', 'preventDefault(', 'stopPropagation(', 'sync_scheduler_client.forward(', 'document.cookie'):
            self.assertNotIn(forbidden, js)
        self.assertNotIn('http://', html)
        self.assertNotIn('https://', html)
        self.assertNotIn('<script', html.lower())

    def test_overlay_does_not_change_inputs(self):
        tree=ast.parse((ROOT / 'overlay/minimax_h3.py').read_text('utf-8'))
        ui=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run_minimax_h3_webui')
        calls=[x for x in ast.walk(ui) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and isinstance(x.func.value,ast.Name) and x.func.value.id=='run_btn' and x.func.attr=='click']
        self.assertEqual(len(calls),1)
        click=calls[0]
        args={kw.arg:kw.value for kw in click.keywords}
        self.assertEqual(len(args['inputs'].elts),15)
        self.assertIsInstance(args['outputs'], ast.Name)
        self.assertEqual(args['outputs'].id, 'video_out')

if __name__ == '__main__':
    unittest.main()
