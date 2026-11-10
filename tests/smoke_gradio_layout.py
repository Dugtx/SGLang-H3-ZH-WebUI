"""Construct the actual Gradio 6 interface without importing SGLang or touching GPU."""
from __future__ import annotations
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch
import gradio as gr

ROOT=Path(__file__).resolve().parents[1]
for module in ('sglang','sglang.multimodal_gen','sglang.multimodal_gen.apps',
               'sglang.multimodal_gen.apps.webui'):
    if module not in sys.modules:
        m=types.ModuleType(module); m.__path__=[]; sys.modules[module]=m

import importlib.util
spec=importlib.util.spec_from_file_location('sglang.multimodal_gen.apps.webui.h3_studio_history',ROOT/'assets/history_store.py')
hmod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(hmod)
sys.modules[spec.name]=hmod

with tempfile.TemporaryDirectory(prefix='h3gradio-smoke-') as td:
    os.environ['H3_OUTPUTS_DIR']=str(Path(td)/'outputs')
    os.environ['H3_HISTORY_DB']=str(Path(td)/'history.sqlite3')
    sample_video=Path(os.environ['H3_OUTPUTS_DIR'])/'smoke.mp4'
    sample_video.parent.mkdir(parents=True, exist_ok=True)
    sample_video.write_bytes(b'fake-test-output')
    callback=[]
    original_button_click=gr.Button.click
    def capture(self, *args, **kwargs):
        if self.elem_id == 'wb-generate':
            callback.append(kwargs.get('fn'))
        return original_button_click(self, *args, **kwargs)
    src='def build_test_ui():\n'+(ROOT/'assets/workbench_layout.pyfrag').read_text()
    ns={
        '__name__':'h3_smoke','os':os,'gr':gr,'tasks':['ref2va'],
        'workbench_css':(ROOT/'assets/workbench.css').read_text(),
        'workbench_js':(ROOT/'assets/workbench.js').read_text(),
        'server_args':types.SimpleNamespace(webui_port=18080),
        'profile':types.SimpleNamespace(default_flow_shift=12.0, default_audio_flow_shift=3.0),
        'MINIMAX_H3_RECOMMENDED_SHORT_EDGE':768,
        'MINIMAX_H3_FINITE_ASPECT_RATIOS':('16:9','9:16','1:1'),
        'generate':lambda *args:str(sample_video),
    }
    exec(compile(src,'workbench_layout.pyfrag','exec'), ns)
    with patch.object(gr.Blocks,'launch',return_value=(None,'http://127.0.0.1:18080',None)):
        with patch.object(gr.Blocks,'block_thread',return_value=None):
            with patch.object(gr.Button,'click',capture):
                ns['build_test_ui']()
    assert len(callback)==1, callback
    args=['smoke prompt','ref2va',None,None,None,None,None,False,
          567,2,768,'16:9',5.0,12.0,3.0]
    assert callback[0](*args)==str(sample_video)
    history=hmod.VideoHistory(os.environ['H3_OUTPUTS_DIR'],os.environ['H3_HISTORY_DB'])
    rows,_,total=history.page()
    assert total==1, rows
    assert rows[0]['source']=='webui' and rows[0]['status']=='completed'
    assert rows[0]['seed']==567 and rows[0]['steps']==2 and rows[0]['duration_seconds']==5.0
    assert rows[0]['prompt']=='smoke prompt'
    assert history.video_path(rows[0]['id'])==str(sample_video)
    print('PASS: Gradio layout and 15-parameter callback, correct persisted history metadata')
