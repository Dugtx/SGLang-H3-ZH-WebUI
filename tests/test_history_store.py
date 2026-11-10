"""Real filesystem + SQLite integration regression for Studio V4.3 history."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "assets/history_store.py"
spec = importlib.util.spec_from_file_location("h3_studio_history", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VideoHistory = module.VideoHistory


class HistoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()
        self.db = self.root / "history" / "history.sqlite3"
        self.history = VideoHistory(str(self.outputs), str(self.db))

    def create_mp4(self, name="example.mp4"):
        p = self.outputs / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("ffmpeg"):
            subprocess.run([
                "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=red:s=128x128:r=5",
                "-t", "1", "-c:v", "mpeg4", str(p),
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=15)
        else:
            # Validity for scanner is patched if ffmpeg is unavailable.
            p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"a" * 2048)
        old = time.time() - 65
        os.utime(p, (old, old))
        return p

    def test_persistent_webui_record_and_original_mp4(self):
        video = self.create_mp4()
        job_id = self.history.create_job(prompt="sky", task="ref2va", seed=123,
                                         steps=2, duration_seconds=5.0)
        self.assertEqual(self.history.get(job_id)["status"], "running")
        self.history.complete_job(job_id, str(video))
        self.assertEqual(self.history.video_path(job_id), str(video.resolve()))
        self.assertEqual(self.history.get(job_id)["steps"], 2)
        reopened = VideoHistory(str(self.outputs), str(self.db))
        self.assertEqual(reopened.get(job_id)["status"], "completed")
        self.assertEqual(reopened.video_path(job_id), str(video.resolve()))
        self.assertEqual(len(list(self.outputs.rglob("*.mp4"))), 1)

    def test_import_existing_mp4_and_deduplicate(self):
        video = self.create_mp4("old/sub.mp4")
        with patch.object(VideoHistory, "_playable", return_value=True):
            self.assertEqual(self.history.scan(force=True), 1)
            self.assertEqual(self.history.scan(force=True), 0)
        rows, page, total = self.history.page()
        self.assertEqual((page, total), (0, 1))
        self.assertEqual(rows[0]["source"], "scan")
        self.assertIsNone(rows[0]["prompt"])
        self.assertEqual(self.history.video_path(rows[0]["id"]), str(video.resolve()))

    def test_ignore_recent_incomplete_and_external_symlink(self):
        fresh = self.outputs / "fresh.mp4"
        fresh.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"z" * 5000)
        outer = self.root / "outside.mp4"
        outer.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"y" * 5000)
        (self.outputs / "escape.mp4").symlink_to(outer)
        with patch.object(VideoHistory, "_playable", return_value=True):
            self.assertEqual(self.history.scan(force=True), 0)
        self.assertEqual(self.history.page()[2], 0)

    def test_reject_traversal_and_missing_file(self):
        self.create_mp4()
        job = self.history.create_job(prompt="x", task="ref2va", seed=1, steps=2,
                                      duration_seconds=5)
        with self.assertRaises(ValueError):
            self.history.complete_job(job, self.root / "escape.mp4")
        self.assertIsNone(self.history.video_path(job))
        self.history.fail_job(job, "failed")
        self.assertEqual(self.history.get(job)["status"], "failed")

    def test_paging_search_and_restarted_job_status(self):
        for i in range(23):
            self.history.create_job(prompt=f"prompt-{i}", task="ref2va", seed=i,
                                    steps=2, duration_seconds=5)
        rows, page, total = self.history.page(1, page_size=10)
        self.assertEqual((len(rows), page, total), (10, 1, 23))
        rows, page, total = self.history.page(999, page_size=10)
        self.assertEqual((len(rows), page, total), (3, 2, 23))
        rows, page, total = self.history.page(0, "prompt-17")
        self.assertEqual((len(rows), total), (1, 1))
        rows, page, total = self.history.page(0, "%")
        self.assertEqual(total, 0)
        reopened = VideoHistory(str(self.outputs), str(self.db))
        self.assertTrue(all(x["status"] == "interrupted" for x in reopened.page(0)[0]))

    def test_promote_scanned_row_to_webui_metadata(self):
        path = self.create_mp4()
        with patch.object(VideoHistory, "_playable", return_value=True):
            self.assertEqual(self.history.scan(force=True), 1)
        job = self.history.create_job(prompt="with prompt", task="ref2va", seed=9,
                                      steps=2, duration_seconds=4)
        self.history.complete_job(job, str(path))
        rows, _, total = self.history.page()
        self.assertEqual(total, 1)
        self.assertEqual(rows[0]["source"], "webui")
        self.assertEqual(rows[0]["prompt"], "with prompt")

    def test_missing_on_disk_shows_no_download(self):
        path = self.create_mp4()
        job = self.history.create_job(prompt="sky", task="ref2va", seed=7,
                                      steps=2, duration_seconds=5)
        self.history.complete_job(job, str(path))
        path.unlink()
        self.assertIsNone(self.history.video_path(job))
        self.assertEqual(self.history.page()[2], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
