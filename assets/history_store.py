# SPDX-License-Identifier: Apache-2.0
"""Persistent video index for the MiniMax-H3 Studio workbench.

The existing /data/minimax-h3/outputs directory stays the authoritative video
store.  SQLite contains small metadata records only; no MP4 files are copied.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
import os
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from pathlib import Path


DEFAULT_OUTPUT_DIR = "/data/minimax-h3/outputs"
DEFAULT_HISTORY_DB = "/data/minimax-h3/history/history.sqlite3"
PAGE_SIZE = 10


class VideoHistory:
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR, database: str = DEFAULT_HISTORY_DB):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.database = Path(database).expanduser().resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._scan_lock = threading.RLock()
        self._last_scan = 0.0
        self._init_db()

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(str(self.database), timeout=15)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA busy_timeout = 15000")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _init_db(self):
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    id TEXT PRIMARY KEY,
                    file_name TEXT UNIQUE,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    prompt TEXT,
                    task TEXT,
                    seed INTEGER,
                    steps INTEGER,
                    duration_seconds REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    size_bytes INTEGER,
                    error TEXT
                )
            """)
            db.execute("CREATE INDEX IF NOT EXISTS idx_videos_created ON videos(created_at DESC, id DESC)")
            # A restart interrupts a synchronous WebUI generation; never leave it 'running' forever.
            db.execute("UPDATE videos SET status='interrupted', updated_at=? WHERE status='running'", (time.time(),))

    def _relative_name(self, value: str | os.PathLike) -> str:
        path = Path(value)
        if not path.is_absolute():
            # SGLang reports output paths relative to the process working directory.
            path = Path.cwd() / path
        path = path.resolve()
        if not path.is_relative_to(self.output_dir) or path.suffix.lower() != ".mp4":
            raise ValueError("Video must be an MP4 inside the configured output directory")
        return path.relative_to(self.output_dir).as_posix()

    def create_job(self, *, prompt: str, task: str, seed: int, steps: int,
                   duration_seconds: float) -> str:
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._connect() as db:
            db.execute("""
                INSERT INTO videos(id,source,status,prompt,task,seed,steps,duration_seconds,created_at,updated_at)
                VALUES (?, 'webui', 'running', ?, ?, ?, ?, ?, ?, ?)
            """, (job_id, prompt[:4000], task, int(seed), int(steps),
                  float(duration_seconds), now, now))
        return job_id

    def complete_job(self, job_id: str, path: str | os.PathLike) -> None:
        name = self._relative_name(path)
        file = self.output_dir / name
        stat = file.stat()
        if not file.is_file() or stat.st_size == 0:
            raise FileNotFoundError(str(file))
        with self._connect() as db:
            # A file scanner might have imported this path while the job finished.
            db.execute("DELETE FROM videos WHERE file_name=? AND source='scan'", (name,))
            cursor = db.execute("""
                UPDATE videos SET file_name=?,status='completed',updated_at=?,size_bytes=?,error=NULL
                WHERE id=?
            """, (name, time.time(), stat.st_size, job_id))
            if cursor.rowcount != 1:
                raise KeyError(job_id)

    def fail_job(self, job_id: str, error: str) -> None:
        with self._connect() as db:
            db.execute("""
                UPDATE videos SET status='failed',updated_at=?,error=? WHERE id=?
            """, (time.time(), str(error)[:500], job_id))

    @staticmethod
    def _playable(path: Path) -> bool:
        probe = shutil.which("ffprobe")
        if not probe:
            # Without ffprobe, accept only settled files with an MP4 ftyp header.
            try:
                with path.open("rb") as fh:
                    header = fh.read(12)
                return len(header) == 12 and header[4:8] == b"ftyp"
            except OSError:
                return False
        try:
            result = subprocess.run(
                [probe, "-v", "error", "-show_entries", "format=format_name",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                timeout=5, check=False, text=True,
            )
            return result.returncode == 0 and "mp4" in result.stdout
        except (OSError, subprocess.TimeoutExpired):
            return False

    def scan(self, *, force: bool = False) -> int:
        """Import settled, valid MP4 files, including those created by the HTTP API.

        File metadata is never claimed to contain prompts for legacy/API files.
        """
        with self._scan_lock:
            now = time.time()
            if not force and time.monotonic() - self._last_scan < 15:
                return 0
            known = set()
            with self._connect() as db:
                known = {row[0] for row in db.execute(
                    "SELECT file_name FROM videos WHERE file_name IS NOT NULL")}
            imported = 0
            for path in self.output_dir.rglob("*.mp4"):
                try:
                    if not path.is_file() or path.is_symlink():
                        continue
                    name = self._relative_name(path)
                    if name in known:
                        continue
                    st = path.stat()
                    if st.st_size < 1024 or now - st.st_mtime < 30:
                        continue
                    if not self._playable(path):
                        continue
                    file_id = "scan_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]
                    with self._connect() as db:
                        changed = db.execute("""
                            INSERT OR IGNORE INTO videos
                            (id,file_name,source,status,created_at,updated_at,size_bytes)
                            VALUES(?, ?, 'scan', 'completed', ?, ?, ?)
                        """, (file_id, name, st.st_mtime, st.st_mtime, st.st_size)).rowcount
                    imported += changed
                    known.add(name)
                except (OSError, ValueError, sqlite3.Error):
                    # A file can disappear or be replaced during a concurrent scan.
                    continue
            self._last_scan = time.monotonic()
            return imported

    def page(self, page: int = 0, search: str = "", page_size: int = PAGE_SIZE):
        page = max(0, int(page))
        page_size = min(50, max(1, int(page_size)))
        search = (search or "").strip()[:120]
        predicate, args = "", []
        if search:
            # LIKE wildcards in user text should match literally.
            escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            predicate = "WHERE (file_name LIKE ? ESCAPE '\\' OR prompt LIKE ? ESCAPE '\\')"
            args = [f"%{escaped}%", f"%{escaped}%"]
        with self._connect() as db:
            count = db.execute(f"SELECT COUNT(*) FROM videos {predicate}", args).fetchone()[0]
            max_page = max(0, (count - 1) // page_size)
            page = min(page, max_page)
            rows = db.execute(f"""
                SELECT id,file_name,source,status,prompt,task,seed,steps,duration_seconds,
                       created_at,updated_at,size_bytes,error FROM videos
                {predicate} ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?
            """, [*args, page_size, page * page_size]).fetchall()
        return [dict(row) for row in rows], page, count

    def get(self, job_id: str):
        if not job_id:
            return None
        with self._connect() as db:
            row = db.execute("SELECT * FROM videos WHERE id=?", (str(job_id),)).fetchone()
        return dict(row) if row else None

    def video_path(self, job_id: str) -> str | None:
        row = self.get(job_id)
        if not row or row["status"] != "completed" or not row["file_name"]:
            return None
        try:
            path = (self.output_dir / row["file_name"]).resolve()
            if not path.is_relative_to(self.output_dir) or path.suffix.lower() != ".mp4":
                return None
            if not path.is_file() or path.stat().st_size == 0:
                return None
            return str(path)
        except OSError:
            return None
