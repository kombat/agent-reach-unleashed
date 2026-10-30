# -*- coding: utf-8 -*-
"""Watch a video: sample still frames into contact sheets and capture its speech.

Work happens in a folder of its own per run under ~/.agent-reach/watch/, which
`agent-reach uninstall` removes. The downloaded video and single frames are
deleted as soon as a run ends, even on failure; if the process exits first
(timeout, Ctrl-C) the whole run folder goes. Only contact sheets outlive a run,
for the agent to open, and the next run deletes anything untouched for an hour.
Each channel passes its own yt-dlp runner and speech source.
"""

from __future__ import annotations

import atexit
import functools
import hashlib
import math
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

from agent_reach import transcribe as tr
from agent_reach.runtime import state
from agent_reach.utils.paths import ensure_no_symlink_path, make_private_dir

MAX_VIDEO_BYTES = 512 * 1024 * 1024
MAX_FRAMES = 48
SHEET_COLUMNS, SHEET_ROWS = 4, 3
STALE_AFTER_SECONDS = 3600
FRAME_WIDTH = 480
VIDEO_FORMAT = "bv*[height<=480]+ba/b[height<=480]/b"
AUDIO_FORMAT = "ba/b"

#: (text, where it came from, why there is none)
Speech = Tuple[Optional[str], Optional[str], Optional[str]]


@dataclass
class Watched:
    folder: Path
    duration: float
    every_seconds: int
    frames: int
    sheets: List[Path]
    text: Optional[str]
    transcript_source: Optional[str]
    transcript_note: Optional[str]


def watch_root() -> Path:
    root = make_private_dir(state.root() / "watch")
    state.remember_path(root, "contact sheets from watched videos")
    return root


def _remove_tree(path: Path) -> None:
    ensure_no_symlink_path(path, "watch folder")
    shutil.rmtree(path, ignore_errors=True)


def _last_touched(folder: Path) -> float:
    times = [folder.stat().st_mtime]
    for entry in folder.rglob("*"):
        try:
            times.append(entry.lstat().st_mtime)
        except FileNotFoundError:
            continue
    return max(times)


def _clear_stale(root: Path) -> None:
    cutoff = time.time() - STALE_AFTER_SECONDS
    for folder in root.iterdir():
        try:
            if folder.is_dir() and not folder.is_symlink() and _last_touched(folder) < cutoff:
                _remove_tree(folder)
        except FileNotFoundError:
            continue


def _discard_media(folder: Path) -> None:
    for media in [*folder.glob("video.*"), *folder.glob("source.*")]:
        media.unlink(missing_ok=True)
    _remove_tree(folder / "frames")


@contextmanager
def run_folder(url: str) -> Iterator[Path]:
    """A private folder for one run. Media is deleted when the run ends; all of it if the process exits first."""
    root = watch_root()
    _clear_stale(root)
    name = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    folder = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=root))
    on_exit = functools.partial(_remove_tree, folder)
    atexit.register(on_exit)
    try:
        yield folder
    finally:
        _discard_media(folder)
        atexit.unregister(on_exit)


class _Budget:
    def __init__(self, seconds: float):
        self.deadline = time.monotonic() + seconds

    def left(self) -> int:
        seconds = int(self.deadline - time.monotonic())
        if seconds < 1:
            raise tr.TranscribeError("ran out of time")
        return seconds


def _download(url: str, folder: Path, fmt: str, stem: str, run_ytdlp: Callable[[list, float], str], seconds: int) -> Path:
    run_ytdlp(
        [
            "-f", fmt,
            "--merge-output-format", "mp4",
            "--no-playlist",
            "--max-filesize", str(MAX_VIDEO_BYTES),
            "--ffmpeg-location", tr._bin("ffmpeg"),
            "-o", str(folder / f"{stem}.%(ext)s"),
            "--", url,
        ],
        seconds,
    )
    found = sorted(p for p in folder.glob(f"{stem}.*") if not p.name.endswith((".part", ".ytdl")))
    if not found:
        raise tr.TranscribeError(f"yt-dlp produced no {stem} (it may exceed the 512 MiB limit)")
    return found[0]


def whisper_speech(video: Path) -> Speech:
    """The video's speech through Whisper, or why there is none, so the frames still come back."""
    try:
        return tr.transcribe(str(video)), "whisper", None
    except tr.NoProviderConfigured:
        return None, None, tr.NO_WHISPER_HINT
    except tr.TranscribeError as exc:
        return None, None, f"transcription failed: {str(exc)[:300]}"


def transcribe_url(url: str, *, run_ytdlp: Callable[[list, float], str], timeout: float) -> str:
    """Download only the audio with the channel's own runner, transcribe it, then delete it."""
    tr._assert_safe_public_url(url)
    budget = _Budget(timeout)
    with run_folder(url) as folder:
        try:
            audio = _download(url, folder, AUDIO_FORMAT, "source", run_ytdlp, budget.left())
            return tr.transcribe(str(audio))
        finally:
            _remove_tree(folder)


def watch(
    url: str,
    *,
    run_ytdlp: Callable[[list, float], str],
    speech: Callable[[Path], Speech],
    timeout: float,
    every: int,
) -> Watched:
    """Download `url`, take one frame every `every` seconds, get its speech, then delete the video."""
    if every < 1:
        raise tr.TranscribeError("every must be at least 1 second")
    tr._assert_safe_public_url(url)
    tr._require("ffmpeg")
    budget = _Budget(timeout)
    with run_folder(url) as folder:
        video = _download(url, folder, VIDEO_FORMAT, "video", run_ytdlp, budget.left())
        duration = tr._probe_audio_duration(video)
        every = max(every, math.ceil(duration / MAX_FRAMES))

        frames_dir = make_private_dir(folder / "frames")
        tr._run(
            [
                tr._bin("ffmpeg"), "-loglevel", "error", "-y", "-i", str(video),
                "-vf", f"fps=1/{every},scale={FRAME_WIDTH}:-2", "-frames:v", str(MAX_FRAMES),
                str(frames_dir / "frame_%03d.jpg"),
            ],
            timeout=budget.left(),
        )
        frames = len(list(frames_dir.glob("frame_*.jpg")))
        if not frames:
            raise tr.TranscribeError("ffmpeg produced no frames")
        tr._run(
            [
                tr._bin("ffmpeg"), "-loglevel", "error", "-y", "-i", str(frames_dir / "frame_%03d.jpg"),
                "-vf", f"tile={SHEET_COLUMNS}x{SHEET_ROWS}:padding=4",
                str(folder / "sheet_%02d.jpg"),
            ],
            timeout=budget.left(),
        )
        text, source, note = speech(video)
    return Watched(folder, duration, every, frames, sorted(folder.glob("sheet_*.jpg")), text, source, note)


def describe(watched: Watched) -> dict:
    """The `raw` block of a watch item: where the sheets are and how the frames were taken."""
    return {
        "folder": str(watched.folder),
        "duration_seconds": round(watched.duration, 1),
        "frame_every_seconds": watched.every_seconds,
        "frames": watched.frames,
        "sheets": [str(p) for p in watched.sheets],
        "sheet_layout": f"{SHEET_COLUMNS}x{SHEET_ROWS}, left to right, top to bottom",
        "sheets_cleared": "by the next watch once untouched for an hour",
        "video_kept": False,
        "transcript_source": watched.transcript_source,
        "transcript_note": watched.transcript_note,
    }
