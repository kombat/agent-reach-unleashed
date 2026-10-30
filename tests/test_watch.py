# -*- coding: utf-8 -*-
"""Tests for agent_reach.watch: frames into contact sheets, speech, and deleting the media."""

import atexit
import os
import time
from pathlib import Path

import pytest

from agent_reach import transcribe as tr
from agent_reach import watch as w
from agent_reach.runtime import state

URL = "https://www.instagram.com/reel/EXAMPLE1/"


def _speech(video):
    assert video.is_file(), "speech runs while the video still exists"
    return "spoken words", "whisper", None


@pytest.fixture
def fake_media(monkeypatch):
    """yt-dlp writes <stem>.mp4 (plus a stray .part); ffmpeg writes three frames, then one sheet."""
    monkeypatch.setattr(tr, "_require", lambda _binary: None)
    monkeypatch.setattr(tr, "_probe_audio_duration", lambda _path: 30.0)
    calls = {"ytdlp": [], "ffmpeg": []}

    def run_ytdlp(args, timeout):
        calls["ytdlp"].append((args, timeout))
        template = Path(args[args.index("-o") + 1])
        stem = template.name.split(".")[0]
        (template.parent / f"{stem}.mp4").write_bytes(b"media")
        (template.parent / f"{stem}.mp4.part").write_bytes(b"partial")
        return ""

    def run(cmd, timeout=600):
        calls["ffmpeg"].append((cmd, timeout))
        out = Path(cmd[-1])
        if out.name.startswith("frame_"):
            for i in (1, 2, 3):
                (out.parent / f"frame_{i:03d}.jpg").write_bytes(b"jpg")
        else:
            (out.parent / "sheet_01.jpg").write_bytes(b"jpg")

    monkeypatch.setattr(tr, "_run", run)
    return run_ytdlp, calls


def _watch(run_ytdlp, **kwargs):
    return w.watch(URL, run_ytdlp=run_ytdlp, speech=kwargs.pop("speech", _speech),
                   timeout=kwargs.pop("timeout", 60), every=kwargs.pop("every", 5))


def _age(path: Path, seconds: float) -> None:
    then = time.time() - seconds
    for entry in [path, *path.rglob("*")]:
        os.utime(entry, (then, then))


def test_watch_returns_sheets_and_speech_from_the_agent_reach_folder(fake_media):
    run_ytdlp, calls = fake_media

    watched = _watch(run_ytdlp)

    assert watched.folder.parent == Path.home() / ".agent-reach" / "watch"
    assert watched.frames == 3
    assert [p.name for p in watched.sheets] == ["sheet_01.jpg"]
    assert watched.text == "spoken words" and watched.transcript_source == "whisper"
    frames_cmd = calls["ffmpeg"][0][0]
    assert "fps=1/5" in frames_cmd[frames_cmd.index("-vf") + 1]
    assert calls["ytdlp"][0][0][-2:] == ["--", URL]


def test_frame_count_is_hard_capped_whatever_the_video_claims(fake_media):
    run_ytdlp, calls = fake_media

    _watch(run_ytdlp)

    frames_cmd = calls["ffmpeg"][0][0]
    assert frames_cmd[frames_cmd.index("-frames:v") + 1] == str(w.MAX_FRAMES)


def test_video_partial_files_and_single_frames_are_deleted_once_the_watch_ends(fake_media):
    run_ytdlp, _calls = fake_media

    watched = _watch(run_ytdlp)

    assert sorted(p.name for p in watched.folder.iterdir()) == ["sheet_01.jpg"]
    assert w.describe(watched)["video_kept"] is False


def test_video_is_deleted_even_when_a_later_step_fails(fake_media):
    run_ytdlp, _calls = fake_media

    def failing_speech(video):
        raise RuntimeError("speech step crashed")

    with pytest.raises(RuntimeError):
        _watch(run_ytdlp, speech=failing_speech)

    (folder,) = list(w.watch_root().iterdir())
    assert not list(folder.glob("video.*"))
    assert not (folder / "frames").exists()


def test_process_exit_mid_run_removes_the_whole_run_folder(fake_media, monkeypatch):
    run_ytdlp, _calls = fake_media
    registered = []
    monkeypatch.setattr(atexit, "register", lambda fn: registered.append(fn))
    monkeypatch.setattr(atexit, "unregister", lambda fn: None)

    def exit_now(video):
        registered[-1]()  # what atexit does if the CLI exits on a timeout or Ctrl-C
        assert not video.parent.exists()
        return None, None, None

    _watch(run_ytdlp, speech=exit_now)


def test_a_finished_run_no_longer_has_an_exit_hook(fake_media, monkeypatch):
    run_ytdlp, _calls = fake_media
    hooks = set()
    monkeypatch.setattr(atexit, "register", hooks.add)
    monkeypatch.setattr(atexit, "unregister", hooks.discard)

    _watch(run_ytdlp)

    assert hooks == set()


def test_each_run_gets_its_own_folder(fake_media):
    run_ytdlp, _calls = fake_media

    first, second = _watch(run_ytdlp), _watch(run_ytdlp)

    assert first.folder != second.folder
    assert all(p.is_file() for p in first.sheets + second.sheets)


def test_every_step_gets_only_the_time_left(fake_media, monkeypatch):
    run_ytdlp, calls = fake_media
    clock = iter([1000.0, 1000.0, 1040.0, 1050.0])
    monkeypatch.setattr(w.time, "monotonic", lambda: next(clock))

    _watch(run_ytdlp, timeout=60)

    assert calls["ytdlp"][0][1] == 60
    assert [timeout for _cmd, timeout in calls["ffmpeg"]] == [20, 10]


def test_running_out_of_time_stops_and_still_deletes_the_video(fake_media, monkeypatch):
    run_ytdlp, _calls = fake_media
    clock = iter([1000.0, 1000.0, 1100.0])
    monkeypatch.setattr(w.time, "monotonic", lambda: next(clock))

    with pytest.raises(tr.TranscribeError, match="ran out of time"):
        _watch(run_ytdlp, timeout=60)

    (folder,) = list(w.watch_root().iterdir())
    assert not list(folder.glob("video.*"))


def test_uninstall_knows_the_watch_folder_is_agent_reachs(fake_media):
    run_ytdlp, _calls = fake_media

    _watch(run_ytdlp)

    assert "watch" in state.load()["owned_paths"]


def test_anything_untouched_for_an_hour_is_cleared_by_the_next_run(fake_media):
    run_ytdlp, _calls = fake_media
    root = w.watch_root()
    old_sheets, left_behind, recent = root / "old-sheets", root / "left-behind", root / "recent"
    for folder in (old_sheets, left_behind, recent):
        folder.mkdir()
    (old_sheets / "sheet_01.jpg").write_bytes(b"jpg")
    (left_behind / "video.mp4.part").write_bytes(b"partial")
    (recent / "sheet_01.jpg").write_bytes(b"jpg")
    _age(old_sheets, 2 * 3600)
    _age(left_behind, 2 * 3600)

    _watch(run_ytdlp)

    assert not old_sheets.exists() and not left_behind.exists()
    assert recent.exists()


def test_a_folder_with_a_recently_touched_file_is_kept(fake_media):
    run_ytdlp, _calls = fake_media
    busy = w.watch_root() / "busy"
    busy.mkdir()
    (busy / "video.mp4.part").write_bytes(b"still downloading")
    _age(busy, 2 * 3600)
    os.utime(busy / "video.mp4.part")

    _watch(run_ytdlp)

    assert busy.exists()

def test_a_folder_removed_by_another_run_mid_sweep_is_ignored(fake_media, monkeypatch):
    run_ytdlp, _calls = fake_media
    gone = w.watch_root() / "gone"
    gone.mkdir()

    def vanished(_folder):
        raise FileNotFoundError

    monkeypatch.setattr(w, "_last_touched", vanished)

    _watch(run_ytdlp)


def test_long_video_spreads_frames_out_to_stay_under_the_cap(fake_media, monkeypatch):
    run_ytdlp, calls = fake_media
    monkeypatch.setattr(tr, "_probe_audio_duration", lambda _path: 600.0)

    watched = _watch(run_ytdlp)

    assert watched.every_seconds == 13
    assert "fps=1/13" in " ".join(calls["ffmpeg"][0][0])


def test_a_download_that_writes_no_video_is_an_error(fake_media):
    with pytest.raises(tr.TranscribeError, match="no video"):
        _watch(lambda args, timeout: "")


@pytest.mark.parametrize("every", [0, -3])
def test_every_must_be_at_least_one_second(fake_media, every):
    run_ytdlp, _calls = fake_media

    with pytest.raises(tr.TranscribeError, match="at least 1"):
        _watch(run_ytdlp, every=every)


def test_private_addresses_are_refused_before_any_download(fake_media):
    run_ytdlp, calls = fake_media

    with pytest.raises(tr.TranscribeError, match="SSRF"):
        w.watch("http://127.0.0.1/video", run_ytdlp=run_ytdlp, speech=_speech, timeout=60, every=5)
    assert calls["ytdlp"] == []


def test_transcribe_url_downloads_audio_only_and_deletes_it(fake_media, monkeypatch):
    run_ytdlp, calls = fake_media
    seen = []
    monkeypatch.setattr(tr, "transcribe", lambda source: seen.append(Path(source)) or "spoken words")

    assert w.transcribe_url(URL, run_ytdlp=run_ytdlp, timeout=60) == "spoken words"

    args = calls["ytdlp"][0][0]
    assert args[args.index("-f") + 1] == w.AUDIO_FORMAT
    assert seen[0].name == "source.mp4"
    assert not seen[0].parent.exists()


def test_whisper_speech_without_any_transcriber_explains_what_to_install(monkeypatch, tmp_path):
    def none_configured(*_a, **_k):
        raise tr.NoProviderConfigured("none")

    monkeypatch.setattr(tr, "transcribe", none_configured)

    text, source, note = w.whisper_speech(tmp_path / "video.mp4")

    assert text is None and source is None
    assert "agent-reach-unleashed update" in note


def test_whisper_speech_failure_is_reported_not_raised(monkeypatch, tmp_path):
    def broken(*_a, **_k):
        raise tr.TranscribeError("ffmpeg failed")

    monkeypatch.setattr(tr, "transcribe", broken)

    assert w.whisper_speech(tmp_path / "video.mp4") == (None, None, "transcription failed: ffmpeg failed")
