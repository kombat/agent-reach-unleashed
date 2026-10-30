# -*- coding: utf-8 -*-
"""Dedicated tests for the ``youtube`` channel.

YouTube's readiness is layered: yt-dlp must probe alive (missing / broken /
unrunnable are distinct), a JS runtime is mandatory (deno works out of the
box, Node needs an explicit ``--js-runtimes`` config), and transcription
readiness (whisper provider + ffmpeg/ffprobe) is surfaced for ``doctor`` without
gating the backend. These tests stub ``probe_command`` / ``shutil.which``
so every branch runs offline. Follow-up to #331 — extends dedicated
channel coverage after rss (#360), github (#361), web (#363),
reddit (#364), xueqiu (#365) and v2ex (#366).
"""

from pathlib import Path

import pytest

from agent_reach.channels.youtube import YouTubeChannel


def _which(*present):
    """shutil.which side effect: returns a path for the named tools present."""
    return lambda name: f"/usr/bin/{name}" if name in present else None


# --- can_handle ---

def test_can_handle_matches_youtube_hosts():
    ch = YouTubeChannel()
    for url in [
        "https://www.youtube.com/watch?v=abc",
        "https://youtu.be/abc",
        "https://m.youtube.com/watch?v=abc",
        "https://YOUTUBE.COM/shorts/x",
    ]:
        assert ch.can_handle(url) is True, url
    for url in ["https://example.com", "https://vimeo.com/123", ""]:
        assert ch.can_handle(url) is False, url


# --- _has_js_runtime_config ---









# --- check(): yt-dlp probe states ---







# --- check(): JS runtime gating (backend stays yt-dlp once the probe is ok) ---











def test_constraints_pin_ytdlp_to_current_stable_release():
    constraints = Path(__file__).parents[1] / "constraints.txt"
    pin = next(
        line
        for line in constraints.read_text(encoding="utf-8").splitlines()
        if line.startswith("yt-dlp==")
    )

    assert pin == "yt-dlp==2026.8.19"


def test_project_installs_official_ytdlp_default_dependencies():
    root = Path(__file__).parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    constraints = (root / "constraints.txt").read_text(encoding="utf-8")

    assert '"yt-dlp[default]==2026.8.19"' in pyproject
    assert "yt-dlp-ejs==0.8.0" in constraints




# --- check(): transcription readiness surfaced (deno present so JS path is ok) ---









# --- subtitle language and watch ---

def _subtitle_action():
    return next(a for a in YouTubeChannel.actions if a.name == "subtitle")


def test_subtitles_default_to_english_only():
    lang = next(p for p in _subtitle_action().params if p.name == "lang")
    assert lang.default.split(",") == ["en", "en-orig"]


def _watch_setup(monkeypatch, tmp_path):
    from agent_reach import watch as w
    from agent_reach.runtime.result import make_item

    monkeypatch.setattr(w, "watch", lambda url, *, speech, **_k: w.Watched(
        tmp_path, 510.0, 10, 47, [tmp_path / "sheet_01.jpg"], *speech(tmp_path / "video.mp4")))
    monkeypatch.setattr(YouTubeChannel, "_ytdlp_read", lambda self, *, target, timeout: [
        make_item(title="An example video", url=target, author="Example Channel", raw={"id": "EXAMPLE0001"})])
    return w


def test_watch_prefers_the_videos_own_subtitles(monkeypatch, tmp_path):
    from agent_reach.runtime.result import make_item

    w = _watch_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(YouTubeChannel, "_ytdlp_subtitle", lambda self, *, target, lang, timeout: [make_item(text="from captions")])
    monkeypatch.setattr(w, "whisper_speech", lambda video: pytest.fail("Whisper ran although subtitles exist"))

    item = YouTubeChannel()._ytdlp_watch(target="EXAMPLE0001", every=10, timeout=60)[0]

    assert item["text"] == "from captions"
    assert item["raw"]["transcript_source"] == "subtitles"
    assert item["raw"]["id"] == "EXAMPLE0001"


def test_watch_falls_back_to_whisper_when_there_are_no_subtitles(monkeypatch, tmp_path):
    from agent_reach.runtime.result import NOT_FOUND, ReachError

    w = _watch_setup(monkeypatch, tmp_path)

    def no_subtitles(self, *, target, lang, timeout):
        raise ReachError(NOT_FOUND, "no subtitles")

    monkeypatch.setattr(YouTubeChannel, "_ytdlp_subtitle", no_subtitles)
    monkeypatch.setattr(w.tr, "transcribe", lambda source: "from whisper")

    item = YouTubeChannel()._ytdlp_watch(target="EXAMPLE0001", every=10, timeout=60)[0]

    assert item["text"] == "from whisper"
    assert item["raw"]["transcript_source"] == "whisper"
    assert item["raw"]["frames"] == 47
    assert item["raw"]["video_kept"] is False
