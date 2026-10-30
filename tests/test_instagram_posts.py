# -*- coding: utf-8 -*-
"""Instagram single posts and reels, read logged out through yt-dlp."""

import json
from pathlib import Path

import pytest

from agent_reach import transcribe as tr
from agent_reach import watch as w
from agent_reach.channels import instagram
from agent_reach.channels.instagram import LOGGED_OUT, InstagramChannel
from agent_reach.channels.web import WebChannel
from agent_reach.runtime.dispatch import run_action
from agent_reach.runtime.result import ReachError, make_item
from agent_reach.runtime.run import CommandFailed

REEL = "https://www.instagram.com/reel/EXAMPLE1/"
PROFILE = "https://www.instagram.com/example_creator/"
INFO = {
    "title": "Video by example_creator",
    "webpage_url": REEL,
    "channel": "example_creator",
    "uploader": "Example Creator",
    "upload_date": "20260101",
    "description": "An example caption",
    "like_count": 100,
    "comment_count": 7,
}


def _write_media(argv):
    if "-o" in argv:
        template = Path(argv[argv.index("-o") + 1])
        (template.parent / f"{template.name.split('.')[0]}.mp4").write_bytes(b"media")


@pytest.fixture
def ytdlp(monkeypatch):
    calls = []

    def fake_run_cmd(argv, **_kwargs):
        calls.append(argv)
        _write_media(argv)
        return json.dumps(INFO)

    monkeypatch.setattr(instagram, "run_cmd", fake_run_cmd)
    return calls


@pytest.fixture
def web_reader(monkeypatch):
    monkeypatch.setattr(WebChannel, "_jina_read", lambda self, *, target, timeout: [make_item(url=target, text="page text")])


def _yt_dlp_fails(monkeypatch, stderr):
    def failing(argv, **_kwargs):
        raise CommandFailed("UPSTREAM_BROKEN", "yt-dlp exited 1", stderr=stderr)

    monkeypatch.setattr(instagram, "run_cmd", failing)


def _assert_logged_out(argv):
    assert list(LOGGED_OUT) == argv[argv.index("--ignore-config"):][:len(LOGGED_OUT)]
    assert not any(arg == "--cookies" or arg == "--cookies-from-browser" for arg in argv)


def test_reel_is_read_with_ytdlp_and_no_login(ytdlp):
    result = run_action(InstagramChannel(), "read", {"target": REEL})

    assert result["ok"] and result["backend"] == "yt-dlp"
    item = result["items"][0]
    assert item["author"] == "example_creator"
    assert item["published_at"] == "2026-01-01"
    assert item["text"] == "An example caption"
    assert item["raw"]["like_count"] == 100
    _assert_logged_out(ytdlp[0])


def test_every_instagram_ytdlp_call_ignores_user_config_and_cookies(ytdlp, monkeypatch):
    monkeypatch.setattr(tr, "_require", lambda _binary: None)
    monkeypatch.setattr(tr, "_probe_audio_duration", lambda _path: 10.0)
    monkeypatch.setattr(tr, "transcribe", lambda source: "spoken words")
    monkeypatch.setattr(tr, "_run", lambda cmd, timeout=600: Path(cmd[-1]).parent.joinpath(
        "frame_001.jpg" if "frame_" in cmd[-1] else "sheet_01.jpg").write_bytes(b"jpg"))

    InstagramChannel()._whisper_transcribe(target=REEL, timeout=60)
    InstagramChannel()._ytdlp_watch(target=REEL, every=5, timeout=60)

    assert len(ytdlp) == 3
    for argv in ytdlp:
        _assert_logged_out(argv)


def test_read_command_sends_reel_links_to_instagram(ytdlp, capsys):
    from agent_reach.runtime.cli import _cmd_read

    _cmd_read([REEL])

    out = json.loads(capsys.readouterr().out)
    assert out["channel"] == "instagram" and out["backend"] == "yt-dlp"


@pytest.mark.parametrize("url", [
    PROFILE,
    "https://l.instagram.com/p/EXAMPLE1/",
    "https://instagr.am/p/EXAMPLE1/",
])
def test_links_yt_dlp_should_not_get_still_go_to_the_web_reader(ytdlp, web_reader, url):
    result = run_action(InstagramChannel(), "read", {"target": url})

    assert result["ok"] and result["backend"] == "jina"
    assert ytdlp == []


def test_login_wall_falls_back_to_the_web_reader_for_the_caption(monkeypatch, web_reader):
    _yt_dlp_fails(monkeypatch, "ERROR: Requested content is not available, rate-limit reached or login required")

    result = run_action(InstagramChannel(), "read", {"target": REEL})

    assert result["ok"] and result["backend"] == "jina"
    assert result["meta"]["attempts"][0]["result"] == "UPSTREAM_BROKEN"


def test_empty_ytdlp_output_is_a_clear_error_not_a_crash(monkeypatch, web_reader):
    monkeypatch.setattr(instagram, "run_cmd", lambda argv, **_k: "")

    result = run_action(InstagramChannel(), "read", {"target": REEL})

    assert result["meta"]["attempts"][0] == {"backend": "yt-dlp", "result": "UPSTREAM_BROKEN",
                                             "message": "yt-dlp returned nothing for this post"}


def test_rate_limit_stops_without_trying_the_web_reader(monkeypatch, web_reader):
    _yt_dlp_fails(monkeypatch, "HTTP Error 429: Too Many Requests")

    result = run_action(InstagramChannel(), "read", {"target": REEL})

    assert result["error"]["code"] == "RATE_LIMITED"
    assert [a["backend"] for a in result["meta"]["attempts"]] == ["yt-dlp"]


@pytest.mark.parametrize("target", [PROFILE, "https://l.instagram.com/reel/EXAMPLE1/"])
@pytest.mark.parametrize("action", ["transcribe", "watch"])
def test_transcribe_and_watch_need_an_instagram_com_post_or_reel(action, target):
    with pytest.raises(ReachError) as exc:
        getattr(InstagramChannel(), {"transcribe": "_whisper_transcribe", "watch": "_ytdlp_watch"}[action])(
            target=target, timeout=5, **({"every": 5} if action == "watch" else {}))
    assert exc.value.code == "BAD_INPUT"


def test_watch_returns_the_spoken_words_with_the_caption_and_frames(ytdlp, monkeypatch, tmp_path):
    monkeypatch.setattr(w.tr, "transcribe", lambda source: "spoken words")
    monkeypatch.setattr(w, "watch", lambda url, *, speech, **_k: w.Watched(
        tmp_path, 99.9, 5, 20, [tmp_path / "sheet_01.jpg"], *speech(tmp_path / "video.mp4")))

    item = InstagramChannel()._ytdlp_watch(target=REEL, every=5, timeout=60)[0]

    assert item["text"] == "spoken words"
    assert item["raw"]["caption"] == "An example caption"
    assert item["raw"]["transcript_source"] == "whisper"
    assert item["raw"]["sheets"] == [str(Path(tmp_path / "sheet_01.jpg"))]
