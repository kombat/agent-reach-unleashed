# -*- coding: utf-8 -*-
"""Instagram users, profiles and posts through OpenCLI in the shared Agent Reach browser.

Single posts and reels are read with yt-dlp, logged out: no login, no cookies and
no user yt-dlp config. search, profile and posts still use the logged-in browser.
"""

import json
import time

from agent_reach.runtime.result import RATE_LIMITED, UPSTREAM_BROKEN, ReachError, make_item
from agent_reach.runtime.run import CommandFailed, run_cmd

from ._opencli_site import OpenCLISiteChannel
from .base import Action, Param
from .youtube import _iso_date, _ytdlp_argv

_POST_PATHS = ("/p/", "/reel/", "/reels/", "/tv/")
_POST_HOSTS = ("instagram.com", "www.instagram.com")
_RAW_KEYS = ("id", "duration", "like_count", "comment_count", "channel")
#: Never load the user's yt-dlp config or any cookies, and use only the Instagram extractor.
LOGGED_OUT = ("--ignore-config", "--no-cookies", "--no-cookies-from-browser", "--use-extractors", "instagram")


def _instagram_url(target: str) -> str:
    from agent_reach.utils.url import host_matches

    url = target if "://" in target else f"https://{target}"
    if not host_matches(url, "instagram.com", "instagr.am"):
        raise ReachError("BAD_INPUT", "target must be an Instagram link")
    return url


def _is_post(url: str) -> bool:
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    return (parts.hostname or "").lower() in _POST_HOSTS and parts.path.startswith(_POST_PATHS)


def _post_url(target: str) -> str:
    url = _instagram_url(target)
    if not _is_post(url):
        raise ReachError("BAD_INPUT", "target must be an instagram.com post or reel link")
    return url


def _run_ytdlp(args: list, timeout: float) -> str:
    try:
        return run_cmd(_ytdlp_argv(*LOGGED_OUT, *args), timeout=timeout, label="yt-dlp")
    except CommandFailed as exc:
        err = exc.stderr.lower()
        if "429" in err or "too many requests" in err:
            raise ReachError(RATE_LIMITED, exc.message) from None
        if "login required" in err or "no video in this post" in err:
            raise ReachError(
                UPSTREAM_BROKEN,
                "Instagram would not serve this post's video without a login",
                hint="Agent Reach Unleashed reads Instagram logged out only; the caption may still be readable.",
            ) from None
        raise


class InstagramChannel(OpenCLISiteChannel):
    name = "instagram"
    description = "Instagram 用户、主页和指定用户帖子"
    site = "instagram"
    domains = ("instagram.com", "instagr.am")
    usage = "opencli instagram search/profile/user/explore -f yaml"
    login_hint = "instagram.com"

    level = "experimental"
    login_url = "https://www.instagram.com/accounts/login/"
    actions = (
        Action(
            "search",
            "Search users (not posts)",
            (
                Param("query", "what to search for"),
                Param("limit", "max results", required=False, type=int, default=5),
            ),
            (("opencli", "_opencli_search"),),
            live_test={"query": "nasa", "limit": 2},
        ),
        Action(
            "profile",
            "A user's profile",
            (Param("username", "Instagram username"),),
            (("opencli", "_opencli_profile"),),
            live_test={"username": "nasa"},
        ),
        Action(
            "posts",
            "A user's recent posts",
            (
                Param("username", "Instagram username"),
                Param("limit", "max posts", required=False, type=int, default=6),
            ),
            (("opencli", "_opencli_posts"),),
            live_test={"username": "nasa", "limit": 2},
        ),
        Action(
            "read",
            "One post or reel: caption, author, date, likes; other pages as web text (no login)",
            (Param("target", "post or reel link"),),
            (("yt-dlp", "_ytdlp_read"), ("jina", "_jina_read"), ("exa", "_exa_read")),
        ),
        Action(
            "transcribe",
            "Speech-to-text of a reel's audio (no login)",
            (Param("target", "reel link"),),
            (("whisper", "_whisper_transcribe"),),
            timeout=1800,
        ),
        Action(
            "watch",
            "A reel's transcript plus still frames saved as images an AI can look at (no login)",
            (
                Param("target", "reel link"),
                Param("every", "seconds between frames", required=False, type=int, default=5),
            ),
            (("yt-dlp", "_ytdlp_watch"),),
            timeout=1800,
        ),
    )

    def _opencli_search(self, *, query: str, limit: int, timeout: float) -> list:
        return self._opencli(["search", query, "--limit", str(limit)], timeout)[:limit]

    def _opencli_profile(self, *, username: str, timeout: float) -> list:
        return self._opencli(["profile", username], timeout, empty_is_error=False)[:1]

    def _opencli_posts(self, *, username: str, limit: int, timeout: float) -> list:
        return self._opencli(["user", username, "--limit", str(limit)], timeout)[:limit]

    def _ytdlp_read(self, *, target: str, timeout: float) -> list:
        url = _instagram_url(target)
        if not _is_post(url):
            raise ReachError(UPSTREAM_BROKEN, "yt-dlp reads single posts and reels only")
        lines = _run_ytdlp(["--dump-json", "--skip-download", "--no-playlist", "--", url], timeout).splitlines()
        if not lines:
            raise ReachError(UPSTREAM_BROKEN, "yt-dlp returned nothing for this post")
        info = json.loads(lines[-1])
        return [
            make_item(
                title=info.get("title"),
                url=info.get("webpage_url") or url,
                author=info.get("channel") or info.get("uploader"),
                published_at=_iso_date(info.get("upload_date")),
                text=info.get("description"),
                raw={k: info.get(k) for k in _RAW_KEYS},
            )
        ]

    def _jina_read(self, *, target: str, timeout: float) -> list:
        from .web import WebChannel

        return WebChannel()._jina_read(target=_instagram_url(target), timeout=timeout)

    def _exa_read(self, *, target: str, timeout: float) -> list:
        from .web import WebChannel

        return WebChannel()._exa_read(target=_instagram_url(target), timeout=timeout)

    def _whisper_transcribe(self, *, target: str, timeout: float) -> list:
        from agent_reach import watch as w
        from agent_reach.transcribe import NO_WHISPER_HINT, NoProviderConfigured, TranscribeError

        url = _post_url(target)
        try:
            text = w.transcribe_url(url, run_ytdlp=_run_ytdlp, timeout=timeout)
        except NoProviderConfigured:
            raise ReachError("NEED_SETUP", "no Whisper configured", hint=NO_WHISPER_HINT) from None
        except TranscribeError as exc:
            raise ReachError(UPSTREAM_BROKEN, str(exc)[:300]) from None
        return [make_item(url=url, text=text)]

    def _ytdlp_watch(self, *, target: str, every: int, timeout: float) -> list:
        from agent_reach import watch as w
        from agent_reach.transcribe import TranscribeError

        url = _post_url(target)
        started = time.monotonic()
        info = self._ytdlp_read(target=url, timeout=timeout)[0]
        left = timeout - (time.monotonic() - started)
        try:
            watched = w.watch(url, run_ytdlp=_run_ytdlp, speech=w.whisper_speech, timeout=left, every=every)
        except TranscribeError as exc:
            raise ReachError(UPSTREAM_BROKEN, str(exc)[:300]) from None
        raw = {**(info.get("raw") or {}), "caption": info.get("text"), **w.describe(watched)}
        return [make_item(title=info.get("title"), url=url, author=info.get("author"),
                          published_at=info.get("published_at"), text=watched.text, raw=raw)]
