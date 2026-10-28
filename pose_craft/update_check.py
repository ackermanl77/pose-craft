"""Is a newer version out? A once-per-launch check against a small JSON feed.

Each release publishes a file like this somewhere users can reach:

    {"version": "0.3.0", "url": "https://example.com/download", "notes": "What changed"}

UPDATE_FEED_URL is empty until there is such a place (the GitHub repository is private),
and while it is empty the app never checks. Set it when cutting a release; the
POSECRAFT_UPDATE_FEED_URL environment variable overrides it (for testing a feed). Only https feeds and
download links are accepted, the reply is capped at 64 KB, and the request carries nothing
but a User-Agent naming the app and version: no identifiers, no telemetry.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

UPDATE_FEED_URL = ""
ENV_VAR = "POSECRAFT_UPDATE_FEED_URL"
APP_NAME = "PoseCraft"
TIMEOUT_SECONDS = 5
MAX_FEED_BYTES = 64 * 1024

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UpdateInfo:
    version: str
    url: str
    notes: str = ""


def feed_url() -> str:
    """The configured feed, or "" when update checks are off."""
    return (os.environ.get(ENV_VAR) or UPDATE_FEED_URL).strip()


def parse_version(text: str) -> tuple[int, ...]:
    """'v1.2.3' -> (1, 2, 3). Anything after the numeric part ('-beta', '+build') is ignored."""
    match = re.match(r"\s*v?(\d+(?:\.\d+)*)", text or "")
    if not match:
        raise ValueError(f"not a version: {text!r}")
    return tuple(int(part) for part in match.group(1).split("."))


def is_newer(candidate: str, current: str) -> bool:
    a, b = list(parse_version(candidate)), list(parse_version(current))
    width = max(len(a), len(b))
    return a + [0] * (width - len(a)) > b + [0] * (width - len(b))


def parse_feed(data: bytes) -> UpdateInfo:
    feed = json.loads(data.decode("utf-8"))
    if not isinstance(feed, dict):
        raise ValueError("the feed is not a JSON object")
    version, url, notes = feed.get("version"), feed.get("url"), feed.get("notes", "")
    if not isinstance(version, str) or not isinstance(url, str) or not isinstance(notes, str):
        raise ValueError("the feed needs string 'version' and 'url' fields")
    parse_version(version)
    if not url.lower().startswith("https://"):
        raise ValueError("the download link must be https")
    return UpdateInfo(version=version.strip(), url=url.strip(), notes=notes.strip()[:2000])


def check_for_update(current_version: str, *, url: str | None = None,
                     opener: Callable = urllib.request.urlopen) -> UpdateInfo | None:
    """The newer release, or None: no feed configured, already up to date, or any failure
    (offline, a bad feed...). Never raises; failures are logged at debug level."""
    url = feed_url() if url is None else url
    if not url:
        return None
    if not url.lower().startswith("https://"):
        logger.warning("Update feed ignored: not an https URL (%s)", url)
        return None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": f"PoseCraft/{current_version}"})
        with opener(request, timeout=TIMEOUT_SECONDS) as response:
            data = response.read(MAX_FEED_BYTES + 1)
        if len(data) > MAX_FEED_BYTES:
            raise ValueError("the feed is larger than 64 KB")
        info = parse_feed(data)
        return info if is_newer(info.version, current_version) else None
    except Exception as error:  # noqa: BLE001 -- an update check must never break the app
        logger.debug("Update check failed: %s", error)
        return None


def check_in_background(current_version: str, on_update: Callable[[UpdateInfo], None]) -> threading.Thread | None:
    """Run check_for_update on a daemon thread and call on_update(info) there when a newer
    version exists. Returns None (and starts nothing) when no feed is configured. The
    callback runs on the worker thread: hand it to the UI thread yourself."""
    if not feed_url():
        return None

    def work() -> None:
        info = check_for_update(current_version)
        if info is not None:
            on_update(info)

    thread = threading.Thread(target=work, name="update-check", daemon=True)
    thread.start()
    return thread
