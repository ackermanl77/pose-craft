"""Client side of PoseCraft's live Blender link: sends a pose document to the
`export/blender_addon/import_pose.py` add-on's "Live Link" server over a plain
TCP socket, so a solved pose can appear on the armature in Blender without a
file dialog on either side -- the same convenience PoseForge's "Open in DAZ
Studio" button gives for its own target application.

Wire format, one request and one reply per connection, each a 4-byte
big-endian length prefix followed by that many bytes of UTF-8 JSON:

- request: a pose_craft_pose document (see export/blender_pose.py) plus
  "live_link_protocol": the lowest protocol that understands the request, so
  the add-on can reject a newer protocol it doesn't understand.
- reply (add-on 0.4.0+), after it applied the pose (or failed to):
  {"ok": bool, "applied": [...], "requested": int, "unmatched": [...],
  "low_confidence": [...], "armature": str | None, "addon_version": [...],
  "protocol": int, "error": str | None}.

An older add-on closes the connection without replying; send_pose_document
then returns None, and the app says it couldn't confirm what was applied.

Animations (protocol 3, add-on 0.7.0+): send_animation_document() sends a
pose_craft_animation document (see blender_pose.build_animation_document)
as several requests ("chunks"), each under ANIMATION_CHUNK_MAX_BYTES -- below
the add-on's 256 KB MAX_LIVE_LINK_PAYLOAD_BYTES, which is kept rather than
raised -- and at most ANIMATION_CHUNK_MAX_FRAMES frames, so no single request
keeps Blender's UI busy for long. Each chunk adds "chunk": {"index", "count"}
and "first_frame_index" (the whole clip's first frame, so every chunk is
spaced from the same anchor); the first reply says which Blender frame the
animation starts on ("start_frame") and later chunks repeat it. A request
carries the LOWEST protocol that understands it: single poses still say 2 (a
0.4-0.6 add-on keeps working), animations say 3 (an older add-on refuses them
with "update the add-on").

DEFAULT_PORT must match DEFAULT_LIVE_LINK_PORT in the Blender add-on, and
PROTOCOL_VERSION its LIVE_LINK_PROTOCOL_VERSION -- they can't share constants
since the add-on runs inside Blender's own Python and can't import this
package (tests/test_blender_addon_source.py checks they match).
"""
from __future__ import annotations

import json
import socket
from typing import Any

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8642
PROTOCOL_VERSION = 4
POSE_REQUEST_PROTOCOL = 2
ANIMATION_REQUEST_PROTOCOL = 3
# F17 (add-on 0.9.0+): {"format": "pose_craft_library", "path": <bundle JSON>} -- Blender reads the
# bundle (pose_library.PoseLibrary.export_blender_bundle) from disk and makes Pose Library assets.
LIBRARY_REQUEST_PROTOCOL = 4
LIBRARY_FORMAT_NAME = "pose_craft_library"
LIBRARY_TIMEOUT_SECONDS = 60.0
# A reply is a few hundred bytes; anything claiming more is not the add-on.
MAX_REPLY_BYTES = 64 * 1024
# Under the add-on's MAX_LIVE_LINK_PAYLOAD_BYTES (256 KB) with room for the
# request's own fields; a body-only frame is ~5 KB, one with fingers ~15 KB.
ANIMATION_CHUNK_MAX_BYTES = 200_000
ANIMATION_CHUNK_MAX_FRAMES = 30
# Blender keys a whole chunk (frame_set + a rig solve per frame) before it
# replies, which can take longer than a single pose on a heavy rig.
ANIMATION_TIMEOUT_SECONDS = 30.0


def _recv_exact_or_none(sock: socket.socket, size: int) -> bytes | None:
    """Reads exactly `size` bytes, or returns None if the peer closes before
    sending anything at all (an add-on that doesn't reply)."""
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            if remaining == size and not chunks:
                return None
            raise ValueError("Live Link reply was cut off")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _exchange(request: dict, host: str, port: int, timeout: float) -> dict | None:
    """One request and its reply over a fresh connection (see the module
    docstring); None when the add-on closes without replying."""
    payload = json.dumps(request).encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(len(payload).to_bytes(4, "big") + payload)
        header = _recv_exact_or_none(sock, 4)
        if header is None:
            return None
        length = int.from_bytes(header, "big")
        if length > MAX_REPLY_BYTES:
            raise ValueError(f"Live Link reply too large ({length} bytes)")
        body = _recv_exact_or_none(sock, length) if length else b""
        try:
            reply = json.loads((body or b"").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed Live Link reply: {exc}") from None
    if not isinstance(reply, dict):
        raise ValueError("malformed Live Link reply: not a JSON object")
    return reply


def send_pose_document(
    document: dict, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 3.0
) -> dict | None:
    """Sends `document` and returns the add-on's reply (see module docstring),
    or None when the add-on predates replies.

    Raises OSError (e.g. ConnectionRefusedError, TimeoutError) if Blender's
    Live Link isn't running or isn't reachable, and ValueError for a malformed
    reply -- callers should catch both and show the user a clear message.
    """
    request = dict(document)
    request["live_link_protocol"] = POSE_REQUEST_PROTOCOL
    return _exchange(request, host, port, timeout)


def send_library_request(
    bundle_path: str, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = LIBRARY_TIMEOUT_SECONDS
) -> dict | None:
    """Asks the add-on to turn the library bundle at `bundle_path` into Pose
    Library assets (F17). Only the path travels: Live Link is local, and a
    library with thumbnails is far over the add-on's request size limit. The
    reply's "library" says how many assets were made."""
    request = {"format": LIBRARY_FORMAT_NAME, "path": str(bundle_path), "live_link_protocol": LIBRARY_REQUEST_PROTOCOL}
    return _exchange(request, host, port, timeout)


def split_animation_frames(
    frames: list[dict], max_bytes: int = ANIMATION_CHUNK_MAX_BYTES, max_frames: int = ANIMATION_CHUNK_MAX_FRAMES
) -> list[list[dict]]:
    """An animation's frames in consecutive chunks of at most `max_frames`
    frames whose JSON adds up to at most `max_bytes`. Raises ValueError for a
    single frame over `max_bytes` (it could never be sent)."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    size = 0
    for frame in frames:
        frame_size = len(json.dumps(frame).encode("utf-8")) + 2  # the ", " between list items
        if frame_size > max_bytes:
            raise ValueError(f"one animation frame is {frame_size} bytes, over the Live Link limit of {max_bytes}")
        if current and (size + frame_size > max_bytes or len(current) >= max_frames):
            chunks.append(current)
            current, size = [], 0
        current.append(frame)
        size += frame_size
    if current:
        chunks.append(current)
    return chunks


def _merge_unique(target: list, items) -> None:
    for item in items or ():
        if item not in target:
            target.append(item)


def send_animation_document(
    document: dict,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = ANIMATION_TIMEOUT_SECONDS,
    max_bytes: int = ANIMATION_CHUNK_MAX_BYTES,
    max_frames: int = ANIMATION_CHUNK_MAX_FRAMES,
    on_chunk=None,
) -> dict:
    """Sends a pose_craft_animation document in chunks (see the module
    docstring) and returns one combined reply: "ok", the union of "applied" /
    "unmatched" / "low_confidence", the largest "requested", "armature",
    "animation_frames" (frames keyed in all), "start_frame" / "end_frame" (the
    Blender frames keyed) and "chunks". Stops at the first chunk Blender
    refuses and returns that error, saying how many frames were keyed before
    it. on_chunk(done, total) is called after each chunk.

    Raises OSError / ValueError like send_pose_document, and ValueError when
    the add-on doesn't answer like a 0.7.0+ add-on (an older one can't key
    animations)."""
    frames = list(document.get("frames") or [])
    if not frames:
        raise ValueError("the animation has no frames")
    chunks = split_animation_frames(frames, max_bytes, max_frames)
    header = {key: value for key, value in document.items() if key != "frames"}
    header["first_frame_index"] = int(frames[0]["frame_index"])
    combined: dict[str, Any] = {
        "ok": True, "applied": [], "unmatched": [], "low_confidence": [], "requested": 0, "armature": None,
        "animation_frames": 0, "start_frame": None, "end_frame": None, "chunks": len(chunks), "error": None,
    }
    for index, chunk in enumerate(chunks):
        request = dict(header, frames=chunk, chunk={"index": index, "count": len(chunks)})
        if combined["start_frame"] is not None:
            request["start_frame"] = combined["start_frame"]
        request["live_link_protocol"] = ANIMATION_REQUEST_PROTOCOL
        reply = _exchange(request, host, port, timeout)
        if reply is None:
            raise ValueError("the Blender add-on didn't reply; update it to 0.7.0 or newer to receive animations")
        if not reply.get("ok", False):
            error = reply.get("error") or "unknown error"
            if combined["animation_frames"]:
                error += f" (after {combined['animation_frames']} of {len(frames)} frames were keyed)"
            return dict(combined, ok=False, error=error, armature=reply.get("armature") or combined["armature"])
        if reply.get("start_frame") is None:
            raise ValueError("the Blender add-on didn't key an animation; update it to 0.7.0 or newer")
        _merge_unique(combined["applied"], reply.get("applied"))
        _merge_unique(combined["unmatched"], reply.get("unmatched"))
        _merge_unique(combined["low_confidence"], reply.get("low_confidence"))
        combined["requested"] = max(combined["requested"], int(reply.get("requested") or 0))
        combined["armature"] = reply.get("armature")
        combined["animation_frames"] += int(reply.get("animation_frames") or 0)
        if combined["start_frame"] is None:
            combined["start_frame"] = reply.get("start_frame")
        if reply.get("end_frame") is not None:  # a short chunk can fall between two scene frames
            combined["end_frame"] = reply.get("end_frame")
        if on_chunk is not None:
            on_chunk(index + 1, len(chunks))
    return combined
