"""Client side of PoseCraft's live Blender link: sends a pose document to the
`export/blender_addon/import_pose.py` add-on's "Live Link" server over a plain
TCP socket, so a solved pose can appear on the armature in Blender without a
file dialog on either side -- the same convenience PoseForge's "Open in DAZ
Studio" button gives for its own target application.

Wire format, one request and one reply per connection, each a 4-byte
big-endian length prefix followed by that many bytes of UTF-8 JSON:

- request: a pose_craft_pose document (see export/blender_pose.py) plus
  "live_link_protocol": PROTOCOL_VERSION, so the add-on can reject a newer
  protocol it doesn't understand.
- reply (add-on 0.4.0+), after it applied the pose (or failed to):
  {"ok": bool, "applied": [...], "requested": int, "unmatched": [...],
  "low_confidence": [...], "armature": str | None, "addon_version": [...],
  "protocol": int, "error": str | None}.

An older add-on closes the connection without replying; send_pose_document
then returns None, and the app says it couldn't confirm what was applied.
DEFAULT_PORT must match DEFAULT_LIVE_LINK_PORT in the Blender add-on, and
PROTOCOL_VERSION its LIVE_LINK_PROTOCOL_VERSION -- they can't share constants
since the add-on runs inside Blender's own Python and can't import this
package (tests/test_blender_addon_source.py checks they match).
"""
from __future__ import annotations

import json
import socket

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8642
PROTOCOL_VERSION = 2
# A reply is a few hundred bytes; anything claiming more is not the add-on.
MAX_REPLY_BYTES = 64 * 1024


def _recv_exact_or_none(sock: socket.socket, size: int) -> bytes | None:
    """Reads exactly `size` bytes, or returns None if the peer closes before
    sending anything at all (an add-on that doesn't reply)."""
    chunks = []
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
    request["live_link_protocol"] = PROTOCOL_VERSION
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
