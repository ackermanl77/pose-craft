"""Client side of PoseCraft's live Blender link: sends a pose document to the
`export/blender_addon/import_pose.py` add-on's "Live Link" server over a plain
TCP socket, so a solved pose can appear on the armature in Blender without a
file dialog on either side -- the same convenience PoseForge's "Open in DAZ
Studio" button gives for its own target application.

Wire format: a 4-byte big-endian length prefix, then that many bytes of UTF-8
JSON (a pose_craft_pose document, see export/blender_pose.py). One message per
connection. DEFAULT_PORT must match LIVE_LINK_PORT in the Blender add-on --
they can't share a constant since the add-on runs inside Blender's own Python
and can't import this package.
"""
from __future__ import annotations

import json
import socket

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8642


def send_pose_document(document: dict, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 3.0) -> None:
    """Raises OSError (e.g. ConnectionRefusedError, TimeoutError) if Blender's
    Live Link isn't running or isn't reachable -- callers should catch OSError
    and show the user a clear message rather than letting it propagate raw.
    """
    payload = json.dumps(document).encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(len(payload).to_bytes(4, "big") + payload)
