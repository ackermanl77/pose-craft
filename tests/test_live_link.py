from __future__ import annotations

import json
import socket
import threading

import pytest

from pose_craft.export import live_link


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("closed early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def test_send_pose_document_wire_format():
    """A minimal stand-in for the Blender add-on's server: accept one connection,
    read the length-prefixed message the same way import_pose.py does, and hand
    the decoded document back to the test.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    received = {}

    def _serve():
        conn, _addr = server.accept()
        with conn:
            header = _recv_exact(conn, 4)
            length = int.from_bytes(header, "big")
            payload = _recv_exact(conn, length)
            received["document"] = json.loads(payload.decode("utf-8"))

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    document = {"format": "pose_craft_pose", "version": 1, "bones": {"spine": {"quaternion_wxyz": [1, 0, 0, 0]}}}
    live_link.send_pose_document(document, host="127.0.0.1", port=port, timeout=2.0)

    thread.join(timeout=2.0)
    server.close()
    assert received["document"] == document


def test_send_pose_document_raises_when_nothing_listening():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.close()  # bound-and-released port, nothing listening on it now

    with pytest.raises(OSError):
        live_link.send_pose_document({"format": "pose_craft_pose"}, host="127.0.0.1", port=port, timeout=1.0)
