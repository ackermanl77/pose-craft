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
    sent = dict(received["document"])
    # A single pose says the lowest protocol that understands it, so a 0.4-0.6 add-on still takes it.
    assert sent.pop("live_link_protocol") == live_link.POSE_REQUEST_PROTOCOL == 2
    assert sent == document


def test_send_pose_document_raises_when_nothing_listening():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.close()  # bound-and-released port, nothing listening on it now

    with pytest.raises(OSError):
        live_link.send_pose_document({"format": "pose_craft_pose"}, host="127.0.0.1", port=port, timeout=1.0)


def _serve_once(reply: dict | None, received: dict, raw_reply: bytes | None = None):
    """A stand-in add-on: read one request, optionally write a length-prefixed reply."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def _serve():
        conn, _addr = server.accept()
        with conn:
            length = int.from_bytes(_recv_exact(conn, 4), "big")
            received["document"] = json.loads(_recv_exact(conn, length).decode("utf-8"))
            if raw_reply is not None:
                conn.sendall(raw_reply)
            elif reply is not None:
                payload = json.dumps(reply).encode("utf-8")
                conn.sendall(len(payload).to_bytes(4, "big") + payload)
        server.close()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    return server.getsockname()[1], thread


def test_send_returns_what_blender_applied():
    """Regression test: the client used to close right after sending, so the app
    said "Pose sent to Blender" even when the add-on applied nothing."""
    received = {}
    reply = {"ok": True, "applied": ["spine"], "requested": 1, "unmatched": [], "armature": "Armature"}
    port, thread = _serve_once(reply, received)

    result = live_link.send_pose_document({"format": "pose_craft_pose", "bones": {}}, port=port, timeout=2.0)

    thread.join(timeout=2.0)
    assert result == reply
    assert received["document"]["live_link_protocol"] == live_link.POSE_REQUEST_PROTOCOL


def test_send_to_an_addon_without_replies_returns_none():
    port, thread = _serve_once(None, {})
    assert live_link.send_pose_document({"format": "pose_craft_pose"}, port=port, timeout=2.0) is None
    thread.join(timeout=2.0)


def test_malformed_reply_raises_value_error():
    port, thread = _serve_once(None, {}, raw_reply=(5).to_bytes(4, "big") + b"nope!")
    with pytest.raises(ValueError):
        live_link.send_pose_document({"format": "pose_craft_pose"}, port=port, timeout=2.0)
    thread.join(timeout=2.0)
