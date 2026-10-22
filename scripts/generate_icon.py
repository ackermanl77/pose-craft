"""Generates assets/icon.png and assets/icon.ico from assets/icon.svg: the stick figure in a
dynamic pose, in the 3D viewport's bone/joint colours (gui/viewport_3d.py's BONE_COLOR /
JOINT_COLOR), with a small X/Y/Z axis gizmo, on a dark rounded-square tile -- the shared style of
the author's app icons. Edit the SVG, then run `python scripts/generate_icon.py`; the generated
files are checked in, this script isn't run at build time.

Each .ico size is rendered from the SVG (not scaled down from 256 px), so 16 and 24 px stay crisp.
Uses Qt's SVG renderer (PySide6 is already a dependency).
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def render(svg: Path, size: int):
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    renderer = QSvgRenderer(str(svg))
    if not renderer.isValid():
        raise SystemExit(f"invalid SVG: {svg}")
    image = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return image


def png_bytes(image) -> bytes:
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice

    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(data)


def ico_bytes(entries: list[tuple[int, bytes]]) -> bytes:
    """An .ico container with one PNG image per size (Windows Vista+)."""
    header = struct.pack("<HHH", 0, 1, len(entries))
    offset = 6 + 16 * len(entries)
    directory, blobs = b"", b""
    for size, blob in entries:
        side = 0 if size >= 256 else size
        directory += struct.pack("<BBBBHHII", side, side, 0, 0, 1, 32, len(blob), offset + len(blobs))
        blobs += blob
    return header + directory + blobs


def main() -> None:
    from PySide6.QtGui import QGuiApplication

    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])  # noqa: F841 (Qt needs one)
    svg = ASSETS / "icon.svg"
    render(svg, 256).save(str(ASSETS / "icon.png"))
    (ASSETS / "icon.ico").write_bytes(ico_bytes([(size, png_bytes(render(svg, size))) for size in ICO_SIZES]))
    print(f"Wrote {ASSETS / 'icon.png'} and {ASSETS / 'icon.ico'}")


if __name__ == "__main__":
    main()
