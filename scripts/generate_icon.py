"""Generates assets/icon.png and assets/icon.ico: a simple stick-figure skeleton
in a dynamic pose, using the same bone/joint colors as the 3D viewport
(gui/viewport_3d.py's BONE_COLOR/JOINT_COLOR), on a dark rounded-square
background. Run with `python scripts/generate_icon.py` whenever the design
needs to change -- the generated files are checked in, this script isn't run
at build time.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 256
BACKGROUND = (30, 32, 41, 255)  # dark slate, matches the viewport's #222-ish background
BONE_COLOR = (77, 204, 255, 255)  # from viewport_3d.BONE_COLOR (0.3, 0.8, 1.0)
JOINT_COLOR = (255, 153, 26, 255)  # from viewport_3d.JOINT_COLOR (1.0, 0.6, 0.1)
BONE_WIDTH = 10
JOINT_RADIUS = 11

# A simple dynamic pose in a 0..1 unit box (x right, y down), hand-placed rather than
# solved -- this is decorative art, not a real skeleton export.
JOINTS = {
    "head": (0.5, 0.16),
    "neck": (0.5, 0.28),
    "hip": (0.5, 0.58),
    "shoulder_l": (0.34, 0.32),
    "shoulder_r": (0.66, 0.32),
    "elbow_l": (0.20, 0.46),
    "elbow_r": (0.78, 0.22),
    "hand_l": (0.14, 0.62),
    "hand_r": (0.88, 0.10),
    "knee_l": (0.38, 0.80),
    "knee_r": (0.64, 0.78),
    "foot_l": (0.30, 0.98),
    "foot_r": (0.74, 0.96),
}

BONES = [
    ("neck", "hip"),
    ("neck", "shoulder_l"),
    ("neck", "shoulder_r"),
    ("shoulder_l", "elbow_l"),
    ("elbow_l", "hand_l"),
    ("shoulder_r", "elbow_r"),
    ("elbow_r", "hand_r"),
    ("hip", "knee_l"),
    ("knee_l", "foot_l"),
    ("hip", "knee_r"),
    ("knee_r", "foot_r"),
]

HEAD_RADIUS = 0.11  # unit-box radius, drawn separately from the joint dots


def _pt(name: str, margin: float, usable: float) -> tuple[float, float]:
    x, y = JOINTS[name]
    return margin + x * usable, margin + y * usable


def build_icon() -> Image.Image:
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    corner_radius = SIZE * 0.18
    draw.rounded_rectangle([0, 0, SIZE - 1, SIZE - 1], radius=corner_radius, fill=BACKGROUND)

    margin = SIZE * 0.12
    usable = SIZE - 2 * margin

    for a, b in BONES:
        draw.line([_pt(a, margin, usable), _pt(b, margin, usable)], fill=BONE_COLOR, width=BONE_WIDTH)

    head_x, head_y = _pt("head", margin, usable)
    r = HEAD_RADIUS * usable
    draw.ellipse([head_x - r, head_y - r, head_x + r, head_y + r], fill=BONE_COLOR)

    for name in JOINTS:
        if name == "head":
            continue
        x, y = _pt(name, margin, usable)
        draw.ellipse(
            [x - JOINT_RADIUS, y - JOINT_RADIUS, x + JOINT_RADIUS, y + JOINT_RADIUS],
            fill=JOINT_COLOR,
        )

    return image


def main() -> None:
    assets_dir = Path(__file__).resolve().parent.parent / "assets"
    assets_dir.mkdir(exist_ok=True)

    icon = build_icon()
    icon.save(assets_dir / "icon.png")
    icon.save(assets_dir / "icon.ico", sizes=[(16, 16), (32, 32), (48, 48), (128, 128), (256, 256)])
    print(f"Wrote {assets_dir / 'icon.png'} and {assets_dir / 'icon.ico'}")


if __name__ == "__main__":
    main()
