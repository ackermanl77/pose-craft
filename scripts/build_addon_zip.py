"""Builds the Blender add-on zip: posecraft_import_pose/__init__.py (the add-on) plus
blender_manifest.toml, so the one zip installs as an extension on Blender 4.2+ ("Install
from Disk") and as a legacy add-on on 3.6-4.1 ("Install..."). Prints the zip's path.

    python scripts/build_addon_zip.py [--out DIR]
"""
from __future__ import annotations

import argparse
import ast
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADDON_DIR = ROOT / "pose_craft" / "export" / "blender_addon"
PACKAGE = "posecraft_import_pose"


def addon_version() -> str:
    """bl_info["version"] of the add-on, as "0.7.1"."""
    tree = ast.parse((ADDON_DIR / "import_pose.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "bl_info":
            return ".".join(str(part) for part in ast.literal_eval(node.value)["version"])
    raise ValueError("import_pose.py has no bl_info")


def build(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"posecraft-blender-addon-{addon_version()}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(ADDON_DIR / "import_pose.py", f"{PACKAGE}/__init__.py")
        archive.write(ADDON_DIR / "blender_manifest.toml", f"{PACKAGE}/blender_manifest.toml")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    print(build(parser.parse_args().out))


if __name__ == "__main__":
    main()
