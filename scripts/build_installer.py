"""Compiles the per-user Windows installer (installer/PoseCraft.iss) for the built app.

    pyinstaller PoseCraft.spec                       # first: dist/PoseCraft/
    python scripts/build_installer.py [--flavor dml|cpu|cuda] [--sign]

Writes dist/installer/PoseCraft-<version>-<flavor>-setup.exe with Inno Setup 6 (ISCC.exe on
PATH or in its default install folders). The flavour only names the file (default: the
POSECRAFT_FLAVOR the build used, else dml). --sign first Authenticode-signs PoseCraft.exe
(scripts/sign_build.py, certificate thumbprint in CODESIGN_THUMBPRINT) and has Inno Setup sign
the installer and its uninstaller with the same certificate. Stdlib only.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sign_build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ISS = ROOT / "installer" / "PoseCraft.iss"
FLAVORS = ("dml", "cpu", "cuda")


def find_iscc() -> str | None:
    """Inno Setup 6's compiler: on PATH, or in its default install folders."""
    found = shutil.which("iscc")
    if found:
        return found
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        if not base:
            continue
        for folder in ("Inno Setup 6", os.path.join("Programs", "Inno Setup 6")):
            candidate = Path(base) / folder / "ISCC.exe"
            if candidate.is_file():
                return str(candidate)
    return None


def app_version() -> str:
    """pose_craft.__version__, read without importing the package (it pulls in Qt and NumPy)."""
    text = (ROOT / "pose_craft" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.M)
    if not match:
        raise RuntimeError("pose_craft/__init__.py has no __version__")
    return match.group(1)


def inno_sign_args(signtool: str, thumbprint: str, timestamp_url: str) -> list[str]:
    """ISCC arguments that make Inno Setup sign the installer and its uninstaller
    (PoseCraft.iss: SignTool=codesign under #ifdef Sign; $q = quote, $f = file)."""
    return ["/DSign", f"/Scodesign=$q{signtool}$q sign /sha1 {thumbprint} /fd SHA256 /tr {timestamp_url} /td SHA256 $f"]


def iscc_command(iscc: str, version: str, flavor: str, sign_args: list[str]) -> list[str]:
    return [iscc, f"/DAppVersion={version}", f"/DFlavor={flavor}", *sign_args, str(ISS)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--flavor", choices=FLAVORS, default=os.environ.get("POSECRAFT_FLAVOR") or "dml")
    parser.add_argument("--sign", action="store_true")
    args = parser.parse_args(argv)
    if not (sign_build.DEFAULT_EXE).is_file():
        print(f"{sign_build.DEFAULT_EXE} does not exist: run `pyinstaller PoseCraft.spec` first.")
        return 1
    iscc = find_iscc()
    if iscc is None:
        print("Inno Setup 6 (ISCC.exe) was not found. Install it from https://jrsoftware.org/isinfo.php")
        return 1
    sign_args: list[str] = []
    if args.sign:
        if sign_build.main(["--exe", str(sign_build.DEFAULT_EXE)]) != 0:
            return 1
        signtool = sign_build.find_signtool()
        thumbprint = os.environ.get("CODESIGN_THUMBPRINT", "").replace(" ", "")
        timestamp = os.environ.get("CODESIGN_TIMESTAMP_URL") or sign_build.DEFAULT_TIMESTAMP_URL
        sign_args = inno_sign_args(signtool, thumbprint, timestamp)
    version = app_version()
    result = subprocess.run(iscc_command(iscc, version, args.flavor, sign_args), cwd=ROOT)
    if result.returncode == 0:
        print(f"Installer: dist/installer/PoseCraft-{version}-{args.flavor}-setup.exe")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
