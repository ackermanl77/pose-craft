"""Authenticode-signs the built app (after `pyinstaller PoseCraft.spec`).

    set CODESIGN_THUMBPRINT=<SHA-1 thumbprint of a code-signing certificate>
    python scripts/sign_build.py [--exe dist/PoseCraft/PoseCraft.exe]

Uses signtool (Windows SDK; PATH, else the newest Windows Kits 10 one) and the certificate in
the Windows certificate store with that thumbprint, timestamped (RFC 3161) by
CODESIGN_TIMESTAMP_URL (default DigiCert's) so the signature outlives the certificate.
Stdlib only.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "dist" / "PoseCraft" / "PoseCraft.exe"
DEFAULT_TIMESTAMP_URL = "http://timestamp.digicert.com"


def find_signtool() -> str | None:
    """signtool.exe: on PATH, or the newest one in the Windows 10/11 SDK."""
    found = shutil.which("signtool")
    if found:
        return found
    kits = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Windows Kits" / "10" / "bin"
    if not kits.is_dir():
        return None
    for version in sorted(kits.iterdir(), reverse=True):
        candidate = version / "x64" / "signtool.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def sign_command(signtool: str, thumbprint: str, path: str, timestamp_url: str) -> list[str]:
    return [signtool, "sign", "/sha1", thumbprint, "/fd", "SHA256", "/tr", timestamp_url, "/td", "SHA256", path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    args = parser.parse_args(argv)
    thumbprint = os.environ.get("CODESIGN_THUMBPRINT", "").replace(" ", "")
    if not thumbprint:
        print("CODESIGN_THUMBPRINT is not set: the SHA-1 thumbprint of a code-signing certificate.")
        return 1
    signtool = find_signtool()
    if signtool is None:
        print("signtool.exe was not found (install the Windows SDK's signing tools).")
        return 1
    if not args.exe.is_file():
        print(f"{args.exe} does not exist: run `pyinstaller PoseCraft.spec` first.")
        return 1
    timestamp = os.environ.get("CODESIGN_TIMESTAMP_URL") or DEFAULT_TIMESTAMP_URL
    print("Signing:", args.exe)
    return subprocess.run(sign_command(signtool, thumbprint, str(args.exe), timestamp), cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
