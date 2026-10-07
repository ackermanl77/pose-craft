"""scripts/sign_build.py: Authenticode signing of the built exe."""
import importlib.util
import os
from pathlib import Path
from unittest import mock

_SPEC = importlib.util.spec_from_file_location("sign_build", Path(__file__).resolve().parents[1] / "scripts" / "sign_build.py")
sign_build = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sign_build)


def test_signtool_command_signs_sha256_with_an_rfc3161_timestamp():
    cmd = sign_build.sign_command("signtool.exe", "ABC", "app.exe", "http://ts.example")
    assert cmd == ["signtool.exe", "sign", "/sha1", "ABC", "/fd", "SHA256", "/tr", "http://ts.example", "/td", "SHA256", "app.exe"]


def test_without_a_thumbprint_nothing_is_signed(tmp_path):
    with mock.patch.dict(os.environ, {"CODESIGN_THUMBPRINT": ""}), mock.patch.object(sign_build.subprocess, "run") as run:
        assert sign_build.main(["--exe", str(tmp_path / "app.exe")]) == 1
    run.assert_not_called()


def test_signs_the_exe_with_the_thumbprint_and_default_timestamp(tmp_path):
    exe = tmp_path / "app.exe"
    exe.write_bytes(b"MZ")
    with mock.patch.dict(os.environ, {"CODESIGN_THUMBPRINT": "AB CD", "CODESIGN_TIMESTAMP_URL": ""}), \
            mock.patch.object(sign_build, "find_signtool", return_value="signtool.exe"), \
            mock.patch.object(sign_build.subprocess, "run", return_value=mock.Mock(returncode=0)) as run:
        assert sign_build.main(["--exe", str(exe)]) == 0
    cmd = run.call_args.args[0]
    assert cmd[3] == "ABCD" and sign_build.DEFAULT_TIMESTAMP_URL in cmd and cmd[-1] == str(exe)


def test_a_missing_build_fails_clearly(tmp_path):
    with mock.patch.dict(os.environ, {"CODESIGN_THUMBPRINT": "AB"}), \
            mock.patch.object(sign_build, "find_signtool", return_value="signtool.exe"), \
            mock.patch.object(sign_build.subprocess, "run") as run:
        assert sign_build.main(["--exe", str(tmp_path / "missing.exe")]) == 1
    run.assert_not_called()
