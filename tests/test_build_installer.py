"""scripts/build_installer.py and installer/PoseCraft.iss (K5)."""
import importlib.util
import os
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("build_installer", ROOT / "scripts" / "build_installer.py")
build_installer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_installer)


def test_version_comes_from_the_package():
    from pose_craft import __version__

    assert build_installer.app_version() == __version__


def test_iscc_command_and_signing_arguments():
    cmd = build_installer.iscc_command("ISCC.exe", "1.2.3", "cpu", ["/DSign", "/Sx"])
    assert cmd[:3] == ["ISCC.exe", "/DAppVersion=1.2.3", "/DFlavor=cpu"] and cmd[3:5] == ["/DSign", "/Sx"]
    assert cmd[-1].endswith("PoseCraft.iss")
    sign = build_installer.inno_sign_args(r"C:\Kits\signtool.exe", "ABC", "http://ts")
    assert sign == ["/DSign", r"/Scodesign=$qC:\Kits\signtool.exe$q sign /sha1 ABC /fd SHA256 /tr http://ts /td SHA256 $f"]


def test_the_installer_script_is_per_user_and_signs_when_asked():
    iss = (ROOT / "installer" / "PoseCraft.iss").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "PrivilegesRequired=lowest" in iss and r"{localappdata}\Programs" in iss
    assert "#ifdef Sign\nSignTool=codesign\nSignedUninstaller=yes\n#endif" in iss
    assert "OutputBaseFilename=PoseCraft-{#AppVersion}-{#Flavor}-setup" in iss


def test_a_missing_build_stops_before_iscc(tmp_path):
    with mock.patch.object(build_installer.sign_build, "DEFAULT_EXE", tmp_path / "missing.exe"), \
            mock.patch.object(build_installer.subprocess, "run") as run:
        assert build_installer.main([]) == 1
    run.assert_not_called()


def test_a_signed_build_signs_the_exe_then_passes_the_tool_to_iscc(tmp_path):
    exe = tmp_path / "PoseCraft.exe"
    exe.write_bytes(b"MZ")
    with mock.patch.object(build_installer.sign_build, "DEFAULT_EXE", exe), \
            mock.patch.object(build_installer, "find_iscc", return_value="ISCC.exe"), \
            mock.patch.object(build_installer.sign_build, "find_signtool", return_value="signtool.exe"), \
            mock.patch.dict(os.environ, {"CODESIGN_THUMBPRINT": "AB", "CODESIGN_TIMESTAMP_URL": ""}), \
            mock.patch("subprocess.run", return_value=mock.Mock(returncode=0)) as run:
        assert build_installer.main(["--sign", "--flavor", "dml"]) == 0
    signtool_call, iscc_call = (call.args[0] for call in run.call_args_list)
    assert signtool_call[:4] == ["signtool.exe", "sign", "/sha1", "AB"] and signtool_call[-1] == str(exe)
    assert iscc_call[0] == "ISCC.exe" and "/DSign" in iscc_call and "/DFlavor=dml" in iscc_call
