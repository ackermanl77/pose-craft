"""The exe's version resource, manifest and icon sizes (report item K7), from scripts/windows_resources.py."""

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import pose_craft

ROOT = Path(__file__).resolve().parent.parent


def _resources():
    spec = importlib.util.spec_from_file_location("windows_resources", ROOT / "scripts" / "windows_resources.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_version_comes_from_the_package_and_names_the_one_publisher(tmp_path):
    res = _resources()
    assert res.app_version() == pose_craft.__version__
    text = Path(res.write_version_file(str(tmp_path), "1.2.3")).read_text(encoding="utf-8")
    assert 'StringStruct(u"CompanyName", u"MELEJ")' in text and "filevers=(1, 2, 3, 0)" in text
    installer = (ROOT / "installer" / "PoseCraft.iss").read_text(encoding="utf-8")
    assert f"AppPublisher={res.PUBLISHER}" in installer


def test_the_version_file_parses_in_pyinstaller(tmp_path):
    versioninfo = pytest.importorskip("PyInstaller.utils.win32.versioninfo")
    path = _resources().write_version_file(str(tmp_path), "0.4.1")
    assert versioninfo.load_version_info_from_text_file(path).ffi.fileVersionLS == 1 << 16


def test_the_manifest_sets_a_utf8_code_page_and_per_monitor_dpi(tmp_path):
    root = ET.parse(_resources().write_manifest(str(tmp_path))).getroot()
    values = {el.tag.split("}")[1]: el.text for el in root.iter() if el.text and el.text.strip()}
    assert values["activeCodePage"] == "UTF-8" and values["dpiAwareness"] == "PerMonitorV2"


def test_the_app_icon_has_every_explorer_size():
    res = _resources()
    assert sorted(res.read_ico((ROOT / "assets" / "icon.ico").read_bytes())) == list(res.ICON_SIZES)


def test_ico_round_trip_keeps_each_image():
    res = _resources()
    images = {16: (32, b"a" * 10), 256: (32, b"b" * 20)}
    assert res.read_ico(res.ico_bytes(images)) == images


def test_the_spec_embeds_the_version_and_manifest():
    spec = (ROOT / "PoseCraft.spec").read_text(encoding="utf-8")
    assert "version=VERSION_PATH" in spec and "manifest=MANIFEST_PATH" in spec
