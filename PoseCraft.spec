# -*- mode: python ; coding: utf-8 -*-
import glob
import importlib.util
import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [('assets/icon.png', 'assets'), ('assets/icon.ico', 'assets')]
binaries = []
hiddenimports = []
# onnxruntime: pose inference (its capi/ folder holds the CPU + CUDA provider DLLs).
# onnx + tflite: the one-time .task -> ONNX model conversion (estimation/tflite_to_onnx.py).
for package in ('onnxruntime', 'onnx', 'cv2', 'pyqtgraph'):
    tmp_ret = collect_all(package)
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
hiddenimports += collect_submodules('tflite')

# NVIDIA CUDA 13 runtime + cuDNN 9 from the pip packages requirements-gpu.txt installs
# (onnxruntime-gpu[cuda,cudnn]). ONNX Runtime's CUDA provider needs them but doesn't
# import them, so PyInstaller can't discover them all on its own. They keep their
# site-packages layout (nvidia/cu13/bin/x86_64/, nvidia/cudnn/bin/) under _internal/,
# which is exactly where onnxruntime.preload_dlls() looks relative to its own package --
# so the frozen app finds them the same way a dev venv does, and PyInstaller's own
# dependency scan (which copies some of them there too) doesn't create duplicates.
# Skipped (CPU-only build) when they aren't installed. cufftw/nvblas are CPU-API shims
# nothing here uses.
_nvidia_spec = importlib.util.find_spec('nvidia')
if _nvidia_spec is not None and _nvidia_spec.submodule_search_locations:
    _nvidia_root = list(_nvidia_spec.submodule_search_locations)[0]
    _site_packages = os.path.dirname(_nvidia_root)
    for _dll in glob.glob(os.path.join(_nvidia_root, '**', 'bin', '**', '*.dll'), recursive=True):
        if not os.path.basename(_dll).startswith(('cufftw', 'nvblas')):
            binaries.append((_dll, os.path.relpath(os.path.dirname(_dll), _site_packages)))


a = Analysis(
    ['pose_craft/app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # mediapipe was the previous inference backend; it may still be installed in an
    # older dev venv, but nothing imports it anymore -- keep it out of the bundle.
    excludes=['mediapipe'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PoseCraft',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX off: compressing the multi-hundred-MB CUDA/cuDNN DLLs takes ages for little
    # gain, and UPX-packed CUDA libraries are a known source of load failures.
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets/icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='PoseCraft',
)
