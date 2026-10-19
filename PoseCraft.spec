# -*- mode: python ; coding: utf-8 -*-
import glob
import importlib.util
import os

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

datas = [('assets/icon.png', 'assets'), ('assets/icon.ico', 'assets')]
# --self-test (pose_craft/selftest.py) solves this CC0 photo and compares it with its golden output.
datas += [('tests/data/golden/yoga_tree.jpg', 'selftest'), ('tests/data/golden/yoga_tree.golden.json', 'selftest')]
binaries = []
hiddenimports = []
excludes = [
    # mediapipe was the previous inference backend; it may still be installed in an
    # older dev venv, but nothing imports it anymore.
    'mediapipe',
    # Pulled in by pyqtgraph's optional extras; nothing here uses them (~28 MB).
    'matplotlib', 'PIL',
    # Qt modules nothing imports (pyqtgraph imports QtSvg/QtTest optionally and
    # falls back cleanly without them).
    'PySide6.QtQuick', 'PySide6.QtQml', 'PySide6.QtNetwork', 'PySide6.QtTest', 'PySide6.QtPdf',
    'PySide6.QtVirtualKeyboard',
]
# onnxruntime: pose inference (its capi/ folder holds the CPU + CUDA provider DLLs).
for package in ('onnxruntime', 'cv2'):
    tmp_ret = collect_all(package)
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
# pyqtgraph: its modules and data files, but not its bundled examples.
hiddenimports += collect_submodules('pyqtgraph', filter=lambda name: '.examples' not in name)
datas += collect_data_files('pyqtgraph', excludes=['**/examples/**'])

# Pre-converted models (K3): scripts/convert_models.py writes them, with a
# manifest, to build/models (or $POSECRAFT_MODELS_DIR). When they're there they're
# bundled under models/ (estimation/landmarker.bundled_onnx_paths finds them) and
# the onnx/tflite converter is left out: the app needs no download or network on
# first run. Without them, the build keeps the converter and converts on first use.
_models_dir = os.environ.get('POSECRAFT_MODELS_DIR', os.path.join('build', 'models'))
if os.path.isfile(os.path.join(_models_dir, 'manifest.json')):
    datas.append((_models_dir, 'models'))
    excludes += ['onnx', 'tflite']
else:
    tmp_ret = collect_all('onnx')
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
# nothing here uses; curand64_10 (~60 MB) isn't imported (statically or delay-loaded)
# by onnxruntime's providers, cuDNN or cuBLAS -- checked with pefile -- and
# onnxruntime.preload_dlls() only preloads curand on Linux.
#
# Also left out (~520 MB): cufft64 (222 MB), nvrtc/nvrtc-builtins (102 MB+), nvJitLink
# (93 MB) and cudnn_adv (101 MB). pefile shows no bundled DLL imports or delay-imports
# them (they're only named as strings for dynamic loading), and with all five removed
# from a dev venv, onnxruntime.preload_dlls(), session creation and inference still ran
# on the CUDA provider for lite/full/heavy with identical results, TF32 on and off,
# HEURISTIC and EXHAUSTIVE algorithm search. Re-check (a CUDA solve per variant) whenever
# onnxruntime, cuDNN or the models change.
_NVIDIA_EXCLUDED_PREFIXES = ('cufftw', 'nvblas', 'curand', 'cufft64', 'nvrtc', 'nvJitLink', 'cudnn_adv')

# Build flavour, from the POSECRAFT_FLAVOR environment variable:
#   dml (default) -- DirectML: build from a venv with requirements-dml.txt (onnxruntime-directml);
#           never bundles the NVIDIA DLLs (508 MB instead of 1.73 GB). The app uses the GPU only
#           when it finds a hardware DirectX 12 GPU whose results match the CPU's at load time
#           (onnx_pose.best_directml_adapter / gpu_session_problem), otherwise the CPU.
#   cuda -- CUDA: bundles the NVIDIA DLLs from requirements-gpu.txt.
#   cpu  -- never bundles the NVIDIA DLLs.
_flavor = os.environ.get('POSECRAFT_FLAVOR', '').strip().lower() or 'dml'
if _flavor not in ('dml', 'cuda', 'cpu'):
    raise SystemExit(f"POSECRAFT_FLAVOR must be dml, cuda or cpu, not {_flavor!r}")
if _flavor == 'dml':
    import onnxruntime as _ort

    if 'DmlExecutionProvider' not in _ort.get_available_providers():
        raise SystemExit(
            "The default DirectML build needs onnxruntime-directml (requirements-dml.txt) in the build "
            "venv. Install it, or set POSECRAFT_FLAVOR=cuda / cpu for another flavour."
        )
_nvidia_spec = importlib.util.find_spec('nvidia') if _flavor == 'cuda' else None
if _flavor == 'cuda' and (_nvidia_spec is None or not _nvidia_spec.submodule_search_locations):
    raise SystemExit("POSECRAFT_FLAVOR=cuda but the nvidia-* packages aren't installed (requirements-gpu.txt)")
if _nvidia_spec is not None and _nvidia_spec.submodule_search_locations:
    _nvidia_root = list(_nvidia_spec.submodule_search_locations)[0]
    _site_packages = os.path.dirname(_nvidia_root)
    for _dll in glob.glob(os.path.join(_nvidia_root, '**', 'bin', '**', '*.dll'), recursive=True):
        if not os.path.basename(_dll).startswith(_NVIDIA_EXCLUDED_PREFIXES):
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
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
# The Qt libraries (and plugins needing them) of the excluded Qt modules can still
# arrive as binaries; drop them by name.
_UNUSED_QT = ('Qt6Quick', 'Qt6Qml', 'Qt6Pdf', 'Qt6VirtualKeyboard', 'Qt6Network', 'Qt6Test', 'qtvirtualkeyboard')
a.binaries = [b for b in a.binaries if not os.path.basename(b[0]).startswith(_UNUSED_QT)]
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
