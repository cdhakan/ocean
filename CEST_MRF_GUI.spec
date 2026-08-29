import sys
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all

ROOT       = Path(SPECPATH)
CEST_LIB   = ROOT / "open-py-cest-mrf"
MAIN_ENTRY = ROOT / "my_gui" / "main.py"

sys.path.insert(0, str(CEST_LIB))

_pyqt6_hidden = [
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "PyQt6.QtPrintSupport",
    "PyQt6.sip",
]

_mpl_hidden = [
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_svg",
    "matplotlib.backends.backend_pdf",
    "matplotlib.figure",
    "matplotlib.font_manager",
]

_scipy_hidden = [
    "scipy.interpolate",
    "scipy.interpolate._interpolate",
    "scipy.optimize",
    "scipy.optimize._minpack",
    "scipy.optimize._minpack_py",
    "scipy.optimize._lsq",
    "scipy.optimize._minimize",
    "scipy.optimize._root",
    "scipy.optimize._zeros_py",
    "scipy.linalg",
    "scipy.linalg.cython_blas",
    "scipy.linalg.cython_lapack",
    "scipy.ndimage",
    "scipy.ndimage._ni_support",
    "scipy.special",
    "scipy.special._ufuncs",
    "scipy.integrate",
    "scipy.io",
    "scipy.io.matlab",
    "scipy.io.matlab._mio5_params",
    "scipy._lib.messagestream",
    "scipy._lib._util",
]

_io_hidden = [
    "h5py",
    "h5py._hl",
    "h5py.defs",
    "h5py.utils",
    "h5py._errors",
    "h5py.h5ac",
]

_app_hidden = (
    collect_submodules("my_gui") +
    collect_submodules("cest_mrf")
)

hidden_imports = (
    _pyqt6_hidden + _mpl_hidden + _scipy_hidden + _io_hidden + _app_hidden
)

datas = [
    (str(ROOT / "T1cm.mat"),          "."),
    (str(ROOT / "T2cm.mat"),          "."),
    (str(ROOT / "differenceMaps.mat"), "."),
    (str(ROOT / "cmp_files.mat"),     "."),

    *collect_data_files("matplotlib"),

    *collect_data_files("PyQt6"),
]

if CEST_LIB.exists():
    datas += collect_data_files("cest_mrf", subdir=str(CEST_LIB))

_PULSEQ_LIB_SRC = os.environ.get("OCEAN_PULSEQ_LIBRARY") or str(
    Path.home() / "Downloads" / "pulseq-cest-library-master" / "seq-library")
if Path(_PULSEQ_LIB_SRC).is_dir():
    for _seq in Path(_PULSEQ_LIB_SRC).glob("*/*.seq"):
        datas.append((str(_seq),
                      str(Path("pulseq-cest-library") / "seq-library" / _seq.parent.name)))

binaries = []
if not os.environ.get("OCEAN_NO_ITK"):
    try:
        _itk_datas, _itk_bins, _itk_hidden = collect_all("itk")
        datas += _itk_datas
        binaries += _itk_bins
        hidden_imports = list(hidden_imports) + list(_itk_hidden)
        print(f"[spec] bundling itk-elastix for motion correction "
              f"({len(_itk_bins)} binaries, {len(_itk_datas)} data files)")
    except Exception as _itk_err:
        print(f"[spec] itk NOT bundled (motion correction disabled at runtime): {_itk_err}")

_hook_dir = ROOT / "_pyinstaller_hooks"
_hook_dir.mkdir(exist_ok=True)
_hook_file = _hook_dir / "rthook_cwd.py"
_hook_file.write_text(
    "# PyInstaller runtime hook — set writable CWD before app starts\n"
    "import sys, os\n"
    "from pathlib import Path\n"
    "if getattr(sys, 'frozen', False):\n"
    "    _out = Path.home() / 'Documents' / 'OCEAN_Output'\n"
    "    _out.mkdir(parents=True, exist_ok=True)\n"
    "    os.chdir(_out)\n"
)

a = Analysis(
    [str(MAIN_ENTRY)],
    pathex=[str(ROOT), str(CEST_LIB)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(_hook_file)],
    excludes=[
        "IPython", "jupyter", "nbformat", "nbconvert",
        "tkinter", "_tkinter",
        "wx", "gi",
        "test", "tests",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OCEAN",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    icon="assets/AppIcon.icns",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="OCEAN",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="OCEAN.app",
        icon="assets/AppIcon.icns",
        bundle_identifier="com.cbdlab.ocean",
        info_plist={
            "CFBundleName":              "OCEAN",
            "CFBundleDisplayName":       "OCEAN",
            "CFBundleShortVersionString": "1.0.0",
            "CFBundleVersion":           "1.0.0",
            "NSHighResolutionCapable":   True,
            "NSHumanReadableCopyright":  "© 2025 CBD Lab",
            "LSMinimumSystemVersion":    "10.14",
        },
    )
