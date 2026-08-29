"""
paths.py  —  Centralised path resolution for CEST-MRF GUI
===========================================================
Handles two execution contexts transparently:

  Development   CWD is the project root; OUTPUT_FILES/ lives beside my_gui/
  Bundled DMG   App runs from inside a read-only .app bundle, so relative
                paths like "OUTPUT_FILES/dict.mat" would resolve to either
                the read-only bundle or "/" — both fail with PermissionError.

In bundled mode all writable output goes to:
    ~/Documents/OCEAN_Output/

…which users can find immediately in Finder, and which is always writable.

Read-only resources (T1cm.mat, T2cm.mat, differenceMaps.mat) are found via
sys._MEIPASS when frozen, or relative to the project root otherwise.

Usage
-----
    from my_gui.paths import output_path, get_output_dir, get_resource_dir

    out = output_path("dict.mat")          # absolute path, dir guaranteed to exist
    d   = get_output_dir()                 # Path object to writable output folder
    r   = get_resource_dir()               # Path object to read-only resources
"""
from __future__ import annotations

import sys
import os
from pathlib import Path


# ── Bundle detection ───────────────────────────────────────────────────────────

def _is_bundled() -> bool:
    """Return True when running inside a PyInstaller .app bundle."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


# ── Resource directory (read-only: .mat colourmaps, etc.) ─────────────────────

def get_resource_dir() -> Path:
    """
    Directory containing bundled read-only assets.

    • Bundled:     sys._MEIPASS   (PyInstaller extraction root)
    • Development: project root   (two levels above my_gui/paths.py)
    """
    if _is_bundled():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    # Development: my_gui/paths.py → my_gui/ → ocean/
    return Path(__file__).resolve().parent.parent


# ── Output directory (writable) ───────────────────────────────────────────────

def get_output_dir() -> Path:
    """
    Return the writable output directory, creating it if necessary.

    • Bundled:     ~/Documents/OCEAN_Output/
    • Development: <project_root>/OUTPUT_FILES/

    Always creates the directory; safe to call multiple times.
    """
    if _is_bundled():
        d = Path.home() / "Documents" / "OCEAN_Output"
    else:
        d = Path(__file__).resolve().parent.parent / "OUTPUT_FILES"
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_path(filename: str) -> str:
    """
    Return an absolute path string for *filename* inside the output directory.
    The output directory is created on first call.

    Example::

        yaml_fn = output_path("scenario.yaml")
        # → "/Users/alice/Documents/OCEAN_Output/scenario.yaml"  (bundled)
        # → "/Users/alice/dev/ocean/OUTPUT_FILES/scenario.yaml"  (dev)
    """
    return str(get_output_dir() / filename)


# ── Convenience: set CWD for legacy relative-path code ────────────────────────

def ensure_cwd() -> None:
    """
    Change the process CWD to the output directory.

    Called once from main.py so that any legacy code that still uses bare
    relative paths (e.g. open("mask.npy")) keeps working without changes.
    In bundled mode this also prevents the app from starting with CWD = "/"
    or the read-only .app bundle directory.
    """
    target = get_output_dir()
    try:
        os.chdir(target)
    except OSError:
        pass  # non-fatal; absolute paths via output_path() are the real fix
