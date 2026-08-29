"""
colormaps_gui.py
Load and register custom MRF colormaps (T1cm, T2cm, differenceMaps)
from the .mat files that ship with ocean.

Based on colormaps_dk.py by DK.

Usage:
    from my_gui.colormaps_gui import get_cmap_list, get_cmap

    names = get_cmap_list()          # list of (name, display_label)
    cmap  = get_cmap("T1cm")         # matplotlib colormap object
"""

from __future__ import annotations
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Paths ──────────────────────────────────────────────────────────────────
# Use get_resource_dir() so bundled .app finds .mat files in sys._MEIPASS
# and development mode finds them in the project root.
from my_gui.paths import get_resource_dir as _get_resource_dir
_ROOT = _get_resource_dir()

_MAT_FILES = {
    "T1":   _ROOT / "T1cm.mat",
    "T2":   _ROOT / "T2cm.mat",
    "diff": _ROOT / "differenceMaps.mat",
}

# ── Internal registry ──────────────────────────────────────────────────────
# name → matplotlib colormap object
_REGISTRY: dict[str, mcolors.Colormap] = {}

# Ordered list of (internal_name, display_label) for the GUI dropdown
_ORDER: list[tuple[str, str]] = []

_LOADED = False


def _build_black_bg(colors: np.ndarray, name: str) -> mcolors.LinearSegmentedColormap:
    """Return a copy of a colormap with the first entry set to black."""
    cb = colors.copy()
    cb[0, :3] = 0.0
    return mcolors.LinearSegmentedColormap.from_list(name, cb)


def _load_colormaps():
    global _LOADED, _REGISTRY, _ORDER
    if _LOADED:
        return
    _LOADED = True

    import scipy.io as sio

    # ── Standard matplotlib colormaps with black-background variants ───────
    for std_name in ("viridis", "winter", "hot", "plasma", "gray"):
        try:
            base = plt.get_cmap(std_name)
            arr  = base(np.linspace(0, 1, base.N))
            cm_b = _build_black_bg(arr, f"b_{std_name}")
            _REGISTRY[f"b_{std_name}"] = cm_b
        except Exception:
            pass

    # ── T1 colormap ────────────────────────────────────────────────────────
    try:
        d = sio.loadmat(str(_MAT_FILES["T1"]))
        T1_colors = d["T1colormap"].astype(float)
        if T1_colors.max() > 1.0:
            T1_colors /= 255.0

        T1cm   = mcolors.LinearSegmentedColormap.from_list("T1cm",   T1_colors)
        T1cm_r = T1cm.reversed()
        b_T1cm = _build_black_bg(T1_colors, "b_T1cm")

        for cm in (T1cm, T1cm_r, b_T1cm):
            _REGISTRY[cm.name] = cm

        _ORDER += [
            ("T1cm",   "T1cm  (MRF T1)"),
            ("T1cm_r", "T1cm reversed"),
            ("b_T1cm", "b_T1cm (black bg)"),
        ]
    except Exception as e:
        warnings.warn(f"colormaps_gui: could not load T1cm.mat — {e}")

    # ── T2 colormap ────────────────────────────────────────────────────────
    try:
        d = sio.loadmat(str(_MAT_FILES["T2"]))
        T2_colors = d["T2colormap"].astype(float)
        if T2_colors.max() > 1.0:
            T2_colors /= 255.0

        T2cm   = mcolors.LinearSegmentedColormap.from_list("T2cm",   T2_colors)
        T2cm_r = T2cm.reversed()
        b_T2cm = _build_black_bg(T2_colors, "b_T2cm")

        for cm in (T2cm, T2cm_r, b_T2cm):
            _REGISTRY[cm.name] = cm

        _ORDER += [
            ("T2cm",   "T2cm  (MRF T2)"),
            ("T2cm_r", "T2cm reversed"),
            ("b_T2cm", "b_T2cm (black bg)"),
        ]
    except Exception as e:
        warnings.warn(f"colormaps_gui: could not load T2cm.mat — {e}")

    # ── Difference colormaps ───────────────────────────────────────────────
    try:
        d = sio.loadmat(str(_MAT_FILES["diff"]))
        cm_colors  = d["cm"].astype(float)
        cm1_colors = d["cm1"].astype(float)
        for arr in (cm_colors, cm1_colors):
            if arr.max() > 1.0:
                arr /= 255.0

        cm_map   = mcolors.LinearSegmentedColormap.from_list("diffcm",   cm_colors)
        cm1_map  = mcolors.LinearSegmentedColormap.from_list("diffcm1",  cm1_colors)
        b_cm_map  = _build_black_bg(cm_colors,  "b_diffcm")
        b_cm1_map = _build_black_bg(cm1_colors, "b_diffcm1")

        for cm in (cm_map, cm1_map, b_cm_map, b_cm1_map):
            _REGISTRY[cm.name] = cm

        _ORDER += [
            ("diffcm",    "diffcm  (difference)"),
            ("diffcm1",   "diffcm1 (difference alt)"),
            ("b_diffcm",  "b_diffcm  (black bg)"),
            ("b_diffcm1", "b_diffcm1 (black bg)"),
        ]
    except Exception as e:
        warnings.warn(f"colormaps_gui: could not load differenceMaps.mat — {e}")

    # ── black-background variants of standard maps (always available) ──────
    _ORDER += [
        ("b_viridis", "b_viridis (black bg)"),
        ("b_winter",  "b_winter  (black bg)"),
        ("b_hot",     "b_hot     (black bg)"),
        ("b_plasma",  "b_plasma  (black bg)"),
        ("b_gray",    "b_gray    (black bg)"),
    ]

    # ── Scientific colormap library (cmp_files.mat) ────────────────────────
    # Crameri perceptually-uniform maps (batlow, vik, roma, …) + parula/CMRmap
    # /B0/difference maps. Each is an N×3 RGB table in [0,1].
    try:
        d = sio.loadmat(str(_ROOT / "cmp_files.mat"))
        builtins = set(plt.colormaps())
        extra: list[tuple[str, str]] = []
        for key in sorted((k for k in d if not k.startswith("__")), key=str.lower):
            arr = np.asarray(d[key], dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] < 2:
                continue
            cols = arr[:, :3]
            if cols.max() > 1.0:
                cols = cols / 255.0
            cols = np.clip(cols, 0.0, 1.0)
            name = str(key)
            # Don't shadow matplotlib built-ins (viridis/magma/…) or our T1cm/T2cm
            if name in _REGISTRY or name in builtins:
                continue
            cm = mcolors.LinearSegmentedColormap.from_list(name, cols)
            _REGISTRY[name] = cm
            extra.append((name, name))
        _ORDER += extra
    except Exception as e:
        warnings.warn(f"colormaps_gui: could not load cmp_files.mat — {e}")

    # Register everything with matplotlib so imshow can use them by name
    for name, cm in _REGISTRY.items():
        try:
            plt.colormaps.register(cm, name=name, force=True)
        except Exception:
            pass


# ── Public API ─────────────────────────────────────────────────────────────

def get_cmap_list() -> list[tuple[str, str]]:
    """
    Return list of (internal_name, display_label) for all custom colormaps.
    Only includes entries whose colormap was successfully loaded.
    """
    _load_colormaps()
    return [(n, lbl) for n, lbl in _ORDER if n in _REGISTRY]


def get_cmap(name: str) -> mcolors.Colormap:
    """Return colormap object by internal name."""
    _load_colormaps()
    if name in _REGISTRY:
        return _REGISTRY[name]
    return plt.get_cmap(name)   # fallback to matplotlib built-ins


# ── Fuderer perceptual "log-like" colormap remap ────────────────────────────
# Port of colorLogRemap() from OpenMRF's get_cmp.m
#   Fuderer M, et al. Color-map recommendation for MR relaxometry maps.
#   Magn Reson Med. 2025 Feb;93(2):490-506.  doi: 10.1002/mrm.30290
# The data and the colourbar ticks stay LINEAR; only the *allocation* of colours
# is warped along a log-like curve (linear below upLev/e, logarithmic above), so
# equal colour steps correspond to roughly equal PERCENT change in the value.

def color_log_remap(colors, loLev: float, upLev: float):
    """Warp an ``N×3`` RGB table (values in [0, 1]) for the window
    ``[loLev, upLev]``.  Returns a new ``N×3`` table.  If the window is invalid
    (``upLev <= 0`` or ``upLev <= loLev``) the input is returned unchanged."""
    ori = np.asarray(colors, dtype=float)
    N = ori.shape[0]
    if N < 2 or not (upLev > 0.0 and upLev > loLev):
        return ori.copy()
    aVal = np.exp(-1.0) * upLev                    # a = upLev / e
    mVal = max(aVal, loLev)
    bVal = 1.0 / N + ((aVal - loLev) / (2 * aVal - loLev) if aVal >= loLev else 0.0)
    bVal += 1e-7
    out = np.zeros_like(ori)
    out[0] = ori[0]
    logPortion = 1.0 / (np.log(mVal) - np.log(upLev))
    for g in range(2, N + 1):                       # 1-based, like the MATLAB source
        x = g * (upLev - loLev) / N + loLev         # linear data value at bar position g
        if x > mVal:                                # logarithmic segment
            f = N * ((np.log(mVal) - np.log(x)) * logPortion * (1 - bVal) + bVal)
        elif (loLev < aVal) and (x > loLev):        # linear segment near the bottom
            f = N * ((x - loLev) / (aVal - loLev) * (bVal - 1.0 / N)) + 1.0
        else:                                       # lowest valid colour
            f = 1.0
        idx = min(N, 1 + int(np.floor(f)))
        out[g - 1] = ori[idx - 1]
    return out


def log_remap_cmap(cmap, vmin: float, vmax: float, N: int = 256) -> mcolors.Colormap:
    """Return a Fuderer log-remapped copy of ``cmap`` for the window
    ``[vmin, vmax]``.  ``cmap`` may be a name or a Colormap object.  When the
    window is invalid the original colormap is returned unchanged (so callers
    can apply it unconditionally to any map, including signed/difference maps)."""
    base = cmap if isinstance(cmap, mcolors.Colormap) else get_cmap(cmap)
    try:
        # Only meaningful for non-negative (magnitude) maps: require vmin >= 0 so
        # signed/difference maps (which straddle 0) are left linear & unchanged.
        if not (vmax is not None and vmin is not None
                and float(vmin) >= 0.0 and float(vmax) > 0.0
                and float(vmax) > float(vmin)):
            return base
        cols = np.asarray(base(np.linspace(0.0, 1.0, N)))[:, :3]
        warped = color_log_remap(cols, float(vmin), float(vmax))
        name = getattr(base, "name", "cmap")
        return mcolors.LinearSegmentedColormap.from_list(f"{name}_log", warped, N=N)
    except Exception:
        return base
