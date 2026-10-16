"""
mat_reader.py
=============
Vendor-neutral MATLAB ``.mat`` reader for OCEAN.

A single ``.mat`` file is treated as a self-contained scan: it holds one image
stack (2-D … 4-D) plus, optionally, the acquisition schedule (TR / TE / TI /
saturation-offset vector).  This lets a user export data from GE, Siemens,
Bruker, MR Solutions — anything that can write a ``.mat`` — and load it straight
into OCEAN alongside the existing DICOM / NIfTI / 2dseq paths.

Return contract (identical to ``bruker_reader`` / ``ge_siemens_reader``)
----------------------------------------------------------------------
    T1  : (Y, X, n_slices, nTR),  trs_ms   ndarray
    T2  : (Y, X, n_slices, nTE),  tes_ms   ndarray
    CEST: (Y, X, nMeas),          ppm      ndarray

Both classic (scipy, ``-v6`` / ``-v7``) and HDF5 (``-v7.3``) ``.mat`` files are
supported.  Variable names are auto-detected but can always be overridden
(``image_key`` / ``schedule_key``) for non-standard files.

Nothing here is Bruker/GE/Siemens specific — the same engine backs every
vendor's ".mat" option.
"""
from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Low-level loading (scipy  +  HDF5 fallback for -v7.3)
# ──────────────────────────────────────────────────────────────────────────────

# Variable-name hints for the acquisition schedule, by modality.  Matched
# case-insensitively as a substring, most-specific first.
_SCHED_HINTS = {
    "t1": ["trs_ms", "tr_ms", "trs", "tr", "ti_ms", "tis", "ti",
           "recovery", "invtime", "inversiontime", "repetitiontime", "sched"],
    "t2": ["tes_ms", "te_ms", "tes", "te", "prep", "spinlock", "tsl",
           "echotime", "echo", "sched"],
    "cest": ["ppm", "offsets_ppm", "offset_ppm", "offsets", "offset",
             "dw", "ppms", "w", "sched", "freq", "satfreq"],
}
# Keys that are never the image or the schedule (metadata that scipy adds).
_SKIP_KEYS = {"__header__", "__version__", "__globals__", "none"}


def _is_numeric(a: np.ndarray) -> bool:
    return a is not None and isinstance(a, np.ndarray) and a.dtype.kind in "fiub"


def _load_scipy(path: str) -> "dict[str, np.ndarray] | None":
    """Classic (<= v7) .mat via scipy.  Returns None for v7.3 (HDF5)."""
    try:
        from scipy.io import loadmat
    except Exception as exc:                                   # pragma: no cover
        raise ImportError("scipy is required to read .mat files") from exc
    try:
        raw = loadmat(path, squeeze_me=False, struct_as_record=False)
    except NotImplementedError:
        return None            # -v7.3 → HDF5, handled by _load_h5py
    except ValueError:
        return None            # not a classic .mat (likely HDF5)
    out: "dict[str, np.ndarray]" = {}
    for k, v in raw.items():
        if k in _SKIP_KEYS:
            continue
        if isinstance(v, np.ndarray) and v.dtype.kind in "fiubc":
            out[k] = np.asarray(v)
    return out


def _load_h5py(path: str) -> "dict[str, np.ndarray]":
    """HDF5 (-v7.3) .mat via h5py.  Transposes to restore MATLAB column order."""
    try:
        import h5py
    except Exception as exc:                                   # pragma: no cover
        raise ImportError(
            "h5py is required to read MATLAB -v7.3 (.mat) files.\n"
            "Install it with:  pip install h5py") from exc
    out: "dict[str, np.ndarray]" = {}
    with h5py.File(path, "r") as f:
        def _visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if name.split("/")[-1].startswith("#"):
                return
            try:
                arr = np.asarray(obj[()])
            except Exception:
                return
            if arr.dtype.kind in "fiubc" and arr.size:
                # h5py reads MATLAB arrays in reversed (row-major) axis order.
                out[name.split("/")[-1]] = arr.T if arr.ndim > 1 else arr
        f.visititems(_visit)
    return out


def load_mat(path: str) -> "dict[str, np.ndarray]":
    """Load every numeric variable from a .mat file (classic or -v7.3)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    mats = _load_scipy(path)
    if mats is None:
        mats = _load_h5py(path)
    if not mats:
        raise ValueError(
            f"No numeric arrays found in {os.path.basename(path)}. "
            "The .mat file may be empty or hold only structs/cells.")
    return mats


def inspect_mat(path: str) -> "list[tuple[str, tuple]]":
    """[(variable_name, shape), …] sorted largest-first — for UI preview."""
    mats = load_mat(path)
    items = [(k, tuple(v.shape)) for k, v in mats.items()]
    items.sort(key=lambda kv: int(np.prod(kv[1]) if kv[1] else 0), reverse=True)
    return items


# ──────────────────────────────────────────────────────────────────────────────
# Heuristic variable selection
# ──────────────────────────────────────────────────────────────────────────────

def _pick_image(mats: "dict[str, np.ndarray]",
                image_key: "str | None" = None) -> "tuple[str, np.ndarray]":
    """Choose the image stack: explicit key, else the largest ndim>=2 array."""
    if image_key:
        if image_key not in mats:
            raise KeyError(f"image variable {image_key!r} not in .mat "
                           f"(have: {', '.join(mats)})")
        return image_key, np.asarray(mats[image_key])
    cands = [(k, v) for k, v in mats.items()
             if _is_numeric(v) and v.ndim >= 2 and v.size >= 16]
    if not cands:
        # Fall back to whatever has the most elements.
        cands = [(k, v) for k, v in mats.items() if _is_numeric(v)]
    if not cands:
        raise ValueError("no numeric image array found in .mat file")
    k, v = max(cands, key=lambda kv: kv[1].size)
    return k, np.asarray(v)


def _complex_to_mag(a: np.ndarray) -> np.ndarray:
    return np.abs(a) if np.iscomplexobj(a) else a


def _pick_schedule(mats: "dict[str, np.ndarray]", n_meas: int, kind: str,
                   exclude: str, schedule_key: "str | None" = None
                   ) -> "np.ndarray | None":
    """Find a 1-D schedule vector of length ``n_meas``.

    Preference: explicit key → hinted name of right length → any 1-D vector of
    the right length.  Returns None when nothing matches (caller supplies a
    fallback axis).
    """
    if schedule_key:
        if schedule_key not in mats:
            raise KeyError(f"schedule variable {schedule_key!r} not in .mat "
                           f"(have: {', '.join(mats)})")
        v = np.asarray(mats[schedule_key]).squeeze().astype(float).ravel()
        return v

    def _as_vec(a):
        a = np.asarray(a).squeeze()
        return a.astype(float).ravel() if a.ndim <= 1 else None

    hints = _SCHED_HINTS.get(kind, [])
    # 1) hinted names, most-specific first, matching length
    for hint in hints:
        for k, v in mats.items():
            if k == exclude or k.lower() != hint and hint not in k.lower():
                continue
            vec = _as_vec(v)
            if vec is not None and vec.size == n_meas:
                return vec
    # 2) any 1-D numeric vector whose length matches the measurement count
    for k, v in mats.items():
        if k == exclude or not _is_numeric(v):
            continue
        vec = _as_vec(v)
        if vec is not None and vec.size == n_meas and n_meas > 1:
            return vec
    return None


def _best_sched_len(mats: "dict[str, np.ndarray]", kind: str, exclude: str,
                    allowed: "set[int]") -> "int | None":
    """Length of the most likely schedule vector, restricted to axis lengths
    that actually occur in the image stack.  Used to pin the measurement axis
    *before* orienting (essential to tell slices from measurements in 4-D)."""
    def _vec_len(a):
        a = np.asarray(a).squeeze()
        return int(a.size) if a.ndim <= 1 else None
    hints = _SCHED_HINTS.get(kind, [])
    for hint in hints:                       # most-specific hinted name first
        for k, v in mats.items():
            if k == exclude or not _is_numeric(v):
                continue
            if k.lower() != hint and hint not in k.lower():
                continue
            n = _vec_len(v)
            if n and n in allowed and n > 1:
                return n
    for k, v in mats.items():                # any 1-D vector of a valid length
        if k == exclude or not _is_numeric(v):
            continue
        n = _vec_len(v)
        if n and n in allowed and n > 1:
            return n
    return None


def _orient_stack(img: np.ndarray, n_meas_hint: "int | None" = None,
                  want_slices: bool = True) -> np.ndarray:
    """Normalise an image array to (Y, X, [n_slices,] n_meas).

    ``n_meas_hint`` (the schedule length) pins the measurement axis; without it
    the shortest non-spatial axis is assumed to be measurements.  A 4-D array
    keeps one slice axis; a 3-D array has no slice axis unless ``want_slices``.
    """
    img = _complex_to_mag(np.asarray(img))
    img = np.squeeze(img)
    if img.ndim == 2:
        return img[:, :, None, None] if want_slices else img[:, :, None]

    if img.ndim == 3:
        # Decide which axis is "measurements": prefer the hint, else the
        # smallest axis (schedules are short relative to image matrix).
        if n_meas_hint and n_meas_hint in img.shape:
            meas_ax = list(img.shape).index(n_meas_hint)
        else:
            meas_ax = int(np.argmin(img.shape))
        img = np.moveaxis(img, meas_ax, -1)        # (Y, X, nMeas)
        return img[:, :, None, :] if want_slices else img

    if img.ndim == 4:
        shp = list(img.shape)
        # Pick the measurement axis from the schedule-length hint when given;
        # else the smallest axis.  (Slices can out- or under-number echoes, so
        # size alone is unreliable — the hint disambiguates.)
        if n_meas_hint and n_meas_hint in shp:
            meas_ax = shp.index(n_meas_hint)
        else:
            meas_ax = int(np.argmin(shp))
        rest = [a for a in range(4) if a != meas_ax]
        # Of the remaining three, the two largest are Y,X; the smallest is slices
        rest_sorted = sorted(rest, key=lambda a: shp[a])
        slice_ax = rest_sorted[0]
        y_ax, x_ax = sorted(rest_sorted[1:])
        img = np.transpose(img, (y_ax, x_ax, slice_ax, meas_ax))
        return img

    raise ValueError(f"unsupported image ndim={img.ndim} (shape {img.shape})")


# ──────────────────────────────────────────────────────────────────────────────
# Public readers — mirror ge_siemens_reader / bruker_reader signatures
# ──────────────────────────────────────────────────────────────────────────────

def read_mat_t1_vtr(path: str, image_key: "str | None" = None,
                    schedule_key: "str | None" = None, schedule=None
                    ) -> "tuple[np.ndarray, np.ndarray]":
    """T1 variable-TR / inversion-recovery stack from a .mat file.

    Returns (image[Y, X, n_slices, nTR], trs_ms).  ``schedule`` (an explicit
    array of TR/TI values) wins over anything auto-detected in the file.
    """
    mats = load_mat(path)
    ik, img = _pick_image(mats, image_key)
    sched = None if schedule is None else np.asarray(schedule, float).ravel()
    n_hint = (sched.size if sched is not None else
              _best_sched_len(mats, "t1", ik, set(np.squeeze(img).shape)))
    img4 = _orient_stack(img, n_meas_hint=n_hint, want_slices=True)
    n_meas = img4.shape[3]
    if sched is None:
        sched = _pick_schedule(mats, n_meas, "t1", ik, schedule_key)
    if sched is None:
        sched = np.arange(1, n_meas + 1, dtype=float)        # positional fallback
    if sched.size != n_meas:
        raise ValueError(f"T1 schedule has {sched.size} values but stack has "
                         f"{n_meas} measurements")
    return img4.astype(np.float32), sched.astype(float)


def read_mat_t2_msme(path: str, image_key: "str | None" = None,
                     schedule_key: "str | None" = None, schedule=None
                     ) -> "tuple[np.ndarray, np.ndarray]":
    """T2 multi-echo stack from a .mat file.

    Returns (image[Y, X, n_slices, nTE], tes_ms).
    """
    mats = load_mat(path)
    ik, img = _pick_image(mats, image_key)
    sched = None if schedule is None else np.asarray(schedule, float).ravel()
    n_hint = (sched.size if sched is not None else
              _best_sched_len(mats, "t2", ik, set(np.squeeze(img).shape)))
    img4 = _orient_stack(img, n_meas_hint=n_hint, want_slices=True)
    n_meas = img4.shape[3]
    if sched is None:
        sched = _pick_schedule(mats, n_meas, "t2", ik, schedule_key)
    if sched is None:
        sched = np.arange(1, n_meas + 1, dtype=float)
    if sched.size != n_meas:
        raise ValueError(f"T2 schedule has {sched.size} values but stack has "
                         f"{n_meas} measurements")
    return img4.astype(np.float32), sched.astype(float)


def read_mat_cest(path: str, image_key: "str | None" = None,
                  schedule_key: "str | None" = None, offsets=None,
                  keep_slices: bool = False
                  ) -> "tuple[np.ndarray, np.ndarray]":
    """CEST / WASSR Z-spectrum stack from a .mat file.

    Returns (z_img, ppm).  ``z_img`` is (Y, X, nMeas) by default; with
    ``keep_slices=True`` it is (Y, X, n_slices, nMeas) — the 4-D layout the
    Z-spectrum tab's ``_finish_cest_load`` expects (a 2-D/3-D file gets a
    singleton slice axis).  ``offsets`` (explicit ppm array) wins over any
    vector found in the file.
    """
    mats = load_mat(path)
    ik, img = _pick_image(mats, image_key)
    ppm = None if offsets is None else np.asarray(offsets, float).ravel()
    n_hint = (ppm.size if ppm is not None else
              _best_sched_len(mats, "cest", ik, set(np.squeeze(img).shape)))
    if keep_slices:
        img4 = _orient_stack(img, n_meas_hint=n_hint, want_slices=True)
        n_meas = img4.shape[3]
        if ppm is None:
            ppm = _pick_schedule(mats, n_meas, "cest", ik, schedule_key)
        if ppm is None:
            ppm = np.arange(n_meas, dtype=float)
        if ppm.size != n_meas:
            raise ValueError(f"CEST offset vector has {ppm.size} values but "
                             f"stack has {n_meas} measurements")
        return img4.astype(np.float32), ppm.astype(float)
    img3 = _orient_stack(img, n_meas_hint=n_hint, want_slices=False)
    if img3.ndim == 4:                      # a slice axis slipped through
        img3 = img3[:, :, img3.shape[2] // 2, :]
    n_meas = img3.shape[2]
    if ppm is None:
        ppm = _pick_schedule(mats, n_meas, "cest", ik, schedule_key)
    if ppm is None:
        ppm = np.arange(n_meas, dtype=float)
    if ppm.size != n_meas:
        raise ValueError(f"CEST offset vector has {ppm.size} values but stack "
                         f"has {n_meas} measurements")
    return img3.astype(np.float32), ppm.astype(float)


def read_mat_stack(path: str, kind: str = "auto", **kw
                   ) -> "tuple[np.ndarray, np.ndarray, str]":
    """Generic entry point.  ``kind`` ∈ {'t1','t2','cest','auto'}.

    'auto' picks by schedule name when possible (returns the detected kind).
    """
    kind = (kind or "auto").lower()
    if kind == "t1":
        img, s = read_mat_t1_vtr(path, **kw); return img, s, "t1"
    if kind == "t2":
        img, s = read_mat_t2_msme(path, **kw); return img, s, "t2"
    if kind == "cest":
        img, s = read_mat_cest(path, **kw); return img, s, "cest"
    # auto: sniff which schedule hint family the file matches
    mats = load_mat(path)
    ik, _img = _pick_image(mats, kw.get("image_key"))
    keys = {k.lower() for k in mats if k != ik}
    for guess in ("cest", "t1", "t2"):
        if any(h in k for k in keys for h in _SCHED_HINTS[guess][:4]):
            return read_mat_stack(path, guess, **kw)
    img, s = read_mat_t1_vtr(path, **kw)        # safe default
    return img, s, "t1"
