"""
matio.py — MAT-file I/O that survives the 4 GB limit of the classic format.
=========================================================================
scipy.io.savemat writes MATLAB v5/v7 files, whose per-variable byte count is a
32-bit field: any variable >= 4 GiB raises
``MatWriteError: Matrix too large to save with Matlab 5 format``.  Large
CEST-MRF dictionaries (``sig`` = n_entries x n_meas float64) hit this.

``savemat_any`` writes the classic format when every variable fits and the
MATLAB **v7.3** (HDF5) format otherwise — with the MATLAB userblock header and
``MATLAB_class`` attributes, so MATLAB's ``load`` reads it natively.
``loadmat_any`` reads either format and returns the same dict layout scipy's
``loadmat`` gives (row vectors as (1, n), full arrays in Python axis order).

Only numeric / logical / char variables are supported in the v7.3 writer —
that covers the dictionaries and quant maps OCEAN produces.  Dicts/cells are
rejected with a clear error rather than silently dropped.
"""
from __future__ import annotations

import datetime as _dt
import platform as _platform

import numpy as np
import scipy.io as sio
from scipy.io.matlab import MatWriteError

# scipy raises when a variable's byte count >= 2**32; keep a safety margin.
MAT5_LIMIT = 2 ** 32 - 2 ** 24

_NP2MATLAB = {
    "float64": "double", "float32": "single",
    "int8": "int8", "uint8": "uint8", "int16": "int16", "uint16": "uint16",
    "int32": "int32", "uint32": "uint32", "int64": "int64", "uint64": "uint64",
    "bool": "logical",
}


def _as_array(value):
    """numpy view of a variable (lists of equal-length arrays → 2-D)."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (str, bytes)):
        return value
    return np.asarray(value)


def needs_v73(d: dict) -> bool:
    """True when any variable would exceed the classic-format limit."""
    for v in d.values():
        a = _as_array(v)
        if isinstance(a, np.ndarray) and a.nbytes >= MAT5_LIMIT:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# v7.3 writer
# ──────────────────────────────────────────────────────────────────────────────

def _matlab_userblock() -> bytes:
    """128-byte MAT header (text, subsystem offset, version 0x0200, 'IM')."""
    text = (f"MATLAB 7.3 MAT-file, Platform: {_platform.system()}, "
            f"Created on: {_dt.datetime.now().strftime('%a %b %d %H:%M:%S %Y')} "
            f"HDF5 schema 1.00 .")
    head = text.encode("ascii", "replace")[:116].ljust(116, b" ")
    head += b"\x00" * 8                     # subsystem data offset
    head += b"\x00\x02"                     # version 0x0200 (little-endian)
    head += b"IM"                           # endian indicator
    return head.ljust(512, b"\x00")


def save_mat73(path: str, d: dict) -> None:
    """Write ``d`` as a MATLAB v7.3 (HDF5) file MATLAB can ``load``."""
    import h5py

    with h5py.File(path, "w", userblock_size=512) as f:
        for name, value in d.items():
            if name.startswith("__"):
                continue
            a = _as_array(value)
            if isinstance(a, (str, bytes)) or (isinstance(a, np.ndarray) and a.dtype.kind in "US"):
                # MATLAB char matrix: one row per string (N x L, space-padded,
                # as MATLAB/scipy do), stored transposed like every v7.3 array.
                # Covers a single string and per-entry string arrays such as the
                # dictionary's `lineshape` ('SuperLorentzian' x n_entries).
                if isinstance(a, (str, bytes)):
                    strs = [a.decode("utf-8", "replace") if isinstance(a, bytes) else a]
                else:
                    strs = [(s.decode("utf-8", "replace") if isinstance(s, bytes) else str(s))
                            for s in a.ravel()]
                L = max((len(s) for s in strs), default=0)
                codes = np.full((len(strs), L), ord(" "), dtype=np.uint16)
                for i, s in enumerate(strs):
                    codes[i, :len(s)] = [ord(c) for c in s]
                ds = f.create_dataset(name, data=np.ascontiguousarray(codes.T))
                ds.attrs["MATLAB_class"] = np.bytes_(b"char")
                ds.attrs["MATLAB_int_decode"] = np.int32(2)
                continue
            if a.dtype.kind == "O" or a.dtype.names:
                raise TypeError(f"variable {name!r}: dtype {a.dtype} is not "
                                "supported by the v7.3 writer (numeric, logical "
                                "and char only)")
            if a.size == 0:
                # scipy's loadmat returns a 1-D empty as (0, 0) — mirror that.
                dims = a.shape if a.ndim >= 2 else (0, 0)
                ds = f.create_dataset(name, data=np.array(dims, dtype=np.uint64))
                ds.attrs["MATLAB_class"] = np.bytes_(_NP2MATLAB.get(a.dtype.name, "double").encode())
                ds.attrs["MATLAB_empty"] = np.int32(1)
                continue
            a2 = np.atleast_2d(a)
            if a2.dtype.kind == "b":
                a2 = a2.astype(np.uint8)
                mclass = "logical"
            elif a2.dtype.kind == "c":
                # MATLAB complex: compound {real, imag}
                base = np.float64 if a2.dtype == np.complex128 else np.float32
                comp = np.empty(a2.shape, dtype=[("real", base), ("imag", base)])
                comp["real"], comp["imag"] = a2.real, a2.imag
                a2, mclass = comp, ("double" if base is np.float64 else "single")
            else:
                mclass = _NP2MATLAB.get(a2.dtype.name)
                if mclass is None:               # e.g. float16 → double
                    a2 = a2.astype(np.float64); mclass = "double"
            # MATLAB stores arrays in reversed axis order → write the transpose.
            ds = f.create_dataset(name, data=np.ascontiguousarray(a2.T))
            ds.attrs["MATLAB_class"] = np.bytes_(mclass.encode())
    with open(path, "r+b") as fh:
        fh.seek(0)
        fh.write(_matlab_userblock())


def savemat_any(path: str, d: dict, **savemat_kw) -> str:
    """savemat that falls back to v7.3 for > 4 GiB variables.  Returns the
    format written: 'v5' or 'v7.3'."""
    if needs_v73(d):
        save_mat73(path, d)
        return "v7.3"
    try:
        sio.savemat(path, d, **savemat_kw)
        return "v5"
    except MatWriteError:
        save_mat73(path, d)
        return "v7.3"


# ──────────────────────────────────────────────────────────────────────────────
# reader
# ──────────────────────────────────────────────────────────────────────────────

def _is_v73(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            head = fh.read(128)
        return len(head) >= 128 and head[124:126] == b"\x00\x02"
    except OSError:
        return False


def load_mat73(path: str) -> dict:
    """Read a v7.3 file into the layout scipy's loadmat returns."""
    import h5py

    out = {"__header__": b"MATLAB 7.3 MAT-file", "__version__": "7.3", "__globals__": []}
    with h5py.File(path, "r") as f:
        for name, obj in f.items():
            if name.startswith("#") or not isinstance(obj, h5py.Dataset):
                continue
            mclass = obj.attrs.get("MATLAB_class", b"")
            mclass = mclass.decode() if isinstance(mclass, bytes) else str(mclass)
            if obj.attrs.get("MATLAB_empty", 0):
                dims = tuple(int(x) for x in np.asarray(obj[()]).ravel())
                out[name] = np.zeros(dims)
                continue
            a = np.asarray(obj[()])
            if mclass == "char":
                # Stored (L, N) → one string per MATLAB row; scipy returns an
                # (N,) array of str (a single string → shape (1,)).
                codes = np.atleast_2d(a).T                     # (N, L)
                out[name] = np.array(["".join(chr(int(c)) for c in row) for row in codes])
                continue
            if a.dtype.names and "real" in a.dtype.names:
                a = a["real"] + 1j * a["imag"]
            if mclass == "logical":
                a = a.astype(bool)
            out[name] = a.T if a.ndim > 1 else a[np.newaxis, :]
    return out


def loadmat_any(path: str, **loadmat_kw) -> dict:
    """``scipy.io.loadmat`` for classic files, HDF5 reader for v7.3 files."""
    if _is_v73(path):
        return load_mat73(path)
    try:
        return sio.loadmat(path, **loadmat_kw)
    except NotImplementedError:          # "Please use HDF reader for matlab v7.3 files"
        return load_mat73(path)
