"""
ge_cest_reader.py
=================
Reader for GE 3T CEST acquisitions that store one combined dynamic series
containing idling (dummy) frames, an S0 reference, a WASSR/B0 block, and the
CEST z-spectrum block — with the saturation offsets supplied separately as a
text file in **Hz**.

典型 GE layout (1-indexed frames), e.g. "SSFSE, CW, DL":
    frame 1            – idling / dummy            (dropped)
    frame 2            – S0 reference              (used as M0)
    frame 3            – idling / dummy            (dropped)
    frame 4 … 4+N-1    – N offset images, ordered as the offsets file:
        first  n_b0    – WASSR / B0 block          (e.g. ±1.88 ppm)
        rest           – CEST z-spectrum block     (e.g. ±7 ppm)

The offsets file holds one value per line in **Hz** (3T).  ppm = Hz / larmor.

Returns a dict the CEST-MRI tab can feed straight into its existing CEST and
WASSR pipelines (MTR-asymmetry, %CEST, B0 correction, dictionary matching).
"""
from __future__ import annotations

import glob
import os

import numpy as np


def _load_4d(data_path: str, log_fn=print) -> np.ndarray:
    """Load a CEST dynamic series as (Y, X, n_slices, n_frames) float array.

    Accepts a NIfTI file (.nii / .nii.gz) or a folder of DICOM frames.
    """
    if os.path.isdir(data_path):
        # DICOM folder — reuse the tab's robust 4-D DICOM loader
        from my_gui.tabs.zspec_tab import _load_dicom_4d
        img_all, _ppm = _load_dicom_4d(data_path, "", log_fn)
        return np.asarray(img_all, dtype=np.float32)

    ext = data_path.lower()
    if ext.endswith(".nii") or ext.endswith(".nii.gz"):
        import nibabel as nib
        data = np.asarray(nib.load(data_path).dataobj, dtype=np.float32)
        if data.ndim == 3:                       # (Y, X, frames) → add slice axis
            data = data[:, :, np.newaxis, :]
        elif data.ndim == 4 and data.shape[2] == 1:
            pass                                  # already (Y, X, 1, frames)
        elif data.ndim == 4:
            pass                                  # (Y, X, slices, frames)
        else:
            raise ValueError(f"Unexpected NIfTI shape {data.shape}")
        return data

    raise ValueError(
        f"Unsupported data path '{data_path}'. Provide a .nii/.nii.gz file "
        "or a folder of DICOM frames."
    )


def peek_cube_slices(data_path: str) -> "int | None":
    """Number of slices per offset for a 3D CUBE series (unique SliceLocations)."""
    try:
        if os.path.isdir(data_path):
            import glob as _g, pydicom
            files = [f for f in _g.glob(os.path.join(data_path, "*"))
                     if os.path.isfile(f) and f.lower().endswith((".dcm", ".ima"))]
            locs = set()
            for f in files:
                ds = pydicom.dcmread(f, force=True, stop_before_pixels=True)
                locs.add(round(float(getattr(ds, "SliceLocation", 0.0)), 2))
            return len(locs) or None
    except Exception:
        pass
    return None


def read_ge_cest_cvs(data_path: str) -> dict:
    """Read GE saturation User-Variables (CVs) from the first DICOM.

    Mapping (0x0019,10Bx): CW flag, duration(ms), WASSR-B1(µT, 10B6),
    CEST-B1(µT, 10B8).  Returns {} if unavailable (e.g. NIfTI).
    """
    try:
        import glob as _g, pydicom
        if os.path.isdir(data_path):
            files = sorted(f for f in _g.glob(os.path.join(data_path, "*"))
                           if os.path.isfile(f) and f.lower().endswith((".dcm", ".ima")))
            ds = pydicom.dcmread(files[0], force=True, stop_before_pixels=True)
        elif data_path.lower().endswith((".dcm", ".ima")):
            ds = pydicom.dcmread(data_path, force=True, stop_before_pixels=True)
        else:
            return {}

        def _cv(off):
            try:
                return float(ds[0x0019, 0x1000 + off].value)
            except Exception:
                return None
        return {
            "cw":          _cv(0xB0),
            "duration_ms": _cv(0xB1),
            "wassr_b1_uT": _cv(0xB6),
            "cest_b1_uT":  _cv(0xB8),
        }
    except Exception:
        return {}


def read_ge_cest_block(
    data_path: str,
    offsets: np.ndarray,
    block: str = "cest",
    acquisition: str = "ssfse",
    larmor_mhz: float = 127.7415,
    offsets_in_hz: bool = True,
    n_slices: "int | None" = None,
    log_fn=print,
) -> dict:
    """
    Extract the CEST or WASSR block (+ S0) from a combined GE acquisition.

    Frame ordering (per offset, S = slices/offset):
      SSFSE (S=1):  idling, S0, idling, WASSR×nW, CEST×nC
      CUBE  (S>1):  idling×S, S0×S, WASSR×(nW·S), CEST×(nC·S)

    block='wassr' → the first len(offsets) offsets after the prefix.
    block='cest'  → the LAST len(offsets) offsets (CEST is always last).
    So each block is extracted independently from its own offsets file.

    Returns dict: s0 (Y,X,S), img (Y,X,S,n_off), ppm (n_off,), n_slices (S).
    """
    data = _load_4d(data_path, log_fn)                  # (Y, X, n_sl, F)
    Y, X, n_sl, F = data.shape

    offs = np.asarray(offsets, dtype=float).ravel()
    n_off = len(offs)
    ppm = (offs / larmor_mhz) if offsets_in_hz else offs.copy()
    acq = acquisition.lower()

    # Direct extraction — NO GE idling/S0 prefix — for a "custom" acquisition
    # (Siemens or any vendor: one image per saturation offset, so N frames = N
    # offsets from the loaded .txt), or for multi-slice input (e.g. a de-tiled
    # Siemens mosaic) that already carries the offsets on the frame axis F.
    # Take the block directly along F.  (The GE single-series combined layout
    # below always arrives with n_sl == 1 and a non-custom acquisition.)
    if acq.startswith("custom") or n_sl > 1:
        if n_off > F:
            raise ValueError(
                f"{block.upper()}: offsets file has {n_off} entries but the "
                f"series has only {F} offset frame(s).")
        if block == "wassr":
            img = data[:, :, :, :n_off]
            where = f"offsets 1..{n_off}"
        else:                                           # CEST → last n_off offsets
            img = data[:, :, :, F - n_off:F]
            where = f"offsets {F - n_off + 1}..{F}"
        _tag = "custom" if acq.startswith("custom") else f"multi-slice S={n_sl}"
        log_fn(f"  {block.upper()} ({_tag}): {where}; {n_off} offsets "
               f"[{ppm.min():.2f}..{ppm.max():.2f} ppm]. No idling/S0 prefix.")
        return {"s0": None, "img": img, "ppm": ppm, "n_slices": n_sl}

    frames = data[:, :, 0, :]                           # (Y, X, F)
    Ftot = frames.shape[2]

    if acq.startswith("cube"):
        S = n_slices or peek_cube_slices(data_path) or 1
        prefix = 2 * S                                  # idling + S0 (no 2nd idling)
        s0 = frames[:, :, S:2 * S]                      # (Y,X,S)
    else:                                               # SSFSE
        S = 1
        prefix = 3                                      # idling, S0, idling
        s0 = frames[:, :, 1:2]                          # (Y,X,1)

    need = n_off * S
    if block == "wassr":
        blk = frames[:, :, prefix:prefix + need]
        where = f"frames {prefix + 1}..{prefix + need}"
    else:                                               # CEST → last n_off offsets
        blk = frames[:, :, Ftot - need:Ftot]
        where = f"frames {Ftot - need + 1}..{Ftot}"

    if blk.shape[2] != need:
        raise ValueError(
            f"{block.upper()}: need {n_off} offsets × {S} slice(s) = {need} frames "
            f"but only {blk.shape[2]} available (series has {Ftot}). Check the "
            f"acquisition type and the offsets file.")

    # (Y,X, n_off*S) ordered offset-major, slice-minor → (Y,X,S,n_off)
    img = blk.reshape(Y, X, n_off, S).transpose(0, 1, 3, 2)
    log_fn(f"  {block.upper()} ({acq}, S={S}): {where}; {n_off} offsets "
           f"[{ppm.min():.2f}..{ppm.max():.2f} ppm].")
    return {"s0": s0, "img": img, "ppm": ppm, "n_slices": S}


def read_ge_cest_hz(
    data_path: str,
    offsets: np.ndarray,
    larmor_mhz: float = 127.7415,
    s0_index1: int = 2,
    first_offset_index1: int = 4,
    n_b0: int = 11,
    offsets_in_hz: bool = True,
    log_fn=print,
) -> dict:
    """
    Parse a combined GE CEST series into S0 / WASSR / CEST blocks.

    Parameters
    ----------
    data_path          : NIfTI file or DICOM folder (combined dynamic series)
    offsets            : 1-D array of saturation offsets (Hz if offsets_in_hz)
    larmor_mhz         : proton Larmor frequency (MHz). 3T GE ≈ 127.7415.
    s0_index1          : 1-based frame index of the S0 reference (default 2)
    first_offset_index1: 1-based frame index of the first offset image (default 4)
    n_b0               : number of leading offsets that form the WASSR/B0 block
    offsets_in_hz      : if True, convert offsets to ppm via /larmor_mhz

    Returns
    -------
    dict with keys:
        s0          : (Y, X, n_sl) S0 reference image
        cest_img    : (Y, X, n_sl, n_cest) CEST offset images
        cest_ppm    : (n_cest,) CEST offsets in ppm
        b0_img      : (Y, X, n_sl, n_b0) WASSR offset images  (or None if n_b0=0)
        b0_ppm      : (n_b0,) WASSR offsets in ppm            (or None)
        larmor_mhz  : echoed Larmor frequency
        n_frames    : total frames in the series
    """
    data = _load_4d(data_path, log_fn)          # (Y, X, n_sl, n_frames)
    Y, X, n_sl, n_frames = data.shape
    offsets = np.asarray(offsets, dtype=float).ravel()
    n_off = len(offsets)

    log_fn(f"  GE CEST series: {Y}x{X}, {n_sl} slice(s), {n_frames} frames; "
           f"{n_off} offsets supplied.")

    # ── Frame-range sanity checks ──────────────────────────────────────────
    s0_idx0    = s0_index1 - 1
    first_idx0 = first_offset_index1 - 1
    if not (0 <= s0_idx0 < n_frames):
        raise ValueError(f"S0 frame index {s0_index1} out of range (1..{n_frames}).")
    last_needed = first_idx0 + n_off
    if last_needed > n_frames:
        raise ValueError(
            f"Need frames {first_offset_index1}..{first_idx0 + n_off} "
            f"({n_off} offsets) but series only has {n_frames} frames. "
            "Check 'first offset frame' / offsets-file length."
        )
    if n_b0 < 0 or n_b0 > n_off:
        raise ValueError(f"# B0 offsets ({n_b0}) must be between 0 and {n_off}.")

    # ── Split frames ───────────────────────────────────────────────────────
    s0_img   = data[:, :, :, s0_idx0]                                  # (Y,X,n_sl)
    off_imgs = data[:, :, :, first_idx0:first_idx0 + n_off]            # (Y,X,n_sl,n_off)

    ppm = (offsets / larmor_mhz) if offsets_in_hz else offsets.copy()

    b0_img = b0_ppm = None
    if n_b0 > 0:
        b0_img = off_imgs[:, :, :, :n_b0]
        b0_ppm = ppm[:n_b0]
    cest_img = off_imgs[:, :, :, n_b0:]
    cest_ppm = ppm[n_b0:]

    log_fn(f"  → S0 = frame {s0_index1};  "
           f"WASSR = {0 if b0_ppm is None else len(b0_ppm)} offsets "
           f"[{'' if b0_ppm is None else f'{b0_ppm.min():.2f}..{b0_ppm.max():.2f} ppm'}];  "
           f"CEST = {len(cest_ppm)} offsets [{cest_ppm.min():.2f}..{cest_ppm.max():.2f} ppm].")

    return {
        "s0":         s0_img,
        "cest_img":   cest_img,
        "cest_ppm":   cest_ppm,
        "b0_img":     b0_img,
        "b0_ppm":     b0_ppm,
        "larmor_mhz": larmor_mhz,
        "n_frames":   n_frames,
    }
