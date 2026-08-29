"""
dicom_mosaic.py
===============
Siemens MOSAIC de-tiling.

Siemens packs every slice of a multi-slice / 3D acquisition into a single
2-D DICOM frame — a square grid of tiles (a "montage").  For example a
116x116 acquisition with 88 slices is stored as a 1160x1160 image: a 10x10
grid (ceil(sqrt(88)) = 10) whose first 88 tiles are the real slices and whose
last 12 tiles are zero padding.

This module detects such frames and splits them back into a proper
(rows, cols, n_slices) volume so every tab shows one slice at a time (with a
slice slider) instead of the tiled montage.

Detection is by the DICOM tags — NOT by ``MRAcquisitionType`` alone, because
Siemens localizers / MPRAGE are also ``MRAcquisitionType == '3D'`` yet are NOT
mosaics.  The reliable signals are ``'MOSAIC' in ImageType`` and the private
Siemens tag ``(0019,100a) NumberOfImagesInMosaic``.
"""
from __future__ import annotations

import math

import numpy as np

# Private Siemens tag: number of real slices packed into the mosaic
_TAG_N_IMAGES_IN_MOSAIC = (0x0019, 0x100A)


def _num_images_in_mosaic(ds) -> "int | None":
    """Return the number of real slices in a Siemens mosaic, or None."""
    # 1. Standard private tag (0019,100a)
    try:
        v = ds[_TAG_N_IMAGES_IN_MOSAIC].value
        n = int(v)
        if n > 0:
            return n
    except Exception:
        pass
    # 2. CSA image header ("NumberOfImagesInMosaic") as a last resort
    try:
        n = _csa_num_images_in_mosaic(ds)
        if n and n > 0:
            return int(n)
    except Exception:
        pass
    return None


def _csa_num_images_in_mosaic(ds) -> "int | None":
    """Best-effort scrape of NumberOfImagesInMosaic from the Siemens CSA header
    (0029,1010).  Avoids a full CSA parser — just finds the field name and the
    first following integer token."""
    try:
        raw = ds[0x0029, 0x1010].value
    except Exception:
        return None
    if isinstance(raw, (bytes, bytearray)):
        blob = bytes(raw)
    else:
        return None
    key = b"NumberOfImagesInMosaic"
    i = blob.find(key)
    if i < 0:
        return None
    # Scan the bytes after the key for the first plausible ASCII integer
    tail = blob[i + len(key): i + len(key) + 256]
    num = b""
    for b in tail:
        c = bytes([b])
        if c.isdigit():
            num += c
        elif num:
            break
    try:
        return int(num) if num else None
    except ValueError:
        return None


def _acq_matrix_inplane(ds) -> "int | None":
    """Base in-plane matrix size from AcquisitionMatrix (e.g. [0,116,116,0]→116)."""
    try:
        am = getattr(ds, "AcquisitionMatrix", None)
        if am:
            vals = [int(x) for x in am if int(x) > 0]
            if vals:
                return max(vals)
    except Exception:
        pass
    return None


def is_mosaic(ds) -> bool:
    """True if the pydicom dataset is a Siemens MOSAIC image."""
    try:
        it = [str(x).upper() for x in (getattr(ds, "ImageType", []) or [])]
        if "MOSAIC" in it:
            return True
    except Exception:
        pass
    return _num_images_in_mosaic(ds) is not None


def mosaic_geometry(ds, rows: int, cols: int) -> "tuple[int, int, int] | None":
    """Return (grid, tile_rows, tile_cols) for a mosaic of shape (rows, cols),
    or None if it does not look like a de-tileable mosaic.

    The grid is ceil(sqrt(n_slices)); if the slice count is unknown it is
    inferred from AcquisitionMatrix (rows // base)."""
    n = _num_images_in_mosaic(ds)
    if n is not None:
        grid = int(math.ceil(math.sqrt(n)))
    else:
        base = _acq_matrix_inplane(ds)
        if not base or base >= rows or rows % base != 0:
            return None
        grid = rows // base
        n = grid * grid
    if grid <= 1:
        return None
    if rows % grid != 0 or cols % grid != 0:
        return None
    return grid, rows // grid, cols // grid


def mosaic_to_volume(arr, ds) -> np.ndarray:
    """De-tile a 2-D Siemens mosaic array into (tile_rows, tile_cols, n_slices).

    Returns the input unchanged (as 2-D) if it is not a de-tileable mosaic.
    The first ``NumberOfImagesInMosaic`` tiles (row-major: left→right,
    top→bottom) are kept; trailing zero-padding tiles are dropped."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        return arr
    rows, cols = arr.shape
    geom = mosaic_geometry(ds, rows, cols)
    if geom is None:
        return arr
    grid, tr, tc = geom
    n = _num_images_in_mosaic(ds) or (grid * grid)
    n = min(n, grid * grid)
    vol = np.empty((tr, tc, n), dtype=arr.dtype)
    for s in range(n):
        r, c = divmod(s, grid)
        vol[:, :, s] = arr[r * tr:(r + 1) * tr, c * tc:(c + 1) * tc]
    return vol


def n_mosaic_slices(ds) -> "int | None":
    """Public helper: number of real slices if ds is a mosaic, else None."""
    return _num_images_in_mosaic(ds)
