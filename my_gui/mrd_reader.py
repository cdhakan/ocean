"""
mrd_reader.py
=============
Reader for MR Solutions (SMIS) ``.MRD`` raw datasets.

Python port of ``Get_mrd_3D5.m`` (R. Garipov) plus CEST/WASSR image-formation
helpers, mirroring ``CEST_UW_MRsolutions.m``:

    [im, dim, par] = Get_mrd_3D5(file, 'cen', 'cen')
    image_per_offset = abs( ifftshift( ifft2( ifftshift( squeeze(im(offset,:,:)) ) ) ) )
    ppm = (par.mtc_freq_max + i*par.mtc_freq_step) / Larmor_MHz

The .MRD payload is *k-space* (phase-encode views are centric-reordered), so
each saturation-offset frame is reconstructed with a 2-D inverse FFT before the
existing CEST-tab pipeline (B0 correction, MTR asymmetry, fitting) takes over.

Public API
----------
read_mrd(filename, reordering1='cen', reordering2='cen') -> (im, dim, par)
read_mrd_cest(filename, larmor_mhz=None, gauss_sigma=0.5)
    -> (img_4d (Y, X, 1, offsets) float32 magnitude, ppm_all, info)

``read_mrd_cest`` is used for both CEST and WASSR scans — they share the same
file layout; the only difference downstream is the ppm range.
"""
from __future__ import annotations

import re
import struct
from typing import Any

import numpy as np

__all__ = ["read_mrd", "read_mrd_cest"]


# ──────────────────────────────────────────────────────────────────────────────
# datatype (second hex nibble of the uint16 at byte 18) → numpy dtype, byte size
# Mirrors the switch() in Get_mrd_3D5.m
# ──────────────────────────────────────────────────────────────────────────────
_DTYPE_MAP = {
    "0": (np.uint8,   1),   # uchar
    "1": (np.int8,    1),   # schar
    "2": (np.int16,   2),   # short
    "3": (np.int16,   2),   # int16
    "4": (np.int32,   4),   # int32
    "5": (np.float32, 4),   # float32
    "6": (np.float64, 8),   # double
}


def read_mrd(filename: str,
             reordering1: str = "cen",
             reordering2: str = "cen") -> tuple[np.ndarray, list[int], dict]:
    """
    Read an MR Solutions ``.MRD`` / ``.SUR`` file.

    Parameters
    ----------
    filename    : path to the .MRD file
    reordering1 : 'cen' or 'seq' — phase-encode (views) reordering
    reordering2 : 'cen' or 'seq' — 2nd phase-encode (views_2) reordering

    Returns
    -------
    im  : complex ndarray, squeezed from
          (no_expts, no_echoes, no_slices, no_views_2, no_views, no_samples)
    dim : the 6-element raw dimension list (pre-squeeze)
    par : dict of parsed PPR parameters (includes 'scaling')
    """
    with open(filename, "rb") as fid:
        raw = fid.read()

    # ── header ────────────────────────────────────────────────────────────────
    xdim, ydim, zdim, dim4 = struct.unpack_from("<4i", raw, 0)
    datatype_val = struct.unpack_from("<H", raw, 18)[0]
    datatype_hex = format(datatype_val, "X")           # dec2hex (no leading zeros)
    scaling      = struct.unpack_from("<f", raw, 48)[0]
    # bitsperpixel = struct.unpack_from("<B", raw, 52)[0]   # unused
    dim5, dim6   = struct.unpack_from("<2i", raw, 152)

    no_samples = xdim
    no_views   = ydim
    no_views_2 = zdim
    no_slices  = dim4
    no_echoes  = dim5
    no_expts   = dim6

    dim = [no_expts, no_echoes, no_slices, no_views_2, no_views, no_samples]

    # ── data format / complex flag (from the hex string) ────────────────────────
    if len(datatype_hex) > 1:
        only = datatype_hex[1]
        iscomplex = 2
    else:
        only = datatype_hex[0] if datatype_hex else "4"
        iscomplex = 1
    np_dtype, dsize = _DTYPE_MAP.get(only, (np.int32, 4))

    num2read = (no_expts * no_echoes * no_slices *
                no_views_2 * no_views * no_samples * iscomplex)

    # Data begins at byte 512 in the MRD format (256-byte header + 256-byte text)
    data_off = 512
    flat = np.frombuffer(raw, dtype=np_dtype, count=num2read, offset=data_off)
    flat = flat.astype(np.float64, copy=False)

    if iscomplex == 2:
        m_C = flat[0::2] + 1j * flat[1::2]
    else:
        m_C = flat.astype(np.complex128)

    # ── reshape in MATLAB fill order  a,b,c,d(views),e(views_2),samples ─────────
    # MATLAB loop nesting is: expts, echoes, slices, views, views_2, samples
    raw6 = m_C.reshape(no_expts, no_echoes, no_slices,
                       no_views, no_views_2, no_samples)

    # Centric reorder indices (0-based).  ord[d] = destination index of acquired d
    ord_v  = _centric_order(no_views)  if reordering1 == "cen" else np.arange(no_views)
    ord_v2 = _centric_order(no_views_2) if reordering2 == "cen" else np.arange(no_views_2)

    # Place acquired view d at position ord_v[d]; same for views_2
    tmp = np.empty_like(raw6)
    tmp[:, :, :, ord_v, :, :] = raw6
    raw6 = tmp
    if no_views_2 > 1:
        tmp = np.empty_like(raw6)
        tmp[:, :, :, :, ord_v2, :] = raw6
        raw6 = tmp

    # MATLAB stores as (expts, echoes, slices, views_2, views, samples) → swap d/e
    im = np.transpose(raw6, (0, 1, 2, 4, 3, 5))
    im = np.squeeze(im)

    # ── PPR parameter block (after the image data) ──────────────────────────────
    ppr_off = data_off + num2read * dsize
    ppr_bytes = raw[ppr_off:]
    par = _parse_ppr(ppr_bytes, filename)
    par["scaling"] = scaling
    par["_dim_raw"] = dim
    return im, dim, par


def _centric_order(n: int) -> np.ndarray:
    """0-based centric phase-encode reorder (MATLAB 'cen' loop in Get_mrd_3D5)."""
    ord_ = np.arange(n)
    if n >= 2:
        for g in range(1, n // 2 + 1):       # MATLAB g = 1..n/2
            ord_[2 * g - 2] = n // 2 + g - 1   # ord(2g-1) = n/2+g   (1-based) → 0-based
            ord_[2 * g - 1] = n // 2 - g       # ord(2g)   = n/2-g+1 (1-based) → 0-based
    return ord_


def _parse_ppr(ppr_bytes: bytes, filename: str) -> dict:
    """
    Parse the PPR text block.  We only need a handful of numeric fields for CEST
    (``mtc_freq_max``, ``mtc_freq_step``, ``OBSERVE_FREQUENCY``), so this is a
    pragmatic line scanner rather than a full re-implementation of the MATLAB
    keyword table.
    """
    par: dict[str, Any] = {"filename": filename}
    try:
        text = ppr_bytes.decode("latin-1", errors="ignore")
    except Exception:
        return par

    # The first 120 bytes are a sample filename; PPR keyword lines start with ':'
    for line in re.split(r"[\r\n]+", text):
        line = line.strip()
        if not line.startswith(":"):
            continue
        body = line[1:]                       # drop leading ':'

        # :VAR name, value   →  par[name] = value
        m = re.match(r"VAR\s+([A-Za-z_]\w*)\s*,\s*([-+0-9.eE]+)", body)
        if m:
            par[m.group(1)] = _to_num(m.group(2))
            continue

        # :KEYWORD value           or   :KEYWORD name, value
        m = re.match(r"([A-Za-z_]\w*)\s+(.*)$", body)
        if m:
            key, rest = m.group(1), m.group(2).strip()
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", rest)
            if "," in rest and len(nums) >= 1:
                # name, value style — keep last number (matches MATLAB type_2)
                par.setdefault(key, _to_num(nums[-1]))
            elif len(nums) == 1:
                par.setdefault(key, _to_num(nums[0]))
            elif rest and not nums:
                par.setdefault(key, rest)

    # Nucleus / observe frequency convenience
    if "OBSERVE_FREQUENCY" in par and "Nucleus" not in par:
        par["Nucleus"] = str(par["OBSERVE_FREQUENCY"])
    return par


def _to_num(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# CEST / WASSR image formation
# ──────────────────────────────────────────────────────────────────────────────

def read_mrd_cest(filename: str,
                  larmor_mhz: float | None = None,
                  gauss_sigma: float = 0.5) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load an MR Solutions CEST / WASSR .MRD scan and reconstruct magnitude images.

    Mirrors ``CEST_UW_MRsolutions.m``:
        per offset:  abs( ifftshift( ifft2( ifftshift( kspace_2d ) ) ) )
        ppm       :  (mtc_freq_max + i*mtc_freq_step) / Larmor_MHz

    Parameters
    ----------
    filename    : path to the .MRD file
    larmor_mhz  : proton Larmor frequency in MHz.  If None, taken from the PPR
                  ``OBSERVE_FREQUENCY`` field when available, else falls back to
                  199.7502 (the MR Solutions 4.7 T value used in the reference
                  script).
    gauss_sigma : Gaussian smoothing applied to each magnitude image
                  (MATLAB ``imgaussfilt`` default σ = 0.5); set 0 to disable.

    Returns
    -------
    img_4d   : (Y, X, 1, n_offsets) float32 magnitude images
    ppm_all  : (n_offsets,) saturation offsets in ppm
    info     : dict with 'par', 'larmor_mhz', 'mtc_freq_max', 'mtc_freq_step'
    """
    im, dim, par = read_mrd(filename, "cen", "cen")

    # Ensure layout is (offsets, views, samples)
    arr = np.asarray(im)
    if arr.ndim == 2:                     # single offset
        arr = arr[np.newaxis, :, :]
    elif arr.ndim > 3:
        arr = arr.reshape(arr.shape[0], arr.shape[-2], arr.shape[-1])
    n_off = arr.shape[0]

    # Per-offset 2-D inverse FFT reconstruction → magnitude
    recon = np.empty((arr.shape[1], arr.shape[2], n_off), dtype=np.float32)
    for k in range(n_off):
        ksp = np.fft.ifftshift(arr[k])
        img = np.fft.ifftshift(np.fft.ifft2(ksp))
        recon[:, :, k] = np.abs(img).astype(np.float32)

    if gauss_sigma and gauss_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            for k in range(n_off):
                recon[:, :, k] = gaussian_filter(recon[:, :, k], sigma=gauss_sigma)
        except Exception:
            pass

    # (Y, X, slices=1, offsets)
    img_4d = recon[:, :, np.newaxis, :]

    # ── ppm offsets ─────────────────────────────────────────────────────────────
    # MR Solutions .MRD files do not store the Larmor frequency in a usable form
    # (OBSERVE_FREQUENCY is typically "1H 0.0"), so the reference MATLAB script
    # hard-codes 199.7502 MHz (4.7 T).  Use the caller-supplied value when it is
    # physically plausible (> 10 MHz), otherwise fall back to that default.
    if larmor_mhz is None or larmor_mhz <= 10.0:
        larmor_mhz = 199.7502

    fmax  = par.get("mtc_freq_max")
    fstep = par.get("mtc_freq_step")
    if fmax is not None and fstep is not None and np.isfinite(fmax) and np.isfinite(fstep):
        sat_hz  = fmax + np.arange(n_off) * fstep
        ppm_all = np.round(sat_hz / larmor_mhz, 3)
    else:
        # mtc fields missing — fall back to sequential indices so the user can
        # supply a ppm sidecar manually.
        ppm_all = np.arange(n_off, dtype=float)

    info = {
        "par": par,
        "larmor_mhz": larmor_mhz,
        "mtc_freq_max": fmax,
        "mtc_freq_step": fstep,
        "size": img_4d.shape,
    }
    return img_4d, ppm_all.astype(float), info
