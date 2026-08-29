"""
wasabi_fit.py
=============
WASABI (Water Shift And B1) simultaneous B0 + B1 mapping.

Ported from CEST_EVAL (Schuenke et al., MRM 2017) — the levmar_fit WASABI
routines ``WASABIFIT`` / ``WASABIFIT_2`` and ``fitmodelfunc_NUM`` (model
numbers 021011 / 021021).

Signal model (4-parameter, WASABIFIT_2)::

    Z(w) = | c - af * sin^2( atan( (B1*gamma/FREQ) / (w - dB0) ) )
                    * sin^2( sqrt( (B1*gamma/FREQ)^2 + (w - dB0)^2 )
                             * FREQ * 2*pi * tp / 2 ) |

Units (from CEST_EVAL ``wipread_modified.m``):
    w      offset axis, ppm          (ACQ_O2_list / ImagingFrequency)
    FREQ   Larmor frequency, MHz     (ImagingFrequency)
    tp     saturation pulse length, seconds (PVM_MagTransPulse1 / 1000)
    B1     saturation amplitude, uT  (fitted); nominal = PVM_MagTransPower
    gamma  42.576375 MHz/T (== Hz/uT)
    dB0    B0 shift, ppm             (fitted)
    c, af  baseline / modulation scaling (fitted)

Maps (from ``main_CEST_bruker.m``)::
    B1map        = B1_fitted / B1_nominal        (relative B1; ×100 → %)
    dB0_stack    = dB0_fitted                     (ppm)
"""
from __future__ import annotations

import numpy as np

GAMMA = 42.576375   # MHz/T == Hz/uT  (CEST_EVAL tools/gamma_.m)


def wasabi_model(w, B1, dB0, c, af, freq_mhz, tp_s, gamma: float = GAMMA):
    """WASABI Z-spectrum (4-parameter WASABIFIT_2). ``w`` in ppm → Z array."""
    w = np.asarray(w, dtype=float)
    gb1 = B1 * gamma / freq_mhz          # B1 expressed in ppm
    d = w - dB0
    # sin^2(atan(gb1/d)) — arctan2 is the safe, div-by-zero-free equivalent
    # (sin^2 is identical for atan(gb1/d) and atan2(gb1, d) across all d).
    s_theta = np.sin(np.arctan2(gb1, d)) ** 2
    phi = np.sqrt(gb1 ** 2 + d ** 2) * freq_mhz * 2.0 * np.pi * tp_s / 2.0
    return np.abs(c - af * s_theta * np.sin(phi) ** 2)


def wasabi_model_3p(w, B1, dB0, c, freq_mhz, tp_s, gamma: float = GAMMA):
    """WASABI Z-spectrum (3-parameter WASABIFIT):  c*|1 - 2*sin^2*sin^2|."""
    w = np.asarray(w, dtype=float)
    gb1 = B1 * gamma / freq_mhz
    d = w - dB0
    s_theta = np.sin(np.arctan2(gb1, d)) ** 2
    phi = np.sqrt(gb1 ** 2 + d ** 2) * freq_mhz * 2.0 * np.pi * tp_s / 2.0
    return c * np.abs(1.0 - 2.0 * s_theta * np.sin(phi) ** 2)


def _bounds_and_p0(z, w, b1_nom, freq_mhz, tp_s, model):
    """Start values + bounds (mirrors fitmodelfunc_NUM 021021 / 021011), with the
    B1 start seeded from the nominal amplitude and dB0 from the spectral dip."""
    z = np.asarray(z, dtype=float)
    dip = float(w[int(np.nanargmin(z))]) if z.size else 0.0
    dip = float(np.clip(dip, -1.5, 1.5))
    # off-resonance plateau ≈ c
    c0 = float(np.clip(np.nanmedian(z[np.abs(w) >= 0.75 * np.nanmax(np.abs(w))]
                                    if w.size else z), 0.05, 1.2))
    if not np.isfinite(c0) or c0 <= 0:
        c0 = float(np.clip(np.nanmax(z), 0.05, 1.2)) if z.size else 0.5
    b1_hi = max(20.0, 4.0 * float(b1_nom))
    if model == "3param":
        #        B1        dB0    c
        lb = [0.0,       -2.0,   0.0]
        ub = [b1_hi,      2.0,   6.0]
        p0 = [float(b1_nom), dip, max(c0, 0.1)]
    else:  # 4param (WASABIFIT_2)
        #        B1        dB0    c     af
        lb = [0.0,       -2.0,   0.0,  0.0]
        ub = [b1_hi,      2.0,   1.2,  2.0]
        p0 = [float(b1_nom), dip, min(c0, 1.1), 1.2]
    return (np.array(p0), (np.array(lb), np.array(ub)))


def build_wasabi_lookup(w, freq_mhz, tp_s, b1_nom, model="4param"):
    """Pre-compute a coarse (dB0, B1) start-value grid (mirrors CEST_EVAL
    ``lookuptable_WASABI`` / ``perform_lookup``). Returns (grid_z, grid_p) where
    grid_z is (K, n_off) model spectra and grid_p is (K, 4) start parameters.
    Built ONCE per fit and reused for every voxel."""
    db0_grid = np.round(np.arange(-0.9, 0.9 + 1e-9, 0.1), 4)
    b1_grid = np.array([0.6, 0.8, 1.0, 1.2, 1.4]) * float(b1_nom)
    c0, af0 = 0.9, 1.3
    zs, ps = [], []
    for b1 in b1_grid:
        for d in db0_grid:
            if model == "3param":
                zs.append(wasabi_model_3p(w, b1, d, 1.0, freq_mhz, tp_s))
                ps.append([b1, d, 1.0, 2.0])
            else:
                zs.append(wasabi_model(w, b1, d, c0, af0, freq_mhz, tp_s))
                ps.append([b1, d, c0, af0])
    return np.asarray(zs), np.asarray(ps)


def fit_wasabi_voxel(z, w, freq_mhz, tp_s, b1_nom, model="4param",
                     max_nfev=400, lookup=None):
    """Fit one voxel's WASABI Z-spectrum. Returns (B1, dB0, c, af, rmse).

    ``lookup`` = (grid_z, grid_p) from build_wasabi_lookup for a robust start
    (recommended); if None a single dip-based seed is used."""
    from scipy.optimize import least_squares

    z = np.asarray(z, dtype=float)
    if not np.all(np.isfinite(z)) or z.size < (4 if model == "3param" else 5):
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    p0, (lb, ub) = _bounds_and_p0(z, w, b1_nom, freq_mhz, tp_s, model)

    # Pick the best start from the lookup grid (nearest model spectrum)
    if lookup is not None:
        grid_z, grid_p = lookup
        k = int(np.argmin(np.abs(grid_z - z[None, :]).sum(axis=1)))
        seed = grid_p[k]
        p0 = np.array([seed[0], seed[1], p0[2], seed[3]] if model != "3param"
                      else [seed[0], seed[1], p0[2]])
        p0 = np.clip(p0, lb, ub)

    if model == "3param":
        def resid(p):
            return wasabi_model_3p(w, p[0], p[1], p[2], freq_mhz, tp_s) - z
    else:
        def resid(p):
            return wasabi_model(w, p[0], p[1], p[2], p[3], freq_mhz, tp_s) - z

    try:
        res = least_squares(resid, p0, bounds=(lb, ub), method="trf",
                            max_nfev=max_nfev, ftol=1e-8, xtol=1e-8)
        p = res.x
        rmse = float(np.sqrt(np.mean(res.fun ** 2)))
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    if model == "3param":
        return (float(p[0]), float(p[1]), float(p[2]), 2.0, rmse)
    return (float(p[0]), float(p[1]), float(p[2]), float(p[3]), rmse)


def prep_wasabi(img4d, ppm, m0=None, m0_thresh=30.0, ref_pct=95.0):
    """Split off far-offset M0/reference frames and normalise a WASABI stack.

    Returns (z, ppm_fit, ref_img):
      * frames whose |offset| exceeds ``m0_thresh`` (e.g. the −300 ppm M0 frame
        in a Pulseq-CEST WASABI list) are REMOVED from the fit offsets — they are
        not part of the z-spectrum shape;
      * with an explicit ``m0`` (Y,X,slices) it is used as the S0 reference;
      * otherwise each voxel is normalised by its own off-resonance PLATEAU (the
        ``ref_pct`` percentile of its WASABI frames), which is the effective
        unsaturated level (Z→c off-resonance). This is robust even when a
        dedicated M0 frame is dimmer than the flanks (as happens with some
        Pulseq WASABI M0/Trec_M0 acquisitions), keeping Z within the model's
        c≤1.2 / af≤2 range so the fit is well-posed.
    """
    img4d = np.asarray(img4d, dtype=float)
    ppm = np.asarray(ppm, dtype=float).ravel()
    n = img4d.shape[-1]
    if len(ppm) != n:
        raise ValueError(f"offsets ({len(ppm)}) must match the {n} acquired frames.")

    keep = np.abs(ppm) <= float(m0_thresh)
    if not keep.any():
        keep = np.ones(n, dtype=bool)
    z_frames = img4d[:, :, :, keep]                    # (Y,X,slices,n_fit)

    if m0 is not None:
        s0 = np.asarray(m0, dtype=float)
        if s0.ndim == 2:
            s0 = s0[:, :, None]
        ref = s0
    else:
        ref = np.percentile(z_frames, ref_pct, axis=3)  # (Y,X,slices) plateau
    z = z_frames / (ref[:, :, :, None] + 1e-9)
    return np.clip(z, 0.0, 2.0), ppm[keep], ref


def normalize_wasabi(img4d, ppm, m0=None):
    """Z = S/S0 for a WASABI stack (Y,X,slices,offsets).

    If ``m0`` (Y,X,slices) is given it is used as S0; otherwise the frame at the
    largest |offset| (most unsaturated) is used per voxel."""
    img4d = np.asarray(img4d, dtype=float)
    ppm = np.asarray(ppm, dtype=float).ravel()
    if m0 is not None:
        s0 = np.asarray(m0, dtype=float)
        if s0.ndim == 2:
            s0 = s0[:, :, None]
        s0 = s0[:, :, :, None]
    else:
        i0 = int(np.argmax(np.abs(ppm)))
        s0 = img4d[:, :, :, i0:i0 + 1]
    z = img4d / (s0 + 1e-9)
    return np.clip(z, 0.0, 2.0)
