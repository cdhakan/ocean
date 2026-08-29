"""
b1_processing.py
================
B1 (transmit field) map computation for GE / Siemens data.

Three independent methods are supported by the T1/T2/B1 tab:

1. Double-angle  (existing `calc_b1_map` in t1t2_tab) — two flip-angle images.
2. Pre-computed   — a single scanner B1 map (e.g. GE Bloch-Siegert product map),
                    normalised to the phantom mean (× 100), matching the MATLAB
                    B1-map normalisation pipeline.
3. Bloch-Siegert  — two BS phase images (+/- off-resonance), calibrated with
                    `calc_b1_bloch_siegert` (port of fitDataBSB1.m).

All maps are returned in **percent of the nominal flip angle** (100 % = on target).
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Bloch-Siegert calibration  (port of fitDataBSB1.m)
# ─────────────────────────────────────────────────────────────────────────────

def calc_b1_bloch_siegert(
    phase_pos: np.ndarray,
    phase_neg: np.ndarray,
    flip_angle: float,
    duration: float,
    pulse_type: str = "fermi",
    raw_phase_scale: float = 4096.0,
) -> np.ndarray:
    """
    Bloch-Siegert B1 map from the +/- off-resonance phase images.

    Mirrors fitDataBSB1.m:
        phase  = raw / 4096 * pi
        dPhi   = |phase_pos - phase_neg|
        B1[G]  = (dPhi<pi)·sqrt(dPhi/2/Kbs) + (dPhi>=pi)·sqrt(|dPhi-2pi|/2/Kbs)
        B1[rad]= (gamma·AmpInt/512·duration)·B1[G]
        B1[deg]= B1[rad]/pi·180
        B1     = B1[deg] / flip_angle          (relative to nominal)

    Parameters
    ----------
    phase_pos, phase_neg : raw BS phase images (integer/scaled, same shape)
    flip_angle           : nominal flip angle of the BS pulse (degrees)
    duration             : BS pulse duration (ms)
    pulse_type           : 'fermi' or 'gauss'
    raw_phase_scale      : raw→radians divisor (GE default 4096 → 0..pi)

    Returns
    -------
    b1_percent : B1 map in percent of the nominal flip angle (100 % = on target)
    """
    pos = np.asarray(phase_pos, dtype=float) / raw_phase_scale * np.pi
    neg = np.asarray(phase_neg, dtype=float) / raw_phase_scale * np.pi

    pt = pulse_type.lower()
    if pt == "fermi":
        Kbs, gamma, AmpInt = 74.01, 26745.0, 356.259361   # rad/G^2/ms, rad/G, —
    elif pt in ("gauss", "gaussian"):
        Kbs, gamma, AmpInt = 39.4, 26747.0, 247.9
    else:
        raise ValueError(f"Unknown BS pulse_type '{pulse_type}' (use 'fermi' or 'gauss').")

    d = np.abs(pos - neg)
    # B1 in Gauss — branch on the pi phase-wrap
    b1_gauss = np.where(
        d < np.pi,
        np.sqrt(np.abs(d) / 2.0 / Kbs),
        np.sqrt(np.abs(d - 2.0 * np.pi) / 2.0 / Kbs),
    )
    b1_rad = (gamma * AmpInt / 512.0 * duration) * b1_gauss   # radians
    b1_deg = b1_rad / np.pi * 180.0                           # degrees
    b1_rel = b1_deg / float(flip_angle)                       # fraction of nominal
    return b1_rel * 100.0                                     # → percent


# ─────────────────────────────────────────────────────────────────────────────
# Phantom auto-detection  (port of the MATLAB Otsu pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def _otsu_threshold(img01: np.ndarray, nbins: int = 256) -> float:
    """Otsu threshold for an image already scaled to [0, 1]."""
    hist, edges = np.histogram(img01[np.isfinite(img01)], bins=nbins, range=(0.0, 1.0))
    hist = hist.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w = np.cumsum(hist)
    if w[-1] == 0:
        return 0.5
    wb = w
    wf = w[-1] - w
    cum = np.cumsum(hist * centers)
    mb = np.divide(cum, wb, out=np.zeros_like(cum), where=wb > 0)
    mf = np.divide(cum[-1] - cum, wf, out=np.zeros_like(cum), where=wf > 0)
    between = wb * wf * (mb - mf) ** 2
    return float(centers[int(np.argmax(between))])


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= radius * radius


def detect_phantom_mask(img: np.ndarray, smooth_sigma: float = 4.0,
                        open_radius: int = 5) -> np.ndarray:
    """
    Auto-detect the phantom outline (largest bright object).

    Mirrors the MATLAB pipeline:
      imgaussfilt(4) → mat2gray → imbinarize(Otsu) → largest connected
      component → imfill(holes) → imopen(disk 5).
    """
    from scipy.ndimage import (gaussian_filter, binary_fill_holes,
                               binary_opening, label)

    a = np.asarray(img, dtype=float)
    if a.ndim == 3:
        a = a[..., 0]
    sm = gaussian_filter(a, smooth_sigma)
    rng = sm.max() - sm.min()
    smn = (sm - sm.min()) / (rng + 1e-12)               # mat2gray → [0,1]
    bw = smn > _otsu_threshold(smn)                     # Otsu threshold

    lbl, n = label(bw)
    if n >= 1:
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0                                    # ignore background
        bw = lbl == int(np.argmax(sizes))               # largest component
    bw = binary_fill_holes(bw)                          # fill holes
    if open_radius and open_radius > 0:
        bw = binary_opening(bw, structure=_disk(open_radius))
    return bw


# ─────────────────────────────────────────────────────────────────────────────
# Pre-computed B1 map — normalise to phantom mean
# ─────────────────────────────────────────────────────────────────────────────

def calc_b1_precomputed(b1_map: np.ndarray,
                        mask: np.ndarray | None = None,
                        auto_phantom: bool = True,
                        normalize: bool = True,
                        scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Turn a scanner-produced B1 map into a percentage map.

    Two modes:
      • normalize=True  (relative, mirrors the MATLAB pipeline):
            B1% = B1map / mean(B1map inside phantom) * 100
        Forces the phantom mean to 100 % — shows spatial inhomogeneity but is
        NOT absolute.
      • normalize=False (absolute, scale-factor):
            B1% = B1map * scale
        For maps already stored as a (scaled) percentage of nominal flip angle,
        e.g. GE `2db1map` stores value/2 = % → use scale = 0.5.
    Outside the phantom mask the result is NaN.

    Parameters
    ----------
    b1_map       : single B1 map (scanner units), 2-D (or (Y,X,1))
    mask         : optional phantom mask; if None and auto_phantom, detected.
    auto_phantom : run Otsu phantom detection when no mask is given.

    Returns
    -------
    (b1_percent, mask) : normalised B1 map (% of phantom mean, NaN outside) and
                         the phantom mask used.
    """
    a = np.asarray(b1_map, dtype=float)
    if a.ndim == 3:
        a = a[..., 0]
    if mask is None:
        mask = detect_phantom_mask(a) if auto_phantom else (np.isfinite(a) & (a != 0))
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != a.shape:
        mask = np.ones_like(a, dtype=bool)

    if normalize:
        inside = a[mask]
        inside = inside[np.isfinite(inside)]
        ref = float(np.mean(inside)) if inside.size else 1.0
        if ref == 0 or not np.isfinite(ref):
            ref = 1.0
        b1_pct = a / ref * 100.0
    else:
        b1_pct = a * float(scale)
    b1_pct[~mask] = np.nan
    return b1_pct, mask
