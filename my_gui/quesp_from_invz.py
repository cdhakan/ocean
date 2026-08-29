"""
quesp_from_invz.py
==================
Spillover-corrected QUESP from inverse-Z (1/Z) CEST peak fits.

Pipeline (mirrors MT_CEST_fit_QUESP_dk.m):
  1. In the 1/Z tab, each B1's Z-spectrum is MT-corrected and the CEST pools
     are fitted in R1·(1/Z−1) space → per-pool peak amplitude A_pool(B1).
     A_pool ≈ R1 · MTR_Rex(B1)  for that pool (spillover/MT removed).
  2. Accumulate A_pool across B1 (one B1 at a time) into a "QUESP stack".
  3. Linear QUESP per pool:  MTR_Rex = A/R1, then
         1/MTR_Rex = a·(1/w1²) + b
         slope a = ksw/fs,  intercept b = 1/(fs·ksw)
         → ksw = √(a/b),    fs = 1/√(a·b)
     (w1 = B1[µT]·42.577·2π  rad/s).

This module is UI-agnostic: build/save a stack in the 1/Z tab, load + fit in
the QUESP tab.
"""
from __future__ import annotations

import numpy as np

_GAMMA_HZ_UT = 42.577          # ¹H gyromagnetic ratio, Hz/µT


def w1_rad_per_s(b1_uT) -> np.ndarray:
    """Saturation field B1 (µT) → ω1 (rad/s)."""
    return np.asarray(b1_uT, dtype=float) * _GAMMA_HZ_UT * 2.0 * np.pi


def linear_quesp(b1_uT, mtr_rex, *, min_points: int = 3):
    """
    Linear inverse-QUESP fit of MTR_Rex vs B1.

    Parameters
    ----------
    b1_uT   : (n,) saturation amplitudes (µT)
    mtr_rex : (n,) MTR_Rex values at each B1 (= A_pool / R1)

    Returns
    -------
    dict {fs, ksw, slope, intercept, r2, n} — fs (proton fraction), ksw (Hz=s⁻¹).
    Returns NaNs if the fit is ill-posed (too few points / non-positive a,b).
    """
    b1  = np.asarray(b1_uT, dtype=float).ravel()
    rex = np.asarray(mtr_rex, dtype=float).ravel()
    ok  = np.isfinite(b1) & np.isfinite(rex) & (b1 > 0) & (rex > 1e-9)
    b1, rex = b1[ok], rex[ok]
    nan = dict(fs=np.nan, ksw=np.nan, slope=np.nan, intercept=np.nan, r2=np.nan, n=int(b1.size))
    if b1.size < min_points:
        return nan

    w1 = w1_rad_per_s(b1)
    x  = 1.0 / w1 ** 2
    y  = 1.0 / rex
    a, b = np.polyfit(x, y, 1)               # y = a·x + b
    if a <= 0 or b <= 0:
        return {**nan, "slope": float(a), "intercept": float(b)}

    ksw = float(np.sqrt(a / b))
    fs  = float(1.0 / np.sqrt(a * b))
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(np.clip(1.0 - ss_res / (ss_tot + 1e-30), 0.0, 1.0))
    return dict(fs=fs, ksw=ksw, slope=float(a), intercept=float(b), r2=r2, n=int(b1.size))


def nonlinear_quesp(b1_uT, mtr_rex, *, min_points: int = 3):
    """
    Non-linear QUESP fit of the CW model directly to MTR_Rex:

        MTR_Rex(w1) = fs·ksw·w1² / (w1² + ksw²)

    More robust than `linear_quesp` when the B1 range sits well below the
    exchange-rate plateau (fast exchange / low B1) — where the linearised
    1/MTR_Rex vs 1/w1² intercept can go non-positive and yield NaNs.

    Returns dict {fs, ksw, r2, n}; NaNs if ill-posed.
    """
    from scipy.optimize import least_squares
    b1  = np.asarray(b1_uT, dtype=float).ravel()
    rex = np.asarray(mtr_rex, dtype=float).ravel()
    ok  = np.isfinite(b1) & np.isfinite(rex) & (b1 > 0) & (rex > 1e-12)
    b1, rex = b1[ok], rex[ok]
    nan = dict(fs=np.nan, ksw=np.nan, r2=np.nan, n=int(b1.size))
    if b1.size < min_points:
        return nan
    w1 = w1_rad_per_s(b1)

    def model(p):
        fs, ksw = p
        return fs * ksw * w1 ** 2 / (w1 ** 2 + ksw ** 2)

    try:
        res = least_squares(lambda p: model(p) - rex,
                            x0=[float(max(rex) * 2.0), float(np.median(w1))],
                            bounds=([1e-9, 10.0], [1.0, 1e5]),
                            method='trf', max_nfev=5000)
        fs, ksw = float(res.x[0]), float(res.x[1])
    except Exception:
        return nan
    yhat = model([fs, ksw])
    ss_res = float(np.sum((rex - yhat) ** 2))
    ss_tot = float(np.sum((rex - rex.mean()) ** 2))
    r2 = float(np.clip(1.0 - ss_res / (ss_tot + 1e-30), 0.0, 1.0))
    return dict(fs=fs, ksw=ksw, r2=r2, n=int(b1.size))


def fs_to_concentration(fs: float, n_exchangeable_H: int = 1,
                        water_proton_M: float = 110_000.0) -> float:
    """proton fraction fs → solute concentration (mM)."""
    if not np.isfinite(fs) or n_exchangeable_H <= 0:
        return np.nan
    return fs * water_proton_M / n_exchangeable_H


def forward_mtr_rex_cw(b1_uT, fs: float, ksw: float) -> np.ndarray:
    """CW MTR_Rex forward model — for tests / overlay: fs·ksw·w1²/(w1²+ksw²)."""
    w1 = w1_rad_per_s(b1_uT)
    return fs * ksw * w1 ** 2 / (w1 ** 2 + ksw ** 2)
