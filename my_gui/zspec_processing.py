"""
zspec_processing.py
Python equivalent of the MATLAB z-spectroscopy pipeline.

Exactly mirrors:
  - B0correction.m       (per-voxel Akima / makima interpolation)
  - calcMTRmap.m         (MTR asymmetry)
  - zspecMultiPeakFit.m  (Pseudo-Voigt peak model)
  - fitAllZspec.m        (voxelwise fitting, same tolerances as MATLAB)
  - zspecSetPVPeakBounds.m / zspecSetLPeakBounds.m

MATLAB reference (zSpec_load_proc.m):
    zppars.pools     = {'water','NOE','MT','amide'}   # 4 pools, amide @ 3.0 ppm
    zppars.peaktype  = 'Pseudo-Voigt'                 # PV only
    zppars.water1st  = false                          # single simultaneous pass

MATLAB tolerances (zspecMultiPeakFit.m → lsqnonlin):
    FunctionTolerance = 1e-12
    StepTolerance     = 1e-12
    MaxFunctionEvaluations = 6000
    MaxIterations          = 4000

Public API:
    b0_correction(b0_map_ppm, ppm, z_img)  -> corrected z_img
    calc_mtr_map(z_img, ppm, sel_ppm)      -> mtr_map, sel_ppm_true
    fit_zspec_single(ppm, z_vox, ...)      -> params, indiv_peaks, sum_peak
    fit_all_zspec(ppm, z_img, ...)         -> ampl_maps, indiv_maps, sum_maps
"""

from __future__ import annotations

import multiprocessing as _mp
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
from scipy.optimize import least_squares


# ─────────────────────────────────────────────────────────────────────────────
# Peak model functions — exact translation of MATLAB formulas
# ─────────────────────────────────────────────────────────────────────────────

def _pseudo_voigt(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Single Pseudo-Voigt peak — exact match to MATLAB zspecSinglePseudoVoigtModel.

    Parameters (p):
        [0] A        — peak amplitude
        [1] alpha    — Gaussian proportion (0=pure Lorentzian, 1=pure Gaussian)
        [2] FWHMl    — Lorentzian FWHM in ppm
        [3] FWHMrat  — Gaussian:Lorentzian FWHM ratio (constrained 1–2)
        [4] omega_0  — displacement from water (ppm)
        [5] phase    — zero-order phase term (rad); fixed at 0 in MATLAB

    MATLAB formula (zspecSinglePseudoVoigtModel.m):
        sigma = p(3)/2/sqrt(2*log(2)) * p(4)
        G     = exp(-(x-p(5)).^2 / 2 / sigma^2)
        Lnum  = sqrt((p(3)/2)^2 + (x-p(5)).^2) * p(3)/2
        Lden  = (x-p(5)).^2 + (p(3)/2)^2
        Lph   = exp(-1i*(atan((x-p(5))/(p(3)/2)) + p(6)))
        L     = Lnum./Lden .* Lph
        Y     = p(1) * real(p(2)*G + (1-p(2))*L)

    With phase=0 this simplifies to standard Gaussian + standard Lorentzian.
    """
    A       = float(p[0]) if len(p) > 0 else 0.0
    alpha   = float(p[1]) if len(p) > 1 else 0.3
    fwhm_l  = float(p[2]) if len(p) > 2 else 1.0
    fwhm_r  = float(p[3]) if len(p) > 3 else 1.0
    offset  = float(p[4]) if len(p) > 4 else 0.0
    phase   = float(p[5]) if len(p) > 5 else 0.0

    d     = x - offset
    half  = fwhm_l / 2.0

    # Gaussian component
    sigma = half / np.sqrt(2.0 * np.log(2.0)) * fwhm_r
    G     = np.exp(-(d ** 2) / (2.0 * sigma ** 2))

    # Lorentzian component — matches MATLAB atan formulation
    # With phase=0: real(L) = half^2 / (half^2 + d^2)  [standard Lorentzian]
    L_mag  = np.sqrt(half ** 2 + d ** 2) * half / (half ** 2 + d ** 2)
    L_ang  = np.arctan(d / half) + phase          # atan(d/half) like MATLAB
    L_real = L_mag * np.cos(L_ang)

    return A * (alpha * G + (1.0 - alpha) * L_real)


def _lorentzian(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Single Lorentzian peak — exact match to MATLAB zspecSingleLorentzianModel.

    Parameters (p):
        [0] A      — amplitude
        [1] FWHM   — full-width half-maximum in ppm
        [2] offset — displacement from water (ppm)
        [3] phase  — zero-order phase (rad); fixed at 0 in MATLAB

    MATLAB formula:
        num = sqrt((p(2)/2)^2 + (x-p(3))^2) * (p(2)/2)
        den = (p(2)/2)^2 + (x-p(3))^2
        ph  = exp(-1i*p(4))
        Y   = p(1)*real(num/den*ph) - min(real(...))

    With phase=0: Y = A * half/sqrt(half^2 + d^2)
    (Note: this is NOT the classic Lorentzian; it falls off as 1/|x|, not 1/x^2)
    """
    A      = float(p[0]) if len(p) > 0 else 0.0
    fwhm   = float(p[1]) if len(p) > 1 else 1.0
    offset = float(p[2]) if len(p) > 2 else 0.0
    phase  = float(p[3]) if len(p) > 3 else 0.0

    half = fwhm / 2.0
    d    = x - offset
    # real(num/den) = half/sqrt(half^2 + d^2) when phase=0
    mag  = np.sqrt(half ** 2 + d ** 2) * half / (half ** 2 + d ** 2)
    y    = A * mag * np.cos(phase)  # with phase=0: y = A*half/sqrt(half^2+d^2)
    return y - np.min(y)            # subtract min so baseline stays at 0


def _gaussian_simple(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Simple 3-parameter Gaussian peak — fast voxelwise model.
    Parameters: [0] A — amplitude, [1] FWHM — width (ppm), [2] x0 — centre (ppm)
    f(x) = A * exp(-4·ln2·(x−x0)²/FWHM²)
    """
    A   = float(p[0]) if len(p) > 0 else 0.0
    w   = max(abs(float(p[1])), 1e-6) if len(p) > 1 else 1.0
    x0  = float(p[2]) if len(p) > 2 else 0.0
    return A * np.exp(-4.0 * np.log(2.0) * (x - x0) ** 2 / (w * w))


def _simple_lorentzian(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Standard 3-param Lorentzian [A, FWHM, offset].
    f(x) = A * (FWHM/2)^2 / ((FWHM/2)^2 + (x-offset)^2)
    Peak = A; goes to 0 at far off-resonance. Used for inv-Z / MTRRex.
    """
    A      = float(p[0]) if len(p) > 0 else 0.0
    fwhm   = float(p[1]) if len(p) > 1 else 1.0
    offset = float(p[2]) if len(p) > 2 else 0.0
    half2  = (fwhm / 2.0) ** 2
    return A * half2 / (half2 + (x - offset) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Peak bounds — directly from MATLAB zspecSetPVPeakBounds.m / zspecSetLPeakBounds.m
# Parameters: [A, alpha, FWHMl, FWHMrat, omega_0, phase]  (Pseudo-Voigt)
#             [A, FWHM, offset, phase]                      (Lorentzian)
# Phase is always FIXED at 0 (lb=ub=0) matching MATLAB
# ─────────────────────────────────────────────────────────────────────────────

# ── Pseudo-Voigt bounds — copied exactly from zspecSetPVPeakBounds.m ──────
_PSEUDOVOIGT_BOUNDS: dict[str, dict] = {
    # key: "water" — large symmetrical pool at 0 ppm
    "water": {
        "st": [0.90, 0.3, 1.4,  1.0,  0.0,  0.0],
        "lb": [0.30, 0.0, 0.3,  1.0, -0.2,  0.0],
        "ub": [1.00, 1.0, 3.0,  2.0,  0.2,  0.0],
    },
    # key: "NOE" — NOE/rNOE at −3.5 ppm
    # omega_0 ub extended to −1.5 ppm (Pre-CAT reference: allows broader NOE detection
    # including relayed NOE / aliphatic peaks closer to water).
    "NOE": {
        "st": [0.10, 0.3, 1.5,  1.0, -3.5,  0.0],
        "lb": [0.00, 0.0, 1.0,  1.0, -4.0,  0.0],
        "ub": [0.40, 1.0, 6.0,  2.0, -1.5,  0.0],
    },
    # key: "MT" — magnetisation transfer (broad, centred near 0 ppm)
    # FWHMl ub extended to 60 ppm (CEST-master uses 100, Pre-CAT uses 60; 50 was
    # too narrow for very broad MT lines in some tissues at low B0 fields).
    "MT": {
        "st": [0.30, 0.3, 20.0, 1.0,  0.0,  0.0],
        "lb": [0.00, 0.0, 15.0, 1.0,  0.0,  0.0],
        "ub": [0.60, 1.0, 60.0, 2.0,  1.0,  0.0],
    },
    # key: "amide" — amide NH pool at 3.5 ppm (CEST amide)
    # Bounds from MATLAB setPVPeakBounds.m: omega_0 centred at 3.5 ppm [3.0–4.0]
    "amide": {
        "st": [0.025, 0.3, 1.0,  1.0,  3.5,  0.0],
        "lb": [0.000, 0.0, 0.2,  1.0,  3.25, 0.0],
        "ub": [0.800, 1.0, 4.0,  2.0,  3.80, 0.0],
    },
    # key: "amine" — amine NH pool at 3.0 ppm (distinct from amide at 3.5 ppm)
    "amine": {
        "st": [0.025, 0.3, 1.0,  1.0,  3.0,  0.0],
        "lb": [0.000, 0.0, 0.2,  1.0,  2.80, 0.0],
        "ub": [0.800, 1.0, 4.0,  2.0,  3.20, 0.0],
    },
    # key: "OH" — hydroxyl pool at ~0.8 ppm
    "OH": {
        "st": [0.010, 0.3, 1.5,  1.0,  0.8,  0.0],
        "lb": [0.000, 0.0, 1.0,  1.0,  0.6,  0.0],
        "ub": [0.500, 1.0, 5.5,  2.0,  1.0,  0.0],
    },
    # key: "trp" — Tryptophan indole N-H at ~5.4 ppm (broad, FWHM ~1–2 ppm)
    # Bounds from zspecSetPVPeakBounds.m (Trp PV entry) — NOT the Lorentzian table
    "trp": {
        "st": [0.010, 0.3,  1.0, 1.0,  5.4,  0.0],
        "lb": [0.000, 0.0,  0.5, 1.0,  5.1,  0.0],
        "ub": [0.500, 1.0,  2.0, 2.0,  5.7,  0.0],
    },
    # key: "ppm4pt4" — 4.4 ppm pool
    # Bounds from zspecSetPVPeakBounds.m (ppm4pt4 PV entry)
    "ppm4pt4": {
        "st": [0.025, 0.3, 1.0,  1.0,  4.5,  0.0],
        "lb": [0.000, 0.0, 0.2,  1.0,  4.0,  0.0],
        "ub": [0.800, 1.0, 1.5,  2.0,  5.0,  0.0],
    },
    # key: "ppm7pt3" — aromatic/downfield pool at ~7.3 ppm
    # Bounds from setPVPeakBounds.m (ppm7pt3 entry)
    "ppm7pt3": {
        "st": [0.025, 0.3, 1.0,  1.0,  7.3,  0.0],
        "lb": [0.000, 0.0, 0.2,  1.0,  6.5,  0.0],
        "ub": [0.800, 1.0, 1.5,  2.0,  8.0,  0.0],
    },
    # key: "ppm9pt8" — aromatic/downfield pool at ~9.8 ppm
    # Bounds from setPVPeakBounds.m (ppm9pt8 entry)
    "ppm9pt8": {
        "st": [0.025, 0.3, 1.0,  1.0, 10.0,  0.0],
        "lb": [0.000, 0.0, 0.2,  1.0,  9.0,  0.0],
        "ub": [0.800, 1.0, 3.0,  2.0, 11.0,  0.0],
    },
    # key: "guanidinium" — guanidinium NH pool at ~2.0 ppm
    "guanidinium": {
        "st": [0.025, 0.3, 1.0,  1.0,  2.0,  0.0],
        "lb": [0.000, 0.0, 0.2,  1.0,  1.5,  0.0],
        "ub": [0.300, 1.0, 4.0,  2.0,  2.5,  0.0],
    },
    # key: "poly_l_lysine" — Poly-L-Lysine –NH3+ side-chain amine at ~3.7 ppm (broad)
    "poly_l_lysine": {
        "st": [0.030, 0.3, 2.0,  1.0,  3.6,  0.0],
        "lb": [0.000, 0.0, 0.8,  1.0,  3.3,  0.0],
        "ub": [0.400, 1.0, 6.0,  2.0,  3.9,  0.0],
    },
    # key: "glucose" — glucose OH at ~1.2 ppm
    "glucose": {
        "st": [0.020, 0.3, 0.8,  1.0,  1.2,  0.0],
        "lb": [0.000, 0.0, 0.2,  1.0,  0.8,  0.0],
        "ub": [0.200, 1.0, 3.0,  2.0,  1.6,  0.0],
    },
    # key: "creatine" — creatine guanidinium NH at ~1.9 ppm
    "creatine": {
        "st": [0.020, 0.3, 0.8,  1.0,  1.9,  0.0],
        "lb": [0.000, 0.0, 0.2,  1.0,  1.5,  0.0],
        "ub": [0.200, 1.0, 3.0,  2.0,  2.3,  0.0],
    },
    # key: "taurine" — taurine NH at ~3.2 ppm
    "taurine": {
        "st": [0.020, 0.3, 0.8,  1.0,  3.2,  0.0],
        "lb": [0.000, 0.0, 0.2,  1.0,  2.8,  0.0],
        "ub": [0.200, 1.0, 3.0,  2.0,  3.6,  0.0],
    },
    # key: "iopamidol_4.2" — Iopamidol amide proton exchange peak at ~4.2 ppm
    # Centre bounds tightened to 4.0–4.75 ppm (start 4.2) for the 0.25-ppm grid.
    "iopamidol_4.2": {
        "st": [0.025, 0.3, 1.0,  1.0,  4.2,  0.0],
        "lb": [0.000, 0.0, 0.2,  1.0,  4.00, 0.0],
        "ub": [0.500, 1.0, 3.0,  2.0,  4.75, 0.0],
    },
    # key: "iopamidol_5.5" — Iopamidol amide proton exchange peak at ~5.5 ppm
    # Centre bounds tightened to 5.25–5.75 ppm (start 5.5) for the 0.25-ppm grid.
    "iopamidol_5.5": {
        "st": [0.025, 0.3, 1.0,  1.0,  5.5,  0.0],
        "lb": [0.000, 0.0, 0.2,  1.0,  5.25, 0.0],
        "ub": [0.500, 1.0, 3.0,  2.0,  5.75, 0.0],
    },
}
_FALLBACK_PV = {
    "st": [0.05, 0.3, 1.0,  1.0,  0.0,  0.0],
    "lb": [0.00, 0.0, 0.2,  1.0, -10.0, 0.0],
    "ub": [0.50, 1.0, 10.0, 2.0,  10.0, 0.0],
}

# ── Lorentzian bounds — copied exactly from zspecSetLPeakBounds.m ──────────
# Parameters: [A, FWHM, offset, phase]  — phase fixed at 0
_LORENTZ_BOUNDS: dict[str, dict] = {
    # water: A lb=0.02, FWHM ub=10.0 from MATLAB setLPeakBounds.m
    "water": {
        "st": [0.90, 1.4,  0.0, 0.0],
        "lb": [0.02, 0.3,  0.0, 0.0],
        "ub": [1.00, 10.0, 0.0, 0.0],
    },
    "NOE": {
        "st": [0.10, 1.5, -3.5, 0.0],
        "lb": [0.00, 1.0, -4.0, 0.0],
        "ub": [0.40, 4.5, -1.5, 0.0],
    },
    "MT": {
        "st": [0.10, 20.0, 0.0, 0.0],
        "lb": [0.00, 10.0, -1.0, 0.0],
        "ub": [0.50, 60.0,  1.0, 0.0],
    },
    # amide: 3.0 ppm — matches MATLAB setLPeakBounds.m (Lorentzian amide at 3.0 ppm)
    "amide": {
        "st": [0.025, 1.0,  3.0, 0.0],
        "lb": [0.000, 0.2,  2.5, 0.0],
        "ub": [0.800, 5.0,  3.5, 0.0],
    },
    # amine: 3.0 ppm amine NH pool
    "amine": {
        "st": [0.025, 1.0,  3.0, 0.0],
        "lb": [0.000, 0.2,  2.5, 0.0],
        "ub": [0.800, 5.0,  3.5, 0.0],
    },
    "OH": {
        "st": [0.010, 1.2,  0.8, 0.0],
        "lb": [0.000, 1.0,  0.6, 0.0],
        "ub": [0.500, 5.0,  1.0, 0.0],
    },
    # Bounds from setLPeakBounds.m (Trp / ppm7pt3 / ppm9pt8 entries)
    "trp": {
        "st": [0.010, 10.0,  5.4, 0.0],
        "lb": [0.000,  0.2,  5.1, 0.0],
        "ub": [0.500, 100.0, 5.7, 0.0],
    },
    "ppm4pt4": {
        "st": [0.025, 1.0,  4.5, 0.0],
        "lb": [0.000, 0.2,  4.0, 0.0],
        "ub": [0.800, 5.0,  5.0, 0.0],
    },
    "ppm7pt3": {
        "st": [0.025,  1.0,  7.3, 0.0],
        "lb": [0.000,  0.2,  6.5, 0.0],
        "ub": [0.800,  5.0,  8.0, 0.0],
    },
    "ppm9pt8": {
        "st": [0.025,  1.0, 10.0, 0.0],
        "lb": [0.000,  0.2,  9.0, 0.0],
        "ub": [0.800,  5.0, 11.0, 0.0],
    },
    "guanidinium":   {"st": [0.025, 1.0,  2.0, 0.0], "lb": [0.000, 0.2,  1.5, 0.0], "ub": [0.300, 5.0,  2.5, 0.0]},
    "poly_l_lysine": {"st": [0.030, 2.0,  3.7, 0.0], "lb": [0.000, 0.8,  3.2, 0.0], "ub": [0.400, 8.0,  4.5, 0.0]},
    "glucose":       {"st": [0.020, 0.8,  1.2, 0.0], "lb": [0.000, 0.2,  0.8, 0.0], "ub": [0.200, 3.0,  1.6, 0.0]},
    "creatine":      {"st": [0.020, 0.8,  1.9, 0.0], "lb": [0.000, 0.2,  1.5, 0.0], "ub": [0.200, 3.0,  2.3, 0.0]},
    "taurine":       {"st": [0.020, 0.8,  3.2, 0.0], "lb": [0.000, 0.2,  2.8, 0.0], "ub": [0.200, 3.0,  3.6, 0.0]},
    "iopamidol_4.2": {"st": [0.025, 1.0,  4.2, 0.0], "lb": [0.000, 0.2,  4.00, 0.0], "ub": [0.500, 3.0,  4.75, 0.0]},
    "iopamidol_5.5": {"st": [0.025, 1.0,  5.5, 0.0], "lb": [0.000, 0.2,  5.25, 0.0], "ub": [0.500, 3.0,  5.75, 0.0]},
}
_FALLBACK_L = {
    "st": [0.05, 1.0,  0.0, 0.0],
    "lb": [0.00, 0.2, -10.0, 0.0],
    "ub": [0.50, 10.0, 10.0, 0.0],
}

# ── Simple Lorentzian bounds [A, FWHM, offset] — for inv-Z / MTRRex ─────────
_SIMPLE_LORENTZ_BOUNDS: dict[str, dict] = {
    "water":  {"st": [0.90, 1.4,  0.0], "lb": [0.30, 0.3,  0.0], "ub": [1.00, 3.0,  0.0]},
    "NOE":    {"st": [0.10, 1.5, -3.5], "lb": [0.00, 1.0, -4.0], "ub": [0.40, 4.5, -1.5]},
    "MT":     {"st": [0.10, 20.0, 0.0], "lb": [0.00, 10.0, -1.0],"ub": [0.50, 60.0, 1.0]},
    "amide":  {"st": [0.025, 1.0, 3.0], "lb": [0.00, 0.2,  2.5], "ub": [0.80, 5.0,  3.5]},
    "amine":  {"st": [0.025, 1.0, 3.0], "lb": [0.00, 0.2,  2.5], "ub": [0.80, 5.0,  3.5]},
    "OH":     {"st": [0.01, 1.5,  0.8], "lb": [0.00, 1.0,  0.6], "ub": [0.50,   5.5,  1.0]},
    # Bounds from setLPeakBounds.m (Trp / ppm7pt3 / ppm9pt8 entries)
    "trp":    {"st": [0.01, 10.0,  5.4], "lb": [0.00, 0.2,  5.1], "ub": [0.50, 100.0,  5.7]},
    "ppm4pt4":{"st": [0.025, 1.0,  4.5], "lb": [0.00, 0.2,  4.0], "ub": [0.80,   5.0,  5.0]},
    "ppm7pt3":{"st": [0.025, 1.0,  7.3], "lb": [0.00, 0.2,  6.5], "ub": [0.80,   5.0,  8.0]},
    "ppm9pt8":     {"st": [0.025, 1.0, 10.0], "lb": [0.00, 0.2,  9.0], "ub": [0.80,   5.0, 11.0]},
    "guanidinium": {"st": [0.025, 1.0,  2.0], "lb": [0.00, 0.2,  1.5], "ub": [0.30,   5.0,  2.5]},
    "poly_l_lysine":{"st": [0.030, 2.0, 3.7], "lb": [0.00, 0.8,  3.2], "ub": [0.40,   8.0,  4.5]},
    "glucose":     {"st": [0.020, 0.8,  1.2], "lb": [0.00, 0.2,  0.8], "ub": [0.20,   3.0,  1.6]},
    "creatine":    {"st": [0.020, 0.8,  1.9], "lb": [0.00, 0.2,  1.5], "ub": [0.20,   3.0,  2.3]},
    "taurine":     {"st": [0.020, 0.8,  3.2], "lb": [0.00, 0.2,  2.8], "ub": [0.20,   3.0,  3.6]},
    "iopamidol_4.2":{"st": [0.025, 1.0, 4.2], "lb": [0.00, 0.2,  4.00], "ub": [0.50,   3.0,  4.75]},
    "iopamidol_5.5":{"st": [0.025, 1.0, 5.5], "lb": [0.00, 0.2,  5.25], "ub": [0.50,   3.0,  5.75]},
}
_FALLBACK_SL = {"st": [0.05, 1.0, 0.0], "lb": [0.00, 0.2, -10.0], "ub": [0.50, 10.0, 10.0]}

# ── Simple Gaussian bounds [A, FWHM, centre] — for fast voxelwise fitting ───
# 3 parameters per pool (vs 6 for PV) → same cost as MPLF
_GAUSSIAN_BOUNDS: dict[str, dict] = {
    "water":  {"st": [0.90, 1.4,  0.0], "lb": [0.30, 0.3, -0.2], "ub": [1.00,  3.0,  0.2]},
    "NOE":    {"st": [0.10, 1.5, -3.5], "lb": [0.00, 1.0, -4.0], "ub": [0.40,  6.0, -1.5]},
    "MT":     {"st": [0.30, 20.0, 0.0], "lb": [0.00,15.0,  0.0], "ub": [0.60, 60.0,  1.0]},
    "amide":  {"st": [0.025, 1.0, 3.5], "lb": [0.00, 0.2,  3.25], "ub": [0.80,  4.0,  3.80]},
    "amine":  {"st": [0.025, 1.0, 3.0], "lb": [0.00, 0.2,  2.80], "ub": [0.80,  4.0,  3.20]},
    "OH":     {"st": [0.010, 1.5,  0.8], "lb": [0.00, 1.0,  0.6], "ub": [0.50,   5.5,  1.0]},
    # Bounds from setLPeakBounds.m (Trp / ppm7pt3 / ppm9pt8 entries)
    "trp":    {"st": [0.010, 10.0,  5.4], "lb": [0.00, 0.2,  5.1], "ub": [0.50, 100.0,  5.7]},
    "ppm4pt4":{"st": [0.025,  1.0,  4.5], "lb": [0.00, 0.2,  4.0], "ub": [0.80,   5.0,  5.0]},
    "ppm7pt3":{"st": [0.025,  1.0,  7.3], "lb": [0.00, 0.2,  6.5], "ub": [0.80,   5.0,  8.0]},
    "ppm9pt8":     {"st": [0.025,  1.0, 10.0], "lb": [0.00, 0.2,  9.0], "ub": [0.80,   5.0, 11.0]},
    "guanidinium": {"st": [0.025,  1.0,  2.0], "lb": [0.00, 0.2,  1.5], "ub": [0.30,   4.0,  2.5]},
    "poly_l_lysine":{"st": [0.030,  2.0,  3.7], "lb": [0.00, 0.8,  3.2], "ub": [0.40,   6.0,  4.5]},
    "glucose":     {"st": [0.020,  0.8,  1.2], "lb": [0.00, 0.2,  0.8], "ub": [0.20,   3.0,  1.6]},
    "creatine":    {"st": [0.020,  0.8,  1.9], "lb": [0.00, 0.2,  1.5], "ub": [0.20,   3.0,  2.3]},
    "taurine":     {"st": [0.020,  0.8,  3.2], "lb": [0.00, 0.2,  2.8], "ub": [0.20,   3.0,  3.6]},
    "iopamidol_4.2":{"st": [0.025,  1.0,  4.2], "lb": [0.00, 0.2,  4.00], "ub": [0.50,   3.0,  4.75]},
    "iopamidol_5.5":{"st": [0.025,  1.0,  5.5], "lb": [0.00, 0.2,  5.25], "ub": [0.50,   3.0,  5.75]},
}
_FALLBACK_G = {"st": [0.05, 1.0, 0.0], "lb": [0.00, 0.2, -10.0], "ub": [0.50, 10.0, 10.0]}

# ── Inverse-Z bounds — for 1/Z − 1 fitting ───────────────────────────────────
_INVZ_BOUNDS: dict[str, dict] = {
    "water":  {"st": [2.0,  1.5,  0.0], "lb": [0.1,  0.1, -1.0], "ub": [30.0,  8.0,  1.0]},
    "NOE":    {"st": [0.10, 2.0, -3.5], "lb": [0.00, 0.5, -5.0], "ub": [2.00,  6.0, -1.0]},
    "MT":     {"st": [0.30, 25.0, -2.0],"lb": [0.00, 5.0, -5.0], "ub": [8.00, 60.0,  0.0]},
    "amide":  {"st": [0.05, 1.0,  3.5], "lb": [0.00, 0.3,  3.0], "ub": [1.50,  4.0,  4.0]},
    "amine":  {"st": [0.05, 1.0,  3.0], "lb": [0.00, 0.3,  2.5], "ub": [1.50,  4.0,  3.5]},
    "OH":     {"st": [0.05, 1.5,  0.8], "lb": [0.00, 0.3,  0.6], "ub": [1.50,  5.5,  1.0]},
    # Bounds from setLPeakBounds.m (Trp / ppm7pt3 / ppm9pt8 entries), scaled for 1/Z space
    "trp":    {"st": [0.02, 10.0,  5.4], "lb": [0.00, 0.2,  5.1], "ub": [2.00, 100.0,  5.7]},
    "ppm4pt4":{"st": [0.05,  1.0,  4.5], "lb": [0.00, 0.2,  4.0], "ub": [2.00,   5.0,  5.0]},
    "ppm7pt3":{"st": [0.05,  1.0,  7.3], "lb": [0.00, 0.2,  6.5], "ub": [2.00,   5.0,  8.0]},
    "ppm9pt8":     {"st": [0.05,  1.0, 10.0], "lb": [0.00, 0.2,  9.0], "ub": [2.00,   5.0, 11.0]},
    "guanidinium": {"st": [0.05,  1.0,  2.0], "lb": [0.00, 0.2,  1.5], "ub": [1.50,   5.0,  2.5]},
    "poly_l_lysine":{"st": [0.05,  2.0,  3.6], "lb": [0.00, 0.8,  3.4], "ub": [2.00,   8.0,  3.9]},
    "glucose":     {"st": [0.05,  0.8,  1.2], "lb": [0.00, 0.2,  0.8], "ub": [1.50,   3.0,  1.6]},
    "creatine":    {"st": [0.05,  0.8,  1.9], "lb": [0.00, 0.2,  1.5], "ub": [1.50,   3.0,  2.3]},
    "taurine":     {"st": [0.05,  0.8,  3.2], "lb": [0.00, 0.2,  2.8], "ub": [1.50,   3.0,  3.6]},
    "iopamidol_4.2":{"st": [0.05,  1.0,  4.2], "lb": [0.00, 0.2,  4.00], "ub": [2.00,   3.0,  4.75]},
    "iopamidol_5.5":{"st": [0.05,  1.0,  5.5], "lb": [0.00, 0.2,  5.25], "ub": [2.00,   3.0,  5.75]},
}
_FALLBACK_INVZ = {"st": [0.05, 1.0, 0.0], "lb": [0.00, 0.2, -10.0], "ub": [5.00, 10.0, 10.0]}


def _get_bounds(pool: str, peak_type: str, fit_invz: bool = False) -> dict:
    if fit_invz:
        return _INVZ_BOUNDS.get(pool, _FALLBACK_INVZ)
    if "Pseudo" in peak_type or peak_type.lower() == "pseudo-voigt":
        return _PSEUDOVOIGT_BOUNDS.get(pool, _FALLBACK_PV)
    if peak_type.lower() == "gaussian":
        # 3-parameter Gaussian [A, FWHM, centre] — fast voxelwise model
        return _GAUSSIAN_BOUNDS.get(pool, _FALLBACK_G)
    if peak_type in ("Lorentzian-simple", "simple", "Standard", "standard"):
        return _SIMPLE_LORENTZ_BOUNDS.get(pool, _FALLBACK_SL)
    return _LORENTZ_BOUNDS.get(pool, _FALLBACK_L)


# ─────────────────────────────────────────────────────────────────────────────
# Single z-spectrum fit — mirrors zspecMultiPeakFit.m
# ─────────────────────────────────────────────────────────────────────────────

def fit_zspec_single(
    ppm: np.ndarray,
    z_spectrum: np.ndarray,
    pools: list[str] | None = None,
    peak_type: str = "Pseudo-Voigt",
    fixed_vals: dict | None = None,
    # Tolerances for voxelwise imaging — MRI noise floor σ≈1% → cost floor ≈1e-4.
    # ftol/xtol=1e-4 stops when improvement < noise floor; tighter values waste time.
    # MATLAB CEST-master uses TolFun=TolX=1e-6 (compiled C); Python scipy is ~10×
    # slower per iteration so we use 1e-4 to match wall-clock at same accuracy.
    # gtol is set equal to ftol; the bounded TRF can trigger gtol spuriously on
    # constrained parameters — using a matching value keeps it from being the
    # sole stopping criterion before ftol is satisfied.
    ftol: float = 1e-4,
    xtol: float = 1e-4,
    gtol: float = 1e-4,
    max_nfev: int = 400,
    fit_invz: bool = False,
) -> tuple[dict, dict, np.ndarray]:
    """
    Fit a single z-spectrum with a multi-peak model.

    Mirrors MATLAB's zspecMultiPeakFit.m:
      - fits 1-Z (dips become peaks pointing up)
      - uses negppmflg: fit only negative ppm if pools ⊆ {water, NOE, MT}

    Args:
        ppm        : (N,) ppm offsets
        z_spectrum : (N,) Z-values in [0, 1]
        pools      : pool names (default: MATLAB default = ['water','NOE','MT','amide'])
        peak_type  : 'Pseudo-Voigt' (MATLAB default) or 'Lorentzian'
        fixed_vals : dict pool -> param array; NaN = free
        ftol/xtol  : convergence tolerances (1e-7 practical for Python/imaging)
        max_nfev   : max function evaluations (600 sufficient for 4 pools, 24 params)
        fit_invz   : fit 1/Z-1 domain (for MTRRex)

    Returns:
        params      : dict pool -> fitted parameter array
        indiv_peaks : dict pool -> (N,) fitted peak curve (in 1-Z domain)
        sum_peak    : (N,) sum of all fitted peaks
    """
    if pools is None:
        pools = ["water", "NOE", "MT", "amide"]
    if fixed_vals is None:
        fixed_vals = {}

    # Clip to physical range.  Z-spectra should be in [0, 1]; slight
    # over-normalisation (z > 1 at far off-resonance) would create a negative
    # 1-Z baseline that MT (broad, flat) would happily absorb, yielding
    # spurious MT amplitudes for samples like PBS.
    z_spectrum = np.clip(z_spectrum, 0.0, 1.0)

    if fit_invz:
        peak_type = "Lorentzian-simple"

    is_gaussian = peak_type.lower() == "gaussian"
    is_pv       = "Pseudo" in peak_type or peak_type.lower() == "pseudo-voigt"
    is_simple   = peak_type in ("Lorentzian-simple", "simple", "Standard", "standard")

    if is_gaussian:
        # 3-parameter simple Gaussian [A, FWHM, centre] — same cost as MPLF
        npar    = 3
        peak_fn = _gaussian_simple
    elif is_pv:
        npar    = 6
        peak_fn = _pseudo_voigt
    elif is_simple:
        npar    = 3
        peak_fn = _simple_lorentzian
    else:
        npar    = 4
        peak_fn = _lorentzian

    # ── Two-step fitting (mirrors MATLAB pflgs.water1st=True) ─────────────
    # MATLAB approach: fit background pools (water, NOE, MT) on negative ppm
    # first to establish the baseline, then fix their displacements and fit
    # all pools simultaneously on the full ppm range.  This is both faster
    # (step 1 has fewer parameters and fewer data points) and more accurate
    # (prevents water/MT from absorbing CEST signal during the joint fit).
    _BG_POOLS = {"water", "NOE", "MT"}
    _bg_pools   = [p for p in pools if p in _BG_POOLS]
    _cest_pools = [p for p in pools if p not in _BG_POOLS]

    # Only apply two-step when there are CEST pools beyond the background,
    # and we are in the normal Z-spectrum domain (not inv-Z, not Gaussian).
    _do_twostep = (
        bool(_bg_pools) and bool(_cest_pools)
        and not fit_invz
        and not is_gaussian and not is_simple
    )

    if _do_twostep and fixed_vals is None:
        # ── Step 1: fit background on negative ppm ────────────────────────
        # Calling fit_zspec_single with only bg pools triggers negppmflg
        # automatically (all pools ⊆ {water, NOE, MT}).
        try:
            # Step 1 only needs coarse convergence — it pins displacements,
            # not final amplitudes. Use 10× looser tolerance than the main fit.
            _bg_ftol = max(ftol * 10, 1e-3)
            _bg_params, _, _ = fit_zspec_single(
                ppm, z_spectrum, _bg_pools, peak_type,
                ftol=_bg_ftol, xtol=_bg_ftol, gtol=_bg_ftol,
                max_nfev=min(max_nfev, 200),
            )
        except Exception:
            _bg_params = {}

        # ── Build fixed_vals: pin the displacement of each bg pool ────────
        # Displacement index: PV → 4 (omega_0), Lorentzian → 2 (offset)
        _disp_idx = 4 if is_pv else 2
        fixed_vals = {}
        for p in _bg_pools:
            if p in _bg_params:
                fv = np.full(npar, np.nan)
                fv[_disp_idx] = float(_bg_params[p][_disp_idx])
                fixed_vals[p] = fv
        # Fall through to the normal single-pass fit with these fixed_vals
        # applied inside the bounds-building loop below.

    # ── Build data to fit ──────────────────────────────────────────────────
    _eps = 1e-9
    if fit_invz:
        data_to_fit = 1.0 / np.clip(z_spectrum, _eps, None) - 1.0
        fit_mask = np.ones(len(ppm), dtype=bool)
    else:
        # MATLAB: zfit = 1 - squeeze(zSpec(ii,:))  →  fit 1-Z domain
        data_to_fit = 1.0 - z_spectrum

        # MATLAB negppmflg: if ALL pools are in {water, NOE, MT}, fit neg ppm only
        # (fits water tail and NOE cleanly without positive-ppm CEST peaks)
        neg_only = all(p in ("water", "NOE", "MT") for p in pools)
        if neg_only:
            fit_mask = (ppm < 0) & (ppm > -5)   # MATLAB: w<0 & w>-5
        else:
            fit_mask = np.ones(len(ppm), dtype=bool)

    ppm_fit  = ppm[fit_mask]
    data_fit = data_to_fit[fit_mask]

    # ── Build x0, lb, ub ──────────────────────────────────────────────────
    x0, lb, ub = [], [], []
    for pool in pools:
        b = _get_bounds(pool, peak_type, fit_invz=fit_invz)
        x0.extend(b["st"])
        lb.extend(b["lb"])
        ub.extend(b["ub"])

    # Apply fixed values (mirrors MATLAB: fixedVals non-NaN → set lb=ub=val)
    for i, pool in enumerate(pools):
        if pool in fixed_vals:
            fv = np.asarray(fixed_vals[pool], dtype=float)
            for j in range(min(npar, len(fv))):
                if not np.isnan(fv[j]):
                    idx = i * npar + j
                    x0[idx] = lb[idx] = ub[idx] = float(fv[j])

    x0 = np.array(x0, dtype=float)
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)

    # scipy.optimize.least_squares requires lb < ub strictly for every parameter.
    # Phase parameters are fixed at 0 (lb == ub == 0).  Expand by a tiny epsilon
    # so scipy accepts the bounds while keeping the parameter effectively fixed.
    _fixed = lb == ub
    if _fixed.any():
        lb = lb.copy(); ub = ub.copy()
        lb[_fixed] -= 1e-10
        ub[_fixed] += 1e-10

    def _residual(x: np.ndarray) -> np.ndarray:
        fit = np.zeros(len(ppm_fit))
        for k in range(len(pools)):
            fit += peak_fn(x[k * npar:(k + 1) * npar], ppm_fit)
        return data_fit - fit

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res   = least_squares(
                _residual, x0,
                bounds=(lb, ub),
                method="trf",
                ftol=ftol, xtol=xtol, gtol=gtol,
                max_nfev=max_nfev,
                verbose=0,
            )
        x_opt = res.x
    except Exception:
        x_opt = x0.copy()

    # ── Build output — evaluate on FULL ppm range (not just fit_mask) ─────
    params:      dict = {}
    indiv_peaks: dict = {}
    sum_peak = np.zeros(len(ppm))

    for i, pool in enumerate(pools):
        p = x_opt[i * npar:(i + 1) * npar]
        params[pool] = p
        curve = peak_fn(p, ppm)
        indiv_peaks[pool] = curve
        sum_peak += curve

    return params, indiv_peaks, sum_peak


# ─────────────────────────────────────────────────────────────────────────────
# Parallel voxelwise fitting — mirrors MATLAB parfor in fitAllZspec.m
# ─────────────────────────────────────────────────────────────────────────────

def _fit_chunk(args: tuple) -> list:
    """Fit a chunk of voxels — module-level so ProcessPoolExecutor can pickle it."""
    indices, ppm, z_chunk, pools, peak_type, ftol, xtol, gtol, max_nfev = args
    results = []
    n = len(ppm)
    for idx, z_vox in zip(indices, z_chunk):
        try:
            params, indiv, sumf = fit_zspec_single(
                ppm, z_vox, pools, peak_type,
                ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev,
            )
            ampls = {pool: float(params[pool][0]) for pool in pools}
            results.append((idx, ampls, indiv, sumf, None))
        except Exception as exc:
            results.append((
                idx,
                {p: 0.0          for p in pools},
                {p: np.zeros(n)  for p in pools},
                np.zeros(n),
                str(exc),
            ))
    return results


def fit_all_zspec(
    ppm: np.ndarray,
    z_img: np.ndarray,
    pools: list[str] | None = None,
    peak_type: str = "Pseudo-Voigt",
    n_workers: int = 8,
    progress_cb=None,
    cancelled_fn=None,
    chunk_size: int | None = None,
    ftol: float = 1e-7,
    xtol: float = 1e-7,
    gtol: float = 1e-7,
    max_nfev: int = 600,
) -> tuple[dict, dict, np.ndarray]:
    """
    Voxelwise z-spectral fitting — true parallel equivalent of MATLAB parfor.

    Uses ProcessPoolExecutor (separate processes, no GIL) so all CPU cores are
    fully utilised.  scipy's TRF loop is pure-Python and does NOT release the
    GIL, so ThreadPoolExecutor gives no speedup — processes are required.

    Args:
        ppm       : (N_off,) offset array in ppm
        z_img     : (n_vox, N_off) z-spectra (values in [0, 1])
        pools     : pool names to fit
        peak_type : 'Pseudo-Voigt' (MATLAB) or 'Lorentzian'
        n_workers : parallel worker processes

    Returns:
        ampl_maps       : dict pool -> (n_vox,) amplitudes
        indiv_peak_maps : dict pool -> (n_vox, N_off) per-pool peak curves (1-Z domain)
        sum_peak_maps   : (n_vox, N_off) sum of all pool curves
    """
    if pools is None:
        pools = ["water", "NOE", "MT", "amide"]

    n_vox, n_off = z_img.shape
    ampl_maps       = {p: np.zeros(n_vox)          for p in pools}
    indiv_peak_maps = {p: np.zeros((n_vox, n_off)) for p in pools}
    sum_peak_maps   = np.zeros((n_vox, n_off))

    if chunk_size is None:
        # Larger chunks for processes (lower IPC overhead) — 2 chunks per worker
        chunk_size = max(1, n_vox // (n_workers * 2))

    chunks = []
    for start in range(0, n_vox, chunk_size):
        end     = min(start + chunk_size, n_vox)
        indices = list(range(start, end))
        chunks.append((indices, ppm, z_img[start:end],
                       pools, peak_type, ftol, xtol, gtol, max_nfev))

    completed = 0
    _last_pct = [-1]
    _cancelled = False

    # 'spawn' context: each worker is a fresh Python interpreter — safe inside Qt
    _ctx = _mp.get_context('spawn')
    try:
        exe_cls = ProcessPoolExecutor
        _exe_kw = dict(max_workers=n_workers, mp_context=_ctx)
    except TypeError:
        # Fallback for Python < 3.7 (mp_context not supported)
        exe_cls = ThreadPoolExecutor
        _exe_kw = dict(max_workers=n_workers)

    with exe_cls(**_exe_kw) as exe:
        futures = {exe.submit(_fit_chunk, c): c[0][0] for c in chunks}
        for fut in as_completed(futures):
            if _cancelled:
                break
            for idx, ampls, indiv, sumf, err in fut.result():
                if err:
                    warnings.warn(f"Voxel {idx} fitting failed: {err}")
                for pool in pools:
                    ampl_maps[pool][idx]       = ampls[pool]
                    indiv_peak_maps[pool][idx] = indiv[pool]
                sum_peak_maps[idx] = sumf
                completed += 1
            if progress_cb and n_vox > 0:
                cur_pct = int(completed * 100 / n_vox)
                if cur_pct > _last_pct[0]:
                    _last_pct[0] = cur_pct
                    progress_cb(completed)
            if cancelled_fn and cancelled_fn():
                _cancelled = True
                # Cancel any futures not yet started
                for f in futures:
                    f.cancel()

    if _cancelled:
        raise InterruptedError("Cancelled by user.")

    return ampl_maps, indiv_peak_maps, sum_peak_maps


# ─────────────────────────────────────────────────────────────────────────────
# Dual fitting — Lorentzian + Pseudo-Voigt in one parallel pass
# (kept for GUI option; uses same MATLAB-matching tolerances)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_chunk_dual(args: tuple) -> list:
    """Fit a chunk of voxels with both Lorentzian + PV — module-level for ProcessPoolExecutor."""
    indices, ppm, z_chunk, pools, ftol, xtol, gtol, max_nfev = args
    results = []
    n = len(ppm)
    zero_n    = np.zeros(n)
    zero_dict = {p: np.zeros(n) for p in pools}

    for idx, z_vox in zip(indices, z_chunk):
        try:
            pl, il, sl = fit_zspec_single(ppm, z_vox, pools, "Lorentzian",
                                          ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev)
            al = {p: float(pl[p][0]) for p in pools}
        except Exception:
            al, il, sl = {p: 0.0 for p in pools}, dict(zero_dict), zero_n.copy()

        try:
            pp, ip, sp = fit_zspec_single(ppm, z_vox, pools, "Pseudo-Voigt",
                                          ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev)
            ap = {p: float(pp[p][0]) for p in pools}
        except Exception:
            ap, ip, sp = {p: 0.0 for p in pools}, dict(zero_dict), zero_n.copy()

        results.append((idx, al, il, sl, ap, ip, sp))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PV + Gaussian combined pass — one ProcessPoolExecutor instead of two
# ─────────────────────────────────────────────────────────────────────────────

def _fit_chunk_pv_gauss(args: tuple) -> list:
    """Fit a chunk with both Pseudo-Voigt AND Gaussian in a single worker call.
    Saves one full ProcessPoolExecutor spawn compared to two separate passes."""
    indices, ppm, z_chunk, pools, ftol, xtol, gtol, max_nfev = args
    results = []
    n = len(ppm)
    zero_n    = np.zeros(n)
    zero_dict = {p: np.zeros(n) for p in pools}
    for idx, z_vox in zip(indices, z_chunk):
        try:
            pp, ip, sp = fit_zspec_single(ppm, z_vox, pools, "Pseudo-Voigt",
                                          ftol=ftol, xtol=xtol, gtol=gtol,
                                          max_nfev=max_nfev)
            ap = {p: float(pp[p][0]) for p in pools}
        except Exception:
            ap, ip, sp = {p: 0.0 for p in pools}, dict(zero_dict), zero_n.copy()
        try:
            pg, ig, sg = fit_zspec_single(ppm, z_vox, pools, "Gaussian",
                                          ftol=ftol, xtol=xtol, gtol=gtol,
                                          max_nfev=max_nfev)
            ag = {p: float(pg[p][0]) for p in pools}
        except Exception:
            ag, ig, sg = {p: 0.0 for p in pools}, dict(zero_dict), zero_n.copy()
        results.append((idx, ap, ip, sp, ag, ig, sg))
    return results


def fit_all_zspec_pv_gauss(
    ppm: np.ndarray,
    z_img: np.ndarray,
    pools: list[str] | None = None,
    n_workers: int = 8,
    progress_cb=None,
    cancelled_fn=None,
    chunk_size: int | None = None,
    ftol: float = 1e-7,
    xtol: float = 1e-7,
    gtol: float = 1e-7,
    max_nfev: int = 600,
) -> tuple[dict, dict, np.ndarray, dict, dict, np.ndarray]:
    """
    Voxelwise fitting with Pseudo-Voigt AND Gaussian in a single parallel pass.

    Returns (pv_ampl, pv_indiv, pv_sum, gauss_ampl, gauss_indiv, gauss_sum).
    Uses one ProcessPoolExecutor instead of two, saving one full spawn overhead
    (~2–3 s on macOS) — makes a measurable difference for small-to-medium images.
    """
    if pools is None:
        pools = ["water", "NOE", "MT", "amide"]

    n_vox, n_off = z_img.shape
    pv_ampl    = {p: np.zeros(n_vox)          for p in pools}
    pv_indiv   = {p: np.zeros((n_vox, n_off)) for p in pools}
    pv_sum     = np.zeros((n_vox, n_off))
    gauss_ampl = {p: np.zeros(n_vox)          for p in pools}
    gauss_indiv= {p: np.zeros((n_vox, n_off)) for p in pools}
    gauss_sum  = np.zeros((n_vox, n_off))

    if chunk_size is None:
        chunk_size = max(1, n_vox // (n_workers * 2))

    chunks = [
        (list(range(start, min(start + chunk_size, n_vox))),
         ppm, z_img[start:min(start + chunk_size, n_vox)],
         pools, ftol, xtol, gtol, max_nfev)
        for start in range(0, n_vox, chunk_size)
    ]

    completed  = 0
    _last_pct  = [-1]
    _cancelled = False

    _ctx = _mp.get_context('spawn')
    try:
        exe_cls = ProcessPoolExecutor
        _exe_kw = dict(max_workers=n_workers, mp_context=_ctx)
    except TypeError:
        exe_cls = ThreadPoolExecutor
        _exe_kw = dict(max_workers=n_workers)

    with exe_cls(**_exe_kw) as exe:
        futures = {exe.submit(_fit_chunk_pv_gauss, c): c[0][0] for c in chunks}
        for fut in as_completed(futures):
            if _cancelled:
                break
            for idx, ap, ip, sp, ag, ig, sg in fut.result():
                for pool in pools:
                    pv_ampl[pool][idx]     = ap[pool]
                    pv_indiv[pool][idx]    = ip[pool]
                    gauss_ampl[pool][idx]  = ag[pool]
                    gauss_indiv[pool][idx] = ig[pool]
                pv_sum[idx]    = sp
                gauss_sum[idx] = sg
                completed += 1
            if progress_cb and n_vox > 0:
                cur_pct = int(completed * 100 / n_vox)
                if cur_pct > _last_pct[0]:
                    _last_pct[0] = cur_pct
                    progress_cb(completed)
            if cancelled_fn and cancelled_fn():
                _cancelled = True
                for f in futures:
                    f.cancel()

    return pv_ampl, pv_indiv, pv_sum, gauss_ampl, gauss_indiv, gauss_sum


def fit_all_zspec_dual(
    ppm: np.ndarray,
    z_img: np.ndarray,
    pools: list[str] | None = None,
    n_workers: int = 8,
    progress_cb=None,
    cancelled_fn=None,
    chunk_size: int | None = None,
    ftol: float = 1e-12,
    xtol: float = 1e-12,
    gtol: float = 1e-10,
    max_nfev: int = 6000,
) -> tuple[dict, dict, np.ndarray, dict, dict, np.ndarray]:
    """
    Voxelwise z-spectral fitting with BOTH Lorentzian AND Pseudo-Voigt models.
    Uses MATLAB-matching tolerances for both models.
    """
    if pools is None:
        pools = ["water", "NOE", "MT", "amide"]

    n_vox, n_off = z_img.shape
    lor_ampl  = {p: np.zeros(n_vox)          for p in pools}
    lor_indiv = {p: np.zeros((n_vox, n_off)) for p in pools}
    lor_sum   = np.zeros((n_vox, n_off))
    pv_ampl   = {p: np.zeros(n_vox)          for p in pools}
    pv_indiv  = {p: np.zeros((n_vox, n_off)) for p in pools}
    pv_sum    = np.zeros((n_vox, n_off))

    if chunk_size is None:
        chunk_size = max(1, n_vox // (n_workers * 2))

    chunks = [
        (list(range(start, min(start + chunk_size, n_vox))),
         ppm, z_img[start:min(start + chunk_size, n_vox)],
         pools, ftol, xtol, gtol, max_nfev)
        for start in range(0, n_vox, chunk_size)
    ]

    completed  = 0
    _last_pct  = [-1]
    _cancelled = False

    _ctx = _mp.get_context('spawn')
    try:
        exe_cls = ProcessPoolExecutor
        _exe_kw = dict(max_workers=n_workers, mp_context=_ctx)
    except TypeError:
        exe_cls = ThreadPoolExecutor
        _exe_kw = dict(max_workers=n_workers)

    with exe_cls(**_exe_kw) as exe:
        futures = {exe.submit(_fit_chunk_dual, c): c[0][0] for c in chunks}
        for fut in as_completed(futures):
            if _cancelled:
                break
            for idx, al, il, sl, ap, ip, sp in fut.result():
                for pool in pools:
                    lor_ampl[pool][idx]  = al[pool]
                    lor_indiv[pool][idx] = il[pool]
                    pv_ampl[pool][idx]   = ap[pool]
                    pv_indiv[pool][idx]  = ip[pool]
                lor_sum[idx] = sl
                pv_sum[idx]  = sp
                completed += 1
            if progress_cb and n_vox > 0:
                cur_pct = int(completed * 100 / n_vox)
                if cur_pct > _last_pct[0]:
                    _last_pct[0] = cur_pct
                    progress_cb(completed)
            if cancelled_fn and cancelled_fn():
                _cancelled = True
                for f in futures:
                    f.cancel()

    if _cancelled:
        raise InterruptedError("Cancelled by user.")

    return lor_ampl, lor_indiv, lor_sum, pv_ampl, pv_indiv, pv_sum


# ─────────────────────────────────────────────────────────────────────────────
# Voxelwise MPLF (Multi-Pool Lorentzian Fitting) — parallel wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _fit_chunk_mplf(args: tuple) -> list:
    """Fit a chunk of voxels with MPLF — module-level for ProcessPoolExecutor."""
    indices, ppm, z_chunk, pools, ftol, max_nfev, n_restarts = args
    n_off = len(ppm)
    results = []
    for idx, z_vox in zip(indices, z_chunk):
        try:
            res = fit_zspec_mplf(ppm, z_vox, pools=pools,
                                 ftol=ftol, max_nfev=max_nfev,
                                 n_restarts=n_restarts)
            pool_curves = res.get('pools', {})
            ampl  = {pn: float(np.nanmax(pc)) if len(pc) > 0 else 0.0
                     for pn, pc in pool_curves.items()}
            indiv = {pn: np.asarray(pc, dtype=float)
                     for pn, pc in pool_curves.items()}
        except Exception:
            _zero = np.zeros(n_off)
            ampl  = {p: 0.0         for p in pools}
            indiv = {p: _zero.copy() for p in pools}
        results.append((idx, ampl, indiv))
    return results


def fit_all_zspec_mplf(
    ppm: np.ndarray,
    z_img: np.ndarray,
    pools: list | None = None,
    n_workers: int = 8,
    progress_cb=None,
    cancelled_fn=None,
    chunk_size: int | None = None,
    ftol: float = 1e-7,
    max_nfev: int = 600,
    n_restarts: int = 1,
) -> tuple[dict, dict]:
    """
    Voxelwise MPLF fitting — parallel equivalent of fit_all_zspec for the
    Multi-Pool Lorentzian model.

    Returns
    -------
    ampl_maps  : dict pool → (n_vox,) peak amplitude  (max of per-pool ΔZ curve)
    indiv_maps : dict pool → (n_vox, N_off) per-pool ΔZ curve
    """
    # Resolve pool list — same logic as fit_zspec_mplf
    if pools is None:
        pools = list(_MPLF_DEFAULT_POOLS)
    else:
        pools = list(pools)
    if 'water' in pools:
        pools = ['water'] + [p for p in pools if p != 'water']
    else:
        pools = ['water'] + pools
    pools = [p for p in pools if p in MPLF_POOL_CATALOG]
    if not pools:
        pools = ['water']

    n_vox, n_off = z_img.shape
    ampl_maps  = {p: np.zeros(n_vox)          for p in pools}
    indiv_maps = {p: np.zeros((n_vox, n_off)) for p in pools}

    if chunk_size is None:
        chunk_size = max(1, n_vox // (n_workers * 2))

    chunks = []
    for start in range(0, n_vox, chunk_size):
        end = min(start + chunk_size, n_vox)
        chunks.append((list(range(start, end)), ppm, z_img[start:end],
                       pools, ftol, max_nfev, n_restarts))

    completed  = 0
    _last_pct  = [-1]
    _cancelled = False

    _ctx = _mp.get_context('spawn')
    try:
        exe_cls = ProcessPoolExecutor
        _exe_kw = dict(max_workers=n_workers, mp_context=_ctx)
    except TypeError:
        exe_cls = ThreadPoolExecutor
        _exe_kw = dict(max_workers=n_workers)

    with exe_cls(**_exe_kw) as exe:
        futures = {exe.submit(_fit_chunk_mplf, c): c[0][0] for c in chunks}
        for fut in as_completed(futures):
            if _cancelled:
                break
            for idx, ampl, indiv in fut.result():
                for pool in pools:
                    if pool in ampl:
                        ampl_maps[pool][idx] = ampl[pool]
                    if pool in indiv:
                        indiv_maps[pool][idx] = indiv[pool]
                completed += 1
            if progress_cb and n_vox > 0:
                cur_pct = int(completed * 100 / n_vox)
                if cur_pct > _last_pct[0]:
                    _last_pct[0] = cur_pct
                    progress_cb(completed)
            if cancelled_fn and cancelled_fn():
                _cancelled = True
                for f in futures:
                    f.cancel()

    if _cancelled:
        raise InterruptedError("Cancelled by user.")

    return ampl_maps, indiv_maps


# ─────────────────────────────────────────────────────────────────────────────
# B0 correction — mirrors MATLAB B0correction.m exactly
#
# MATLAB uses:
#   interp1(w_ppm - Current_B0, Original_Z, w_ppm, 'makima')
#   per-voxel, with parfor
#
# Python uses Akima1DInterpolator (scipy's nearest equivalent to MATLAB makima).
# We process per-voxel in a vectorised loop after sorting ppm ascending.
# ─────────────────────────────────────────────────────────────────────────────


def _lorentz_dip(delta, center, amp, width, offset):
    """Inverted-Lorentzian saturation dip (MATLAB lorentz_iN in WASSR_load_proc.m).

    y = offset + amp / (1 + (width/(delta-center))^2)
    At delta = center the term → 0, so y = offset (the dip minimum); far from
    resonance y → offset + amp.  ``center`` is the B0 offset we want.
    """
    return offset + amp / (1.0 + (width / (delta - center + 1e-12)) ** 2)


def _fit_b0_voxel(z_vox, w_hz, w_dense, width0, ftol, max_nfev):
    """Lorentzian-fit one WASSR voxel; return B0 offset in Hz (np.nan on failure)."""
    from scipy.optimize import curve_fit
    from scipy.interpolate import CubicSpline

    mx = np.max(z_vox)
    if mx < 1e-9 or not np.all(np.isfinite(z_vox)):
        return np.nan
    z_n = z_vox / mx

    # Initial centre guess = location of the spline minimum (MATLAB: min(spline(...)))
    try:
        x0 = float(w_dense[int(np.argmin(CubicSpline(w_hz, z_n)(w_dense)))])
    except Exception:
        x0 = float(w_hz[int(np.argmin(z_n))])

    #              center,  amp,   width,   offset
    p0 = [x0,    1.0,   width0,  0.05]
    lb = [x0 - 200.0, 1e-3,   1.0,    0.0]
    ub = [x0 + 200.0, 5.0,  1000.0,   1.0]
    try:
        popt, _ = curve_fit(
            _lorentz_dip, w_hz, z_n, p0=p0, bounds=(lb, ub),
            ftol=ftol, xtol=ftol, max_nfev=max_nfev,
        )
        return float(popt[0])
    except Exception:
        return x0


def compute_b0_map_wassr(
    z_norm: np.ndarray,
    ppm: np.ndarray,
    m0_img: np.ndarray | None = None,
    snr_thresh: float = 3.0,
    larmor_mhz: float = 400.0,
    n_workers: int = 1,
    ftol: float = 1e-3,
    max_nfev: int = 400,
    max_fit_dim: int = 192,
) -> np.ndarray:
    """
    Compute a B0 map from WASSR Z-spectra by per-voxel Lorentzian fitting.

    Mirrors MATLAB WASSR_load_proc.m:
      1. Build an SNR mask from M0 (only fit real-signal voxels).
      2. Fit each voxel's Z-spectrum (in Hz) to an inverted Lorentzian; the
         fitted centre is the B0 offset — continuous, sub-Hz, robust to noise.

    This replaces the bare ``argmin`` approach, which left the phantom interior
    flat (argmin can't resolve sub-offset shifts) and snapped noisy background /
    edge voxels to the extreme offsets (the saturated speckles).

    Parameters
    ----------
    z_norm     : (H, W, n_sl, n_off) normalised WASSR Z-spectra
    ppm        : (n_off,) saturation offsets in ppm (any order)
    m0_img     : (H, W, n_sl) M0 image for SNR masking; if None, all
                 non-empty voxels are fit.
    snr_thresh : SNR multiplier for the M0 noise floor (GUI "SNR threshold").
    larmor_mhz : proton Larmor frequency (MHz) — converts ppm ↔ Hz for fitting.
    n_workers  : parallel jobs (joblib); 1 = serial.
    ftol       : fit tolerance (from the "Fit quality" preset).
    max_nfev   : max function evaluations per fit (from "Fit quality").

    Returns
    -------
    b0_map     : (H, W, n_sl) B0 offset in ppm (masked-out voxels = 0).
    """
    z_norm = np.asarray(z_norm)
    if z_norm.ndim == 3:                       # (H, W, n_off) → add slice axis
        z_norm = z_norm[:, :, np.newaxis, :]

    # ── Speed: for high-resolution images the per-voxel Lorentzian fit on the
    # full grid is very slow (e.g. 512×512 ≈ 260k voxels). The B0 field is
    # spatially smooth, so fit on a downsampled grid and upsample the result. ─
    H, W = z_norm.shape[0], z_norm.shape[1]
    if max_fit_dim and max(H, W) > max_fit_dim:
        from scipy.ndimage import zoom as _zoom
        f = max_fit_dim / float(max(H, W))
        z_small = _zoom(z_norm, (f, f, 1, 1), order=1)
        m0_small = (_zoom(np.asarray(m0_img, float), (f, f, 1), order=1)
                    if m0_img is not None else None)
        b0_small = compute_b0_map_wassr(
            z_small, ppm, m0_img=m0_small, snr_thresh=snr_thresh,
            larmor_mhz=larmor_mhz, n_workers=n_workers, ftol=ftol,
            max_nfev=max_nfev, max_fit_dim=0,        # no further downsampling
        )
        # Upsample the smooth B0 map back to the original resolution
        zy = H / float(b0_small.shape[0])
        zx = W / float(b0_small.shape[1])
        b0_full = _zoom(b0_small, (zy, zx, 1), order=1)
        return b0_full[:H, :W, :]

    # Ensure ppm ascending
    sort_idx = np.argsort(ppm)
    ppm_s    = np.asarray(ppm, dtype=float)[sort_idx]
    w_hz     = ppm_s * larmor_mhz                      # offsets in Hz

    orig_shape = z_norm.shape
    n_off      = orig_shape[-1]
    z_flat     = z_norm.reshape(-1, n_off)[:, sort_idx]
    n_vox      = z_flat.shape[0]

    # SNR mask (same logic as the CEST fit path) ─ which voxels to fit
    if m0_img is not None:
        m0_f  = np.asarray(m0_img, dtype=float).ravel()
        low   = m0_f[m0_f < np.percentile(m0_f, 10)]
        noise = np.mean(low) if low.size else 0.0
        if np.isfinite(noise) and noise > 0:
            fit_mask = m0_f > snr_thresh * noise
        else:
            # Degenerate M0 (uniform / no noise floor) → fit all real-signal voxels
            fit_mask = np.max(z_flat, axis=1) > 1e-6
    else:
        fit_mask = np.max(z_flat, axis=1) > 1e-6

    # Dense Hz grid for the initial-minimum guess
    w_dense = np.linspace(w_hz[0], w_hz[-1], n_off * 20)
    width0  = max(1.0, 0.25 * (w_hz[-1] - w_hz[0]))    # ~quarter of the sweep

    idx_fit = np.flatnonzero(fit_mask)
    b0_hz   = np.zeros(n_vox, dtype=float)

    def _do(i):
        return _fit_b0_voxel(z_flat[i], w_hz, w_dense, width0, ftol, max_nfev)

    if n_workers and n_workers > 1 and idx_fit.size:
        try:
            from joblib import Parallel, delayed
            vals = Parallel(n_jobs=n_workers, prefer="processes")(
                delayed(_do)(i) for i in idx_fit
            )
        except Exception:
            vals = [_do(i) for i in idx_fit]
    else:
        vals = [_do(i) for i in idx_fit]

    for i, v in zip(idx_fit, vals):
        b0_hz[i] = 0.0 if (v is None or not np.isfinite(v)) else v

    # Return in ppm (display path re-derives Hz from larmor)
    b0_ppm = b0_hz / larmor_mhz
    return b0_ppm.reshape(orig_shape[:-1])


def b0_correction(
    b0_map_ppm: np.ndarray,
    ppm: np.ndarray,
    z_img: np.ndarray,
) -> np.ndarray:
    """
    Per-voxel B0 correction using Akima interpolation (≈ MATLAB 'makima').

    MATLAB B0correction.m:
        B0corr_z(ind,:) = interp1(w_ppm - Current_B0,
                                  Original_Z(ind,:), w_ppm, 'makima')

    Args:
        b0_map_ppm : (n_vox,) B0 shift per voxel in ppm
        ppm        : (N_off,) offset vector (any order)
        z_img      : (n_vox, N_off) z-spectra

    Returns:
        corrected : (n_vox, N_off) B0-corrected z-spectra (same order as input)
    """
    try:
        from scipy.interpolate import Akima1DInterpolator
        _HAS_AKIMA = True
    except ImportError:
        from scipy.interpolate import interp1d as _interp1d_fallback
        _HAS_AKIMA = False

    b0_vals   = np.asarray(b0_map_ppm, dtype=float)
    corrected = z_img.copy()

    # Sort ppm ascending (required by all 1D interpolators)
    asc_idx  = np.argsort(ppm)
    ppm_asc  = ppm[asc_idx]
    z_asc    = z_img[:, asc_idx]
    corr_asc = z_asc.copy()

    for i in range(len(b0_vals)):
        b0 = float(b0_vals[i])
        if b0 == 0.0:
            continue                    # MATLAB: only correct if B0 ≠ 0
        ppm_shifted = ppm_asc - b0
        z_vox       = z_asc[i]
        try:
            if _HAS_AKIMA:
                f = Akima1DInterpolator(ppm_shifted, z_vox)
                result = f(ppm_asc)
                # Replace NaN (outside interpolation range) with original values
                nan_mask = np.isnan(result)
                if nan_mask.any():
                    result[nan_mask] = z_vox[nan_mask]
                corr_asc[i] = result
            else:
                # Fallback: cubic spline
                from scipy.interpolate import interp1d
                f = interp1d(ppm_shifted, z_vox, kind="cubic",
                             bounds_error=False, fill_value="extrapolate")
                corr_asc[i] = f(ppm_asc)
        except Exception:
            pass  # leave this voxel uncorrected

    # Restore original ppm order
    inv_idx   = np.argsort(asc_idx)
    corrected = corr_asc[:, inv_idx]
    return corrected


# ─────────────────────────────────────────────────────────────────────────────
# MTR asymmetry — mirrors MATLAB calcMTRmap.m
# ─────────────────────────────────────────────────────────────────────────────

def calc_mtr_map(
    z_img: np.ndarray,
    ppm: np.ndarray,
    sel_ppm: float = 3.5,
    interpolate: bool = True,
) -> tuple[np.ndarray, float]:
    """
    MTR asymmetry: Z(−ppm) − Z(+ppm).

    MATLAB calcMTRmap.m:
        MTRmap = zImgNeg − zImgPos  (Z at negative ppm minus Z at positive ppm)

    When interpolate=True (default) uses cubic spline on a dense 0.01 ppm grid
    before computing the asymmetry, matching CEST-LDA-MTRasym-main and
    eliminating quantisation errors from the original sparse ppm grid.

    Args:
        z_img       : (..., N_off) — last dimension is spectral
        ppm         : (N_off,) offset vector
        sel_ppm     : target ppm value
        interpolate : use cubic-spline interpolation (True) or nearest-point (False)

    Returns:
        mtr_map      : (...) MTR values
        sel_ppm_true : actual ppm offset used (on interpolated grid if interpolate=True)
    """
    from scipy.interpolate import CubicSpline

    orig_shape = z_img.shape
    z_flat     = z_img.reshape(-1, orig_shape[-1])

    sort_idx = np.argsort(ppm)
    ppm_s    = ppm[sort_idx]
    z_s      = z_flat[:, sort_idx]

    if interpolate and len(ppm_s) >= 4:
        # Build dense 0.01 ppm grid over the ppm range (CEST-LDA approach)
        ppm_dense = np.arange(ppm_s[0], ppm_s[-1] + 0.005, 0.01)
        # Cubic spline for each voxel — vectorised across voxels
        cs       = CubicSpline(ppm_s, z_s, axis=1, extrapolate=False)
        z_dense  = cs(ppm_dense)                        # (n_vox, n_dense)
        np.nan_to_num(z_dense, copy=False, nan=0.0)

        pos_i = int(np.argmin(np.abs(ppm_dense - sel_ppm)))
        neg_i = int(np.argmin(np.abs(ppm_dense + sel_ppm)))

        sel_ppm_true = float(ppm_dense[pos_i])
        mtr          = z_dense[:, neg_i] - z_dense[:, pos_i]
    else:
        pos_idx      = int(np.argmin(np.abs(ppm_s - sel_ppm)))
        neg_idx      = int(np.argmin(np.abs(ppm_s + sel_ppm)))
        sel_ppm_true = float(ppm_s[pos_idx])
        mtr          = z_s[:, neg_idx] - z_s[:, pos_idx]

    return mtr.reshape(orig_shape[:-1]), sel_ppm_true


def calc_mtrrex_map(
    z_img: np.ndarray,
    ppm: np.ndarray,
    sel_ppm: float = 3.5,
    interpolate: bool = True,
    eps: float = 0.02,
) -> tuple[np.ndarray, float]:
    """
    MTR_Rex (inverse-difference) map:  MTRRex(Δω) = 1/Z(+Δω) − 1/Z(−Δω).

    This is the R1-independent inverse metric (AREX = R1 · MTRRex). It removes
    the spillover/MT bias that MTR-asymmetry carries by working in 1/Z space.
    Z at ±sel_ppm is cubic-spline interpolated to a dense 0.01 ppm grid (same
    as calc_mtr_map), so any offset within the acquired range works.

    Args:
        z_img    : (..., N_off) normalised Z-spectra (last axis spectral)
        ppm      : (N_off,) offset vector
        sel_ppm  : target ppm
        eps      : floor on Z to avoid divide-by-zero in 1/Z

    Returns:
        mtrrex_map   : (...) MTRRex values
        sel_ppm_true : actual ppm offset used
    """
    from scipy.interpolate import CubicSpline

    orig_shape = z_img.shape
    z_flat     = z_img.reshape(-1, orig_shape[-1])
    sort_idx   = np.argsort(ppm)
    ppm_s      = ppm[sort_idx]
    z_s        = z_flat[:, sort_idx]

    if interpolate and len(ppm_s) >= 4:
        ppm_dense = np.arange(ppm_s[0], ppm_s[-1] + 0.005, 0.01)
        cs        = CubicSpline(ppm_s, z_s, axis=1, extrapolate=False)
        z_dense   = cs(ppm_dense)
        np.nan_to_num(z_dense, copy=False, nan=0.0)
        pos_i = int(np.argmin(np.abs(ppm_dense - sel_ppm)))
        neg_i = int(np.argmin(np.abs(ppm_dense + sel_ppm)))
        sel_ppm_true = float(ppm_dense[pos_i])
        zp, zn = z_dense[:, pos_i], z_dense[:, neg_i]
    else:
        pos_idx = int(np.argmin(np.abs(ppm_s - sel_ppm)))
        neg_idx = int(np.argmin(np.abs(ppm_s + sel_ppm)))
        sel_ppm_true = float(ppm_s[pos_idx])
        zp, zn = z_s[:, pos_idx], z_s[:, neg_idx]

    # 1/Z(label, +ppm) − 1/Z(reference, −ppm); guard against tiny/zero Z
    zp = np.where(np.abs(zp) < eps, np.nan, zp)
    zn = np.where(np.abs(zn) < eps, np.nan, zn)
    mtrrex = (1.0 / zp) - (1.0 / zn)
    return mtrrex.reshape(orig_shape[:-1]), sel_ppm_true


def calc_mtr_spectrum(
    z_img: np.ndarray,
    ppm: np.ndarray,
    interpolate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full MTR asymmetry spectrum across all positive-negative offset pairs.

    When interpolate=True uses cubic spline on a 0.01 ppm grid (CEST-LDA method),
    giving a smooth continuous asymmetry curve.

    Returns:
        mtr_spectra : (..., n_pairs) array
        mtr_ppm     : (n_pairs,) positive ppm values used
    """
    from scipy.interpolate import CubicSpline

    orig_shape = z_img.shape
    z_flat     = z_img.reshape(-1, orig_shape[-1])

    sort_idx = np.argsort(ppm)
    ppm_s    = ppm[sort_idx]
    z_s      = z_flat[:, sort_idx]

    if interpolate and len(ppm_s) >= 4:
        ppm_dense = np.arange(ppm_s[0], ppm_s[-1] + 0.005, 0.01)
        cs        = CubicSpline(ppm_s, z_s, axis=1, extrapolate=False)
        z_dense   = cs(ppm_dense)
        np.nan_to_num(z_dense, copy=False, nan=0.0)

        pos_mask    = ppm_dense > 0
        ppm_pos     = ppm_dense[pos_mask]
        z_pos       = z_dense[:, pos_mask]
        # Mirror to negative side
        neg_ppm_idx = np.array([int(np.argmin(np.abs(ppm_dense + p))) for p in ppm_pos])
        z_neg       = z_dense[:, neg_ppm_idx]
        mtr         = z_neg - z_pos
        mtr_ppm     = ppm_pos
    else:
        pos_idx = np.where(ppm_s > 0)[0]
        neg_idx_full = np.array([
            np.where(ppm_s < 0)[0][int(np.argmin(np.abs(ppm_s[ppm_s < 0] + ppm_s[i])))]
            for i in pos_idx
        ])
        mtr     = z_s[:, neg_idx_full] - z_s[:, pos_idx]
        mtr_ppm = ppm_s[pos_idx]

    return mtr.reshape((*orig_shape[:-1], len(mtr_ppm))), mtr_ppm


# =============================================================================
# Physical Z-spectrum models — PLOF and DROF
# =============================================================================

_GAMMA_HZ_UT = 42.5764   # proton gyromagnetic ratio / 2π  [Hz/µT]


def _cos2theta(satpwr_uT: float, B0_MHz: float, x_ppm: np.ndarray) -> np.ndarray:
    """cos²θ for continuous-wave saturation.  ω₁ = γ·B₁."""
    satHz = satpwr_uT * _GAMMA_HZ_UT
    return 1.0 - satHz ** 2 / (satHz ** 2 + (B0_MHz * x_ppm) ** 2)


def fit_zspec_plof(
    ppm: np.ndarray,
    z_spectrum: np.ndarray,
    satpwr_uT: float = 2.5,
    B0_MHz: float = 400.0,
    R1: float = 0.5,
    tsat: float = 2.0,
    peak_ppm: float = 3.5,
    ftol: float = 1e-7,
    max_nfev: int = 800,
) -> dict:
    """
    PLOF (Polynomial and Lorentzian O-Field) two-step physical-model fitting.

    Step 1: Fit background (water Lorentzian + baseline + slope) using only
            far off-resonance offsets (|ppm| > 1).
    Step 2: Fit background + CEST peak across the full spectrum with background
            parameters constrained to ±2 % of step-1 result.

    Physical Z-spectrum model (Henkelman steady-state):
        R1ρ = pool contributions (Lorentzian)
        Z   = (1 − cos²θ · R1/R1ρ) · exp(−R1ρ · tsat) + cos²θ · R1/R1ρ

    Reference: Khlebnikov et al., PLOF (github.com/kherz/PLOF-master)

    Returns
    -------
    dict with keys: 'fit', 'bg', 'delta_z', 'dw_peak', 'bg_params', 'peak_params'
    """
    ppm = np.asarray(ppm,        dtype=float)
    z   = np.asarray(z_spectrum, dtype=float)
    z   = np.clip(z, 0.0, 1.0)

    def _physical_z(R1rho_arr, x):
        c2 = _cos2theta(satpwr_uT, B0_MHz, x)
        R1r = np.maximum(R1rho_arr, 1e-4)
        return (1.0 - c2 * R1 / R1r) * np.exp(-R1r * tsat) + c2 * R1 / R1r

    def _R1rho_bg(p, x):
        A, G, baseline, slope = p
        G = max(abs(G), 1e-6)
        return A * G ** 2 / (G ** 2 + 4.0 * x ** 2) + baseline + slope * 0.001 * (x - peak_ppm)

    # Step 1 — background only (|ppm| > 1 ppm)
    bg_mask = np.abs(ppm) > 1.0
    if bg_mask.sum() < 4:
        bg_mask = np.ones(len(ppm), dtype=bool)
    ppm_bg, z_bg = ppm[bg_mask], z[bg_mask]

    def _resid_bg(p):
        return _physical_z(_R1rho_bg(p, ppm_bg), ppm_bg) - z_bg

    x0_bg = [0.1,  1.0,  0.5,  -190.0]
    lb_bg = [0.0,  1e-6, 0.0,  -1000.0]
    ub_bg = [100.0, 100.0, 1000.0, 0.0]
    try:
        res_bg = least_squares(_resid_bg, x0_bg, bounds=(lb_bg, ub_bg),
                               method='trf', ftol=ftol, max_nfev=max_nfev)
        p_bg = res_bg.x
    except Exception:
        p_bg = np.array(x0_bg)

    bg_curve = _physical_z(_R1rho_bg(p_bg, ppm), ppm)

    # Step 2 — background + CEST peak
    def _R1rho_full(p, x):
        A_pk, G_pk, dw_pk = p[:3]
        A_b, G_b, base, slope = p[3:]
        G_pk = max(abs(G_pk), 1e-6)
        G_b  = max(abs(G_b),  1e-6)
        R_bg = A_b * G_b ** 2 / (G_b ** 2 + 4.0 * x ** 2) + base + slope * 0.001 * (x - peak_ppm)
        R_pk = A_pk * G_pk ** 2 / (G_pk ** 2 + 4.0 * (x - dw_pk) ** 2)
        return R_bg + R_pk

    def _resid_full(p):
        return _physical_z(_R1rho_full(p, ppm), ppm) - z

    # Background constrained ±2 % around step-1 values
    def _bg_bound(v, sign):
        if v > 0:
            return v * (1 - 0.02 * sign)
        return v * (1 + 0.02 * sign)

    x0_full = [0.1, 0.5, peak_ppm] + list(p_bg)
    lb_full = [1e-4, 0.1, peak_ppm - 1.5] + [_bg_bound(v, -1) for v in p_bg]
    ub_full = [100., 5.0, peak_ppm + 1.5] + [_bg_bound(v,  1) for v in p_bg]
    try:
        res_full = least_squares(_resid_full, x0_full, bounds=(lb_full, ub_full),
                                 method='trf', ftol=ftol, max_nfev=max_nfev * 2)
        p_full = res_full.x
    except Exception:
        p_full = np.array(x0_full)

    fit_curve = _physical_z(_R1rho_full(p_full, ppm), ppm)
    idx_pk   = int(np.argmin(np.abs(ppm - peak_ppm)))
    delta_z  = float(bg_curve[idx_pk] - fit_curve[idx_pk])

    return {
        'fit':         fit_curve,
        'bg':          bg_curve,
        'delta_z':     delta_z,
        'dw_peak':     float(p_full[2]),
        'bg_params':   p_bg,
        'peak_params': p_full[:3],
    }


# ---------------------------------------------------------------------------
# MPLF pool catalogue — amplitudes are in Z-attenuation units (0-1), not R1ρ.
# lorentzMultipool model:  Z = Z0 − Σ Aᵢ·Gᵢ²/4 / (Gᵢ²/4 + (ω−dw₀−dwᵢ)²)
# ---------------------------------------------------------------------------
MPLF_POOL_CATALOG: dict = {
    'water':       {'A_iv': 0.90, 'A_lb': 0.40, 'A_ub': 1.05,
                    'G_iv': 1.5,  'G_lb': 0.3,  'G_ub': 6.0,
                    'ppm_iv': 0.0,  'ppm_lb': -0.5, 'ppm_ub': 0.5},
    'amide':       {'A_iv': 0.03, 'A_lb': 0.0,  'A_ub': 0.30,
                    'G_iv': 0.5,  'G_lb': 0.1,  'G_ub': 3.0,
                    'ppm_iv': 3.5,  'ppm_lb': 3.25, 'ppm_ub': 3.80},
    'amine':       {'A_iv': 0.03, 'A_lb': 0.0,  'A_ub': 0.30,
                    'G_iv': 0.5,  'G_lb': 0.1,  'G_ub': 3.0,
                    'ppm_iv': 3.0,  'ppm_lb': 2.80, 'ppm_ub': 3.20},
    'NOE':         {'A_iv': 0.03, 'A_lb': 0.0,  'A_ub': 0.30,
                    'G_iv': 1.0,  'G_lb': 0.1,  'G_ub': 5.0,
                    'ppm_iv': -3.5, 'ppm_lb': -3.9, 'ppm_ub': -3.1},
    'MT':          {'A_iv': 0.05, 'A_lb': 0.0,  'A_ub': 0.50,
                    'G_iv': 15.0, 'G_lb': 5.0,  'G_ub': 60.0,
                    'ppm_iv': -2.5, 'ppm_lb': -3.0, 'ppm_ub': -2.0},
    'guanidinium': {'A_iv': 0.03, 'A_lb': 0.0,  'A_ub': 0.30,
                    'G_iv': 0.5,  'G_lb': 0.1,  'G_ub': 3.0,
                    'ppm_iv': 2.0,  'ppm_lb': 1.5,  'ppm_ub': 2.5},
    'OH':          {'A_iv': 0.02, 'A_lb': 0.0,  'A_ub': 0.20,
                    'G_iv': 0.5,  'G_lb': 0.1,  'G_ub': 3.0,
                    'ppm_iv': 0.8,  'ppm_lb': 0.5,  'ppm_ub': 1.1},
    'glucose':     {'A_iv': 0.02, 'A_lb': 0.0,  'A_ub': 0.20,
                    'G_iv': 0.5,  'G_lb': 0.1,  'G_ub': 3.0,
                    'ppm_iv': 1.2,  'ppm_lb': 0.8,  'ppm_ub': 1.6},
    'creatine':    {'A_iv': 0.02, 'A_lb': 0.0,  'A_ub': 0.20,
                    'G_iv': 0.5,  'G_lb': 0.1,  'G_ub': 3.0,
                    'ppm_iv': 1.9,  'ppm_lb': 1.5,  'ppm_ub': 2.3},
    'taurine':     {'A_iv': 0.02, 'A_lb': 0.0,  'A_ub': 0.20,
                    'G_iv': 0.5,  'G_lb': 0.1,  'G_ub': 3.0,
                    'ppm_iv': 3.2,  'ppm_lb': 2.8,  'ppm_ub': 3.6},
    'poly_l_lysine': {'A_iv': 0.03, 'A_lb': 0.0,  'A_ub': 0.40,
                    'G_iv': 2.0,  'G_lb': 0.8,  'G_ub': 6.0,
                    'ppm_iv': 3.7,  'ppm_lb': 3.2,  'ppm_ub': 4.5},
    # Bounds from setLPeakBounds.m (Trp indole NH, 7.3 ppm, 9.8 ppm).
    # G_ub 100→4: a 100-ppm-wide Trp is a flat pedestal that swamps 5–6 ppm
    # (Trp 5.4 ≈ iopamidol 5.5).
    'trp':         {'A_iv': 0.010, 'A_lb': 0.0,  'A_ub': 0.50,
                    'G_iv': 2.0,  'G_lb': 0.2,  'G_ub': 4.0,
                    'ppm_iv': 5.4,  'ppm_lb': 5.1,  'ppm_ub': 5.7},
    'ppm4pt4':     {'A_iv': 0.025, 'A_lb': 0.0,  'A_ub': 0.80,
                    'G_iv': 1.0,  'G_lb': 0.2,  'G_ub': 5.0,
                    'ppm_iv': 4.5,  'ppm_lb': 4.0,  'ppm_ub': 5.0},
    'ppm7pt3':     {'A_iv': 0.025, 'A_lb': 0.0,  'A_ub': 0.80,
                    'G_iv': 1.0,  'G_lb': 0.2,  'G_ub': 5.0,
                    'ppm_iv': 7.5,  'ppm_lb': 7.0,  'ppm_ub': 8.0},
    'ppm9pt8':     {'A_iv': 0.025, 'A_lb': 0.0,  'A_ub': 0.80,
                    'G_iv': 1.0,  'G_lb': 0.2,  'G_ub': 5.0,
                    'ppm_iv': 10.0, 'ppm_lb': 9.0,  'ppm_ub': 11.0},
    # Iopamidol amide proton exchange peaks (CT agent / pH probe).  Narrow
    # (~1 ppm); G_ub capped at 1.5 so the 5.5-ppm pool can't broaden into the
    # 4.2-ppm pool (mirrors the pseudo-Voigt FWHM cap).
    'iopamidol_4.2': {'A_iv': 0.025, 'A_lb': 0.0,  'A_ub': 0.50,
                      'G_iv': 1.0,   'G_lb': 0.2,  'G_ub': 1.5,
                      'ppm_iv': 4.2,  'ppm_lb': 4.00, 'ppm_ub': 4.75},
    'iopamidol_5.5': {'A_iv': 0.025, 'A_lb': 0.0,  'A_ub': 0.50,
                      'G_iv': 1.0,   'G_lb': 0.2,  'G_ub': 1.5,
                      'ppm_iv': 5.5,  'ppm_lb': 5.25, 'ppm_ub': 5.75},
}

_MPLF_DEFAULT_POOLS: list = ['water', 'amide', 'NOE', 'MT', 'guanidinium']


def fit_zspec_mplf(
    ppm: np.ndarray,
    z_spectrum: np.ndarray,
    pools: list | None = None,
    ftol: float = 1e-7,
    max_nfev: int = 600,
    n_restarts: int = 1,
) -> dict:
    """
    MPLF — Multi-Pool Lorentzian Fitting (lorentzMultipool model).

    Purely empirical model — no physics parameters (B0/B1/R1/tsat) required.
    Fits the Z-spectrum directly as a sum of Lorentzian dips:

        Z(x) = Z0 − Σᵢ  Aᵢ · Gᵢ²/4 / (Gᵢ²/4 + (x − dw₀ − dwᵢ)²)

    where:
        Z0   — baseline (far off-resonance Z value, ≈ 1)
        dw₀  — water centre / B0 shift  (pool 0)
        Aᵢ   — peak amplitude in Z-attenuation units
        Gᵢ   — full-width-at-half-maximum in ppm
        dwᵢ  — peak offset in ppm (relative to water for i > 0)

    Reference: Zhang et al., lorentzMultipool.m (DROF-main toolbox)

    Parameters
    ----------
    pools : list[str] | None
        Pool names to include (must be keys in MPLF_POOL_CATALOG).
        Water is always prepended automatically.
        Defaults to ['water', 'amide', 'NOE', 'MT', 'guanidinium'].

    Returns
    -------
    dict with keys:
        'fit'        – fitted Z-spectrum
        'pools'      – dict name → per-pool ΔZ contribution curve (positive = attenuates Z)
        'params'     – optimised parameter vector
        'dw_b0'      – estimated B0 shift [ppm]
        'pool_names' – ordered list of pool names used
    """
    ppm = np.asarray(ppm,        dtype=float)
    z   = np.asarray(z_spectrum, dtype=float)
    z   = np.clip(z, 0.0, 1.0)

    # ── Resolve pool list ────────────────────────────────────────────────
    if pools is None:
        pools = list(_MPLF_DEFAULT_POOLS)
    else:
        pools = list(pools)
    if 'water' in pools:
        pools = ['water'] + [p for p in pools if p != 'water']
    else:
        pools = ['water'] + pools
    pools = [p for p in pools if p in MPLF_POOL_CATALOG]
    if not pools:
        pools = ['water']

    def _Z_model(p, x):
        # Layout: p = [Z0,  A0,G0,dw0,  A1,G1,dw1,  ...]
        # Pool 0 (water) is centred at dw0; pool i>0 at (dw0 + dwi).
        Z0   = p[0]
        dw_0 = p[3]
        y    = Z0
        n_p  = (len(p) - 1) // 3
        for i in range(n_p):
            A  = max(p[3 * i + 1], 0.0)
            G  = max(abs(p[3 * i + 2]), 1e-6)
            dw = p[3 * i + 3]
            if i == 0:
                y = y - A * G ** 2 / 4.0 / (G ** 2 / 4.0 + (x - dw_0) ** 2)
            else:
                y = y - A * G ** 2 / 4.0 / (G ** 2 / 4.0 + (x - dw_0 - dw) ** 2)
        return y

    # ── Build iv / lb / ub from catalog ─────────────────────────────────
    iv = [1.0]   # Z0
    lb = [0.8]
    ub = [1.05]
    for name in pools:
        pc = MPLF_POOL_CATALOG[name]
        iv += [pc['A_iv'], pc['G_iv'], pc['ppm_iv']]
        lb += [pc['A_lb'], pc['G_lb'], pc['ppm_lb']]
        ub += [pc['A_ub'], pc['G_ub'], pc['ppm_ub']]

    def _resid(p):
        return _Z_model(p, ppm) - z

    # ── Optimise with n_restarts restarts ────────────────────────────────
    # n_restarts=1: use catalog initial values directly — fast, sufficient for
    #               voxelwise maps (good starting point from catalog).
    # n_restarts>1: additional random restarts improve robustness for single
    #               spectra or noisy data (use 3 for publication quality).
    best_p, best_cost = np.array(iv), np.inf
    rng = np.random.default_rng(42)
    for i in range(max(1, n_restarts)):
        if i == 0:
            x0 = np.clip(np.array(iv), lb, ub)  # first: catalog values
        else:
            noise = rng.uniform(-0.05, 0.05, len(iv)) * (np.array(ub) - np.array(lb))
            x0 = np.clip(np.array(iv) + noise, lb, ub)
        try:
            res = least_squares(_resid, x0, bounds=(lb, ub), method='trf',
                                ftol=ftol, max_nfev=max_nfev)
            if res.cost < best_cost:
                best_p, best_cost = res.x, res.cost
        except Exception:
            pass

    p_opt     = best_p
    fit_curve = _Z_model(p_opt, ppm)
    dw_0      = float(p_opt[3])

    # ── Per-pool ΔZ contributions ────────────────────────────────────────
    pool_curves: dict[str, np.ndarray] = {}
    for i, name in enumerate(pools):
        p_sans = p_opt.copy()
        p_sans[3 * i + 1] = 0.0        # zero out this pool's amplitude
        z_sans = _Z_model(p_sans, ppm)
        pool_curves[name] = z_sans - fit_curve  # positive = pool attenuates Z

    return {
        'fit':        fit_curve,
        'pools':      pool_curves,
        'params':     p_opt,
        'dw_b0':      dw_0,
        'pool_names': list(pools),
    }


# ---------------------------------------------------------------------------
# DROF pool catalogue — each entry defines initial value / bounds for the
# three per-pool parameters (A, G, ppm offset).
# Water is always pool 0; all others are optional CEST pools.
# ---------------------------------------------------------------------------
DROF_POOL_CATALOG: dict = {
    # name          A_iv  A_lb  A_ub    G_iv   G_lb   G_ub  ppm_iv ppm_lb ppm_ub
    'water':       {'A_iv': 3.0,  'A_lb': 0.1,  'A_ub': 40.0,
                    'G_iv': 1.0,  'G_lb': 0.3,  'G_ub': 5.0,
                    'ppm_iv': 0.0,  'ppm_lb': -0.5, 'ppm_ub': 0.5},
    'amide':       {'A_iv': 0.1,  'A_lb': 0.0,  'A_ub': 0.8,
                    'G_iv': 0.5,  'G_lb': 0.1,  'G_ub': 8.0,
                    'ppm_iv': 3.5,  'ppm_lb': 3.1,  'ppm_ub': 3.9},
    'amine':       {'A_iv': 0.03, 'A_lb': 0.0,  'A_ub': 0.30,
                    'G_iv': 0.5,  'G_lb': 0.1,  'G_ub': 3.5,
                    'ppm_iv': 3.0,  'ppm_lb': 2.5,  'ppm_ub': 3.5},
    'NOE':         {'A_iv': 0.1,  'A_lb': 0.0,  'A_ub': 0.8,
                    'G_iv': 0.8,  'G_lb': 0.1,  'G_ub': 15.0,
                    'ppm_iv': -3.5, 'ppm_lb': -3.9, 'ppm_ub': -3.1},
    'MT':          {'A_iv': 0.1,  'A_lb': 0.0,  'A_ub': 0.8,
                    'G_iv': 80.0, 'G_lb': 20.0, 'G_ub': 600.0,
                    'ppm_iv': -2.5, 'ppm_lb': -3.0, 'ppm_ub': -2.0},
    'guanidinium': {'A_iv': 0.1,  'A_lb': 0.0,  'A_ub': 0.8,
                    'G_iv': 0.5,  'G_lb': 0.1,  'G_ub': 13.0,
                    'ppm_iv': 2.0,  'ppm_lb': 1.5,  'ppm_ub': 2.5},
    'OH':          {'A_iv': 0.05, 'A_lb': 0.0,  'A_ub': 0.5,
                    'G_iv': 0.5,  'G_lb': 0.1,  'G_ub': 5.0,
                    'ppm_iv': 0.8,  'ppm_lb': 0.5,  'ppm_ub': 1.1},
    'glucose':     {'A_iv': 0.05, 'A_lb': 0.0,  'A_ub': 0.5,
                    'G_iv': 0.5,  'G_lb': 0.1,  'G_ub': 5.0,
                    'ppm_iv': 1.2,  'ppm_lb': 0.8,  'ppm_ub': 1.6},
    'creatine':    {'A_iv': 0.05, 'A_lb': 0.0,  'A_ub': 0.5,
                    'G_iv': 0.5,  'G_lb': 0.1,  'G_ub': 5.0,
                    'ppm_iv': 1.9,  'ppm_lb': 1.5,  'ppm_ub': 2.3},
    'taurine':     {'A_iv': 0.05, 'A_lb': 0.0,  'A_ub': 0.5,
                    'G_iv': 0.5,  'G_lb': 0.1,  'G_ub': 5.0,
                    'ppm_iv': 3.2,  'ppm_lb': 2.8,  'ppm_ub': 3.6},
    'poly_l_lysine': {'A_iv': 0.05, 'A_lb': 0.0,  'A_ub': 0.6,
                    'G_iv': 2.0,  'G_lb': 0.8,  'G_ub': 8.0,
                    'ppm_iv': 3.7,  'ppm_lb': 3.2,  'ppm_ub': 4.5},
    # Bounds from setLPeakBounds.m scaled for DROF R1ρ model
    # (Trp indole NH, 7.3 ppm, 9.8 ppm)
    'trp':         {'A_iv': 0.05,  'A_lb': 0.0,  'A_ub': 0.8,
                    'G_iv': 10.0,  'G_lb': 0.5,  'G_ub': 100.0,
                    'ppm_iv': 5.4,  'ppm_lb': 5.1,  'ppm_ub': 5.7},
    'ppm4pt4':     {'A_iv': 0.1,   'A_lb': 0.0,  'A_ub': 0.8,
                    'G_iv': 1.0,   'G_lb': 0.2,  'G_ub': 8.0,
                    'ppm_iv': 4.5,  'ppm_lb': 4.0,  'ppm_ub': 5.0},
    'ppm7pt3':     {'A_iv': 0.1,   'A_lb': 0.0,  'A_ub': 0.8,
                    'G_iv': 1.0,   'G_lb': 0.2,  'G_ub': 8.0,
                    'ppm_iv': 7.5,  'ppm_lb': 7.0,  'ppm_ub': 8.0},
    'ppm9pt8':     {'A_iv': 0.1,   'A_lb': 0.0,  'A_ub': 0.8,
                    'G_iv': 1.0,   'G_lb': 0.2,  'G_ub': 8.0,
                    'ppm_iv': 10.0, 'ppm_lb': 9.0,  'ppm_ub': 11.0},
}

# Default pool set used when `pools=None`
_DROF_DEFAULT_POOLS: list = ['water', 'amide', 'NOE', 'MT', 'guanidinium']


def fit_zspec_drof(
    ppm: np.ndarray,
    z_spectrum: np.ndarray,
    satpwr_uT: float = 2.5,
    B0_MHz: float = 400.0,
    R1: float = 0.5,
    tsat: float = 2.0,
    pools: list | None = None,
    ftol: float = 1e-7,
    max_nfev: int = 2000,
) -> dict:
    """
    DROF (Double-step R1ρ Fitting) multi-pool Lorentzian Z-spectrum fitting.

    The pool list is fully user-configurable via `pools`.  Water is always
    included as pool 0 (B0 reference).  All pool parameters (A, G, ppm) are
    drawn from DROF_POOL_CATALOG.

    Physical model (DROF formula):
        R1ρ = Z0 + Σ_i  Aᵢ · Gᵢ²/4 / (Gᵢ²/4 + (ω − dw_B0 − dwᵢ)²)
        Z   = (cos²θ − cos²θ·R1/R1ρ)·exp(−R1ρ·tsat) + cos²θ·R1/R1ρ

    Parameters
    ----------
    pools : list[str] | None
        Pool names to include (must be keys in DROF_POOL_CATALOG).
        Water is always prepended automatically.
        Defaults to ['water', 'amide', 'NOE', 'MT', 'guanidinium'].

    Returns
    -------
    dict with keys:
        'fit'        – fitted Z-spectrum
        'pools'      – dict name → per-pool ΔZ contribution curve
        'params'     – optimised parameter vector
        'dw_b0'      – estimated B0 shift [ppm]
        'pool_names' – ordered list of pool names used
    """
    ppm = np.asarray(ppm,        dtype=float)
    z   = np.asarray(z_spectrum, dtype=float)
    z   = np.clip(z, 0.0, 1.0)

    # ── Resolve pool list ────────────────────────────────────────────────
    if pools is None:
        pools = list(_DROF_DEFAULT_POOLS)
    else:
        pools = list(pools)
    # Water must be first
    if 'water' in pools:
        pools = ['water'] + [p for p in pools if p != 'water']
    else:
        pools = ['water'] + pools
    # Keep only known pools
    pools = [p for p in pools if p in DROF_POOL_CATALOG]
    if not pools:
        pools = ['water']

    satHz = satpwr_uT * _GAMMA_HZ_UT

    def _c2(x):
        return 1.0 - satHz ** 2 / (satHz ** 2 + (B0_MHz * x) ** 2)

    def _Z_model(p, x):
        # Layout: p = [Z0,  A0,G0,dw0,  A1,G1,dw1,  A2,G2,dw2, ...]
        # Pool 0 (water): centred at dw0 (B0 shift).
        # Pool i>0: centred at (dw0 + dw_i).
        Z0   = p[0]
        dw_0 = p[3]          # water centre == B0 shift
        R1r  = Z0
        n_p  = (len(p) - 1) // 3
        for i in range(n_p):
            A  = p[3 * i + 1]
            G  = max(abs(p[3 * i + 2]), 1e-6)
            dw = p[3 * i + 3]
            if i == 0:
                R1r = R1r + A * G ** 2 / 4.0 / (G ** 2 / 4.0 + (x - dw_0) ** 2)
            else:
                R1r = R1r + A * G ** 2 / 4.0 / (G ** 2 / 4.0 + (x - dw_0 - dw) ** 2)
        R1r = np.maximum(R1r, 1e-4)
        c2  = _c2(x)
        return (c2 - c2 * R1 / R1r) * np.exp(-R1r * tsat) + c2 * R1 / R1r

    # ── Build iv / lb / ub from catalog ─────────────────────────────────
    iv = [0.5]   # Z0
    lb = [0.1]
    ub = [4.0]
    for name in pools:
        pc = DROF_POOL_CATALOG[name]
        iv += [pc['A_iv'], pc['G_iv'], pc['ppm_iv']]
        lb += [pc['A_lb'], pc['G_lb'], pc['ppm_lb']]
        ub += [pc['A_ub'], pc['G_ub'], pc['ppm_ub']]

    def _resid(p):
        return _Z_model(p, ppm) - z

    # ── Optimise with 3 random restarts ─────────────────────────────────
    best_p, best_cost = np.array(iv), np.inf
    rng = np.random.default_rng(42)
    for _ in range(3):
        noise = rng.uniform(-0.05, 0.05, len(iv)) * (np.array(ub) - np.array(lb))
        x0 = np.clip(np.array(iv) + noise, lb, ub)
        try:
            res = least_squares(_resid, x0, bounds=(lb, ub), method='trf',
                                ftol=ftol, max_nfev=max_nfev)
            if res.cost < best_cost:
                best_p, best_cost = res.x, res.cost
        except Exception:
            pass

    p_opt     = best_p
    fit_curve = _Z_model(p_opt, ppm)
    dw_0      = float(p_opt[3])

    # ── Per-pool ΔZ contributions ────────────────────────────────────────
    pool_curves: dict[str, np.ndarray] = {}
    for i, name in enumerate(pools):
        p_sans = p_opt.copy()
        p_sans[3 * i + 1] = 0.0        # zero out amplitude for this pool
        z_sans = _Z_model(p_sans, ppm)
        pool_curves[name] = z_sans - fit_curve   # positive = pool attenuates Z

    return {
        'fit':        fit_curve,
        'pools':      pool_curves,
        'params':     p_opt,
        'dw_b0':      dw_0,
        'pool_names': list(pools),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CEST Denoising  (wraps CEST-Denoise-main algorithms)
# ─────────────────────────────────────────────────────────────────────────────

def _get_cest_denoise_root() -> str:
    """Return the absolute path to the CEST-Denoise-main folder."""
    import os
    candidates = [
        os.path.expanduser("~/Downloads/CEST-Denoise-main"),
        os.path.join(os.path.dirname(__file__), "..", "..", "CEST-Denoise-main"),
        os.path.join(os.path.dirname(__file__), "..", "CEST-Denoise-main"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "Cannot find CEST-Denoise-main.\n"
        "Place the folder at ~/Downloads/CEST-Denoise-main  or next to the project root."
    )


def _estimate_working_sigma(img01: np.ndarray) -> float:
    """Estimate the image noise σ on the [0, 255] working scale for a [0, 1] image.

    BM3D and NLM operate on an internally ×255-rescaled ([0, 255]) copy of the
    input, so their strength parameters must be expressed on that scale.  Because
    the Z-image is min–max normalised to [0, 1] first, the true noise on the
    [0, 255] scale depends on the volume's dynamic range and is usually far below
    the values a user would guess by hand.  We estimate it robustly with Donoho's
    MAD on the Haar diagonal (HH) detail — edge-robust and dependency-free — per
    offset frame, and take the median across offsets.
    """
    x = np.clip(np.asarray(img01, dtype=np.float64), 0.0, 1.0) * 255.0
    if x.ndim == 2:
        x = x[:, :, None]
    sigmas = []
    for k in range(x.shape[-1]):
        f = x[..., k]
        if f.ndim != 2 or f.shape[0] < 2 or f.shape[1] < 2:
            continue
        # Haar diagonal detail; for i.i.d. noise std σ each coeff has std σ.
        hh = 0.5 * (f[:-1, :-1] - f[:-1, 1:] - f[1:, :-1] + f[1:, 1:])
        med = float(np.median(np.abs(hh)))
        if np.isfinite(med) and med > 0:
            sigmas.append(med / 0.6745)          # MAD → Gaussian σ
    return float(np.median(sigmas)) if sigmas else 10.0


def estimate_bm3d_sigma(img01: np.ndarray, strength: float = 1.0) -> float:
    """BM3D config ``sigma`` (on the [0,255] scale) = estimated noise × strength."""
    return float(np.clip(_estimate_working_sigma(img01) * float(strength), 1.0, 60.0))


def bm3d_denoise(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Denoise a Z-image with the reference BM3D package (Tampere `bm3d`), per frame.

    Accepts a single 2-D frame (Y, X) or a stack (Y, X, off); each offset frame is
    denoised independently with 2-D BM3D.  Works in [0, 1] for conditioning; the
    noise σ is estimated from the data (Donoho MAD) and scaled by ``strength``.
    """
    try:
        import bm3d as _bm3dlib
    except ImportError as _e:
        raise ImportError(
            "BM3D denoising requires the 'bm3d' package.  Run:  pip install bm3d"
        ) from _e
    x = np.asarray(img, dtype=np.float64)
    single = (x.ndim == 2)
    if single:
        x = x[:, :, None]
    vmin, vmax = float(x.min()), float(x.max())
    rng = (vmax - vmin) if (vmax - vmin) > 0 else 1.0
    x01 = (x - vmin) / rng
    sigma01 = estimate_bm3d_sigma(x01, strength) / 255.0   # [0,255] est → [0,1] scale
    out = np.empty_like(x01)
    for k in range(x01.shape[-1]):
        out[..., k] = _bm3dlib.bm3d(x01[..., k], sigma_psd=sigma01)
    out = out * rng + vmin
    return out[:, :, 0] if single else out


# Lazily-compiled numba NLM kernel — cached so numba is only required (and the
# ~1 s JIT only paid) the first time NLM actually runs, not at import time.
_NLM_FRAME_FN = None


def _get_nlm_frame_fn():
    global _NLM_FRAME_FN
    if _NLM_FRAME_FN is not None:
        return _NLM_FRAME_FN
    from numba import njit

    @njit(cache=True)
    def _nlm_frame(image, big, small, h):
        if big % 2 == 0:
            big += 1
        if small % 2 == 0:
            small += 1
        pw = big // 2
        sw = small // 2
        padded = np.zeros((image.shape[0] + big, image.shape[1] + big))
        padded[pw:pw + image.shape[0], pw:pw + image.shape[1]] = image
        out = image.copy()
        h2 = h * h
        for ix in range(pw, pw + image.shape[1]):
            for iy in range(pw, pw + image.shape[0]):
                ox = ix - pw
                oy = iy - pw
                comp = padded[iy - sw:iy + sw + 1, ix - sw:ix + sw + 1]
                val = 0.0
                nf = 0.0
                for wx in range(ox, ox + big - small + 1):
                    for wy in range(oy, oy + big - small + 1):
                        d2 = 0.0
                        for a in range(small):
                            for b in range(small):
                                diff = padded[wy + a, wx + b] - comp[a, b]
                                d2 += diff * diff
                        w = np.exp(-d2 / h2)          # Gaussian bandwidth h
                        nf += w
                        val += w * padded[wy + sw, wx + sw]
                out[oy, ox] = val / nf
        return out

    _NLM_FRAME_FN = _nlm_frame
    return _NLM_FRAME_FN


def _nlm_denoise(img01: np.ndarray, big_window: int, small_window: int,
                 strength: float = 1.0) -> np.ndarray:
    """Non-Local Means on a [0, 1] Z-image (Y, X, off) with a noise-scaled bandwidth.

    The vendored ``nlm_CEST`` uses ``exp(−distance)`` with no bandwidth, so on the
    [0, 255] scale every real patch distance underflows to weight 0 and nothing is
    denoised.  This drop-in uses ``exp(−d²/h²)`` with ``h = strength · σ · patch``,
    where σ is the estimated working-scale noise — giving genuine denoising.
    """
    fn = _get_nlm_frame_fn()
    x255 = np.clip(np.asarray(img01, dtype=np.float64), 0.0, 1.0) * 255.0
    if x255.ndim == 2:
        x255 = x255[:, :, None]
    small = int(small_window) + (1 - int(small_window) % 2)   # force odd
    big = int(big_window) + (1 - int(big_window) % 2)
    sigma = _estimate_working_sigma(img01)
    # Pure-noise patch RMS distance ≈ σ·sqrt(N) = σ·small; h at that scale weights
    # like-patches strongly while suppressing edges.
    h = max(1e-3, float(strength) * sigma * small)
    out = np.empty_like(x255)
    for k in range(x255.shape[-1]):
        out[..., k] = fn(np.ascontiguousarray(x255[..., k]), big, small, h)
    return (out / 255.0).reshape(np.asarray(img01).shape)


def denoise_zimg(
    z_img: np.ndarray,
    method: str = 'pca',
    mask: np.ndarray | None = None,
    pca_criteria: str = 'malinowski',
    bm3d_strength: float = 1.0,
    nlm_big_window: int = 21,
    nlm_small_window: int = 5,
) -> np.ndarray:
    """
    Denoise a CEST Z-spectrum image using CEST-Denoise-main algorithms.

    Parameters
    ----------
    z_img : np.ndarray
        Raw (unnormalized) Z-spectrum array.
        Accepted shapes:
          (Y, X, n_off)          — single slice
          (Y, X, slices, n_off)  — multi-slice; each slice denoised independently
    method : str
        'pca', 'bm3d', or 'nlm'
    mask : np.ndarray or None
        2-D binary mask (Y, X).  Used to focus PCA / BM3D on voxels-of-interest.
    pca_criteria : str | int
        For PCA: 'malinowski' (default), 'nelson', or 'median';
        or an integer giving the exact number of components to keep.
    bm3d_strength : float
        For BM3D: denoising strength (default 1.0).  The noise σ is estimated
        automatically from the data (on the [0,255] working scale); this value
        multiplies it.  1.0 = use the estimate, >1 stronger, <1 gentler.
    nlm_big_window : int
        For NLM: size of the large search window (default 21, must be odd).
    nlm_small_window : int
        For NLM: size of the small patch comparison window (default 5, must be odd).

    Returns
    -------
    np.ndarray
        Denoised array with the same shape and dtype as the input.
    """
    import sys

    root = _get_cest_denoise_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    method = method.lower().strip()
    orig_dtype = z_img.dtype

    # Handle 4-D (Y, X, slices, n_off) — denoise each slice independently
    if z_img.ndim == 4:
        out = np.empty_like(z_img)
        for sl in range(z_img.shape[2]):
            out[:, :, sl, :] = denoise_zimg(
                z_img[:, :, sl, :], method=method, mask=mask,
                pca_criteria=pca_criteria, bm3d_strength=bm3d_strength,
                nlm_big_window=nlm_big_window, nlm_small_window=nlm_small_window,
            )
        return out

    # z_img is now (Y, X, n_off)
    img = z_img.astype(np.float64)

    if method == 'pca':
        try:
            from PCA.src.denoise import pca as _pca
        except ImportError as _e:
            raise ImportError(
                f"PCA denoising requires scikit-image.  "
                f"Run:  pip install scikit-image\n(original error: {_e})"
            ) from _e
        result = _pca(img, criteria=pca_criteria, mask=mask)

    elif method == 'bm3d':
        # Reference BM3D (Tampere `bm3d` package) — fast, C/PyWavelets-backed and
        # robust.  The vendored pure-Python BM3D was buggy (zeroed blocks on real
        # textured data) and slow, so it is no longer used.
        result = bm3d_denoise(img, strength=bm3d_strength)

    elif method == 'nlm':
        try:
            import numba  # noqa: F401
        except ImportError as _e:
            raise ImportError(
                "NLM denoising requires numba.  Run:  pip install numba"
            ) from _e
        # Normalise to [0, 1], denoise with the noise-scaled-bandwidth NLM
        # (_nlm_denoise), then scale back.  The vendored nlm_CEST has no bandwidth
        # and no normalisation, so it either collapses the image to ~0 (raw input
        # overflows int16) or does nothing (weights underflow) — hence the in-house
        # kernel here.
        _vmin, _vmax = img.min(), img.max()
        _rng = (_vmax - _vmin) if (_vmax - _vmin) > 0 else 1.0
        img_01 = (img - _vmin) / _rng          # → [0, 1]
        result_01 = _nlm_denoise(img_01, nlm_big_window, nlm_small_window)
        result = result_01 * _rng + _vmin      # back to original scale

    else:
        raise ValueError(
            f"Unknown denoising method {method!r}. "
            "Choose 'pca', 'bm3d', or 'nlm'."
        )

    return result.astype(orig_dtype)
