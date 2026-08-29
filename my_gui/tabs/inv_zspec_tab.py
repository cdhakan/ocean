"""
inv_zspec_tab.py  —  1/Z Spectroscopy Tab
==========================================
CEST Z-spectrum analysis in 1/Z (AREX) space for a single B1 saturation power.

Three-step pipeline (ports cest_1divZ_fit.py / MT_CEST_fit_QUESP_dk.m):
  Step 1 — MT fitting: Super-Lorentzian + Henkelman 1993 two-pool model
            fitted to far off-resonance wings (|ppm| > MT_excl, default ±8)
  Step 2 — CEST peak fitting in R₁·cos²θ·(1/Z−1) space
            pools: water (~0 ppm), OH (~0.8 ppm), amine (~3 ppm)
            lineshapes: Lorentzian or pseudo-Voigt
  Step 3 — Z-spectrum reconstruction from all fitted components

Default parameters for 9.4 T data (−9.5 to +9.5 ppm):
  B0 = 400 MHz,  B1 = 2.5 µT,  R1 from T1 map or manual entry
  MT excl. ±8 ppm (water re-incl. ±0.5 ppm),  CEST range −2 to +5 ppm

Reference: cest_1divZ_fit.py, Henkelman et al. MRM 1993,
           Bieri & Scheffler MRM 2006 (Super-Lorentzian)
"""
from __future__ import annotations

import os
import threading
import warnings
import multiprocessing as _mp
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)

import numpy as np
from scipy.optimize import least_squares

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QGroupBox,
    QDoubleSpinBox, QSpinBox, QComboBox, QTextEdit,
    QProgressBar, QFileDialog, QScrollArea,
    QListWidget, QListWidgetItem, QDialog,
    QFormLayout, QMessageBox, QProgressDialog,
    QApplication, QCheckBox, QFrame,
    QDialogButtonBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QEvent
from my_gui.pool_dialog import PoolSelectionWidget, POOL_CATALOG, default_pools

from my_gui.roi_tools import ROICanvas, roi_union_mask
from my_gui.plot_custom_bar import PlotCustomBar, WindowLevelToolButton


# ── Physical constants ─────────────────────────────────────────────────────────
_GAMMA_HZ_UT = 42.577          # proton gyromagnetic ratio / 2π  [Hz/µT]
_GAMMA_2PI   = _GAMMA_HZ_UT * 2.0 * np.pi   # rad/s per µT  (w1 = B1[µT]·γ·2π)

# ── Pool display colours (matching reference PNG style) ────────────────────────
POOL_COLORS = {
    'water':  '#2ca02c',  # green
    'OH':     '#d62728',  # red
    'amine':  '#e377c2',  # magenta / pink
    'MT':     '#1f77b4',  # blue
    'amide':  '#ff7f0e',  # orange
    'NOE':    '#17becf',  # cyan
    'Trp':    '#8c4a2f',  # brown
    '4.4ppm': '#9467bd',  # purple
    '7.3ppm': '#e24bc9',  # pink-magenta
    '9.8ppm': '#808000',  # olive
    'glucose':      '#bcbd22',  # yellow-green
    'creatine':     '#7f7f7f',  # grey
    'taurine':      '#c49c94',  # tan
    'iopamidol43':  '#aec7e8',  # light blue
    'iopamidol55':  '#e6550d',  # dark orange (kept distinct from NOE cyan)
    '3omg':         '#ffbb78',  # light orange
    'polylysine':   '#98df8a',  # light green
    'GAG':          '#c5b0d5',  # light purple
    'myo_inositol': '#f7b6d2',  # light pink
    'guanidinium':  '#dbdb8d',  # khaki
}

# ── Pool-key aliases ────────────────────────────────────────────────────────────
# The CEST MRI / DROF pool catalog uses different key spellings than the 1/Z
# _L_DEFS bounds tables. Map them so pools picked in CEST Processing Options
# (e.g. "7.3 ppm pool" → 'ppm7pt3') resolve to the 1/Z fit key ('7.3ppm').
_POOL_ALIASES = {
    'trp':            'Trp',
    'ppm4pt4':        '4.4ppm',
    'ppm7pt3':        '7.3ppm',
    'ppm9pt8':        '9.8ppm',
    'poly_l_lysine':  'polylysine',
    'iopamidol_4.2':  'iopamidol43',
    'iopamidol_5.5':  'iopamidol55',
}


def _canonical_invz_pool(name: str) -> str:
    """Resolve a CEST-MRI/DROF pool key to its 1/Z _L_DEFS equivalent."""
    return _POOL_ALIASES.get(name, name)


# Merged catalog keys that expand to several 1/Z fit sub-pools.  The "Iopamidol
# (4.2 & 5.5 ppm)" checkbox is one entry but fits both amide pools.
_POOL_EXPANSIONS = {
    'iopamidol': ['iopamidol43', 'iopamidol55'],
}


def _expand_invz_pools(names) -> list[str]:
    """Expand merged pool keys (e.g. 'iopamidol') into their fit sub-keys,
    preserving order and dropping duplicates."""
    out: list[str] = []
    for n in names:
        for k in _POOL_EXPANSIONS.get(n, [n]):
            if k not in out:
                out.append(k)
    return out


class _WheelScrollFilter(QObject):
    """Forward mouse-wheel events from a matplotlib canvas up to an enclosing
    QScrollArea.  Without this the canvas swallows the wheel, so tall ROI-spectra
    figures can't be scrolled with the wheel (only by dragging the scrollbar)."""

    def __init__(self, scroll_area):
        super().__init__(scroll_area)
        self._sa = scroll_area

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and self._sa is not None:
            sb = self._sa.verticalScrollBar()
            sb.setValue(sb.value() - int(event.angleDelta().y()))
            return True
        return False


# =============================================================================
#  Physics — Super-Lorentzian + Henkelman MT model
# =============================================================================

# Super-Lorentzian powder-average grid — precomputed ONCE (it is constant
# across every fit iteration). 400 angular steps converges the integral to
# <1 % vs 1000 while cutting the per-call cost ~2× (this function is evaluated
# on every least_squares residual call, so it dominates the MT fit).
_MT_POWDER_STEPS = 400
_PA_CTHETA   = np.arange(1, _MT_POWDER_STEPS + 1) / float(_MT_POWDER_STEPS)
_PA_3CT2M1SQ = (3.0 * _PA_CTHETA ** 2 - 1.0) ** 2
_PA_DENOM    = np.where(np.abs(3.0 * _PA_CTHETA ** 2 - 1.0) < 1e-12, 1e-12,
                        np.abs(3.0 * _PA_CTHETA ** 2 - 1.0))
_PA_CTHETA_C = _PA_CTHETA[:, np.newaxis]          # (steps, 1) for broadcasting
_PA_3CT2_C   = _PA_3CT2M1SQ[:, np.newaxis]


def _RF_superlorentzian(T2b: float, chems_Hz: float,
                         w1_rad: float, delta_Hz: np.ndarray) -> np.ndarray:
    """
    Vectorized powder-average Super-Lorentzian RF absorption rate.
    Returns rrfb = π·w1²·G [s⁻¹].  Uses the precomputed angular grid.

    (Bieri & Scheffler, MRM 2006)
    """
    delta   = np.atleast_1d(np.float64(delta_Hz))      # (N,)
    delta_diff = 2.0 * np.pi * delta[np.newaxis, :] - 2.0 * np.pi * chems_Hz
    exp_arg    = -2.0 * (delta_diff * T2b) ** 2 / _PA_3CT2_C
    exp_arg    = np.clip(exp_arg, -500.0, 0.0)

    f1     = T2b * np.sqrt(2.0 / np.pi) / _MT_POWDER_STEPS / _PA_DENOM  # (steps,)
    result = np.sum(f1[:, np.newaxis] * np.exp(exp_arg), axis=0)        # (N,)
    return np.pi * w1_rad ** 2 * result


def _mt_model(t2a: float, ra: float, rb: float, mb0: float, r: float,
               t2b: float, delta_MT_Hz: float,
               delta_Hz: np.ndarray, w1_rad: float) -> np.ndarray:
    """
    Henkelman 1993 two-pool MT steady-state Z-spectrum (Eq. 9–10).
    Returns Z values clipped to [0, 1].
    """
    d     = np.atleast_1d(np.float64(delta_Hz))
    rrfa  = t2a * w1_rad**2 / (1.0 + (t2a * 2.0 * np.pi * d)**2)
    rrfb  = _RF_superlorentzian(t2b, delta_MT_Hz, w1_rad, d)
    num   = r * (ra + mb0 * rb) + ra * (rb + rrfb)
    den   = (ra + rrfa) * (rb + rrfb) + r * (ra + rrfa + mb0 * (rb + rrfb))
    return np.clip(num / (den + 1e-30), 0.0, 1.0)


# =============================================================================
#  Peak lineshapes
# =============================================================================

def _lorentzian(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Single Lorentzian peak (matches MATLAB ufzsSingleLorentzianModel).
    p = [amplitude, FWHM_ppm, offset_ppm, phase_rad]

    The real part of the complex Lorentzian requires the arctan2 dispersion
    phase — WITHOUT it, ``real(num/den)`` collapses to h/√(h²+Δ²), a
    *square-root* Lorentzian whose effective FWHM is ~1.6× too wide and whose
    tails are ~2.4× too heavy, so neighbouring peaks bleed into far pools (e.g.
    amine swallowing iopamidol at 4.3/5.5 ppm).  Including arctan2 makes the
    real part the true absorption Lorentzian h²/(h²+Δ²), consistent with the
    pseudo-Voigt L-component below.
    """
    amp, hwhm, off = float(p[0]), float(p[1]) / 2.0 + 1e-20, float(p[2])
    ph  = float(p[3]) if len(p) > 3 else 0.0
    d   = x - off
    num = np.sqrt(hwhm**2 + d**2) * hwhm
    den = hwhm**2 + d**2
    y   = amp * np.real(num / den * np.exp(-1j * (np.arctan2(d, hwhm) + ph)))
    return y - float(y.min())


def _pseudovoigt(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Single pseudo-Voigt peak (matches MATLAB ufzsSinglePseudoVoigtModel).
    p = [amplitude, alpha (Gauss fraction), FWHM_L, FWHM_ratio, offset_ppm, phase_rad]
    """
    amp, alpha, fwhm_l, fwhm_rat, off = float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
    ph    = float(p[5]) if len(p) > 5 else 0.0
    sigma = fwhm_l / (2.0 * np.sqrt(2.0 * np.log(2.0)) + 1e-20) * fwhm_rat
    G     = np.exp(-(x - off)**2 / (2.0 * sigma**2 + 1e-40))
    hwhm  = fwhm_l / 2.0 + 1e-20
    Lnum  = np.sqrt(hwhm**2 + (x - off)**2) * hwhm
    Lden  = (x - off)**2 + hwhm**2
    Lph   = np.exp(-1j * (np.arctan2(x - off, hwhm) + ph))
    L     = Lnum / Lden * Lph
    return amp * np.real(alpha * G + (1.0 - alpha) * L)


def _pfn(peak_type: str):
    """Return the peak function for the given peak type string."""
    return _pseudovoigt if peak_type == 'pseudovoigt' else _lorentzian


# =============================================================================
#  Step 1 — MT fitting
# =============================================================================

_MT_X0 = [1.0,  0.05, 40.0, 10e-6, -0.5]   # [rb, mb0, r, t2b_s, delta_MT_ppm]
_MT_LB = [0.1,  0.005, 1.0,  5e-6,  -2.0]   # T2b min = 5 µs (realistic tissue)
_MT_UB = [10.0, 0.30, 200.0, 100e-6, 2.0]


def _fit_MT(ppm: np.ndarray, zspec: np.ndarray,
             satpwr_uT: float, R1: float, B0_MHz: float,
             ppm_exclude: tuple = (-8.0, 8.0),
             ppm_reinclude: tuple = (-0.5, 0.5)) -> dict:
    """
    Fit the MT pool from Z-spectrum far off-resonance wings.

    Fitting mask: (|ppm| > ppm_exclude) OR (|ppm| < |ppm_reinclude|) — water.
    Parameters: rb, mb0, r, t2b_s, delta_MT_ppm  (t2a fixed at 0.02 s for fit,
    then t2a = 0 used to generate the MT-only curve so direct saturation = 0).

    Returns dict with keys: mt_Z, fit_mask, params (dict), converged (bool).
    """
    w1_rad = satpwr_uT * _GAMMA_HZ_UT * 2.0 * np.pi
    dHz    = ppm * B0_MHz

    lo, hi   = sorted(ppm_exclude)
    rlo, rhi = sorted(ppm_reinclude)
    mask = (ppm < lo) | (ppm > hi) | ((ppm > rlo) & (ppm < rhi))
    if mask.sum() < 6:    # auto-widen if too few points
        mask = (ppm < lo * 0.5) | (ppm > hi * 0.5) | ((ppm > rlo) & (ppm < rhi))

    d_fit = dHz[mask]
    z_fit = zspec[mask]

    def resid(params):
        rb, mb0, r, t2b, dmt = params
        zp  = _mt_model(0.02, R1, rb, mb0, r, t2b, dmt * B0_MHz, d_fit, w1_rad)
        res = zp - z_fit
        return np.where(np.isfinite(res), res, 1.0)

    mt_Z      = np.ones_like(ppm)
    params_d  = dict(rb=1.0, mb0=0.05, r=40.0, t2b_s=10e-6, delta_MT_ppm=0.0)
    converged = False

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = least_squares(resid, _MT_X0, bounds=(_MT_LB, _MT_UB),
                                method='trf', max_nfev=1500,
                                ftol=1e-8, xtol=1e-8, gtol=1e-8)
        rb, mb0, r, t2b, dmt = res.x
        params_d  = dict(rb=rb, mb0=mb0, r=r, t2b_s=t2b, delta_MT_ppm=dmt)
        # MT-only curve: t2a = 0 removes free-pool direct-saturation contribution
        mt_Z      = _mt_model(0.0, R1, rb, mb0, r, t2b, dmt * B0_MHz, dHz, w1_rad)
        converged = True
    except Exception:
        pass

    return dict(mt_Z=mt_Z, fit_mask=mask, params=params_d, converged=converged)


# =============================================================================
#  Step 2 — CEST peak fitting in 1/Z space
# =============================================================================

POOL_NAMES = ('water', 'OH', 'amine', 'amide')

# Pools exempt from the narrow-pool width cap in _fit_CEST: the legitimately
# broad water/MT/NOE, plus OH (kept at its original wide bounds per user request).
_INVZ_BROAD_POOLS = frozenset({'water', 'MT', 'NOE', 'OH'})

# Lorentzian bounds: [amp, FWHM_ppm, offset_ppm, phase]
_L_DEFS = {
    # Water = direct-saturation (DS) peak.  FWHM free from 0.1 ppm (reference
    # cest_1divZ_fit.py uses 0.01) so the fit can recover the tall, sharp central
    # DS spike — the dominant peak near 0 ppm in R₁cos²θ(1/Z−1) space.  The old
    # 1.5-ppm floor (a broad, low water hump) was a workaround for the sqrt-
    # Lorentzian tail bug that let far pools bleed together; now that _lorentzian
    # carries the arctan2 phase (true absorption Lorentzian, light 1/Δ² tails), a
    # narrow water no longer leaks into iopamidol 4.3/5.5, and forcing it broad
    # only suppressed the water peak the user expects to see.
    'water':  dict(x0=[5.0, 1.0,  0.0, 0.], lb=[0.1, 0.1, -0.3,-1e-8], ub=[500., 10.,  0.3, 1e-8]),
    # OH kept at its original FWHM ub 10 (user request).  amine/amide/NOE FWHM
    # ub tightened 10→4 ppm: those mobile CEST pools are narrow, and a broad
    # neighbour otherwise bleeds into far pools (amine/amide swallowing
    # iopamidol at 4.3/5.5 ppm).
    'OH':     dict(x0=[0.8, 3.0,  0.8, 0.], lb=[0.,  0.2,  0.6,-1e-8], ub=[  5., 10.,  1.0, 1e-8]),
    'amine':  dict(x0=[1.0, 1.5,  3.0, 0.], lb=[0.,  0.2,  2.5,-1e-8], ub=[ 50.,  4.,  3.5, 1e-8]),
    'amide':  dict(x0=[0.5, 1.0,  3.5, 0.], lb=[0.,  0.1,  3.0,-1e-8], ub=[ 50.,  4.,  4.2, 1e-8]),
    'NOE':    dict(x0=[0.5, 1.5, -3.5, 0.], lb=[0.,  0.2, -4.5,-1e-8], ub=[ 30.,  4., -2.5, 1e-8]),
    'MT':     dict(x0=[1.0,20.0, -2.0, 0.], lb=[0.,  5.0, -5.0,-1e-8], ub=[100., 60.,  0.0, 1e-8]),
    # Extra pools — bounds from setLPeakBounds.m, amplitudes scaled to 1/Z space.
    # Trp FWHM start 10→2 and ub 100→4: an ~100-ppm-wide Trp is a flat pedestal
    # that swamps the 5–6 ppm region (Trp 5.4 ≈ iopamidol 5.5).
    'Trp':    dict(x0=[0.5, 2.0,  5.4, 0.], lb=[0.,  0.2,  5.1,-1e-8], ub=[ 30.,  4.,  5.7, 1e-8]),
    '4.4ppm': dict(x0=[0.5, 1.0,  4.5, 0.], lb=[0.,  0.2,  4.0,-1e-8], ub=[ 30.,  5.,  5.0, 1e-8]),
    '7.3ppm': dict(x0=[0.5, 1.0,  7.5, 0.], lb=[0.,  0.2,  7.0,-1e-8], ub=[ 30.,  5.,  8.0, 1e-8]),
    '9.8ppm': dict(x0=[0.5, 1.0, 10.0, 0.], lb=[0.,  0.2,  9.0,-1e-8], ub=[ 30.,  5., 11.0, 1e-8]),
    # Additional CEST agents (offset windows from setLPeakBounds.m / _INVZ_BOUNDS)
    'glucose':      dict(x0=[0.5, 1.0,  1.2, 0.], lb=[0., 0.2, 0.8, -1e-8], ub=[30.,  5.,  1.6,  1e-8]),
    'creatine':     dict(x0=[0.5, 1.0,  1.9, 0.], lb=[0., 0.2, 1.5, -1e-8], ub=[30.,  5.,  2.3,  1e-8]),
    'taurine':      dict(x0=[0.5, 1.0,  3.2, 0.], lb=[0., 0.2, 2.8, -1e-8], ub=[30.,  5.,  3.6,  1e-8]),
    'iopamidol43':  dict(x0=[1.0, 1.0,  4.3, 0.], lb=[0., 0.2, 4.0, -1e-8], ub=[30.,  5.,  4.75, 1e-8]),
    'iopamidol55':  dict(x0=[1.0, 1.0,  5.5, 0.], lb=[0., 0.2, 5.25,-1e-8], ub=[30.,  5.,  5.75, 1e-8]),
    '3omg':         dict(x0=[0.5, 1.0,  1.2, 0.], lb=[0., 0.2, 0.8, -1e-8], ub=[30.,  5.,  1.6,  1e-8]),
    'polylysine':   dict(x0=[0.5, 2.0,  3.6, 0.], lb=[0., 0.5, 3.4, -1e-8], ub=[30.,  8.,  3.9,  1e-8]),
    'GAG':          dict(x0=[0.5, 1.0,  1.0, 0.], lb=[0., 0.2, 0.6, -1e-8], ub=[30.,  5.,  1.4,  1e-8]),
    'myo_inositol': dict(x0=[0.5, 1.5,  3.5, 0.], lb=[0., 0.3, 3.1, -1e-8], ub=[30.,  6.,  3.9,  1e-8]),
    'guanidinium':  dict(x0=[0.5, 1.0,  2.0, 0.], lb=[0., 0.2, 1.5, -1e-8], ub=[30.,  5.,  2.5,  1e-8]),
}
# Pseudo-Voigt bounds: [amp, alpha, FWHM_L, FWHM_ratio, offset, phase]
_PV_DEFS = {
    # Water/DS FWHM_L free from 0.1 ppm so the sharp central DS peak is recovered
    # (see the Lorentzian _L_DEFS['water'] note above — matches the reference).
    'water':  dict(x0=[5.,0.3,1.0,1., 0.0,0.], lb=[0.1,0.,0.1, 1.,-0.3,-1e-8], ub=[500.,1., 5.,2.,  0.3,1e-8]),
    'OH':     dict(x0=[0.3,0.3,1., 1., 1.0,0.], lb=[0., 0.,0.2, 1., 0.1,-1e-8], ub=[ 20.,1., 5.,2.,  1.4,1e-8]),
    'amine':  dict(x0=[1.,0.3,1., 1., 3.0,0.], lb=[0., 0.,0.5, 1., 2.5,-1e-8], ub=[ 50.,1., 5.,2.,  3.5,1e-8]),
    'amide':  dict(x0=[0.3,0.3,0.8,1., 3.5,0.], lb=[0., 0.,0.2, 1., 3.0,-1e-8], ub=[ 50.,1., 5.,2.,  4.2,1e-8]),
    'NOE':    dict(x0=[0.5,0.3,1.5,1.,-3.5,0.], lb=[0., 0.,0.5, 1.,-4.5,-1e-8], ub=[ 30.,1., 4.,2., -2.5,1e-8]),
    'MT':     dict(x0=[1.,0.3,20.,1.,-2.0,0.], lb=[0., 0.,5.0, 1.,-5.0,-1e-8], ub=[100.,1.,60.,2.,  0.0,1e-8]),
    # Extra pools — bounds from setPVPeakBounds.m, amplitudes scaled to 1/Z space
    'Trp':    dict(x0=[0.5,0.3,1.0,1., 5.4,0.], lb=[0., 0.,0.5, 1., 5.1,-1e-8], ub=[ 30.,1., 2.,2.,  5.7,1e-8]),
    '4.4ppm': dict(x0=[0.5,0.3,1.0,1., 4.5,0.], lb=[0., 0.,0.2, 1., 4.0,-1e-8], ub=[ 30.,1., 5.,2.,  5.0,1e-8]),
    '7.3ppm': dict(x0=[0.5,0.3,1.0,1., 7.5,0.], lb=[0., 0.,0.2, 1., 7.0,-1e-8], ub=[ 30.,1., 1.5,2., 8.0,1e-8]),
    '9.8ppm': dict(x0=[0.5,0.3,1.0,1.,10.0,0.], lb=[0., 0.,0.2, 1., 9.0,-1e-8], ub=[ 30.,1., 3.,2., 11.0,1e-8]),
    # Additional CEST agents (offset windows from setPVPeakBounds.m / _INVZ_BOUNDS)
    'glucose':      dict(x0=[0.5,0.3,1.0,1., 1.2,0.], lb=[0.,0.,0.2,1., 0.8,-1e-8], ub=[30.,1.,5.,2., 1.6, 1e-8]),
    'creatine':     dict(x0=[0.5,0.3,1.0,1., 1.9,0.], lb=[0.,0.,0.2,1., 1.5,-1e-8], ub=[30.,1.,5.,2., 2.3, 1e-8]),
    'taurine':      dict(x0=[0.5,0.3,1.0,1., 3.2,0.], lb=[0.,0.,0.2,1., 2.8,-1e-8], ub=[30.,1.,5.,2., 3.6, 1e-8]),
    # FWHM_L ub 5→1.5, FWHM_ratio ub 2→1.3: keep the pseudo-Voigt iopamidol peaks
    # narrow so one can't balloon into a wide Gaussian and swallow the other.
    'iopamidol43':  dict(x0=[1.0,0.3,1.0,1., 4.3,0.], lb=[0.,0.,0.2,1., 4.0,-1e-8], ub=[30.,1.,1.5,1.3, 4.75,1e-8]),
    'iopamidol55':  dict(x0=[1.0,0.3,1.0,1., 5.5,0.], lb=[0.,0.,0.2,1., 5.25,-1e-8],ub=[30.,1.,1.5,1.3, 5.75,1e-8]),
    '3omg':         dict(x0=[0.5,0.3,1.0,1., 1.2,0.], lb=[0.,0.,0.2,1., 0.8,-1e-8], ub=[30.,1.,5.,2., 1.6, 1e-8]),
    'polylysine':   dict(x0=[0.5,0.3,2.0,1., 3.6,0.], lb=[0.,0.,0.5,1., 3.4,-1e-8], ub=[30.,1.,8.,2., 3.9, 1e-8]),
    'GAG':          dict(x0=[0.5,0.3,1.0,1., 1.0,0.], lb=[0.,0.,0.2,1., 0.6,-1e-8], ub=[30.,1.,5.,2., 1.4, 1e-8]),
    'myo_inositol': dict(x0=[0.5,0.3,1.5,1., 3.5,0.], lb=[0.,0.,0.3,1., 3.1,-1e-8], ub=[30.,1.,6.,2., 3.9, 1e-8]),
    'guanidinium':  dict(x0=[0.5,0.3,1.0,1., 2.0,0.], lb=[0.,0.,0.2,1., 1.5,-1e-8], ub=[30.,1.,5.,2., 2.5, 1e-8]),
}


def _fit_CEST(ppm: np.ndarray, invZ_data: np.ndarray, invZ_MT: np.ndarray,
               satpwr_uT: float, R1: float, B0_MHz: float,
               peak_type: str = 'lorentzian',
               ppm_include: tuple = (-2.0, 5.0),
               pool_names: tuple | None = None) -> dict:
    """
    Fit CEST pools (water, OH, amine, amide) in R₁·cos²θ·(1/Z−1) space.

    The cos²θ weighting is applied INSIDE the fit (each pool × cos²θ),
    so the y-axis of the fitting problem is R₁·(1/Z−1) (without cos²θ),
    while the displayed/returned curves are in R₁·cos²θ·(1/Z−1) space.

    Returns dict: coeffs, ind_fits, ppm_fit, target, cos2, mt_cos2.
    """
    if pool_names is None:
        pool_names = POOL_NAMES

    satHz  = satpwr_uT * _GAMMA_HZ_UT
    cos2th = (ppm * B0_MHz)**2 / ((ppm * B0_MHz)**2 + satHz**2)

    lo, hi = sorted(ppm_include)
    mask   = (ppm >= lo) & (ppm <= hi)
    pf     = ppm[mask]
    c2     = cos2th[mask]

    # Target: (data − MT) × cos²θ in the CEST fitting range
    target = np.maximum((invZ_data[mask] - invZ_MT[mask]) * c2, 0.0)
    target = np.where(np.isfinite(target), target, 0.0)

    pfn  = _pfn(peak_type)
    defs = _PV_DEFS if peak_type == 'pseudovoigt' else _L_DEFS
    np_  = 6 if peak_type == 'pseudovoigt' else 4

    # Filter out any pools that don't have bounds defined — silently skip unknowns
    pool_names = [p for p in pool_names if p in defs]
    if not pool_names:
        pool_names = list(POOL_NAMES)

    # Cap narrow-pool width so a pseudo-Voigt (or heavy-wing Lorentzian) pool
    # can't balloon and swallow a neighbour (e.g. iopamidol 5.5 eating 4.3).
    # water/MT/NOE are legitimately broad and exempt.
    _is_pv = (peak_type == 'pseudovoigt')

    def _cap_ub(p):
        ub_v = list(defs[p]['ub'])
        if p not in _INVZ_BROAD_POOLS:
            if _is_pv:
                ub_v[2] = min(ub_v[2], 1.5)   # FWHM_L
                ub_v[3] = min(ub_v[3], 1.3)   # FWHM_ratio
            else:
                ub_v[1] = min(ub_v[1], 2.0)   # FWHM
        return ub_v

    x0 = np.concatenate([defs[p]['x0'] for p in pool_names])
    lb = np.concatenate([defs[p]['lb'] for p in pool_names])
    ub = np.concatenate([_cap_ub(p) for p in pool_names])
    x0 = np.clip(x0, lb, ub)   # keep the start point inside the capped bounds

    # Initial coeffs (default start point)
    coeffs = {nm: np.array(defs[nm]['x0']) for nm in pool_names}

    def resid(params):
        tot = np.zeros_like(pf)
        for i, nm in enumerate(pool_names):
            tot += c2 * pfn(params[i * np_:(i + 1) * np_], pf)
        return tot - target

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = least_squares(resid, x0, bounds=(lb, ub), method='trf', max_nfev=4000,
                              ftol=1e-8, xtol=1e-8, gtol=1e-8)
        for i, nm in enumerate(pool_names):
            coeffs[nm] = r.x[i * np_:(i + 1) * np_]
    except Exception:
        pass

    ind_fits = {nm: c2 * pfn(coeffs[nm], pf) for nm in pool_names}
    mt_cos2  = invZ_MT[mask] * c2

    return dict(coeffs=coeffs, ind_fits=ind_fits,
                ppm_fit=pf, target=target, cos2=c2, mt_cos2=mt_cos2)


# =============================================================================
#  Full pipeline
# =============================================================================

def _expand_cest_range(ppm_include: tuple, pool_names, margin: float = 0.7) -> tuple:
    """Widen the CEST fit window so every selected pool's resonance is covered.

    A pool selected far off-resonance (e.g. 7.3 ppm) would otherwise fall
    outside the default (−2, 5) window and fit to zero. The offset bounds come
    from _L_DEFS (index 2 = offset, same values as _PV_DEFS).
    """
    lo, hi = sorted(ppm_include)
    for pn in (pool_names or ()):
        d = _L_DEFS.get(pn)
        if d is None:
            continue
        lo = min(lo, float(d['lb'][2]) - margin)
        hi = max(hi, float(d['ub'][2]) + margin)
    return (lo, hi)


def _run_pipeline(ppm: np.ndarray, zspec: np.ndarray,
                   satpwr_uT: float, R1: float, B0_MHz: float,
                   peak_type: str = 'lorentzian',
                   ppm_exclude_MT: tuple = (-8.0, 8.0),
                   ppm_reinclude_MT: tuple = (-0.5, 0.5),
                   ppm_include_CEST: tuple = (-2.0, 5.0),
                   pool_names: tuple | None = None,
                   fit_mt: bool = True) -> dict:
    """
    Full three-step 1/Z fitting pipeline.

    Returns a flat dict with all intermediate results needed for
    plotting the three diagnostic panels (MT fit, 1/Z decomp, Z recon).
    """
    _active = pool_names if pool_names is not None else POOL_NAMES
    # Make sure the CEST window covers every selected pool (e.g. 7.3 ppm).
    ppm_include_CEST = _expand_cest_range(ppm_include_CEST, _active)

    sort_i = np.argsort(ppm)
    ppm_s  = ppm[sort_i].astype(float)
    z_s    = np.clip(zspec[sort_i].astype(float), 1e-6, 1.0)

    # ── Step 1: MT ────────────────────────────────────────────────────────────
    if fit_mt:
        mt = _fit_MT(ppm_s, z_s, satpwr_uT, R1, B0_MHz,
                      ppm_exclude_MT, ppm_reinclude_MT)
        mt_Z, fit_mask = mt['mt_Z'], mt['fit_mask']
    else:
        # MT disabled → flat background (Z_MT = 1 → invZ_MT = 0), as in the
        # reference driver (FitResult.MT.Z = ones).  The "fit mask" still marks
        # the far-offset wings so the diagnostic panel can highlight them.
        lo, hi = sorted(ppm_exclude_MT)
        rlo, rhi = sorted(ppm_reinclude_MT)
        fit_mask = (ppm_s < lo) | (ppm_s > hi) | ((ppm_s > rlo) & (ppm_s < rhi))
        mt_Z = np.ones_like(ppm_s)
        mt = dict(mt_Z=mt_Z, fit_mask=fit_mask,
                  params=dict(rb=0., mb0=0., r=0., t2b_s=0., delta_MT_ppm=0.),
                  converged=True)

    # ── Step 2: 1/Z transform + CEST ─────────────────────────────────────────
    invZ_data = R1 * (1.0 / np.clip(z_s,    1e-6, None) - 1.0)
    invZ_MT   = R1 * (1.0 / np.clip(mt_Z,   1e-6, None) - 1.0)
    cest = _fit_CEST(ppm_s, invZ_data, invZ_MT,
                      satpwr_uT, R1, B0_MHz, peak_type, ppm_include_CEST,
                      pool_names=_active)

    # ── Step 3: Reconstruct Z-spectrum ───────────────────────────────────────
    satHz    = satpwr_uT * _GAMMA_HZ_UT
    cos2_all = (ppm_s * B0_MHz)**2 / ((ppm_s * B0_MHz)**2 + satHz**2)
    pfn      = _pfn(peak_type)

    rcos2 = invZ_MT * cos2_all                                # MT contribution
    for nm in _active:
        if nm in cest['coeffs']:
            rcos2 += cos2_all * pfn(cest['coeffs'][nm], ppm_s)   # CEST contributions

    z_fit = np.clip(R1 * cos2_all / (rcos2 + R1 * cos2_all + 1e-20), 0.0, 1.0)

    return dict(
        # inputs (sorted)
        ppm=ppm_s, zspec=z_s,
        satpwr_uT=satpwr_uT, R1=R1, B0_MHz=B0_MHz, peak_type=peak_type,
        # MT step
        mt_Z=mt_Z, mt_fit_mask=fit_mask, mt_params=mt['params'],
        mt_converged=mt['converged'],
        # 1/Z step
        invZ_data=invZ_data, invZ_MT=invZ_MT,
        cest_ppm_fit=cest['ppm_fit'],
        cest_target=cest['target'],
        cest_cos2=cest['cos2'],
        cest_mt_cos2=cest['mt_cos2'],
        cest_ind_fits=cest['ind_fits'],
        cest_coeffs=cest['coeffs'],
        # reconstruction
        z_fit=z_fit, cos2_all=cos2_all,
    )


# =============================================================================
#  QUESP — fs / ksw from per-B1 CEST peak amplitudes
#  (MTRasym / MTRRex / Ω-plot, ported from QUESPfcn.m + the amine steady-state
#   Z reconstruction in MT_CEST_fit_QUESP_dk.m, Chen et al. NMR Biomed 2017 Eq.7)
# =============================================================================

def _rsq(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ss_res = np.nansum((y - yhat) ** 2)
    ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')


def _fit_quesp_variant(model, w1: np.ndarray, y: np.ndarray, *,
                        b1_plot: np.ndarray,
                        p0=(1e-3, 2000.0), lo=(1e-5, 0.0), hi=(1e-1, 12000.0)):
    """Fit a 2-parameter QUESP model  y(w1) = model(w1, fs, ksw).

    w1 is in rad/s; b1_plot is the matching B1 axis in µT (for plotting the
    fitted curve).  Returns a dict with x/y (measured), xfit/yfit (curve in B1
    units), and the recovered fs, ksw, R².
    """
    from scipy.optimize import curve_fit as _cf
    d = dict(x=np.asarray(b1_plot, float), y=np.asarray(y, float),
             fs=float('nan'), ksw=float('nan'), rsq=float('nan'),
             xfit=None, yfit=None)
    ok = np.isfinite(w1) & np.isfinite(y)
    if ok.sum() < 2:
        return d
    try:
        popt, _ = _cf(model, w1[ok], y[ok], p0=p0, bounds=(lo, hi), maxfev=20000)
        fs, ksw = float(popt[0]), float(popt[1])
        xg = np.linspace(float(b1_plot[ok].min()), float(b1_plot[ok].max()), 120)
        d.update(fs=fs, ksw=ksw, xfit=xg,
                 yfit=model(xg * _GAMMA_2PI, fs, ksw),
                 rsq=_rsq(y[ok], model(w1[ok], fs, ksw)))
    except Exception:
        pass
    return d


def _quesp_pool(coeffs_list: list, satpwr_list: list,
                R1: float, B0_MHz: float, tsat: float, trec: float) -> dict | None:
    """Run the three QUESP variants for ONE pool.

    `coeffs_list[k]` is that pool's fitted peak-parameter vector at B1 index k
    (amplitude = p[0], offset[ppm] = p[-2], for both Lorentzian and
    pseudo-Voigt).  Reconstructs the pool's clean steady-state Z (Chen 2017
    Eq.7) using Z_ref = 1, then fits Regular (MTRasym), Inverse (MTRRex) and the
    Ω-plot.  Returns None if fewer than 2 usable B1 points.
    """
    b1 = np.asarray(satpwr_list, float)
    order = np.argsort(b1)
    b1 = b1[order]
    cl = [np.asarray(coeffs_list[i], float) for i in order]
    if len(b1) < 2:
        return None

    satHz = b1 * _GAMMA_HZ_UT
    w1    = b1 * _GAMMA_2PI                                 # rad/s

    amp = np.array([c[0]  for c in cl], float)
    off = np.array([c[-2] for c in cl], float)             # peak offset [ppm]
    cos2 = (B0_MHz * off) ** 2 / ((B0_MHz * off) ** 2 + satHz ** 2 + 1e-20)
    Rpeak = amp * cos2                                     # undo cos²θ fitting

    ZssPeak = R1 * cos2 / (Rpeak + R1 * cos2 + 1e-20)      # Chen 2017 Eq.7
    Zref    = np.ones_like(ZssPeak)

    out = dict(b1=b1, w1=w1, ZssPeak=ZssPeak, Rpeak=Rpeak, off=off)

    Zi = 1.0 - np.exp(-R1 * trec)

    # ── Regular QUESP — MTRasym = Zref − Zlab ──────────────────────────────────
    MTRasym = Zref - ZssPeak

    def _reg(w, fs, ksw):
        Rex  = fs * ksw * w ** 2 / (w ** 2 + ksw ** 2)
        Reff = R1 + Rex
        return (Rex / Reff
                - (Zi - R1 / Reff) * np.exp(-Reff * tsat)
                + (Zi - 1.0) * np.exp(-R1 * tsat))

    out['regular'] = _fit_quesp_variant(_reg, w1, MTRasym, b1_plot=b1)

    # ── Inverse QUESP — MTR_Rex = 1/Zlab − 1/Zref ──────────────────────────────
    MTRrex = 1.0 / np.clip(ZssPeak, 1e-6, None) - 1.0 / Zref

    def _inv(w, fs, ksw):
        return fs * ksw * w ** 2 / (w ** 2 + ksw ** 2) / R1

    out['inverse'] = _fit_quesp_variant(_inv, w1, MTRrex, b1_plot=b1)

    # ── Ω-plot — 1/MTR_Rex vs 1/w1²  (linear, weighted) ────────────────────────
    # Drop noise-dominated points where MTR_Rex ≈ 0 (1/MTR_Rex → ∞): these are
    # B1 powers with no measurable CEST effect and blow up both the fit and the
    # y-axis.  The reference downweights them via Weights = 1/yData; the extra
    # threshold keeps the plot legible (standard Ω-plot practice, Dixon 2010).
    xw = 1.0 / w1 ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        yw = 1.0 / MTRrex
    _mtr_max = float(np.nanmax(MTRrex)) if np.isfinite(MTRrex).any() else 0.0
    _thr = 0.05 * _mtr_max if _mtr_max > 0 else 0.0
    good = np.isfinite(yw) & (MTRrex > _thr) & (MTRrex > 0)
    om = dict(x=xw, y=yw, good=good, fs=float('nan'), ksw=float('nan'),
              rsq=float('nan'), xfit=None, yfit=None)
    if good.sum() >= 2:
        from scipy.optimize import curve_fit as _cf

        def _om(x, fs, ksw):
            return R1 * (1.0 / (fs * ksw) + ksw / fs * x)
        try:
            popt, _ = _cf(_om, xw[good], yw[good], p0=(1e-3, 2000.0),
                          bounds=((1e-6, 1.0), (1e-1, 3e5)),
                          sigma=yw[good], absolute_sigma=False, maxfev=20000)
            fs, ksw = float(popt[0]), float(popt[1])
            xg = np.linspace(0.0, float(xw[good].max()) * 1.05, 120)
            om.update(fs=fs, ksw=ksw, xfit=xg, yfit=_om(xg, fs, ksw),
                      rsq=_rsq(yw[good], _om(xw[good], fs, ksw)))
        except Exception:
            pass
    out['omega'] = om
    return out


# =============================================================================
#  Worker — fast AREX map (full 1/Z fit done per-ROI in the dialog)
# =============================================================================

class InvZSpecWorker(QThread):
    """
    Computes per-voxel AREX = (1/Z(+ppm) − 1/Z(−ppm)) × R1 for every loaded
    dataset (one per B1 saturation power).  Results are returned as a list so
    the tab can show one AREX map per B1 power and compare them side-by-side.
    """
    log       = pyqtSignal(str)
    progress  = pyqtSignal(int)
    finished  = pyqtSignal(dict)
    error     = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, datasets: list, eval_ppm: float,
                 R1: float, b0_mhz: float, eval_offsets=None, parent=None):
        super().__init__(parent)
        self._datasets = datasets
        # AREX is evaluated at the SELECTED-pool offsets (one AREX map per pool,
        # e.g. 4.2 & 5.5 for iopamidol).  Only when NO pools are selected do we
        # fall back to the manual eval_ppm — so ±3.0 no longer appears just
        # because the spin-box defaults to 3.0 when amine isn't selected.
        src = list(eval_offsets) if eval_offsets else [float(eval_ppm)]
        uniq: list[float] = []
        for e in src:
            e = round(abs(float(e)), 2)
            if e > 0.05 and e not in uniq:
                uniq.append(e)
        self._evals = uniq or [round(abs(float(eval_ppm)), 2)]
        self._eval  = self._evals[0]
        self._R1    = R1
        self._b0    = b0_mhz
        self._stop  = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            all_results = []
            n_ds = len(self._datasets)
            for i, ds in enumerate(self._datasets):
                if self._stop.is_set():
                    self.cancelled.emit()
                    return

                b1_ut = float(ds.get('b1_ut', 2.5))
                z4    = ds['z_img']
                ppm   = np.array(ds['ppm'], dtype=float)
                sl    = ds.get('slice', 0)
                R1    = self._R1

                z_sl  = z4[:, :, sl, :] if z4.ndim == 4 else z4
                Y, X, N = z_sl.shape

                sort_i = np.argsort(ppm)
                ppm_s  = ppm[sort_i]
                z_flat = np.clip(
                    z_sl.reshape(-1, N)[:, sort_i].astype(float), 1e-9, None)
                z_mean = z_flat.mean(axis=0)

                for ep in self._evals:
                    if self._stop.is_set():
                        self.cancelled.emit()
                        return
                    self.log.emit(
                        f"[{i+1}/{n_ds}] B1={b1_ut:.2f} µT — AREX at ±{ep:.2f} ppm"
                        f"  ({Y}×{X} voxels)…"
                    )
                    Zp = np.array([np.interp( ep, ppm_s, z_flat[v]) for v in range(Y * X)])
                    Zn = np.array([np.interp(-ep, ppm_s, z_flat[v]) for v in range(Y * X)])
                    Zp = np.clip(Zp, 1e-9, None)
                    Zn = np.clip(Zn, 1e-9, None)
                    arex = (1.0 / Zp - 1.0 / Zn) * R1

                    all_results.append(dict(
                        arex_map  = arex.reshape(Y, X),
                        eval_ppm  = ep,
                        R1        = R1,
                        b1_ut     = b1_ut,
                        ppm       = ppm_s,
                        z_mean    = z_mean,
                        img_shape = (Y, X),
                        path      = ds.get('path', ''),
                    ))

                self.progress.emit(int((i + 1) * 100 / n_ds))

            self.finished.emit({'all_results': all_results})

        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


# =============================================================================
#  Worker — per-voxel QUESP maps from the full 1/Z fit
# =============================================================================

def _block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """Spatially block-average a (Y,X[,N]) array by integer factor f."""
    if f <= 1:
        return a
    Y, X = a.shape[:2]
    Yc, Xc = (Y // f) * f, (X // f) * f
    a = a[:Yc, :Xc]
    new_shape = (Yc // f, f, Xc // f, f) + a.shape[2:]
    return a.reshape(new_shape).mean(axis=(1, 3))


def _block_any(m: np.ndarray, f: int) -> np.ndarray:
    """Block-reduce a boolean mask by factor f (a block is kept if any voxel is)."""
    if f <= 1:
        return m
    Y, X = m.shape[:2]
    Yc, Xc = (Y // f) * f, (X // f) * f
    m = m[:Yc, :Xc]
    return m.reshape(Yc // f, f, Xc // f, f).any(axis=(1, 3))


def _upsample_to(small: np.ndarray, shape: tuple, f: int) -> np.ndarray:
    """Nearest-neighbour upsample a small map back to `shape` by factor f."""
    if f <= 1 and small.shape == shape:
        return small
    up = np.repeat(np.repeat(small, f, axis=0), f, axis=1)
    out = np.full(shape, np.nan, dtype=float)
    yy = min(up.shape[0], shape[0])
    xx = min(up.shape[1], shape[1])
    out[:yy, :xx] = up[:yy, :xx]
    return out


# ── Parallel per-voxel fitting for the fs/ksw maps ──────────────────────────
# These are module-level (picklable) so a ProcessPoolExecutor can fan the
# per-voxel 1/Z fit out across CPU cores.  The whole call-chain (_run_pipeline
# → _fit_MT / _fit_CEST) is pure numpy/scipy; results are deterministic, so the
# parallel maps are bit-identical to the serial ones.

def _invz_pool_init():
    """Worker-process init: pin BLAS to 1 thread so N processes don't
    oversubscribe the CPU (each fit is already single-threaded work)."""
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")


def _fit_invz_amp_chunk(chunk):
    """Fit a chunk of (bi, yy, xx, ppm, zspec, b1) voxels; return
    [(bi, yy, xx, {pool: amplitude}), …].  Picklable, Qt-free at run time."""
    tasks, R1, B0, peak_type, ex_mt, re_mt, inc_cest, pools, qpools, fit_mt = chunk
    out = []
    for (bi, yy, xx, ppm, zspec, b1u) in tasks:
        amps = {}
        try:
            r = _run_pipeline(ppm, zspec, b1u, R1, B0, peak_type,
                              ex_mt, re_mt, inc_cest,
                              pool_names=pools, fit_mt=fit_mt)
            cc = r['cest_coeffs']
            for pn in qpools:
                if pn in cc:
                    amps[pn] = float(cc[pn][0])
        except Exception:
            pass
        out.append((bi, yy, xx, amps))
    return out


class InvZQUESPWorker(QThread):
    """
    Per-voxel QUESP from the full 1/Z fit.

    For every masked voxel and every loaded B1 dataset, runs the three-step
    MT + CEST pipeline, extracts each CEST pool's peak amplitude A (in
    R1·(1/Z−1) space → MTR_Rex = A/R1), accumulates MTR_Rex across B1, then
    fits the linear inverse-QUESP model per voxel → fs, ksw, R² maps per pool.
    """
    log       = pyqtSignal(str)
    progress  = pyqtSignal(int)
    finished  = pyqtSignal(dict)
    error     = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, datasets, params, pools, mask, *,
                 n_H: int = 1, max_fit_dim: int = 128, parent=None):
        super().__init__(parent)
        self._datasets    = datasets
        self._p           = params
        self._pools       = list(pools)
        self._mask        = mask           # full-res bool (Y,X) or None
        self._n_H         = int(n_H)
        self._max_fit_dim = int(max_fit_dim)
        self._stop        = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            from my_gui.quesp_from_invz import linear_quesp, fs_to_concentration

            p   = self._p
            R1  = float(p['R1'])
            B0  = float(p['B0_MHz'])
            ptp = p['peak_type']
            # CEST exchange pools only — water (direct sat) and MT (semisolid)
            # are fit for spillover removal but not QUESP-quantified.
            qpools = [pn for pn in self._pools if pn not in ('water', 'MT')]
            if not qpools:
                self.error.emit("No CEST exchange pools selected for QUESP.")
                return

            ds0   = self._datasets[0]
            z0    = np.array(ds0['z_img'])
            sl0   = ds0.get('slice', 0)
            z0sl  = z0[:, :, sl0, :] if z0.ndim == 4 else z0
            Yf, Xf = z0sl.shape[:2]

            # Downsample factor for tractable voxelwise fitting
            f = max(1, int(np.ceil(max(Yf, Xf) / float(self._max_fit_dim))))

            # Mask (full-res) → working-res
            if self._mask is not None and self._mask.shape == (Yf, Xf):
                mask_w = _block_any(self._mask.astype(bool), f)
            else:
                mask_w = None

            # Per-dataset working-res z-volumes + ppm + B1
            b1_list, z_works, ppm_works = [], [], []
            for ds in self._datasets:
                zv = np.array(ds['z_img'])
                sl = ds.get('slice', 0)
                zsl = zv[:, :, sl, :] if zv.ndim == 4 else zv
                if zsl.shape[:2] != (Yf, Xf):
                    self.log.emit(
                        f"  ⚠ skipping B1={ds.get('b1_ut','?')} µT "
                        f"(shape {zsl.shape[:2]} ≠ {(Yf, Xf)})")
                    continue
                zsl = np.clip(zsl.astype(float), 1e-6, 1.0)
                z_works.append(_block_mean(zsl, f))
                ppm_works.append(np.array(ds['ppm'], dtype=float))
                b1_list.append(float(ds.get('b1_ut', p['satpwr_uT'])))

            n_b1 = len(z_works)
            if n_b1 < 3:
                self.error.emit(
                    f"QUESP needs ≥3 B1 powers (got {n_b1}). "
                    "Add more saturation-power datasets.")
                return

            Yw, Xw = z_works[0].shape[:2]
            b1_arr = np.array(b1_list, dtype=float)

            if mask_w is None:
                # validity mask: voxels whose mean Z has signal across B1
                vmean = np.mean([z.mean(axis=2) for z in z_works], axis=0)
                mask_w = vmean > 0.05
            vox = np.argwhere(mask_w)
            n_vox = len(vox)
            if n_vox == 0:
                self.error.emit("Mask is empty — draw an ROI or load a phantom.")
                return

            self.log.emit(
                f"Per-voxel QUESP: {n_vox} voxels × {n_b1} B1 powers "
                f"(grid {Yw}×{Xw}, downsample ×{f}) — pools: {', '.join(qpools)}")

            # MTR_Rex stack: per pool, (n_b1, Yw, Xw)
            rex = {pn: np.full((n_b1, Yw, Xw), np.nan) for pn in qpools}

            # Flatten (B1 × voxel) into one task list and fit them across CPU
            # cores.  Falls back to serial if the process pool can't start.
            n_workers = int(getattr(self, '_n_workers', 0) or (os.cpu_count() or 4))
            n_workers = max(1, min(n_workers, 16))
            tasks = []
            for bi in range(n_b1):
                ppm_s = ppm_works[bi]; zw = z_works[bi]; b1u = b1_list[bi]
                for (yy, xx) in vox:
                    tasks.append((bi, int(yy), int(xx), ppm_s, zw[yy, xx], b1u))
            shared = (R1, B0, ptp, p['ppm_exclude_MT'], p['ppm_reinclude_MT'],
                      p['ppm_include_CEST'], tuple(self._pools), tuple(qpools),
                      p.get('fit_mt', True))
            n_tasks = len(tasks)
            cs = max(1, n_tasks // max(n_workers * 3, 1))     # ~3 chunks/worker
            chunks = [(tasks[i:i + cs], *shared) for i in range(0, n_tasks, cs)]

            def _store(results):
                for bi, yy, xx, amps in results:
                    for pn, A in amps.items():
                        rex[pn][bi, yy, xx] = A / R1 if R1 > 0 else A

            used_parallel = False
            if n_workers > 1 and len(chunks) > 1:
                try:
                    exe = ProcessPoolExecutor(max_workers=n_workers,
                                              mp_context=_mp.get_context('spawn'),
                                              initializer=_invz_pool_init)
                    with exe:
                        futs = [exe.submit(_fit_invz_amp_chunk, c) for c in chunks]
                        done = 0
                        for fut in as_completed(futs):
                            if self._stop.is_set():
                                exe.shutdown(wait=False, cancel_futures=True)
                                self.cancelled.emit(); return
                            _store(fut.result())
                            done += 1
                            self.progress.emit(int(done * 90 / len(chunks)))
                    used_parallel = True
                    self.log.emit(
                        f"  {n_tasks} voxel-fits done on {n_workers} workers.")
                except Exception as _exc:
                    self.log.emit(
                        f"  parallel pool unavailable ({_exc}); running serially.")

            if not used_parallel:
                done = 0
                for t in tasks:
                    if self._stop.is_set():
                        self.cancelled.emit(); return
                    _store(_fit_invz_amp_chunk(([t], *shared)))
                    done += 1
                    if done % 25 == 0 or done == n_tasks:
                        self.progress.emit(int(done * 90 / n_tasks))

            # ── Per-voxel linear QUESP ────────────────────────────────────────
            quesp_maps = {}
            for pi, pn in enumerate(qpools):
                if self._stop.is_set():
                    self.cancelled.emit(); return
                fs_m  = np.full((Yw, Xw), np.nan)
                ksw_m = np.full((Yw, Xw), np.nan)
                r2_m  = np.full((Yw, Xw), np.nan)
                stack = rex[pn]                       # (n_b1, Yw, Xw)
                for (yy, xx) in vox:
                    r = linear_quesp(b1_arr, stack[:, yy, xx])
                    fs_m[yy, xx]  = r['fs']
                    ksw_m[yy, xx] = r['ksw']
                    r2_m[yy, xx]  = r['r2']
                conc_m = fs_m * 110_000.0 / max(self._n_H, 1)
                quesp_maps[pn] = dict(
                    fs   = _upsample_to(fs_m,   (Yf, Xf), f),
                    ksw  = _upsample_to(ksw_m,  (Yf, Xf), f),
                    r2   = _upsample_to(r2_m,   (Yf, Xf), f),
                    conc = _upsample_to(conc_m, (Yf, Xf), f),
                )
                self.progress.emit(int(90 + (pi + 1) * 10 / len(qpools)))

            self.finished.emit(dict(
                quesp_maps = quesp_maps,
                b1_list    = b1_list,
                pools      = qpools,
                R1         = R1,
                n_H        = self._n_H,
                img_shape  = (Yf, Xf),
            ))

        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


# =============================================================================
#  Add-scan dialog
# =============================================================================

class _AddScanDialog(QDialog):
    """Dialog to select a Z-spectrum dataset (path + slice)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Z-spectrum scan")
        self.setMinimumWidth(420)
        form = QFormLayout(self)

        # Platform selector (above the data path) — Bruker shows the ParaVision
        # version; GE / Siemens shows only the data path.
        self.combo_platform = QComboBox()
        self.combo_platform.addItems(["Bruker", "GE / Siemens"])
        self.combo_platform.setToolTip(
            "Bruker → also pick the ParaVision version.\n"
            "GE / Siemens → data path only.")
        form.addRow("Platform:", self.combo_platform)

        # Data path
        path_row = QHBoxLayout()
        self.edit_path = QLineEdit_()
        self.edit_path.setPlaceholderText("Select scan folder or data file…")
        self.edit_path.setReadOnly(True)
        path_row.addWidget(self.edit_path, stretch=1)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        path_row.addWidget(btn_browse)
        form.addRow("Data path:", path_row)

        self.combo_pv = QComboBox()
        self.combo_pv.addItems(["PV360", "PV6 / PV7"])
        self.combo_pv.setToolTip(
            "PV360 = ParaVision 360  |  PV6 / PV7 = ParaVision 6 or 7"
        )
        self.combo_pv.setFixedWidth(110)
        form.addRow("Bruker version:", self.combo_pv)

        # Show the Bruker-version row only when the platform is Bruker.
        def _on_platform(*_):
            is_bruker = self.combo_platform.currentText() == "Bruker"
            form.setRowVisible(self.combo_pv, is_bruker)
        self.combo_platform.currentIndexChanged.connect(_on_platform)
        _on_platform()

        self.spin_b1 = QDoubleSpinBox()
        self.spin_b1.setRange(0.01, 50.0)
        self.spin_b1.setValue(2.5)
        self.spin_b1.setSuffix(" µT")
        self.spin_b1.setDecimals(2)
        self.spin_b1.setToolTip(
            "Saturation B1 power for this scan.\n"
            "Auto-detected from Bruker method file when loading; "
            "enter manually for .mat / .npz files."
        )
        form.addRow("B1 power:", self.spin_b1)

        self.spin_slice = QSpinBox()
        self.spin_slice.setRange(1, 99)
        self.spin_slice.setValue(1)
        form.addRow("Slice #:", self.spin_slice)

        btns = QDialogButtonBox_(
            QDialogButtonBox_.StandardButton.Ok | QDialogButtonBox_.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select scan folder (Bruker pdata/1 or GE/Siemens DICOM)", "")
        if not d:
            d, _ = QFileDialog.getOpenFileName(
                self, "Or select .mat / .npz file", "",
                "Data files (*.mat *.npz)"
            )
        if d:
            self.edit_path.setText(d)

    def get_values(self):
        return (
            self.edit_path.text().strip(),
            self.combo_pv.currentText() == "PV360",
            self.spin_slice.value() - 1,   # 0-indexed
            self.spin_b1.value(),           # NEW: B1 in µT
        )


# Inline QLineEdit / QCheckBox / QDialogButtonBox imports for the dialog
from PyQt6.QtWidgets import (
    QLineEdit as QLineEdit_, QCheckBox as QCheckBox_,
    QDialogButtonBox as QDialogButtonBox_,
)


# =============================================================================
#  Main tab widget
# =============================================================================

class InvZSpecTab(QWidget):
    """
    1/Z Spectroscopy tab.

    Loads a single-B1 CEST Z-spectrum dataset, computes a fast AREX map,
    and (via ROI Spectra dialog) runs the full 3-step fitting pipeline to
    show MT fit / 1/Z decomposition / Z reconstruction per ROI.
    """

    def __init__(self):
        super().__init__()
        self._datasets: list[dict] = []
        self._worker:   InvZSpecWorker | None = None
        self._results:  dict = {}
        self._all_results: list = []
        self._quesp_display: list = []   # per-voxel QUESP maps for Display combo
        self._loaded_maps: list = []     # arbitrary maps loaded from a .mat/.npz
        self._roi_spectra_dlg = None     # cached ROI-spectra dialog (avoid re-fit)
        self._roi_spectra_key = None     # validity key for the cached dialog
        self._last_1z_key     = None     # validity key for the cached 1/Z fits
        self._quesp_plot_pools = None    # pools to draw QUESP for (None → default)
        self._last_rois: list = []
        self._roi_bg_img = None          # custom grayscale bkg for "ROIs + Bkg"
        self._scan_paths_getter = None   # callback set by app.py → scan paths
        self._r1_getter = None          # callback set by app.py → From T1 map
        self._dc_annot  = None
        self._dc_cid:   int | None = None

        # CEST fitting options (configured via dialog)
        # Default base set: water, OH, NOE, MT.  Additional pools come from
        # the main CEST MRI tab's pool selector (injected via set_main_pools_getter).
        self._inv_selected_pools: list[str] = ['water', 'OH', 'NOE', 'MT']
        # Callback that returns self.zspec_tab._global_pools; set by app.py
        self._main_pools_getter = None

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer    = QHBoxLayout(self)
        outer.addWidget(splitter)

        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setSizes([380, 620])

    # ── Build panels ─────────────────────────────────────────────────────────

    def _build_left(self) -> QScrollArea:
        left = QWidget()
        vl   = QVBoxLayout(left)
        vl.setSpacing(6)
        vl.setContentsMargins(6, 6, 6, 6)

        def _row(lbl, wid, lbl_w=140):
            r = QHBoxLayout()
            l = QLabel(lbl)
            l.setFixedWidth(lbl_w)
            r.addWidget(l)
            r.addWidget(wid)
            r.addStretch()
            return r

        # ── Z-spectrum Dataset ─────────────────────────────────────────────
        grp_data = QGroupBox("Z-Spectrum Dataset")
        # Only as tall as its contents — otherwise the panel stretches the box and
        # leaves a large empty gap below the buttons.
        from PyQt6.QtWidgets import QSizePolicy as _QSP
        grp_data.setSizePolicy(_QSP.Policy.Preferred, _QSP.Policy.Maximum)
        gd = QVBoxLayout(grp_data)

        self.scan_list = QListWidget()
        self.scan_list.setMaximumHeight(80)
        self.scan_list.setToolTip("Double-click to remove")
        self.scan_list.itemDoubleClicked.connect(self._remove_scan)
        gd.addWidget(self.scan_list)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("＋ Add scan…")
        btn_add.setFixedHeight(28)
        btn_add.clicked.connect(self._add_scan)
        btn_row.addWidget(btn_add)
        btn_rem = QPushButton("✕ Remove")
        btn_rem.setFixedHeight(28)
        btn_rem.clicked.connect(self._remove_selected)
        btn_row.addWidget(btn_rem)
        gd.addLayout(btn_row)

        self.lbl_status_data = QLabel("No scan loaded.")
        self.lbl_status_data.setStyleSheet("font-size: 11px; color: gray;")
        gd.addWidget(self.lbl_status_data)
        vl.addWidget(grp_data)

        # ── Scan-parameter + CEST-fitting spin boxes (non-displayed holders) ──
        # These are kept as lightweight data holders — they are edited through the
        # "Scan Parameters" and "Processing & Fitting" buttons below and read back
        # by _get_params().  Keeping them off the main panel declutters the tab.
        self.spin_b1 = QDoubleSpinBox()
        self.spin_b1.setRange(0.01, 50.0); self.spin_b1.setValue(2.5)
        self.spin_b1.setSuffix(" µT"); self.spin_b1.setDecimals(2)
        self.spin_b1.setToolTip("Saturation B1 power used in the CEST scan")

        self.spin_b0 = QDoubleSpinBox()
        self.spin_b0.setRange(50.0, 1500.0); self.spin_b0.setValue(400.0)
        self.spin_b0.setSuffix(" MHz"); self.spin_b0.setDecimals(1)
        self.spin_b0.setToolTip("Proton Larmor frequency (9.4 T → 400 MHz, 7 T → 298 MHz)")

        self.spin_eval = QDoubleSpinBox()
        self.spin_eval.setRange(0.0, 15.0); self.spin_eval.setValue(3.0)
        self.spin_eval.setSuffix(" ppm"); self.spin_eval.setSingleStep(0.5)
        self.spin_eval.setToolTip("Ppm offset for the AREX map")

        self.spin_r1 = QDoubleSpinBox()
        self.spin_r1.setRange(0.01, 20.0); self.spin_r1.setValue(1.0)
        self.spin_r1.setSuffix(" s⁻¹"); self.spin_r1.setSingleStep(0.05)
        self.spin_r1.setToolTip("R1 = 1/T1 [s⁻¹].  Get from T1/T2/B1 tab → T1 map.")

        self.spin_mt_excl = QDoubleSpinBox()
        self.spin_mt_excl.setRange(1.0, 20.0); self.spin_mt_excl.setValue(8.0)
        self.spin_mt_excl.setSuffix(" ppm"); self.spin_mt_excl.setSingleStep(0.5)

        self.spin_mt_rein = QDoubleSpinBox()
        self.spin_mt_rein.setRange(0.0, 3.0); self.spin_mt_rein.setValue(0.5)
        self.spin_mt_rein.setSuffix(" ppm"); self.spin_mt_rein.setSingleStep(0.1)

        self.spin_cest_lo = QDoubleSpinBox()
        self.spin_cest_lo.setRange(-20.0, 0.0); self.spin_cest_lo.setValue(-2.0)
        self.spin_cest_lo.setSuffix(" ppm"); self.spin_cest_lo.setSingleStep(0.5)

        self.spin_cest_hi = QDoubleSpinBox()
        self.spin_cest_hi.setRange(0.0, 20.0); self.spin_cest_hi.setValue(5.0)
        self.spin_cest_hi.setSuffix(" ppm"); self.spin_cest_hi.setSingleStep(0.5)

        self.combo_peak = QComboBox()
        self.combo_peak.addItems(["Lorentzian", "Pseudo-Voigt"])

        # MT fitting toggle (edited via Processing & Fitting).  When unchecked the
        # MT background is treated as flat (Z_MT = 1) — matches the reference
        # pipeline default (MT_CEST_fit_QUESP_dk.m: FitResult.MT.Z = ones).
        self.chk_mt_fit = QCheckBox("MT Fitting")
        self.chk_mt_fit.setChecked(True)

        # Saturation timing (edited via Scan Parameters).  Not used by the 1/Z
        # fit; required by QUESP (tsat → exp term, trec → Zi = 1−exp(−R1·trec)).
        self.spin_tsat = QDoubleSpinBox()
        self.spin_tsat.setRange(0.0, 60.0); self.spin_tsat.setValue(2.0)
        self.spin_tsat.setSuffix(" s"); self.spin_tsat.setDecimals(3)
        self.spin_tsat.setToolTip("Saturation pulse length (QUESP only)")

        self.spin_trec = QDoubleSpinBox()
        self.spin_trec.setRange(0.0, 120.0); self.spin_trec.setValue(8.0)
        self.spin_trec.setSuffix(" s"); self.spin_trec.setDecimals(3)
        self.spin_trec.setToolTip("Recovery delay before saturation (QUESP only)")

        # ── Scan Parameters button (opens dialog) — above Processing & Fitting ──
        btn_scan = QPushButton("Inverse Z Scan Parameters")
        btn_scan.setStyleSheet(
            "background:#555;color:white;border:none;border-radius:4px;padding:6px 14px;"
        )
        btn_scan.setToolTip("Set saturation B1, B0 field, AREX eval offset and R1 (water)")
        btn_scan.clicked.connect(self._open_scan_params)
        vl.addWidget(btn_scan)

        # ── Processing & Fitting box (mirrors the QUESP tab): the Processing
        #     dialog opener + Run + Cancel all live inside one titled group. ─────
        _proc_grp = QGroupBox("Processing && Fitting")
        _pv = QVBoxLayout(_proc_grp)
        _pv.setSpacing(6)

        self.lbl_inv_pools = QLabel(f"{len(self._inv_selected_pools)} pools selected")
        self.lbl_inv_pools.setStyleSheet("font-size: 11px; color: #aaa;")
        _pv.addWidget(self.lbl_inv_pools)

        btn_cest_opts = QPushButton("Inverse Z Processing && Fitting")
        btn_cest_opts.setFixedHeight(34)
        # Match the CEST-MRI tab's "CEST MRI Processing" button (solid blue).
        btn_cest_opts.setStyleSheet(
            "QPushButton{background:#1565c0;color:white;font-weight:bold;"
            "border:none;border-radius:4px;padding:4px 12px;font-size:12px;}"
            "QPushButton:hover{background:#1976d2;}"
        )
        btn_cest_opts.clicked.connect(self._open_cest_options)
        _pv.addWidget(btn_cest_opts)

        # ── Run / Cancel ──────────────────────────────────────────────────
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Inverse Z Analysis")
        self.btn_run.setFixedHeight(40)
        # Match the CEST-MRI tab's "Run Z-Spectroscopy Analysis" button (solid green).
        self.btn_run.setStyleSheet(
            "QPushButton{background:#27ae60;color:white;font-weight:bold;border-radius:4px;}"
            "QPushButton:hover{background:#2ecc71;}"
            "QPushButton:disabled{background:#555;color:#999;}"
        )
        self.btn_run.clicked.connect(self._run)
        run_row.addWidget(self.btn_run, stretch=1)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setFixedHeight(40)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet(
            "QPushButton{background:#c0392b;color:white;font-weight:bold;"
            "border:none;border-radius:6px;padding:4px 10px;}"
            "QPushButton:hover{background:#e74c3c;}"
            "QPushButton:disabled{background:#555;color:#999;}"
        )
        self.btn_cancel.clicked.connect(self._cancel)
        run_row.addWidget(self.btn_cancel)
        _pv.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        _pv.addWidget(self.progress_bar)

        self.lbl_run_status = QLabel("")
        self.lbl_run_status.setStyleSheet("font-size: 11px;")
        _pv.addWidget(self.lbl_run_status)

        vl.addWidget(_proc_grp)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        from my_gui.theme import mono_font
        self.log_edit.setFont(mono_font(10))
        self.log_edit.setMaximumHeight(100)
        vl.addWidget(self.log_edit)
        vl.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(420)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    def _build_right(self) -> QWidget:
        right = QWidget()
        vl    = QVBoxLayout(right)
        vl.setSpacing(4)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Display:"))
        self.combo_display = QComboBox()
        self.combo_display.addItems(["AREX map (s⁻¹)"])
        self.combo_display.currentIndexChanged.connect(self._refresh_display)
        ctrl.addWidget(self.combo_display, stretch=1)

        self.btn_export = QPushButton("Export figure…")
        self.btn_export.clicked.connect(self._export)
        ctrl.addWidget(self.btn_export)

        self.chk_roi_bg = QCheckBox("ROIs + Bkg")
        self.chk_roi_bg.setToolTip(
            "Show the colored map only inside the ROIs, over the 1st raw image "
            "as a gray background.")
        self.chk_roi_bg.toggled.connect(self._refresh_display)
        ctrl.addWidget(self.chk_roi_bg)

        self.btn_roi_bg = QPushButton("Bkg…")
        self.btn_roi_bg.setToolTip(
            "Pick the grayscale background image (from the Scan Directory) for "
            "the 'ROIs + Bkg' overlay.")
        self.btn_roi_bg.clicked.connect(self._pick_roi_bg)
        ctrl.addWidget(self.btn_roi_bg)

        self.btn_hide_rois = QPushButton("Hide ROIs")
        self.btn_hide_rois.setCheckable(True)
        self.btn_hide_rois.clicked.connect(self._toggle_hide_rois)
        # Window/Level (brightness–contrast) drag tool — OsiriX-style.
        self.btn_contrast = WindowLevelToolButton()
        self.btn_contrast.toggled.connect(
            lambda checked: self.canvas.set_wl_active(
                checked, on_change=lambda a, b: self.plot_bar.set_clim(a, b)))
        ctrl.addWidget(self.btn_contrast)
        ctrl.addWidget(self.btn_hide_rois)
        vl.addLayout(ctrl)

        # ── Figure Customization (collapsible — mirrors the MRF Viewer) ───────
        from PyQt6.QtWidgets import QGroupBox as _QGB
        self.grp_fig_custom = _QGB("Figure Customization")
        _gfc = QVBoxLayout(self.grp_fig_custom); _gfc.setContentsMargins(8, 6, 8, 6)
        self.chk_fig_custom = QCheckBox("Enable Figure Customization")
        self.chk_fig_custom.setToolTip(
            "Show the title, colormap, colour-bar limit and font controls "
            "(including Bg and Log map).")
        _gfc.addWidget(self.chk_fig_custom)
        self._fig_custom_panel = QWidget()
        self._fcp_lay = QVBoxLayout(self._fig_custom_panel)
        self._fcp_lay.setContentsMargins(0, 0, 0, 0)
        _gfc.addWidget(self._fig_custom_panel)
        self._fig_custom_panel.setVisible(False)
        self.chk_fig_custom.toggled.connect(self._fig_custom_panel.setVisible)
        vl.addWidget(self.grp_fig_custom)

        # Custom title
        from PyQt6.QtWidgets import QLineEdit as _QLE
        _title_row = QHBoxLayout()
        _title_row.addWidget(QLabel("Title:"))
        self.edit_map_title = _QLE()
        self.edit_map_title.setPlaceholderText("Custom map title (leave blank for default)")
        _title_row.addWidget(self.edit_map_title, stretch=1)
        self._fcp_lay.addLayout(_title_row)

        from my_gui.format_bar import add_title_format_bar, connect_title_debounced
        connect_title_debounced(self.edit_map_title, self._refresh_display)

        # Fonts line on top (with B/I/x²/x₂ title buttons), colour-bar below.
        self.plot_bar = PlotCustomBar(default_cmap="viridis", fonts_first=True)
        self.plot_bar.applied.connect(self._refresh_display)
        add_title_format_bar(self.edit_map_title, None,
                             target_row=self.plot_bar.font_row(),
                             default_getter=lambda: getattr(self.canvas, "_last_title", ""))
        self._fcp_lay.addWidget(self.plot_bar)

        # ── Load Maps — load any .mat/.npz and view/customize its variables ────
        _lm_row = QHBoxLayout()
        self.btn_load_maps = QPushButton("Load Maps")
        self.btn_load_maps.setToolTip(
            "Load any .mat / .npz file and view its variables as maps here — each\n"
            "variable is added to the Display selector by its own name, and the\n"
            "colormap, colour-bar limits, Bg and Export controls all apply to it.")
        self.btn_load_maps.clicked.connect(self._load_maps_file)
        _lm_row.addWidget(self.btn_load_maps)
        _lm_row.addStretch()
        self._fcp_lay.addLayout(_lm_row)

        # ── "Bg" black-background toggle (for slides) — flips only the white
        #    surround + labels; the map/colormap stays identical. ────────────
        self.chk_dark_bg = QCheckBox("Bg")
        self.chk_dark_bg.setToolTip(
            "Black background for the figure (for slides). Only the white "
            "surround and labels flip - the maps stay identical.")
        self.chk_dark_bg.toggled.connect(self._refresh_display)
        _frow = self.plot_bar.font_row()
        _b_idx = _frow.count()
        for _i in range(_frow.count()):
            _wd = _frow.itemAt(_i).widget()
            if isinstance(_wd, QPushButton) and _wd.text() == "B":
                _b_idx = _i; break
        _frow.insertWidget(_b_idx, self.chk_dark_bg)

        # "Log map" — Fuderer perceptual log-like colour scaling (MRM 2025),
        # placed right after "Bg".  Warps the colormap so low values get more
        # contrast; data & colour-bar ticks stay linear; signed maps unchanged.
        self.chk_logmap = QCheckBox("Log map")
        self.chk_logmap.setToolTip(
            "Log-color scaling, redistribute the colormaps so equal color "
            "steps = equal % change in value.")
        self.chk_logmap.toggled.connect(self._refresh_display)
        _frow.insertWidget(_b_idx + 1, self.chk_logmap)

        self.canvas = ROICanvas()
        vl.addWidget(self.canvas, stretch=1)

        btn_row2 = QHBoxLayout()
        self.btn_roi_spectra = QPushButton("ROI Spectra (1/Z fit)")
        self.btn_roi_spectra.clicked.connect(self._show_roi_spectra)
        btn_row2.addWidget(self.btn_roi_spectra)

        self.btn_roi_table = QPushButton("ROI Statistics")
        self.btn_roi_table.clicked.connect(self._show_roi_table)
        btn_row2.addWidget(self.btn_roi_table)

        # fs/ksw/R² parametric maps are generated automatically from the QUESP
        # ROI fits (see _show_roi_spectra → _run_quesp_maps(silent=True)); no
        # standalone button — they appear in the Display selector.

        from PyQt6.QtWidgets import QCheckBox as _QCB
        from my_gui.plot_custom_bar import DataCursorToolButton
        self.chk_datacursor = DataCursorToolButton()
        self.chk_datacursor.toggled.connect(self._toggle_datacursor)
        ctrl.insertWidget(ctrl.indexOf(self.btn_contrast) + 1, self.chk_datacursor)
        vl.addLayout(btn_row2)
        return right

    # ── Scan Parameters dialog ────────────────────────────────────────────────

    def _open_scan_params(self):
        dlg = _InvZScanParamsDialog(
            b1=self.spin_b1.value(),
            b0=self.spin_b0.value(),
            eval_ppm=self.spin_eval.value(),
            r1=self.spin_r1.value(),
            tsat=self.spin_tsat.value(),
            trec=self.spin_trec.value(),
            r1_getter=self._r1_getter,
            b1_list=[float(ds.get('b1_ut', 0.0)) for ds in self._datasets],
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            s = dlg.get_settings()
            self.spin_b1.setValue(s["b1"])
            self.spin_b0.setValue(s["b0"])
            self.spin_eval.setValue(s["eval_ppm"])
            self.spin_r1.setValue(s["r1"])
            self.spin_tsat.setValue(s["tsat"])
            self.spin_trec.setValue(s["trec"])

    # ── CEST Fitting Options dialog ───────────────────────────────────────────

    def _open_cest_options(self):
        dlg = _InvZSpecOptionsDialog(
            selected_pools=self._inv_selected_pools,
            cest_lo=self.spin_cest_lo.value(),
            cest_hi=self.spin_cest_hi.value(),
            peak_type=self.combo_peak.currentText(),
            mt_excl=self.spin_mt_excl.value(),
            mt_rein=self.spin_mt_rein.value(),
            fit_mt=self.chk_mt_fit.isChecked(),
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            s = dlg.get_settings()
            _pools = s["pools"]
            # Pool dialog does not show water (always included) — re-add it
            if 'water' not in _pools:
                _pools = ['water'] + _pools
            self._inv_selected_pools = _pools
            self.spin_cest_lo.setValue(s["cest_lo"])
            self.spin_cest_hi.setValue(s["cest_hi"])
            self.combo_peak.setCurrentText(s["peak_type"])
            self.spin_mt_excl.setValue(s["mt_excl"])
            self.spin_mt_rein.setValue(s["mt_rein"])
            self.chk_mt_fit.setChecked(bool(s["fit_mt"]))
            n = len(self._inv_selected_pools)
            self.lbl_inv_pools.setText(f"{n} pool{'s' if n != 1 else ''} selected")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        self.log_edit.append(msg)
        self.log_edit.verticalScrollBar().setValue(
            self.log_edit.verticalScrollBar().maximum()
        )

    def _get_params(self) -> dict:
        """Collect all fitting parameters from UI into a dict."""
        peak_type = ('pseudovoigt'
                     if self.combo_peak.currentText() == 'Pseudo-Voigt'
                     else 'lorentzian')
        return dict(
            satpwr_uT      = self.spin_b1.value(),
            B0_MHz         = self.spin_b0.value(),
            R1             = self.spin_r1.value(),
            eval_ppm       = self.spin_eval.value(),
            ppm_exclude_MT = (-self.spin_mt_excl.value(), self.spin_mt_excl.value()),
            ppm_reinclude_MT = (-self.spin_mt_rein.value(), self.spin_mt_rein.value()),
            ppm_include_CEST = (self.spin_cest_lo.value(), self.spin_cest_hi.value()),
            peak_type      = peak_type,
            fit_mt         = bool(self.chk_mt_fit.isChecked()),
            tsat           = self.spin_tsat.value(),
            trec           = self.spin_trec.value(),
        )

    # ── T1 map integration ───────────────────────────────────────────────────

    def set_main_pools_getter(self, fn):
        """Register a callback that returns the main CEST MRI tab's _global_pools.
        Called by app.py so the 1/Z tab inherits the user's pool selection."""
        self._main_pools_getter = fn

    def _get_active_pools(self) -> list[str]:
        """Return the effective pool list for 1/Z fitting — EXACTLY the pools the
        user picked in this tab's CEST Fitting Options (expanded, e.g. iopamidol
        → 4.3 + 5.5).

        It deliberately does NOT merge in the main CEST-MRI tab's pools: pulling
        in default amine/amide there was adding peaks that aren't in the sample
        (e.g. a pure iopamidol phantom), and those spurious pools absorb signal
        from the real pools and corrupt the QUESP fs/ksw.
        """
        return [p for p in _expand_invz_pools(self._inv_selected_pools)
                if p in _L_DEFS]

    def set_r1_getter(self, fn):
        """Register a callback that returns mean R1 from T1 map. Called by app.py."""
        self._r1_getter = fn

    def set_r1_from_map(self, r1_value: float):
        """Set R1 directly (callable from app.py after T1 map computation)."""
        if r1_value > 0:
            self.spin_r1.setValue(float(r1_value))
            self._log(f"R1 set from T1 map: {r1_value:.4f} s⁻¹")

    def _from_t1_map(self):
        fn = self._r1_getter
        if fn is None:
            QMessageBox.information(
                self, "T1 map",
                "No T1/T2/B1 tab connected.\n"
                "Enter R1 = 1/T1 manually."
            )
            return
        r1 = fn()
        if r1 is not None and r1 > 0:
            self.spin_r1.setValue(float(r1))
            self._log(f"R1 from T1 map: {r1:.4f} s⁻¹  (T1 = {1/r1*1000:.1f} ms)")

    # ── Dataset management ───────────────────────────────────────────────────

    def _add_scan(self):
        dlg = _AddScanDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        path, pv360, sl, b1_dialog = dlg.get_values()
        if not path:
            QMessageBox.warning(self, "No path", "Select a data path.")
            return
        try:
            z_img, ppm, b1_auto, pwr_key = self._load_zspec(path, pv360)
            # Prefer auto-detected B1 (from Bruker method file); fall back to dialog entry
            b1_ut = b1_auto if b1_auto > 0.0 else b1_dialog
            self._datasets.append(dict(
                z_img=z_img, ppm=ppm, slice=sl, path=path, b1_ut=b1_ut,
            ))
            self.scan_list.addItem(
                QListWidgetItem(
                    f"B1={b1_ut:.2f} µT  |  {path.split('/')[-1]}  (sl {sl+1})"
                )
            )
            # Sync the "Sat. power (B1)" spin box to the detected value so the
            # user can see it and so it acts as the default for any manual override
            if b1_ut > 0.0:
                self.spin_b1.setValue(b1_ut)
            n = len(self._datasets)
            if b1_auto > 0.0:
                key_hint = f" [{pwr_key}]" if pwr_key else ""
                self._log(f"Loaded: B1={b1_ut:.3f} µT (auto-detected{key_hint})  |  {path}")
            else:
                self._log(f"Loaded: B1={b1_ut:.3f} µT (from dialog)  |  {path}")
            self.lbl_status_data.setText(f"{n} scan(s) loaded.")
            self.lbl_status_data.setStyleSheet("font-size: 11px; color: green;")
            ref = (z_img[:, :, sl, z_img.shape[-1]//2]
                   if z_img.ndim == 4 else z_img[..., 0])
            self.canvas.show_map(ref, "Reference image", cmap="gray")
        except Exception as exc:
            self.lbl_status_data.setText(f"Error: {exc}")
            self.lbl_status_data.setStyleSheet("font-size: 11px; color: red;")
            self._log(f"ERROR loading scan: {exc}")

    def _load_zspec(self, path: str, pv360: bool):
        import os
        if os.path.isdir(path):
            # Bruker pdata/1 (has a 2dseq) → ParaVision reader; any other folder is
            # treated as a GE / Siemens DICOM CEST series (reuses the vetted
            # multi-vendor reader, which also de-tiles Siemens mosaics).
            if os.path.exists(os.path.join(path, '2dseq')):
                from my_gui.bruker_reader import read_2dseq_cest
                image, M0image, info = read_2dseq_cest(path, pv360=pv360)
                z = np.clip(image / (M0image[:, :, :, np.newaxis] + 1e-9), 0, 1)
                b1_ut = float(info.get('satpwr_uT', 0.0))
                pwr_key = info.get('pwr_key_used', '')
                return z, info['w_offsetPPM'], b1_ut, pwr_key
            else:
                import glob
                from my_gui.tabs.zspec_tab import _load_dicom_4d
                # Offsets: a .txt sidecar in the folder if present, else the
                # reader falls back to DICOM-header auto-detection.
                _txt = sorted(glob.glob(os.path.join(path, "*.txt")))
                ppm_sidecar = _txt[0] if _txt else ""
                img4d, ppm = _load_dicom_4d(path, ppm_sidecar,
                                            lambda m: self._log(str(m)))
                # Normalise to a Z-spectrum using the brightest offset per voxel as
                # the unsaturated reference (S0 / far-offset frame).
                s0 = np.nanmax(img4d, axis=-1, keepdims=True)
                z = np.clip(img4d / (s0 + 1e-9), 0.0, 1.0)
                return z.astype(np.float32), np.asarray(ppm, float).ravel(), 0.0, ""
        else:
            if path.endswith('.mat'):
                from scipy.io import loadmat
                d = loadmat(path)
                z_img = np.array(d.get('z_img', d.get('Zlab', list(d.values())[-1])),
                                 dtype=float)
                ppm   = np.array(d.get('ppm', d.get('w_offset',
                                 np.arange(z_img.shape[-1]))), dtype=float).ravel()
                b1_arr = d.get('b1_ut', d.get('B1', np.array([0.0])))
                b1_ut  = float(np.array(b1_arr).ravel()[0])
            else:
                d    = np.load(path, allow_pickle=True)
                z_img = np.array(d['z_img'], dtype=float)
                ppm   = np.array(d.get('ppm', np.arange(z_img.shape[-1])),
                                 dtype=float).ravel()
                b1_ut = float(np.array(d.get('b1_ut', [0.0])).ravel()[0])
            if z_img.ndim == 3:
                z_img = z_img[:, :, np.newaxis, :]
            return z_img.astype(np.float32), ppm, b1_ut, ""

    def _remove_selected(self):
        row = self.scan_list.currentRow()
        if row >= 0:
            self.scan_list.takeItem(row)
            if row < len(self._datasets):
                self._datasets.pop(row)
            self.lbl_status_data.setText(f"{len(self._datasets)} scan(s) loaded.")

    def _remove_scan(self, item):
        self._remove_selected()

    # ── Run / Cancel ─────────────────────────────────────────────────────────

    def _run(self):
        if not self._datasets:
            QMessageBox.warning(self, "No data", "Add at least one scan.")
            return
        p = self._get_params()

        # One AREX map per pool the user EXPLICITLY selected in CEST Fitting
        # Options (not the merged main-tab defaults, so amine/amide don't appear
        # unless chosen).  Positive/CEST-side offsets only (AREX is an asymmetry).
        pool_offsets: list[float] = []
        try:
            for pool in _expand_invz_pools(self._inv_selected_pools):
                d = _L_DEFS.get(pool)
                if d is None:
                    continue
                off = float(d['x0'][2])
                if off > 0.3:          # CEST side only (skip water/MT/NOE)
                    pool_offsets.append(off)
        except Exception:
            pool_offsets = []

        self._worker = InvZSpecWorker(
            datasets=self._datasets,
            eval_ppm=p['eval_ppm'],
            R1=p['R1'],
            b0_mhz=p['B0_MHz'],
            eval_offsets=pool_offsets,
        )
        self._worker.log.connect(self._log)
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.cancelled.connect(self._on_cancelled)

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_run_status.setText("Running…")
        self._worker.start()

    def _cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
        self.btn_cancel.setEnabled(False)

    def _rebuild_display_combo(self):
        """Rebuild the Display combo from AREX results + per-voxel QUESP maps."""
        self.combo_display.blockSignals(True)
        self.combo_display.clear()
        for r in self._all_results:
            b1 = r.get('b1_ut', 0.0)
            ep = r.get('eval_ppm', 3.0)
            self.combo_display.addItem(f"AREX map — B1={b1:.2f} µT  (±{ep:.2f} ppm)")
        for entry in self._quesp_display:
            self.combo_display.addItem(entry['label'])
        for entry in getattr(self, '_loaded_maps', []):
            self.combo_display.addItem(entry['label'])
        if self.combo_display.count() == 0:
            self.combo_display.addItem("AREX map (s⁻¹)")
        self.combo_display.blockSignals(False)

    def _load_maps_file(self):
        """Load any .mat/.npz file and add each 2-D map variable to the Display
        selector, so it can be viewed and customized with the map controls
        (colormap, colour-bar limits, Bg, title, Export) like any AREX/QUESP map."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Maps  (.mat / .npz)", "",
            "Map data (*.mat *.npz);;MATLAB (*.mat);;NumPy (*.npz);;All files (*)")
        if not path:
            return
        try:
            from my_gui.fig_export import _load_any
            data = _load_any(path)
        except Exception as e:                        # noqa: BLE001
            QMessageBox.critical(self, "Load failed", str(e))
            return
        maps = []
        for k, v in data.items():
            key = str(k)
            if key.startswith('__') or key.endswith('axes_info'):
                continue
            arr = np.asarray(v)
            if arr.ndim == 2 and arr.size > 4 and np.issubdtype(arr.dtype, np.number):
                maps.append({'label': key, 'data': arr.astype(float)})
            elif arr.ndim == 3 and arr.shape[2] in (3, 4):     # RGB(A) image
                maps.append({'label': key, 'data': arr})
        if not maps:
            QMessageBox.information(
                self, "No maps",
                "No 2-D map variables were found in that file.")
            return
        self._loaded_maps = maps
        self._rebuild_display_combo()
        if not self.chk_fig_custom.isChecked():
            self.chk_fig_custom.setChecked(True)      # reveal the customization panel
        first = self.combo_display.count() - len(maps)
        self.combo_display.setCurrentIndex(max(0, first))   # jump to first loaded map
        self._refresh_display()

    def _on_done(self, results: dict):
        self._results = results
        self._all_results = results.get('all_results', [])
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)

        self._rebuild_display_combo()

        ep_list = [r.get('eval_ppm', 3.0) for r in self._all_results]
        ep_str  = f"±{ep_list[0]:.2f} ppm" if ep_list else ""
        n       = len(self._all_results)
        self.lbl_run_status.setText(
            f"{n} AREX map(s) computed at {ep_str}."
        )
        self._refresh_display()

        # ── Auto-precompute the per-ROI 1/Z spectra (no window) ────────────────
        # Mirrors "Run Z-Spectroscopy Analysis": the fits run now so that
        # clicking "ROI Spectra (1/Z fit)" shows them instantly.  Then offer the
        # QUESP-pool chooser so the MTRasym / MTRRex / Ω-plot appear beneath the
        # 1/Z panels.  Both are silent no-ops when no ROIs are drawn yet.
        self._auto_precompute_1z()
        # Auto-generate the per-voxel fs / ksw / R² parametric maps right after
        # Run (background, silent; needs ≥3 B1 powers).  They appear in the
        # Display selector when ready.
        try:
            self._run_quesp_maps(silent=True)
        except Exception as _exc:
            self._log(f"(fs/ksw map generation skipped: {_exc})")

    def _auto_precompute_1z(self):
        rois = [r for r in getattr(self, '_last_rois', [])
                if r.name != 'Phantom_outline']
        if not rois or not self._datasets:
            if not rois:
                self._log("Tip: draw ROIs, then Run again to auto-compute the "
                          "1/Z spectra (or click “ROI Spectra (1/Z fit)”).")
            return
        try:
            ok = self._compute_roi_1z(silent=True)
        except Exception as exc:
            self._log(f"1/Z precompute skipped: {exc}")
            return
        if not ok:
            return
        nroi = len(self._last_1z_roi_results)
        self.lbl_run_status.setText(
            self.lbl_run_status.text() +
            f"  1/Z fits ready for {nroi} ROI(s).")
        self._log(f"1/Z spectra precomputed for {nroi} ROI(s) — "
                  f"click “ROI Spectra (1/Z fit)” to view.")
        # QUESP needs ≥2 saturation powers; the Ω-plot really wants ≥3.
        if len(self._datasets) >= 2 and self._quesp_candidate_pools():
            self._ask_quesp_pools()

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.lbl_run_status.setText("Error!")
        self._log(f"ERROR: {msg}")

    def _on_cancelled(self):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.lbl_run_status.setText("Cancelled.")

    # ── Per-voxel QUESP maps ──────────────────────────────────────────────────

    def _run_quesp_maps(self, silent: bool = False):
        """Launch per-voxel QUESP fitting → fs / ksw / R² / conc parametric maps.

        Called automatically (``silent=True``) after the ROI-Spectra QUESP fits,
        so the maps are generated "from" the same QUESP analysis and land in the
        Display selector.  Fits over the ROI/phantom mask for speed and produces
        maps ONLY for the user's explicitly-selected exchange pools."""
        if getattr(self, '_quesp_worker', None) is not None and \
                self._quesp_worker.isRunning():
            return                                  # already running
        if len(self._datasets) < 3:
            if not silent:
                QMessageBox.warning(
                    self, "Need ≥3 B1 powers",
                    "Per-voxel QUESP requires at least 3 saturation-power "
                    f"datasets.\nCurrently loaded: {len(self._datasets)}.")
            return

        p = self._get_params()
        # Maps are produced ONLY for the pools the user selected in CEST Fitting
        # Options — not the merged main-tab defaults (amine/amide etc.).
        sel_q = [pn for pn in _expand_invz_pools(self._inv_selected_pools)
                 if pn not in ('water', 'MT') and pn in _L_DEFS]
        if not sel_q:
            if not silent:
                QMessageBox.warning(
                    self, "No CEST pools",
                    "Select at least one CEST exchange pool (OH, amine, iopamidol,"
                    " …) in CEST Fitting Options before generating maps.")
            return
        # Fit with the full active-pool background for accuracy; restrict the
        # OUTPUT maps to the selected pools (see _on_quesp_done).
        pools = self._get_active_pools()
        self._quesp_output_pools = sel_q

        # Full-res image shape from first dataset
        z0  = np.array(self._datasets[0]['z_img'])
        sl0 = self._datasets[0].get('slice', 0)
        z0sl = z0[:, :, sl0, :] if z0.ndim == 4 else z0
        Yf, Xf = z0sl.shape[:2]

        # Build mask.  PREFER the whole-region outline (Phantom_outline /
        # Brain_outline) so the fs/ksw maps cover the ENTIRE phantom/brain — not
        # just the small tube ROIs.  Fall back to the union of drawn ROIs, then
        # to None (worker uses the intensity fallback / whole image).
        rois = getattr(self, '_last_rois', [])
        roi_mask = None
        outline = next((r for r in rois
                        if r.name in ('Phantom_outline', 'Brain_outline')
                        and getattr(r, 'mask', None) is not None
                        and np.asarray(r.mask).shape == (Yf, Xf)), None)
        if outline is not None:
            roi_mask = np.asarray(outline.mask, dtype=bool)
        else:
            named = [r for r in rois
                     if r.name not in ('Phantom_outline', 'Brain_outline')
                     and getattr(r, 'mask', None) is not None]
            if named:
                roi_mask = np.zeros((Yf, Xf), dtype=bool)
                for r in named:
                    m = np.asarray(r.mask, dtype=bool)
                    if m.shape == (Yf, Xf):
                        roi_mask |= m

        n_mask = int(roi_mask.sum()) if roi_mask is not None else (Yf * Xf)
        # Fit-grid cap: the per-voxel MT+CEST fit runs on a grid no larger than
        # this (then the maps are upsampled to full resolution).  96 fits ~1.8-4x
        # fewer voxels than 128 — a big speed-up for noisy per-voxel QUESP maps
        # with little visible detail loss.  Override via self._map_fit_dim.
        max_dim = int(getattr(self, "_map_fit_dim", 96))

        # Skip a redundant re-run when nothing relevant changed (auto mode fires
        # on every ROI-Spectra open).
        _key = (len(self._datasets), tuple(sel_q), n_mask,
                p['peak_type'], round(p['B0_MHz'], 3), round(p['R1'], 4))
        if silent and getattr(self, '_quesp_maps_key', None) == _key:
            return
        self._quesp_maps_key = _key

        if not silent:
            f = max(1, int(np.ceil(max(Yf, Xf) / float(max_dim))))
            est_vox = max(1, n_mask // (f * f))
            ans = QMessageBox.question(
                self, "Generate fs / ksw maps?",
                f"Fit the full MT + CEST 1/Z pipeline per voxel, then QUESP.\n\n"
                f"  • B1 powers: {len(self._datasets)}\n"
                f"  • Pools: {', '.join(sel_q)}\n"
                f"  • ~{est_vox} voxels at ×{f} downsample\n\n"
                f"This can take a few minutes. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                return

        self._quesp_worker = InvZQUESPWorker(
            self._datasets, p, pools, roi_mask, n_H=1, max_fit_dim=max_dim)
        self._quesp_worker.log.connect(self._log)
        self._quesp_worker.progress.connect(self.progress_bar.setValue)
        self._quesp_worker.finished.connect(self._on_quesp_done)
        self._quesp_worker.error.connect(self._on_error)
        self._quesp_worker.cancelled.connect(self._on_cancelled)
        # Reuse Cancel button → stop QUESP worker
        self._worker = self._quesp_worker

        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._log("Generating fs / ksw / R² maps for the selected pools…")
        self.lbl_run_status.setText("Generating fs / ksw / R² maps…")
        self._quesp_worker.start()

    def _on_quesp_done(self, results: dict):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)

        qmaps = results.get('quesp_maps', {})
        nH    = results.get('n_H', 1)
        # Only surface the pools the user explicitly selected (CEST Fitting
        # Options), even though the fit used the full background pool set.
        out_pools = getattr(self, '_quesp_output_pools', None)
        pools = [pl for pl in results.get('pools', [])
                 if out_pools is None or pl in out_pools]
        # Replace any prior fs/ksw/conc/R² maps for these pools so re-runs don't
        # pile up duplicate Display entries.
        self._quesp_display = [e for e in self._quesp_display
                               if e.get('pool') not in pools]
        for pool in pools:
            mp = qmaps.get(pool, {})
            if not mp:
                continue
            self._quesp_display.append(dict(
                label=f"fs map — {pool}", data=mp['fs'],
                cmap='viridis', vmin=0, pool=pool, kind='fs'))
            self._quesp_display.append(dict(
                label=f"ksw map — {pool} (s⁻¹)", data=mp['ksw'],
                cmap='hot', vmin=0, pool=pool, kind='ksw'))
            self._quesp_display.append(dict(
                label=f"conc map — {pool} (mM, nH={nH})", data=mp['conc'],
                cmap='viridis', vmin=0, pool=pool, kind='conc'))
            self._quesp_display.append(dict(
                label=f"R² map — {pool}", data=mp['r2'],
                cmap='cividis', vmin=0, pool=pool, kind='r2'))

        self._rebuild_display_combo()
        npool = len(pools)
        self.lbl_run_status.setText(
            f"fs / ksw / R² maps ready — {npool} pool(s).")

        # Report NaN-only maps in the log rather than a modal popup (auto-run).
        any_finite = any(
            np.isfinite(qmaps.get(pl, {}).get('fs', np.array([np.nan]))).any()
            for pl in pools)
        if not any_finite:
            self._log("⚠ fs/ksw maps: every voxel fit returned NaN — check that "
                      "≥3 B1 powers are loaded and the selected pools are present.")
        else:
            self._log(f"fs / ksw / R² maps ready for: {', '.join(pools)} "
                      f"(pick them in the Display selector).")
        self._refresh_display()

    # ── ROIs + Bkg background picker ──────────────────────────────────────────

    def set_scan_paths_getter(self, fn):
        self._scan_paths_getter = fn

    def _pick_roi_bg(self):
        from my_gui.roi_tools import choose_background_image
        sp = {}
        _g = getattr(self, "_scan_paths_getter", None)
        if callable(_g):
            try: sp = _g() or {}
            except Exception: sp = {}
        mode, img = choose_background_image(self, sp)
        if mode == "set":
            self._roi_bg_img = img
        elif mode == "default":
            self._roi_bg_img = None
        else:
            return
        self.chk_roi_bg.setChecked(True)
        self._refresh_display()

    # ── Display ──────────────────────────────────────────────────────────────

    def _refresh_display(self):
        if hasattr(self, "chk_dark_bg"):
            self.canvas._dark_bg = self.chk_dark_bg.isChecked()
        if hasattr(self, "chk_logmap"):
            self.canvas._log_map = self.chk_logmap.isChecked()
        self._dc_annot = None
        all_res = getattr(self, '_all_results', [])
        quesp   = getattr(self, '_quesp_display', [])
        loaded  = getattr(self, '_loaded_maps', [])
        if not all_res and not quesp and not loaded:
            return

        cmap       = self.plot_bar.get_cmap()
        vmin, vmax = self.plot_bar.get_clim()
        fs         = self.plot_bar.get_font_sizes()
        ct         = (self.edit_map_title.text().strip()
                      if hasattr(self, 'edit_map_title') else "")

        # Phantom outline masking
        phantom_roi = next(
            (r for r in getattr(self, '_last_rois', []) if r.name == "Phantom_outline"),
            None,
        )
        def _ph_mask(arr):
            if phantom_roi is None or arr is None:
                return arr
            msk = phantom_roi.mask
            if msk.shape == arr.shape:
                return np.where(msk, arr, np.nan)
            elif msk.shape[:2] == arr.shape[:2]:
                return np.where(msk, arr, np.nan)
            return arr

        idx = max(0, self.combo_display.currentIndex())

        # ── QUESP per-voxel map selected ──────────────────────────────────────
        if idx >= len(all_res):
            qi = idx - len(all_res)
            if 0 <= qi < len(quesp):
                entry = quesp[qi]
                data  = _ph_mask(entry['data'])
                fin   = data[np.isfinite(data)]
                hi    = (float(np.nanpercentile(fin, 98)) if fin.size else 1.0)
                _q_cmap = cmap if cmap != 'viridis' else entry.get('cmap', 'viridis')
                _q_vmin = vmin if vmin is not None else entry.get('vmin', 0)
                _q_vmax = vmax if vmax is not None else hi
                if self.chk_roi_bg.isChecked():
                    _base = None
                    if getattr(self, "_datasets", None):
                        _ds = self._datasets[0]
                        _zi = np.asarray(_ds.get("z_img"))
                        if _zi is not None and _zi.ndim >= 3:
                            _sl = int(_ds.get("slice", 0))
                            _base = _zi[:, :, _sl, 0] if _zi.ndim == 4 else _zi[:, :, 0]
                    if getattr(self, "_roi_bg_img", None) is not None:
                        _base = self._roi_bg_img
                    _union = roi_union_mask(getattr(self, "_last_rois", []), data.shape)
                    if _base is not None and _union is not None:
                        _ovc = _q_cmap if str(_q_cmap).lower() not in ("gray", "greys", "greys_r") else "jet"
                        self.canvas.show_map_over_raw(
                            data, _base, _union, ct or entry['label'],
                            cmap=_ovc, vmin=_q_vmin, vmax=_q_vmax, **fs)
                        return
                self.canvas.show_map(
                    data, ct or entry['label'],
                    cmap=_q_cmap,
                    vmin=_q_vmin,
                    vmax=_q_vmax, **fs,
                )
                return
            # ── Loaded .mat/.npz map selected (via 'Load Maps') ───────────────
            li = qi - len(quesp)
            if 0 <= li < len(loaded):
                entry = loaded[li]
                data  = _ph_mask(np.asarray(entry['data']))
                arr   = np.asarray(data)
                if np.issubdtype(arr.dtype, np.floating):
                    fin = arr[np.isfinite(arr)]
                    lo  = float(np.nanpercentile(fin, 2))  if fin.size else 0.0
                    hi  = float(np.nanpercentile(fin, 98)) if fin.size else 1.0
                else:
                    lo, hi = None, None
                _l_vmin = vmin if vmin is not None else lo
                _l_vmax = vmax if vmax is not None else hi
                if self.chk_roi_bg.isChecked():
                    _base = None
                    if getattr(self, "_datasets", None):
                        _ds = self._datasets[0]
                        _zi = np.asarray(_ds.get("z_img"))
                        if _zi is not None and _zi.ndim >= 3:
                            _sl = int(_ds.get("slice", 0))
                            _base = _zi[:, :, _sl, 0] if _zi.ndim == 4 else _zi[:, :, 0]
                    if getattr(self, "_roi_bg_img", None) is not None:
                        _base = self._roi_bg_img
                    _union = roi_union_mask(getattr(self, "_last_rois", []), data.shape)
                    if _base is not None and _union is not None:
                        _ovc = cmap if str(cmap).lower() not in ("gray", "greys", "greys_r") else "jet"
                        self.canvas.show_map_over_raw(
                            data, _base, _union, ct or entry['label'],
                            cmap=_ovc, vmin=_l_vmin, vmax=_l_vmax, **fs)
                        return
                self.canvas.show_map(
                    data, ct or entry['label'], cmap=cmap,
                    vmin=_l_vmin,
                    vmax=_l_vmax, **fs,
                )
            return

        # ── AREX map selected ─────────────────────────────────────────────────
        if not all_res:
            return
        res  = all_res[idx] if idx < len(all_res) else all_res[0]
        data = res.get('arex_map')
        if data is not None:
            ep  = res.get('eval_ppm', self.spin_eval.value())
            R1  = res.get('R1', self.spin_r1.value())
            b1  = res.get('b1_ut', self.spin_b1.value())
            data = _ph_mask(data)
            fin  = data[np.isfinite(data)]
            lim  = float(np.nanpercentile(np.abs(fin), 99)) if fin.size else 1.0
            _a_title = ct or f"AREX  (±{ep:.2f} ppm,  B1={b1:.2f} µT,  R1={R1:.3g} s⁻¹)"
            _a_cmap  = cmap if cmap != 'viridis' else 'hot'
            _a_vmin  = vmin or 0
            _a_vmax  = vmax or lim
            if self.chk_roi_bg.isChecked():
                _base = None
                if getattr(self, "_datasets", None):
                    _ds = self._datasets[0]
                    _zi = np.asarray(_ds.get("z_img"))
                    if _zi is not None and _zi.ndim >= 3:
                        _sl = int(_ds.get("slice", 0))
                        _base = _zi[:, :, _sl, 0] if _zi.ndim == 4 else _zi[:, :, 0]
                if getattr(self, "_roi_bg_img", None) is not None:
                    _base = self._roi_bg_img
                _union = roi_union_mask(getattr(self, "_last_rois", []), data.shape)
                if _base is not None and _union is not None:
                    _ovc = _a_cmap if str(_a_cmap).lower() not in ("gray", "greys", "greys_r") else "jet"
                    self.canvas.show_map_over_raw(
                        data, _base, _union, _a_title,
                        cmap=_ovc, vmin=_a_vmin, vmax=_a_vmax, **fs)
                    return
            self.canvas.show_map(
                data,
                _a_title,
                cmap=_a_cmap,
                vmin=_a_vmin, vmax=_a_vmax, **fs,
            )

    # ── ROI Spectra dialog — full 1/Z pipeline per ROI ────────────────────────

    def _save_quesp_stack(self, parent=None):
        """Export the cached per-ROI/pool 1/Z amplitudes vs B1 to a QUESP .mat.

        Uses results from the most recent ROI Spectra (1/Z fit) run. If none
        exist yet, tells the user to run the analysis first.
        """
        import os
        from scipy.io import savemat
        par = parent if parent is not None else self
        roi_results = getattr(self, '_last_1z_roi_results', None)
        roi_spectra = getattr(self, '_last_1z_roi_spectra', None)
        p           = getattr(self, '_last_1z_params', None)
        if not roi_results or not roi_spectra or p is None:
            QMessageBox.information(
                par, "Run 1/Z fitting first",
                "No 1/Z fit results to export yet.\n\n"
                "Open “ROI Spectra (1/Z fit)”, run the per-ROI analysis, then\n"
                "use this button to save the QUESP stack.")
            return
        # Pool list = union of fitted CEST pools across results (water excluded)
        pools = []
        for lst in roi_results.values():
            for res in lst:
                if res and 'cest_coeffs' in res:
                    for pn in res['cest_coeffs']:
                        if pn not in pools and pn != 'water':
                            pools.append(pn)
        rnames = list(roi_results.keys())
        if not rnames or not pools:
            QMessageBox.information(par, "Nothing to export",
                                    "No fitted CEST pools available.")
            return
        n_b1 = max(len(roi_spectra[r]) for r in rnames)
        R1   = float(p['R1'])
        b1_arr = np.full(n_b1, np.nan)
        out = {'R1': R1, 'B0_MHz': float(p['B0_MHz']),
               'pools': np.array(pools, dtype=object),
               'roi_names': np.array(rnames, dtype=object)}
        for pool in pools:
            # MTR_Rex(B1) = A_pool / R1  (A = R1·(1/Z−1) peak height)
            M = np.full((len(rnames), n_b1), np.nan)
            for ri, rname in enumerate(rnames):
                for bi, (res, spec) in enumerate(zip(roi_results[rname],
                                                     roi_spectra[rname])):
                    b1_arr[bi] = spec[2]
                    if res and pool in res.get('cest_coeffs', {}):
                        A = float(res['cest_coeffs'][pool][0])
                        M[ri, bi] = A / R1 if R1 > 0 else A
            out[f"mtrrex__{pool}"] = M
        out['satpwr_uT'] = b1_arr
        path, _ = QFileDialog.getSaveFileName(
            par, "Save 1/Z → QUESP stack", "quesp_1z_stack.mat",
            "MATLAB (*.mat)")
        if not path:
            return
        if not path.lower().endswith('.mat'):
            path += '.mat'
        try:
            savemat(path, out)
            QMessageBox.information(
                par, "Saved",
                f"QUESP stack saved:\n{os.path.basename(path)}\n\n"
                f"{len(rnames)} ROI(s) × {len(pools)} pool(s) × "
                f"{int(np.sum(np.isfinite(b1_arr)))} B1 power(s).\n"
                "Load it in the QUESP tab to fit fs / ksw.")
        except Exception as exc:
            QMessageBox.warning(par, "Save failed", str(exc))

    # ── Shared 1/Z precompute (used by Run + ROI Spectra) ─────────────────────

    def _roi_1z_key(self, rois, datasets, p, active_pools):
        """Validity key: recompute the per-ROI 1/Z fits only when one of these
        fit-relevant inputs changes."""
        return (
            tuple(r.name for r in rois),
            tuple(round(float(ds.get('b1_ut', 0.0)), 4) for ds in datasets),
            round(p['R1'], 6), round(p['B0_MHz'], 4), p['peak_type'],
            tuple(p['ppm_exclude_MT']), tuple(p['ppm_reinclude_MT']),
            tuple(p['ppm_include_CEST']), tuple(active_pools),
            bool(p.get('fit_mt', True)),
        )

    def _compute_roi_1z(self, parent=None, silent=False) -> bool:
        """Extract per-ROI mean Z-spectra and run the full 1/Z pipeline for every
        ROI × B1.  Caches the result on the tab so that opening the ROI-spectra
        window (or exporting the QUESP stack) is instant.  Returns True on
        success.  Reuses the cache when nothing fit-relevant changed.

        When `silent`, ``None`` is returned instead of popping info dialogs for
        the "no data / no ROI" cases (used by the auto-precompute after Run).
        """
        rois = [r for r in getattr(self, '_last_rois', [])
                if r.name != 'Phantom_outline']
        datasets = self._datasets
        if not datasets or not rois:
            return False

        p            = self._get_params()
        active_pools = self._get_active_pools()
        _key         = self._roi_1z_key(rois, datasets, p, active_pools)

        # Cache hit → nothing to do.
        if (getattr(self, '_last_1z_key', None) == _key
                and getattr(self, '_last_1z_roi_results', None)):
            return True

        # ── Extract ROI mean Z-spectra for every dataset ───────────────────────
        roi_spectra: dict[str, list] = {roi.name: [] for roi in rois}
        for ds in datasets:
            z_vol  = np.array(ds['z_img'])
            ppm_ds = np.array(ds['ppm'], dtype=float)
            sl     = ds.get('slice', 0)
            b1_ut  = float(ds.get('b1_ut', p['satpwr_uT']))
            z_sl   = z_vol[:, :, sl, :] if z_vol.ndim == 4 else z_vol
            z_sl   = np.clip(z_sl.astype(float), 1e-6, 1.0)
            H, W   = z_sl.shape[:2]
            for roi in rois:
                msk = roi.mask
                if msk.shape != (H, W):
                    try:
                        from scipy.ndimage import zoom as _zm
                        msk = _zm(msk.astype(float),
                                   (H / msk.shape[0], W / msk.shape[1]), order=1) > 0.5
                    except Exception:
                        continue
                if not msk.any():
                    continue
                z_flat = z_sl.reshape(-1, z_sl.shape[-1])
                z_mean = z_flat[msk.ravel()].mean(axis=0)
                roi_spectra[roi.name].append((ppm_ds, z_mean, b1_ut))

        roi_spectra = {k: v for k, v in roi_spectra.items() if v}
        if not roi_spectra:
            if not silent:
                QMessageBox.information(self, "No valid ROIs",
                                        "ROI masks don't match image dimensions.")
            return False

        # ── Run pipeline per ROI × per B1 (with progress dialog) ───────────────
        total_fits = sum(len(v) for v in roi_spectra.values())
        prog = QProgressDialog("Running 1/Z fitting…", "Cancel",
                               0, total_fits, parent or self)
        prog.setWindowTitle("Computing per-ROI, per-B1 fits")
        prog.setMinimumDuration(0)
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.show()

        roi_results: dict[str, list] = {
            rname: [None] * len(v) for rname, v in roi_spectra.items()}

        def _fit_one(rname, idx, ppm_ds, z_mean, b1_ut):
            return rname, idx, _run_pipeline(
                ppm_ds, z_mean, satpwr_uT=b1_ut, R1=p['R1'], B0_MHz=p['B0_MHz'],
                peak_type=p['peak_type'], ppm_exclude_MT=p['ppm_exclude_MT'],
                ppm_reinclude_MT=p['ppm_reinclude_MT'],
                ppm_include_CEST=p['ppm_include_CEST'], pool_names=active_pools,
                fit_mt=p.get('fit_mt', True))

        jobs = [(rname, i, s[0], s[1], s[2])
                for rname, lst in roi_spectra.items()
                for i, s in enumerate(lst)]
        fit_count = 0
        try:
            import os as _os
            from concurrent.futures import ThreadPoolExecutor, as_completed
            _nw = max(2, min(8, (_os.cpu_count() or 4)))
            with ThreadPoolExecutor(max_workers=_nw) as _ex:
                futs = [_ex.submit(_fit_one, *j) for j in jobs]
                for fut in as_completed(futs):
                    if prog.wasCanceled():
                        _ex.shutdown(wait=False, cancel_futures=True)
                        prog.close()
                        return False
                    try:
                        rname, idx, res = fut.result()
                        roi_results[rname][idx] = res
                    except Exception as exc:
                        self._log(f"Pipeline failed: {exc}")
                    fit_count += 1
                    prog.setValue(fit_count)
                    QApplication.processEvents()
        except Exception:
            for rname, idx, ppm_ds, z_mean, b1_ut in jobs:
                if prog.wasCanceled():
                    prog.close(); return False
                try:
                    roi_results[rname][idx] = _fit_one(
                        rname, idx, ppm_ds, z_mean, b1_ut)[2]
                except Exception as exc:
                    self._log(f"Pipeline failed ROI={rname!r}: {exc}")
                fit_count += 1
                prog.setValue(fit_count); QApplication.processEvents()
        prog.setValue(total_fits)
        prog.close()

        # Cache — reused by the ROI-spectra window, the QUESP plots, and the
        # "Save 1/Z → QUESP (.mat)…" export.  Storing the key lets us skip the
        # (slow) refit when the window is reopened with unchanged parameters.
        self._last_1z_roi_results = roi_results
        self._last_1z_roi_spectra = roi_spectra
        self._last_1z_params      = p
        self._last_1z_key         = _key
        return True

    # ── QUESP pool selection ───────────────────────────────────────────────────

    def _quesp_candidate_pools(self) -> list[str]:
        """Exchange pools eligible for QUESP — the active 1/Z pools minus water
        (the reference pool) and MT (the semisolid background)."""
        return [pl for pl in self._get_active_pools()
                if pl not in ('water', 'MT')]

    def _resolved_quesp_pools(self) -> list[str]:
        """Pools to actually draw QUESP for: the user's choice if set, else all
        candidates (kept in sync with the current pool selection)."""
        cand = self._quesp_candidate_pools()
        if self._quesp_plot_pools is None:
            return cand
        return [pl for pl in self._quesp_plot_pools if pl in cand]

    def _ask_quesp_pools(self, parent=None) -> bool:
        """Pop the "Plot QUESP for which pools:" chooser.  Returns True if the
        user accepted (selection stored in ``self._quesp_plot_pools``)."""
        cand = self._quesp_candidate_pools()
        if not cand:
            return False
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                     QLabel, QCheckBox, QPushButton)
        dlg = QDialog(parent or self)
        dlg.setWindowTitle("QUESP pools")
        vl = QVBoxLayout(dlg)
        vl.addWidget(QLabel("<b>Plot QUESP for which pools:</b>"))
        vl.addWidget(QLabel(
            "<span style='color:#888;font-size:11px;'>MTR<sub>asym</sub>, "
            "MTR<sub>Rex</sub> and the Ω-plot are drawn below the 1/Z spectra "
            "for each selected pool.</span>"))
        pre = self._resolved_quesp_pools()
        checks: dict[str, QCheckBox] = {}
        for pl in cand:
            c = QCheckBox(pl)
            c.setChecked(pl in pre)
            checks[pl] = c
            vl.addWidget(c)
        btn_row = QHBoxLayout()
        btn_all = QPushButton("All"); btn_none = QPushButton("None")
        btn_all.clicked.connect(lambda: [c.setChecked(True) for c in checks.values()])
        btn_none.clicked.connect(lambda: [c.setChecked(False) for c in checks.values()])
        btn_row.addWidget(btn_all); btn_row.addWidget(btn_none); btn_row.addStretch()
        btn_ok = QPushButton("OK"); btn_cancel = QPushButton("Cancel")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(dlg.accept); btn_cancel.clicked.connect(dlg.reject)
        btn_row.addWidget(btn_ok); btn_row.addWidget(btn_cancel)
        vl.addLayout(btn_row)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._quesp_plot_pools = [pl for pl, c in checks.items() if c.isChecked()]
            return True
        return False

    def _show_roi_spectra(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        import matplotlib.pyplot as _plt
        from my_gui.fig_theme import apply_fig_dark_theme
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
            QWidget as _QW, QScrollArea as _QSA,
            QLabel as _QL, QDoubleSpinBox as _QDSP, QSpinBox as _QSB,
            QPushButton as _QPB, QComboBox as _QCB,
        )

        rois     = [r for r in getattr(self, '_last_rois', [])
                    if r.name != 'Phantom_outline']
        datasets = self._datasets

        if not datasets:
            QMessageBox.information(self, "No Data", "Load a Z-spectrum scan first.")
            return
        if not rois:
            QMessageBox.information(self, "No ROIs", "Draw ROIs in the ROI Manager tab first.")
            return

        p = self._get_params()
        n_b1 = len(datasets)

        # ── Reuse the cached dialog if nothing relevant changed ───────────────
        # (avoids re-running the slow per-ROI fit every time the window reopens)
        _active_pools = self._get_active_pools()
        _key = self._roi_1z_key(rois, datasets, p, _active_pools)
        _cached = self._roi_spectra_dlg
        if _cached is not None and self._roi_spectra_key == _key:
            try:
                _cached.show(); _cached.raise_(); _cached.activateWindow()
                return
            except RuntimeError:
                self._roi_spectra_dlg = None   # was deleted — rebuild below
        # Different data/params → discard the stale dialog before rebuilding
        if _cached is not None:
            try:
                _cached.close(); _cached.deleteLater()
            except RuntimeError:
                pass
            self._roi_spectra_dlg = None

        # Assign one colour per B1 power
        _cmap_fn = _plt.cm.get_cmap('tab10', max(n_b1, 2))
        B1_COLORS = [_cmap_fn(i) for i in range(n_b1)]

        # ── Ensure the per-ROI 1/Z fits are computed (reuses the cache when the
        #     precompute already ran on "Run Inverse Z Analysis") ──────────────
        if not self._compute_roi_1z():
            return
        # (fs/ksw/R² parametric maps are generated on the main Run — see _on_done)
        roi_results = self._last_1z_roi_results
        roi_spectra = self._last_1z_roi_spectra
        p           = self._last_1z_params
        total_fits  = sum(len(v) for v in roi_spectra.values())

        # ── Build display dialog ────────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle(
            "ROI Spectra — 1/Z Fitting  "
            "[MT fit | 1/Z decomposition | Z reconstruction]"
        )
        dlg.resize(1280, 660)
        vl = QVBoxLayout(dlg)
        vl.setSpacing(4)

        # ── Top bar (always visible): Enable Plot Customization + Export ────────
        top_row = QHBoxLayout()
        chk_customize = QCheckBox("Enable Plot Customization")
        chk_customize.setChecked(False)
        chk_customize.setStyleSheet("font-weight:bold;")
        chk_customize.setToolTip(
            "Show the peak-type / fit-window / pool-colour / font / title controls "
            "for the ROI-spectra figures.")
        top_row.addWidget(chk_customize)
        top_row.addStretch()
        btn_export = _QPB("Export…")
        btn_export.setToolTip(
            "Export the current panel view — PNG / JPEG at 300 dpi (or "
            "PDF / SVG / TIFF / .mat / .npz).")
        top_row.addWidget(btn_export)
        vl.addLayout(top_row)

        # ── Customization panel (hidden until 'Enable Plot Customization') ──────
        custom_panel = _QW()
        _cp_lay = QVBoxLayout(custom_panel)
        _cp_lay.setContentsMargins(0, 0, 0, 0)
        _cp_lay.setSpacing(4)
        custom_panel.setVisible(False)
        chk_customize.toggled.connect(custom_panel.setVisible)
        vl.addWidget(custom_panel)

        # Row 1: Refit parameters
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(_QL("<b>Peak type:</b>"))
        combo_pk = _QCB()
        combo_pk.addItems(["Lorentzian", "Pseudo-Voigt"])
        combo_pk.setCurrentText("Pseudo-Voigt" if p['peak_type'] == 'pseudovoigt'
                                else "Lorentzian")
        ctrl_row.addWidget(combo_pk)
        ctrl_row.addWidget(_QL("   MT excl. ±"))
        spin_mtex = _QDSP(); spin_mtex.setRange(1, 20)
        spin_mtex.setValue(self.spin_mt_excl.value()); spin_mtex.setSuffix(" ppm")
        spin_mtex.setFixedWidth(90)
        ctrl_row.addWidget(spin_mtex)
        ctrl_row.addWidget(_QL("Water reincl. ±"))
        spin_mtrein = _QDSP(); spin_mtrein.setRange(0, 3)
        spin_mtrein.setValue(self.spin_mt_rein.value()); spin_mtrein.setSuffix(" ppm")
        spin_mtrein.setFixedWidth(80)
        ctrl_row.addWidget(spin_mtrein)
        ctrl_row.addWidget(_QL("   CEST"))
        spin_clo = _QDSP(); spin_clo.setRange(-20, 0)
        spin_clo.setValue(self.spin_cest_lo.value())
        spin_clo.setSuffix(" ppm"); spin_clo.setFixedWidth(80)
        ctrl_row.addWidget(spin_clo)
        ctrl_row.addWidget(_QL("to"))
        spin_chi = _QDSP(); spin_chi.setRange(0, 20)
        spin_chi.setValue(self.spin_cest_hi.value())
        spin_chi.setSuffix(" ppm"); spin_chi.setFixedWidth(80)
        ctrl_row.addWidget(spin_chi)
        btn_refit = _QPB("Refit")
        ctrl_row.addWidget(btn_refit)
        ctrl_row.addStretch()
        _cp_lay.addLayout(ctrl_row)

        # Row 2: Pool visibility checkboxes + per-pool colour pickers
        pool_row = QHBoxLayout()
        pool_row.addWidget(_QL("<b>Show pools:</b>"))
        # Build one checkbox per ACTUALLY-fitted pool (from the user's selection),
        # so iopamidol etc. get a toggle and unselected defaults (amine/amide)
        # never appear.  Each real pool also gets a colour swatch → QColorDialog
        # so the user can recolour it (stored in _pool_colors, used by the draw).
        _pool_checks: dict[str, QCheckBox] = {}
        _pool_colors: dict[str, str] = {}     # user-overridable pool colours
        _fallback_chk = ['#2ca02c', '#d62728', '#e377c2', '#ff7f0e', '#9467bd',
                         '#17becf', '#8c564b', '#bcbd22', '#7f7f7f']
        _chk_items = [('raw', 'Raw', '#888888'), ('sum', 'Sum', '#000000')]
        for _i, _pk in enumerate(self._get_active_pools()):
            _chk_items.append(
                (_pk, _pk, POOL_COLORS.get(_pk, _fallback_chk[_i % len(_fallback_chk)])))
        for _key, _lbl, _col in _chk_items:
            _pool_colors[_key] = _col
            chk = QCheckBox(_lbl)
            chk.setChecked(True)
            chk.setStyleSheet(f"QCheckBox::indicator:checked {{background: {_col};"
                              f"border: 2px solid {_col}; border-radius: 3px;}}")
            _pool_checks[_key] = chk
            pool_row.addWidget(chk)
            # Colour picker swatch — real pools only (not Raw/Sum aggregates).
            if _key not in ('raw', 'sum'):
                sw = _QPB(); sw.setFixedSize(16, 16)
                sw.setStyleSheet(f"background:{_col};border:1px solid #888;border-radius:3px;")
                sw.setToolTip(f"Change {_lbl} colour")
                def _pick_color(_checked=False, k=_key, s=sw, c=chk):
                    from PyQt6.QtWidgets import QColorDialog
                    from PyQt6.QtGui import QColor
                    col = QColorDialog.getColor(QColor(_pool_colors[k]), dlg,
                                                f"{k} colour")
                    if col.isValid():
                        hexc = col.name()
                        _pool_colors[k] = hexc
                        s.setStyleSheet(f"background:{hexc};border:1px solid #888;"
                                        f"border-radius:3px;")
                        c.setStyleSheet(f"QCheckBox::indicator:checked {{background:{hexc};"
                                        f"border:2px solid {hexc};border-radius:3px;}}")
                        _redraw_all()
                sw.clicked.connect(_pick_color)
                pool_row.addWidget(sw)
        pool_row.addStretch()
        _cp_lay.addLayout(pool_row)

        # Row 3: Font / display controls — full Title / Main / Axes / Ticks /
        # Legend size set + font family, matching the "Per-ROI T1/T2 Fit Curves"
        # toolbar so the ROI-spectra (1/Z + QUESP) figures are fully stylable.
        font_row = QHBoxLayout()
        font_row.addWidget(_QL("Font:"))
        combo_ff = _QCB(); combo_ff.setFixedWidth(140)
        combo_ff.setToolTip("Font family for titles, axis labels and legends")
        for _ff in ("Default", "Arial", "Times New Roman",
                    "Helvetica", "DejaVu Sans", "DejaVu Serif"):
            combo_ff.addItem(_ff)
        font_row.addWidget(combo_ff)

        def _mkfs(label, default, lo, hi, tip):
            font_row.addWidget(_QL(label))
            sp = _QSB(); sp.setRange(lo, hi); sp.setValue(default)
            sp.setFixedWidth(50); sp.setToolTip(tip)
            font_row.addWidget(sp)
            return sp

        spin_tfs    = _mkfs("Title:",  11, 5, 32, "Subplot title font size")
        spin_main   = _mkfs("Main:",   13, 5, 32, "Overall figure title font size")
        spin_axes   = _mkfs("Axes:",   10, 5, 28, "X / Y axis label font size")
        spin_tkfs   = _mkfs("Ticks:",   9, 5, 24, "Tick label font size")
        spin_legend = _mkfs("Legend:",  9, 5, 24, "Legend font size")
        spin_dot    = _mkfs("Dot:",    15, 1, 30, "Data point size")
        font_row.addWidget(_QL("   "))
        chk_dark_bg = QCheckBox("Bg")
        chk_dark_bg.setToolTip(
            "Black background for the figures (for slides). Only the surround "
            "and labels flip - the plotted curves stay identical.")
        font_row.addWidget(chk_dark_bg)
        dlg._chk_dark_bg = chk_dark_bg
        def _dark_on() -> bool:
            cb = getattr(dlg, "_chk_dark_bg", None)
            try:
                return bool(cb.isChecked()) if cb is not None else False
            except Exception:
                return False
        font_row.addStretch()
        _cp_lay.addLayout(font_row)

        # Editable plot title row
        title_row = QHBoxLayout()
        title_row.addWidget(_QL("  Plot title:"))
        from PyQt6.QtWidgets import QLineEdit as _QLE2
        edit_plot_title = _QLE2()
        edit_plot_title.setPlaceholderText("Leave blank for default  (ROI: <name>)")
        edit_plot_title.setFixedHeight(26)
        title_row.addWidget(edit_plot_title, stretch=1)
        _cp_lay.addLayout(title_row)

        # Shared font-properties helper — honours the chosen family + size (+ bold
        # for titles).  Used by every ROI-spectra / QUESP draw below so the Font /
        # Title / Main / Axes / Legend controls apply uniformly.
        from matplotlib.font_manager import FontProperties as _FP
        def _fp(size, bold=False):
            ff = combo_ff.currentText()
            kw = {'size': size}
            if ff and ff.lower() != "default":
                kw['family'] = ff
            if bold:
                kw['weight'] = 'bold'
            return _FP(**kw)

        # Tab widget — one tab per ROI
        tab_w = QTabWidget()
        vl.addWidget(tab_w, stretch=1)

        tab_figs:    dict[str, list] = {}   # per-ROI [panel][b1] single-panel figs
        tab_panelwidget: dict       = {}   # per-ROI outer QTabWidget (one tab per panel)
        tab_b1widgets: dict         = {}   # per-ROI list of B1 QTabWidgets (one per panel)
        tab_inners:  dict           = {}
        tab_layouts: dict[str, QVBoxLayout] = {}
        tab_q_holder: dict          = {}   # QUESP holder (widget, layout) per ROI
        tab_q_figs:  dict[str, object] = {}   # QUESP Figure per ROI (or None)
        tab_wheel:   dict           = {}   # per-ROI wheel→scroll forwarder

        # Panel metadata (index → title, y-label) shared by the drawing and the
        # per-ROI panel tabs.  Panel 0 = Z-spectrum + MT fit, 1 = 1/Z decomp,
        # 2 = Z reconstruction.
        _PANEL_META = [
            ('Z-spectrum  +  MT fit',       'Z'),
            ('CEST  Fitting  in  1/Z',      r'$R_1\cos^2\theta\,(1/Z-1)$'),
            ('Z-spectrum',                  'Z'),
        ]

        # ── Drawing function — draws the requested panel subset for a ROI ─────
        # ``panels`` selects which of the three panels to render (defaults to all
        # three).  Passing a single index yields a one-panel figure, which the
        # per-panel sub-tabs use so each panel type gets its own All-B1 / per-B1
        # views.
        def _draw_roi_tab(rname: str, res_list: list, visible: set,
                          panels=(0, 1, 2)) -> Figure:
            tfs  = spin_tfs.value()
            tkfs = spin_tkfs.value()
            dts  = spin_dot.value()
            mfs  = spin_main.value()      # overall figure (suptitle) size
            afs  = spin_axes.value()      # x/y axis-label size
            lfs  = spin_legend.value()    # legend size

            panels = tuple(panels)
            n_panels = max(len(panels), 1)

            n_entries = max(len(res_list), 1)  # actual number of entries in this ROI
            n_valid   = sum(1 for r in res_list if r is not None)
            # Build a colour list from the actual res_list length — avoids IndexError
            # when ROIs have duplicate names (each appends separately) or when
            # len(res_list) != n_b1 for any other reason.
            _tab10     = _plt.cm.get_cmap('tab10', max(n_entries, 2))
            _RES_COLORS = [_tab10(i) for i in range(n_entries)]
            # Single-B1 view → use the SAME colour scheme as the Quantitative Z
            # Analysis tab's 1/Z subtab: data/curves in tab10-blue, summed fit in
            # black (white on a dark background).  Multi-B1 keeps per-power colours.
            _single   = (n_valid == 1)
            _ZBLUE    = _plt.cm.get_cmap('tab10', 10)(0)     # zspec 1/Z data colour
            _SUM_SINGLE = ('white' if _dark_on() else 'black')

            _fig_w = 5.0 * n_panels if n_panels > 1 else 6.8
            fig  = Figure(figsize=(_fig_w, 4.6 + 0.3 * max(n_entries - 1, 0)),
                          facecolor='white')
            # axes[k] is the axis for panel k, or None when that panel isn't drawn
            axes = [None, None, None]
            for _slot, _pidx in enumerate(panels):
                axes[_pidx] = fig.add_subplot(1, n_panels, _slot + 1)
            _custom_title = edit_plot_title.text().strip()
            fig.suptitle(
                _custom_title if _custom_title else f"ROI: {rname}",
                fontproperties=_fp(mfs, bold=True))

            for b1_idx, res in enumerate(res_list):
                if res is None:
                    continue
                color   = _ZBLUE if _single else _RES_COLORS[b1_idx]
                b1_lbl  = f"B1={res['satpwr_uT']:.2f} µT"
                ppm_s   = res['ppm']
                z_s     = res['zspec']
                mt_Z    = res['mt_Z']
                fit_mask= res['mt_fit_mask']
                ppm_fit = res['cest_ppm_fit']
                target  = res['cest_target']
                mt_cos2 = res['cest_mt_cos2']
                ind_fits= res['cest_ind_fits']
                R1_val  = res['R1']

                ppm_fine = np.linspace(ppm_s.min(), ppm_s.max(), 600)
                satHz_f  = res['satpwr_uT'] * _GAMMA_HZ_UT
                cos2_f   = (ppm_fine * res['B0_MHz'])**2 / (
                             (ppm_fine * res['B0_MHz'])**2 + satHz_f**2)
                mt_Z_f    = np.interp(ppm_fine, ppm_s, mt_Z)
                invZ_MT_f = R1_val * (1.0 / np.clip(mt_Z_f, 1e-6, None) - 1.0)
                mt_cos2_f = invZ_MT_f * cos2_f

                cest_lo = float(ppm_fit.min())
                cest_hi = float(ppm_fit.max())
                cmask_f = (ppm_fine >= cest_lo) & (ppm_fine <= cest_hi)
                ppm_fc  = ppm_fine[cmask_f]
                cos2_fc = cos2_f[cmask_f]

                pfn_fn = _pfn(res['peak_type'])
                coeffs = res['cest_coeffs']

                # Use actual fitted pool names (not hardcoded POOL_NAMES)
                _fit_pool_names = list(coeffs.keys())
                ind_fits_f: dict[str, np.ndarray] = {}
                for nm in _fit_pool_names:
                    if nm in coeffs:
                        ind_fits_f[nm] = cos2_fc * pfn_fn(coeffs[nm], ppm_fc)

                total_f = mt_cos2_f[cmask_f] + sum(
                    ind_fits_f.get(nm, np.zeros_like(ppm_fc)) for nm in _fit_pool_names)

                rcos2_f = invZ_MT_f * cos2_f
                for nm in _fit_pool_names:
                    if nm in coeffs:
                        rcos2_f += cos2_f * pfn_fn(coeffs[nm], ppm_fine)
                z_fit_f = np.clip(R1_val * cos2_f / (rcos2_f + R1_val * cos2_f + 1e-20),
                                   0.0, 1.0)

                raw_1z = target + mt_cos2
                alpha_fill = 0.55 if n_b1 > 1 else 0.9

                # Panel A — Z-spectrum + MT fit
                ax = axes[0]
                if ax is not None:
                    # Show all data points (dim), then highlight the points used
                    # for MT fitting (brighter / larger) so the user can see both
                    ax.scatter(ppm_s, z_s, c=[color],
                               s=dts * 0.45, alpha=0.35, zorder=3)
                    ax.scatter(ppm_s[fit_mask], z_s[fit_mask], c=[color],
                               s=dts * 0.9, alpha=alpha_fill, zorder=4,
                               label=b1_lbl)
                    ax.plot(ppm_fine, mt_Z_f, '-', color=color, lw=1.8,
                            zorder=5)

                # Panel B — 1/Z decomposition
                ax = axes[1]
                if ax is not None:
                    if 'raw' in visible:
                        ax.scatter(ppm_fit, raw_1z, c=[color], s=dts * 0.8,
                                   alpha=alpha_fill, zorder=3)
                    if 'MT' in visible:
                        # MT: solid line with its own fixed colour (user-overridable)
                        _mtc = _pool_colors.get('MT') or POOL_COLORS['MT']
                        ax.plot(ppm_fine, mt_cos2_f, '-', color=_mtc,
                                lw=1.5, alpha=0.85, label='MT')
                    # ── Individual pool contributions (solid lines, no shading) ─
                    # Build dynamically from fitted pools so newly selected pools
                    # (7.3ppm, Trp, …) are drawn automatically.  Colours prefer the
                    # user-picked _pool_colors, then POOL_COLORS, then a fallback.
                    _fallback_colors = [
                        '#2ca02c','#d62728','#e377c2','#ff7f0e','#9467bd',
                        '#17becf','#8c564b','#bcbd22','#7f7f7f','#e377c2',
                    ]
                    _pool_draw = [
                        (_nm, _pool_colors.get(_nm)
                              or POOL_COLORS.get(_nm, _fallback_colors[i % len(_fallback_colors)]))
                        for i, _nm in enumerate(_fit_pool_names)
                    ]
                    for _nm, _col in _pool_draw:
                        if _nm in visible and _nm in ind_fits_f:
                            _curve = ind_fits_f[_nm]
                            ax.plot(ppm_fc, _curve, color=_col, lw=1.8,
                                    alpha=0.9, label=_nm, zorder=5)
                    # ── Total fit on top ─ single-B1 → black (zspec 1/Z style),
                    #    multi-B1 → the B1-power colour so the powers stay distinct.
                    if 'sum' in visible:
                        _sum_col = _SUM_SINGLE if _single else color
                        ax.plot(ppm_fc, total_f, '-', color=_sum_col, lw=2.2,
                                alpha=0.9 if _single else 0.65,
                                label='sum', zorder=6)

                # Panel C — Reconstructed Z-spectrum
                ax = axes[2]
                if ax is not None:
                    if 'raw' in visible:
                        ax.scatter(ppm_s, z_s, c=[color], s=dts * 0.8,
                                   alpha=alpha_fill)
                    ax.plot(ppm_fine, z_fit_f, '-', color=color, lw=2,
                            label=b1_lbl, zorder=5)

            # Axes decoration — only for the panels actually drawn
            for ax_i in range(3):
                ax = axes[ax_i]
                if ax is None:
                    continue
                title, ylabel = _PANEL_META[ax_i]
                ax.set_facecolor('#f9f9f9')
                ax.set_xlabel('Δω  (ppm)', fontproperties=_fp(afs))
                ax.set_ylabel(ylabel, fontproperties=_fp(afs))
                ax.set_title(title, fontproperties=_fp(tfs, bold=True))
                ax.tick_params(labelsize=tkfs)
                ax.grid(True, alpha=0.2, ls='--')
                if n_valid > 0:
                    all_ppm = np.concatenate([r['ppm'] for r in res_list if r])
                    ax.set_xlim(all_ppm.max() + 0.3, all_ppm.min() - 0.3)
                if ax_i in (0, 2):
                    ax.set_ylim(0, 1.05)
                if ax_i == 1:
                    # Restrict panel B x-axis to the CEST fitting window
                    _b_lo = spin_clo.value()
                    _b_hi = spin_chi.value()
                    ax.set_xlim(_b_hi + 0.5, _b_lo - 0.5)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    # De-duplicate so each series (pool / MT / sum / B1 curve)
                    # appears once — mirrors the 1/Z subtab's clean legend even
                    # when several B1 powers are overlaid.
                    _seen: dict = {}
                    for _h, _l in zip(handles, labels):
                        if _l not in _seen:
                            _seen[_l] = _h
                    leg = ax.legend(_seen.values(), _seen.keys(),
                                    prop=_fp(lfs),
                                    loc='upper right', framealpha=0.85)
                    leg.set_draggable(True)

            fig.tight_layout(pad=1.2, rect=[0, 0, 1, 0.93])
            apply_fig_dark_theme(fig, _dark_on())
            return fig

        def _visible_set() -> set:
            return {k for k, c in _pool_checks.items() if c.isChecked()}

        # ── QUESP drawing — MTRasym / MTRRex / Ω-plot per chosen pool ─────────
        def _draw_quesp_fig(rname: str, res_list: list):
            pools = self._resolved_quesp_pools()
            valid = [r for r in res_list if r is not None]
            if not pools or len(valid) < 2:
                return None
            satpwr = [r['satpwr_uT'] for r in valid]
            R1v  = p['R1']; B0v = p['B0_MHz']
            tsat = float(p.get('tsat', 2.0)); trec = float(p.get('trec', 8.0))
            tfs  = spin_tfs.value(); tkfs = spin_tkfs.value(); dts = spin_dot.value()
            mfs  = spin_main.value(); afs = spin_axes.value()

            n_pool = len(pools)
            fig = Figure(figsize=(15, 3.3 * n_pool), facecolor='white')
            fig.suptitle(f"QUESP — ROI: {rname}",
                         fontproperties=_fp(mfs, bold=True))

            # Common QUESP style for every pool: blue open-circle data + red fit
            # (matches the QUESP tab), rather than a per-pool colour.
            _D_CLR, _F_CLR = '#1f77b4', '#d62728'
            for pi, pool in enumerate(pools):
                axes = [fig.add_subplot(n_pool, 3, pi * 3 + j + 1) for j in range(3)]
                col  = _D_CLR

                coeffs_list, ok = [], True
                for r in valid:
                    c = r['cest_coeffs'].get(pool)
                    if c is None:
                        ok = False; break
                    coeffs_list.append(c)

                q = _quesp_pool(coeffs_list, satpwr, R1v, B0v, tsat, trec) if ok else None
                if q is None:
                    for ax in axes:
                        ax.text(0.5, 0.5, f"{pool}: not fitted", ha='center',
                                va='center', transform=ax.transAxes, fontsize=tkfs)
                        ax.set_xticks([]); ax.set_yticks([])
                    continue

                def _annot(ax, d):
                    ax.text(0.03, 0.97,
                            f"f$_s$={d['fs']:.2e}\n"
                            f"k$_{{sw}}$={d['ksw']:.0f} s$^{{-1}}$\n"
                            f"R²={d['rsq']:.3f}",
                            transform=ax.transAxes, va='top', ha='left',
                            fontsize=max(6, tkfs - 1),
                            bbox=dict(boxstyle='round', fc='white', ec=col, alpha=0.9))

                def _pts(ax, x, y):
                    ax.scatter(x, y, s=dts * 2.4, facecolors='white',
                               edgecolors=_D_CLR, linewidths=1.8, zorder=4,
                               label='Data')

                # Panel 1 — Regular QUESP (MTRasym)
                reg = q['regular']; ax = axes[0]
                _pts(ax, reg['x'], reg['y'])
                if reg['xfit'] is not None:
                    ax.plot(reg['xfit'], reg['yfit'], '-', color=_F_CLR, lw=2)
                ax.set_title(f"{pool} • MTR$_{{asym}}$", fontproperties=_fp(tfs, bold=True))
                ax.set_xlabel("B$_1$ (µT)", fontproperties=_fp(afs))
                ax.set_ylabel("MTR$_{asym}$", fontproperties=_fp(afs))
                _annot(ax, reg)

                # Panel 2 — Inverse QUESP (MTRRex)
                inv = q['inverse']; ax = axes[1]
                _pts(ax, inv['x'], inv['y'])
                if inv['xfit'] is not None:
                    ax.plot(inv['xfit'], inv['yfit'], '-', color=_F_CLR, lw=2)
                ax.set_title(f"{pool} • MTR$_{{Rex}}$", fontproperties=_fp(tfs, bold=True))
                ax.set_xlabel("B$_1$ (µT)", fontproperties=_fp(afs))
                ax.set_ylabel("MTR$_{Rex}$", fontproperties=_fp(afs))
                _annot(ax, inv)

                # Panel 3 — Ω-plot (1/MTRRex vs 1/ω1²)
                om = q['omega']; ax = axes[2]
                good = om['good']
                _pts(ax, om['x'][good], om['y'][good])
                if om['xfit'] is not None:
                    ax.plot(om['xfit'], om['yfit'], '--', color=_F_CLR, lw=2)
                ax.set_title(f"{pool} • Ω-plot", fontproperties=_fp(tfs, bold=True))
                ax.set_xlabel("1 / ω$_1^2$  (s²·rad$^{-2}$)", fontproperties=_fp(afs))
                ax.set_ylabel("1 / MTR$_{Rex}$", fontproperties=_fp(afs))
                _annot(ax, om)

                for ax in axes:
                    ax.tick_params(labelsize=tkfs)
                    ax.grid(True, alpha=0.2, ls='--')
                    ax.set_facecolor('#f9f9f9')

            fig.tight_layout(pad=1.1, rect=[0, 0, 1, 0.95])
            apply_fig_dark_theme(fig, _dark_on())
            return fig

        def _clear_layout(l):
            while l.count():
                it = l.takeAt(0)
                w = it.widget()
                if w is not None:
                    w.setParent(None); w.deleteLater()

        def _populate_quesp(rname: str):
            holder, q_lay = tab_q_holder[rname]
            _clear_layout(q_lay)
            qfig = _draw_quesp_fig(rname, roi_results[rname])
            tab_q_figs[rname] = qfig
            if qfig is None:
                lbl = _QL("<i>QUESP needs ≥2 B1 powers and at least one exchange "
                          "pool — choose pools via “QUESP pools…” below.</i>")
                lbl.setStyleSheet("color:#888; padding:6px;")
                q_lay.addWidget(lbl)
                holder.setMaximumHeight(44)
                return
            from matplotlib.backends.backend_qt import NavigationToolbar2QT as _NTB
            npool = max(len(self._resolved_quesp_pools()), 1)
            lbl = _QL("<b>QUESP</b>  —  f<sub>s</sub> / k<sub>sw</sub> from "
                      "MTR<sub>asym</sub>, MTR<sub>Rex</sub> and the Ω-plot")
            q_lay.addWidget(lbl)
            qcv = FigureCanvas(qfig)
            qcv.setMinimumHeight(230 * npool)
            _wf = tab_wheel.get(rname)
            if _wf is not None:
                qcv.installEventFilter(_wf)      # wheel scrolls the ROI panel
            qtb = _NTB(qcv, holder)
            q_lay.addWidget(qtb)
            q_lay.addWidget(qcv)
            holder.setMaximumHeight(16777215)

        # Short tab labels for the three panel types (one panel per tab, exactly
        # as requested: each panel gets its own All-B1 / per-B1 sub-tabs).
        _PANEL_TAB_LABELS = ['Z-spectrum + MT Fit', 'CEST Fitting in 1/Z', 'Z-spectrum']
        # Panels actually shown/exported: drop "Z-spectrum + MT Fit" (index 0)
        # when the MT background is NOT fitted (it would just be a flat Z_MT=1).
        _FULL_PANELS = (0, 1, 2) if p.get('fit_mt', True) else (1, 2)

        def _make_b1_tabs(rname: str, res_list: list, visible: set, panel_idx: int):
            """Build a QTabWidget of B1 pages for ONE panel type: an 'All B1'
            overlay (when >1 power) then one page per B1 saturation power, each
            showing only ``panel_idx``.  Returns ``(tabwidget, [figs])``."""
            from matplotlib.backends.backend_qt import NavigationToolbar2QT as _NTB
            sub = QTabWidget()
            sub.setTabPosition(QTabWidget.TabPosition.North)
            sub.setDocumentMode(True)
            figs: list = []

            def _add_page(figure, label):
                page = _QW()
                ply  = QVBoxLayout(page)
                ply.setContentsMargins(2, 2, 2, 2)
                cv = FigureCanvas(figure)
                cv.setMinimumHeight(380)
                _wf = tab_wheel.get(rname)
                if _wf is not None:
                    cv.installEventFilter(_wf)   # wheel scrolls the ROI panel
                tb = _NTB(cv, page)
                ply.addWidget(tb)
                ply.addWidget(cv)
                sub.addTab(page, label)
                figs.append(figure)

            n_nonnull = sum(1 for r in res_list if r is not None)
            # Overlay of all B1 powers first (only meaningful when >1 power).
            if n_nonnull > 1:
                _add_page(_draw_roi_tab(rname, res_list, visible, panels=(panel_idx,)),
                          "All B1")
            # One page per B1 saturation power (single-power view).
            for i, res in enumerate(res_list):
                if res is None:
                    continue
                single = [None] * len(res_list)
                single[i] = res           # keep index → same colour as the overlay
                _add_page(_draw_roi_tab(rname, single, visible, panels=(panel_idx,)),
                          f"B1 = {res['satpwr_uT']:.2f} µT")
            if sub.count() == 0:          # no valid power → single placeholder page
                _add_page(_draw_roi_tab(rname, res_list, visible, panels=(panel_idx,)),
                          rname)
            return sub, figs

        def _make_panel_tabs(rname: str, res_list: list, visible: set):
            """Build the outer QTabWidget with one tab per panel type
            (Z+MT / 1/Z / Z-recon), each holding its own B1 sub-tabs.  Returns
            ``(panel_tabwidget, [b1widget×3], [figs×3])``."""
            outer = QTabWidget()
            outer.setTabPosition(QTabWidget.TabPosition.North)
            b1_widgets: list = []
            fig_lists:  list = []
            # Show "Z-spectrum + MT Fit" ONLY when the MT background is actually
            # fitted — with MT off it is a flat Z_MT=1 line, so drop that panel.
            for _pidx in _FULL_PANELS:
                _b1w, _figs = _make_b1_tabs(rname, res_list, visible, _pidx)
                outer.addTab(_b1w, _PANEL_TAB_LABELS[_pidx])
                b1_widgets.append(_b1w)
                fig_lists.append(_figs)
            return outer, b1_widgets, fig_lists

        def _rebuild_b1(rname: str, res_list: list, vis: set):
            """Replace a ROI's panel/B1 tab tree in place, preserving the current
            panel tab and each panel's current B1 page, refreshing every figure."""
            old = tab_panelwidget[rname]
            panel_idx = old.currentIndex()
            b1_idx = [w.currentIndex() for w in tab_b1widgets.get(rname, [])]
            new, b1ws, figs = _make_panel_tabs(rname, res_list, vis)
            tab_layouts[rname].replaceWidget(old, new)
            old.setParent(None)
            old.deleteLater()
            tab_panelwidget[rname] = new
            tab_b1widgets[rname]   = b1ws
            tab_figs[rname]        = figs
            if 0 <= panel_idx < new.count():
                new.setCurrentIndex(panel_idx)
            for _w, _i in zip(b1ws, b1_idx):
                if 0 <= _i < _w.count():
                    _w.setCurrentIndex(_i)

        def _add_roi_tab(rname: str, res_list: list):
            scroll = _QSA()
            scroll.setWidgetResizable(True)
            tab_wheel[rname] = _WheelScrollFilter(scroll)   # wheel→scroll (canvases eat it)
            inner  = _QW()
            lay    = QVBoxLayout(inner)
            lay.setContentsMargins(4, 4, 4, 4)

            panel_w, b1ws, figs = _make_panel_tabs(rname, res_list, _visible_set())
            lay.addWidget(panel_w)
            tab_panelwidget[rname] = panel_w
            tab_b1widgets[rname]   = b1ws
            tab_figs[rname]        = figs

            # QUESP block — sits directly beneath the per-B1 spectra
            q_holder = _QW()
            q_lay    = QVBoxLayout(q_holder)
            q_lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(q_holder)
            tab_q_holder[rname] = (q_holder, q_lay)
            _populate_quesp(rname)

            scroll.setWidget(inner)
            tab_w.addTab(scroll, rname)
            tab_inners[rname]  = inner
            tab_layouts[rname] = lay

        for rname in roi_results:
            _add_roi_tab(rname, roi_results[rname])

        # ── Export (top button) — saves the CURRENT ROI / panel / B1 view as an
        #     image (PNG/JPEG @300 dpi, or PDF/SVG/TIFF), or bundles every ROI's
        #     full figure to .mat/.npz. ─────────────────────────────────────────
        def _export_current():
            import os
            from my_gui.fig_export import (save_figure, save_figures,
                                           FIG_EXPORT_FILTER)
            _ti = tab_w.currentIndex()
            _r  = tab_w.tabText(_ti) if _ti >= 0 else (
                next(iter(roi_results), "roi"))
            fp, _flt = QFileDialog.getSaveFileName(
                dlg, "Export figure", f"1z_{_r}.png", FIG_EXPORT_FILTER)
            if not fp:
                return
            ext = os.path.splitext(fp)[1].lower()
            if ext in ('.mat', '.npz'):
                _bundle = {}
                for _rn, _rl in roi_results.items():
                    try:
                        _bundle[_rn] = _draw_roi_tab(_rn, _rl, _visible_set(),
                                                     panels=_FULL_PANELS)
                    except Exception:
                        pass
                save_figures(_bundle, fp)
            else:
                fig_to_save = None
                if _r in tab_panelwidget:
                    _pi = tab_panelwidget[_r].currentIndex()
                    _b1ws = tab_b1widgets[_r]
                    _bi = _b1ws[_pi].currentIndex() if 0 <= _pi < len(_b1ws) else 0
                    _figs = tab_figs[_r]
                    if 0 <= _pi < len(_figs) and 0 <= _bi < len(_figs[_pi]):
                        fig_to_save = _figs[_pi][_bi]
                if fig_to_save is None:
                    fig_to_save = _draw_roi_tab(_r, roi_results.get(_r, []),
                                                _visible_set(), panels=_FULL_PANELS)
                # PNG / JPEG (and every image format) written at 300 dpi.
                save_figure(fig_to_save, fp, dpi=300)
        btn_export.clicked.connect(_export_current)

        def _redraw_all():
            vis = _visible_set()
            for rname, res_list in roi_results.items():
                _rebuild_b1(rname, res_list, vis)

        def _refresh_all_quesp():
            for _rn in roi_results:
                _populate_quesp(_rn)

        for _chk in _pool_checks.values():
            _chk.toggled.connect(lambda _: _redraw_all())
        # Font/size changes affect both the 1/Z panels and the QUESP panels.
        # Debounced so holding a spin arrow (or scrubbing the font combo) coalesces
        # into a single rebuild rather than one heavy rebuild per intermediate value.
        from PyQt6.QtCore import QTimer as _QTimer
        _font_timer = _QTimer(dlg)
        _font_timer.setSingleShot(True)
        _font_timer.setInterval(180)
        _font_timer.timeout.connect(lambda: (_redraw_all(), _refresh_all_quesp()))
        def _sched_font(*_):
            _font_timer.start()
        for _sp in (spin_tfs, spin_main, spin_axes, spin_tkfs, spin_legend, spin_dot):
            _sp.valueChanged.connect(_sched_font)
        combo_ff.currentIndexChanged.connect(_sched_font)
        edit_plot_title.editingFinished.connect(_redraw_all)
        # "Bg" toggle re-themes both the 1/Z spectra and the QUESP panels.
        chk_dark_bg.toggled.connect(lambda _: (_redraw_all(), _refresh_all_quesp()))

        def _do_refit():
            new_pt    = ('pseudovoigt' if combo_pk.currentText() == 'Pseudo-Voigt'
                         else 'lorentzian')
            new_mtex   = spin_mtex.value()
            new_mtrein = spin_mtrein.value()
            new_clo    = spin_clo.value()
            new_chi    = spin_chi.value()

            prog2 = QProgressDialog("Refitting…", "Cancel",
                                     0, total_fits, dlg)
            prog2.setMinimumDuration(0)
            prog2.setWindowModality(Qt.WindowModality.ApplicationModal)
            prog2.show()

            fit_count2 = 0
            vis = _visible_set()
            for rname, spectra_list in roi_spectra.items():
                new_list = []
                for ppm_ds, z_mean, b1_ut in spectra_list:
                    prog2.setValue(fit_count2); QApplication.processEvents()
                    if prog2.wasCanceled():
                        return
                    try:
                        new_res = _run_pipeline(
                            ppm_ds, z_mean,
                            satpwr_uT        = b1_ut,
                            R1               = p['R1'],
                            B0_MHz           = p['B0_MHz'],
                            peak_type        = new_pt,
                            ppm_exclude_MT   = (-new_mtex, new_mtex),
                            ppm_reinclude_MT = (-new_mtrein, new_mtrein),
                            ppm_include_CEST = (new_clo, new_chi),
                            pool_names       = self._get_active_pools(),
                            fit_mt           = p.get('fit_mt', True),
                        )
                        new_list.append(new_res)
                    except Exception as exc:
                        self._log(f"Refit failed ROI={rname!r} B1={b1_ut:.2f}: {exc}")
                        new_list.append(None)
                    fit_count2 += 1
                roi_results[rname] = new_list
                _rebuild_b1(rname, new_list, vis)
                _populate_quesp(rname)          # coeffs changed → refresh QUESP
            prog2.setValue(total_fits); prog2.close()

        btn_refit.clicked.connect(_do_refit)

        # Bottom row — just Close (Save / QUESP-pools / Save-1/Z buttons removed;
        # exporting is now the single "Export…" button at the top of the dialog).
        bot_row = QHBoxLayout()
        bot_row.addStretch()
        btn_close = _QPB("Close")
        btn_close.clicked.connect(dlg.hide)   # hide (keep cached) instead of destroy
        bot_row.addWidget(btn_close)
        vl.addLayout(bot_row)

        # Cache the built dialog so reopening with unchanged data is instant
        self._roi_spectra_dlg = dlg
        self._roi_spectra_key = _key
        dlg.show()

    # ── ROI Stats table ───────────────────────────────────────────────────────

    def _show_roi_table(self):
        from my_gui.roi_table_dialog import show_roi_table
        rois    = getattr(self, '_last_rois', [])
        all_res = getattr(self, '_all_results', [])
        maps    = []
        for r in all_res:
            arex = r.get('arex_map')
            if arex is not None:
                ep  = r.get('eval_ppm', self.spin_eval.value())
                b1  = r.get('b1_ut', self.spin_b1.value())
                maps.append((f"AREX ±{ep:.2f} ppm  B1={b1:.2f} µT (s⁻¹)", np.array(arex)))
        if not maps:
            # Backward compat: try legacy flat results dict
            results = self._results
            arex = results.get('arex_map')
            if arex is not None:
                ep = results.get('eval_ppm', self.spin_eval.value())
                maps.append((f"AREX ±{ep:.2f} ppm (s⁻¹)", np.array(arex)))
        # Per-voxel QUESP maps (fs / ksw / conc / R²)
        for entry in getattr(self, '_quesp_display', []):
            lbl = entry['label'].replace('QUESP ', '')
            maps.append((lbl, np.array(entry['data'])))
        img = self.canvas._img_data
        if img is not None and not maps:
            maps.append(("Current Map", img))
        show_roi_table(self, rois, maps, title="1/Z Spectroscopy — ROI Statistics")

    # ── ROI manager integration ───────────────────────────────────────────────

    def connect_roi_manager(self, roi_manager):
        roi_manager.connect_canvas(self.canvas)
        roi_manager.rois_changed.connect(self._update_roi_stats)
        self._last_rois = []

    def _update_roi_stats(self, rois: list):
        self._last_rois = list(rois)

    # ── Misc UI helpers ───────────────────────────────────────────────────────

    def _toggle_hide_rois(self, checked: bool):
        # Use the canvas's own toggle method (same as zspec_tab) — it handles
        # clearing old contour artists correctly via clf() + redraw.
        self.canvas.toggle_rois_visible()
        self.btn_hide_rois.setText("Show ROIs" if checked else "Hide ROIs")

    def _toggle_datacursor(self, on: bool):
        if not on:
            if self._dc_cid is not None:
                self.canvas._fig.canvas.mpl_disconnect(self._dc_cid)
                self._dc_cid = None
            if self._dc_annot is not None:
                try:
                    self._dc_annot.remove()
                except Exception:
                    pass
                self._dc_annot = None
            self.canvas.draw_idle()
            return

        ax = getattr(self.canvas, '_ax', None)
        if ax is None:
            return

        annot = ax.annotate(
            "", xy=(0, 0), xytext=(8, 8), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
            fontsize=8,
        )
        annot.set_visible(False)
        self._dc_annot = annot

        def _hover(event):
            if event.inaxes != ax:
                return
            img = self.canvas._img_data
            if img is None:
                return
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            H, W  = img.shape[:2]
            if 0 <= x < W and 0 <= y < H:
                val   = img[y, x]
                annot.xy = (event.xdata, event.ydata)
                annot.set_text(f"({x},{y}) = {val:.4g}")
                annot.set_visible(True)
                self.canvas.draw_idle()

        self._dc_cid = self.canvas._fig.canvas.mpl_connect('motion_notify_event', _hover)

    def _export(self):
        if not self._results:
            QMessageBox.information(self, "No results", "Run the analysis first.")
            return
        path, filt = QFileDialog.getSaveFileName(
            self, "Export map", "quant_map.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif);;PDF (*.pdf);;"
            "SVG (*.svg);;NumPy (*.npz);;MATLAB (*.mat)"
        )
        if not path:
            return
        low = path.lower()
        if low.endswith('.npz'):
            save_dict = {k: v for k, v in self._results.items()
                         if isinstance(v, np.ndarray)}
            np.savez(path, **save_dict)
        elif low.endswith('.mat'):
            from scipy.io import savemat
            savemat(path, {"arex_map": self.canvas._img_data})
        else:
            # Images at 300 dpi via the shared helper (handles JPEG facecolor +
            # near-lossless quality centrally).
            from my_gui.fig_export import save_figure
            save_figure(self.canvas._fig, path, dpi=300)
        self._log(f"Exported to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1/Z CEST fitting options dialog
# ─────────────────────────────────────────────────────────────────────────────

class _InvZSpecOptionsDialog(QDialog):
    """Dialog for configuring 1/Z CEST fitting options including pool selection."""

    def __init__(self, selected_pools: list, cest_lo: float, cest_hi: float,
                 peak_type: str, mt_excl: float, mt_rein: float,
                 fit_mt: bool = True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CEST Fitting Options")
        self.setMinimumWidth(480)

        vl = QVBoxLayout(self)

        # Pool selection
        self._pool_widget = PoolSelectionWidget(initial_selection=selected_pools, parent=self)
        vl.addWidget(self._pool_widget)

        form = QFormLayout()

        # MT fitting toggle — when off, the MT background is flat (Z_MT = 1) and
        # the MT-window controls below are disabled.
        self._chk_mt_fit = QCheckBox("Fit MT background (super-Lorentzian)")
        self._chk_mt_fit.setChecked(bool(fit_mt))
        self._chk_mt_fit.setToolTip(
            "When checked, the semisolid MT pool is fitted from the far-offset\n"
            "wings and subtracted before the CEST peak fit.  Uncheck to treat MT\n"
            "as flat (Z_MT = 1).")
        form.addRow("MT fitting:", self._chk_mt_fit)

        # Peak type
        self._combo_peak = QComboBox()
        self._combo_peak.addItems(["Lorentzian", "Pseudo-Voigt"])
        self._combo_peak.setCurrentText(peak_type)
        form.addRow("Peak type:", self._combo_peak)

        # CEST fit range
        cest_range_row = QHBoxLayout()
        self._spin_cest_lo = QDoubleSpinBox()
        self._spin_cest_lo.setRange(-20.0, 0.0)
        self._spin_cest_lo.setValue(cest_lo)
        self._spin_cest_lo.setSuffix(" ppm")
        self._spin_cest_lo.setSingleStep(0.5)
        cest_range_row.addWidget(self._spin_cest_lo)
        cest_range_row.addWidget(QLabel("to"))
        self._spin_cest_hi = QDoubleSpinBox()
        self._spin_cest_hi.setRange(0.0, 20.0)
        self._spin_cest_hi.setValue(cest_hi)
        self._spin_cest_hi.setSuffix(" ppm")
        self._spin_cest_hi.setSingleStep(0.5)
        cest_range_row.addWidget(self._spin_cest_hi)
        cest_range_row.addStretch()
        form.addRow("CEST fit range:", cest_range_row)

        # MT exclusion
        self._spin_mt_excl = QDoubleSpinBox()
        self._spin_mt_excl.setRange(1.0, 20.0)
        self._spin_mt_excl.setValue(mt_excl)
        self._spin_mt_excl.setSuffix(" ppm")
        self._spin_mt_excl.setSingleStep(0.5)
        self._spin_mt_excl.setToolTip(
            "Points with |ppm| > this are used to fit the MT pool."
        )
        form.addRow("MT:", self._spin_mt_excl)

        # MT water re-inclusion
        self._spin_mt_rein = QDoubleSpinBox()
        self._spin_mt_rein.setRange(0.0, 3.0)
        self._spin_mt_rein.setValue(mt_rein)
        self._spin_mt_rein.setSuffix(" ppm")
        self._spin_mt_rein.setSingleStep(0.1)
        self._spin_mt_rein.setToolTip(
            "Re-include the water peak within ±this offset for MT fitting."
        )
        form.addRow("Water:", self._spin_mt_rein)

        # Hide the MT-window rows entirely when MT fitting is disabled — with
        # MT off there is no MT fit, so the exclude / water-reinclude windows are
        # meaningless (matches the reference driver where MT.Z is just ones).
        def _sync_mt_enabled(on):
            try:
                form.setRowVisible(self._spin_mt_excl, on)
                form.setRowVisible(self._spin_mt_rein, on)
            except Exception:
                # Fallback for older Qt without QFormLayout.setRowVisible
                for _w in (self._spin_mt_excl, self._spin_mt_rein,
                           form.labelForField(self._spin_mt_excl),
                           form.labelForField(self._spin_mt_rein)):
                    if _w is not None:
                        _w.setVisible(on)
        self._chk_mt_fit.toggled.connect(_sync_mt_enabled)
        _sync_mt_enabled(self._chk_mt_fit.isChecked())

        vl.addLayout(form)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        vl.addWidget(btns)

    def get_settings(self) -> dict:
        return {
            "pools":     self._pool_widget.selected_pools(),
            "peak_type": self._combo_peak.currentText(),
            "cest_lo":   self._spin_cest_lo.value(),
            "cest_hi":   self._spin_cest_hi.value(),
            "mt_excl":   self._spin_mt_excl.value(),
            "mt_rein":   self._spin_mt_rein.value(),
            "fit_mt":    bool(self._chk_mt_fit.isChecked()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 1/Z scan-parameters dialog
# ─────────────────────────────────────────────────────────────────────────────

class _InvZScanParamsDialog(QDialog):
    """Dialog for editing the 1/Z scan parameters (B1, B0, eval offset, R1)."""

    def __init__(self, b1: float, b0: float, eval_ppm: float, r1: float,
                 tsat: float = 2.0, trec: float = 8.0,
                 r1_getter=None, b1_list=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scan Parameters")
        self.setMinimumWidth(360)
        self._r1_getter = r1_getter
        self._b1_value  = float(b1)     # preserved for get_settings()

        form = QFormLayout(self)

        # B1 is auto-detected per loaded scan and the whole set is fit together
        # (one fit per power), so a single editable B1 is meaningless once ≥2
        # scans are loaded — show the per-scan powers read-only instead.
        _b1_list = list(b1_list or [])
        _multi = len({round(float(x), 2) for x in _b1_list}) >= 2
        if _multi:
            _txt = ",  ".join(f"{v:.2f}" for v in _b1_list)
            _lbl = QLabel(f"per scan (auto):  {_txt} µT")
            _lbl.setStyleSheet("color:#4caf50;")
            _lbl.setToolTip("Each loaded scan uses its own auto-detected B1 power;\n"
                            "all powers are fit together for the QUESP fs/ksw maps.")
            form.addRow("Sat. power (B1):", _lbl)
            self._spin_b1 = None
        else:
            self._spin_b1 = QDoubleSpinBox()
            self._spin_b1.setRange(0.01, 50.0); self._spin_b1.setValue(b1)
            self._spin_b1.setSuffix(" µT"); self._spin_b1.setDecimals(2)
            self._spin_b1.setToolTip("Saturation B1 power used in the CEST scan")
            form.addRow("Sat. power (B1):", self._spin_b1)

        self._spin_b0 = QDoubleSpinBox()
        self._spin_b0.setRange(50.0, 1500.0); self._spin_b0.setValue(b0)
        self._spin_b0.setSuffix(" MHz"); self._spin_b0.setDecimals(1)
        self._spin_b0.setToolTip("Proton Larmor frequency (9.4 T → 400 MHz, 7 T → 298 MHz)")
        form.addRow("B0 field:", self._spin_b0)

        self._spin_eval = QDoubleSpinBox()
        self._spin_eval.setRange(0.0, 15.0); self._spin_eval.setValue(eval_ppm)
        self._spin_eval.setSuffix(" ppm"); self._spin_eval.setSingleStep(0.5)
        self._spin_eval.setToolTip("Ppm offset for the AREX map")
        form.addRow("Eval offset (map):", self._spin_eval)

        r1_row = QHBoxLayout()
        self._spin_r1 = QDoubleSpinBox()
        self._spin_r1.setRange(0.01, 20.0); self._spin_r1.setValue(r1)
        self._spin_r1.setSuffix(" s⁻¹"); self._spin_r1.setSingleStep(0.05)
        self._spin_r1.setToolTip("R1 = 1/T1 [s⁻¹].  Get from T1/T2/B1 tab → T1 map.")
        r1_row.addWidget(self._spin_r1)
        btn_r1 = QPushButton("From T1 map")
        btn_r1.setToolTip("Use mean R1 from the T1 map in the T1/T2/B1 tab")
        btn_r1.clicked.connect(self._from_t1_map)
        r1_row.addWidget(btn_r1)
        form.addRow("R1 (water):", r1_row)

        # ── Saturation timing (QUESP only) ─────────────────────────────────────
        # Not used by the 1/Z peak fit; required by QUESP:  tsat sets the
        # saturation-time exponential, trec sets Zi = 1 − exp(−R1·trec).
        _hdr = QLabel("<b>Saturation timing</b>  <span style='color:#888;'>(QUESP only)</span>")
        form.addRow(_hdr)

        self._spin_tsat = QDoubleSpinBox()
        self._spin_tsat.setRange(0.0, 60.0); self._spin_tsat.setValue(tsat)
        self._spin_tsat.setSuffix(" s"); self._spin_tsat.setDecimals(3)
        self._spin_tsat.setToolTip("Saturation pulse length (tsat / tp)")
        form.addRow("Sat. time (tsat):", self._spin_tsat)

        self._spin_trec = QDoubleSpinBox()
        self._spin_trec.setRange(0.0, 120.0); self._spin_trec.setValue(trec)
        self._spin_trec.setSuffix(" s"); self._spin_trec.setDecimals(3)
        self._spin_trec.setToolTip("Recovery delay before saturation (trec / rd)")
        form.addRow("Recovery (trec):", self._spin_trec)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def _from_t1_map(self):
        fn = self._r1_getter
        if fn is None:
            QMessageBox.information(
                self, "T1 map",
                "No T1/T2/B1 tab connected.\nEnter R1 = 1/T1 manually."
            )
            return
        r1 = fn()
        if r1 is not None and r1 > 0:
            self._spin_r1.setValue(float(r1))

    def get_settings(self) -> dict:
        return {
            "b1":       (self._spin_b1.value() if self._spin_b1 is not None
                         else self._b1_value),
            "b0":       self._spin_b0.value(),
            "eval_ppm": self._spin_eval.value(),
            "r1":       self._spin_r1.value(),
            "tsat":     self._spin_tsat.value(),
            "trec":     self._spin_trec.value(),
        }
