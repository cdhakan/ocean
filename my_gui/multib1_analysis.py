"""
multib1_analysis.py — shared MTR-asymmetry helpers for OCEAN.
================================================================
One place for the MTR-asymmetry maths used across the app:

  • Quantitative-Z ("Z-spectroscopy") ROI-spectra dialog — single-B1 MTRasym
    sub-tab shown beside the pool-fit tabs (Pseudo-Voigt / MPLF / Gaussian /
    PLOF / DROF / 1-Z).
  • QUESP and Inverse-Z tabs — multi-B1 MTRasym (one curve per B1 power) plus a
    per-B1 pool-fit (Pseudo-Voigt / MPLF) amplitude-vs-B1 view.

MTR asymmetry is defined as   MTRasym(Δω) = Z(−Δω) − Z(+Δω)   plotted on the
positive-offset axis, matching the CEST-master / literature convention.
"""
from __future__ import annotations

import numpy as np

# Proton gyromagnetic ratio → ω1 (rad/s) for a given B1 (µT).  Matches the QUESP
# tab constant so B1↔rad/s labels agree across the app.
GAMMA_RAD_PER_UT = 267.522    # rad s⁻¹ µT⁻¹


def b1_to_rad_per_s(b1_uT: float) -> float:
    """ω1 = γ·B1 in rad/s for a B1 peak amplitude in µT."""
    return GAMMA_RAD_PER_UT * float(b1_uT)


def b1_legend_label(b1_uT: float) -> str:
    """Legend text like '2.0 µT (534 rad/s)' — matches the reference figure."""
    return f"{b1_uT:g} µT ({b1_to_rad_per_s(b1_uT):.0f} rad/s)"


# ─────────────────────────────────────────────────────────────────────────────
# Core MTR-asymmetry computation
# ─────────────────────────────────────────────────────────────────────────────
def mtr_asym(ppm: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos_ppm, asym) with asym = Z(−Δω) − Z(+Δω) on the positive axis.

    `ppm`/`z` are a 1-D Z-spectrum (any offset order).  The negative side is
    interpolated onto the acquired positive offsets so the two branches are
    matched even for slightly asymmetric sampling.
    """
    ppm = np.asarray(ppm, float).ravel()
    z   = np.asarray(z,   float).ravel()
    o   = np.argsort(ppm)
    ppm_s, z_s = ppm[o], z[o]
    pos = ppm_s[ppm_s > 1e-6]
    if pos.size == 0:
        return np.array([]), np.array([])
    z_pos = np.interp(pos,  ppm_s, z_s)
    z_neg = np.interp(-pos, ppm_s, z_s)
    return pos, (z_neg - z_pos)


def roi_zspectrum(zstack: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean Z-spectrum over a boolean ROI mask.

    `zstack` is (H, W, n_off) already normalised by M0; `mask` is (H, W) bool.
    """
    zstack = np.asarray(zstack, float)
    m = np.asarray(mask, bool)
    if m.shape != zstack.shape[:2] or not m.any():
        return np.full(zstack.shape[-1], np.nan)
    return np.nanmean(zstack.reshape(-1, zstack.shape[-1])[m.ravel()], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset reader — one B1-power scan → (ppm, normalised Z-stack)
# ─────────────────────────────────────────────────────────────────────────────
def read_dataset_zstack(ds: dict, slice_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Read one multi-B1 dataset dict → (ppm (n,), Zstack (H, W, n) / M0).

    `ds` = {'path': str, 'b1_ut': float, 'pv360': bool}.  Handles Bruker
    folders and .mat/.npz exports (image + w_offsetPPM + M0image), mirroring
    the QUESP tab's own loader so results are consistent.
    """
    path  = ds['path']
    pv360 = bool(ds.get('pv360', False))

    if path.endswith('.npz') or path.endswith('.mat'):
        if path.endswith('.npz'):
            d = dict(np.load(path, allow_pickle=True))
        else:
            import scipy.io as sio
            d = sio.loadmat(path)
        if 'image' not in d or 'w_offsetPPM' not in d:
            raise ValueError(f"{path}: missing 'image'/'w_offsetPPM'")
        img = np.array(d['image'], dtype=float)
        ppm = np.array(d['w_offsetPPM'], dtype=float).ravel()
        M0_key = next((k for k in ('M0image', 'M0', 'm0') if k in d), None)
        M0 = np.array(d[M0_key], dtype=float) if M0_key else None
        if img.ndim == 4:
            si = min(slice_idx, img.shape[2] - 1)
            imgs_2d = img[:, :, si, :]
            M0_2d = (M0[:, :, si] if (M0 is not None and M0.ndim >= 3)
                     else (M0 if M0 is not None else None))
        else:
            imgs_2d = img
            M0_2d = M0 if (M0 is not None and M0.ndim >= 2) else None
    else:
        from my_gui.bruker_reader import read_2dseq_cest
        img, M0, info = read_2dseq_cest(path, pv360=pv360)
        ppm = np.array(info['w_offsetPPM'], dtype=float).ravel()
        si  = min(slice_idx, img.shape[2] - 1)
        imgs_2d = img[:, :, si, :].astype(float)
        M0_2d = (M0[:, :, min(slice_idx, M0.shape[2] - 1)].astype(float)
                 if M0.ndim >= 3 else M0.astype(float))

    if M0_2d is None:
        # Fallback M0: highest-signal frame (true unsaturated reference).
        fs = imgs_2d.reshape(-1, imgs_2d.shape[-1]).mean(0)
        M0_2d = imgs_2d[:, :, int(np.argmax(fs))]
    M0_safe = M0_2d.astype(float) + 1e-9
    zstack = np.clip(imgs_2d / M0_safe[:, :, None], 0.0, 2.0)
    return ppm, zstack


# ─────────────────────────────────────────────────────────────────────────────
# Per-B1 pool-fit amplitudes (Pseudo-Voigt / MPLF) for the amplitude-vs-B1 view
# ─────────────────────────────────────────────────────────────────────────────
def pool_amplitudes(ppm: np.ndarray, z: np.ndarray, method: str,
                    pools: list[str] | None = None) -> dict[str, float]:
    """Fit one ROI Z-spectrum and return {pool: peak amplitude (1−Z units)}.

    `method` ∈ {'Pseudo-Voigt', 'Gaussian', 'Lorentzian', 'MPLF'}.  The peak
    amplitude is the max of each pool's fitted contribution curve — a robust,
    model-independent readout suitable for plotting vs B1.
    """
    ppm = np.asarray(ppm, float).ravel()
    z   = np.asarray(z,   float).ravel()
    good = np.isfinite(ppm) & np.isfinite(z)
    ppm, z = ppm[good], z[good]
    if ppm.size < 6:
        return {}
    try:
        if method.upper() == 'MPLF':
            from my_gui.zspec_processing import fit_zspec_mplf
            res = fit_zspec_mplf(ppm, z, pools=pools)
            curves = res.get('pools', {}) or {}
        else:
            from my_gui.zspec_processing import fit_zspec_single
            res = fit_zspec_single(ppm, z, pools=pools, peak_type=method)
            curves = res.get('indiv_peaks', {}) or {}
    except Exception:
        return {}
    out = {}
    for name, curve in curves.items():
        c = np.asarray(curve, float)
        if c.size:
            out[name] = float(np.nanmax(c))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Multi-B1 MTR-asymmetry dialog (QUESP + Inverse-Z tabs)
# ─────────────────────────────────────────────────────────────────────────────
def _rois_for_plot(rois):
    """Drawn ROIs to plot: tubes first, skip outlines; fall back to any valid ROI."""
    out = []
    for r in (rois or []):
        nm = getattr(r, 'name', '')
        if nm in ('Phantom_outline', 'Brain_outline'):
            continue
        if getattr(r, 'mask', None) is not None and np.asarray(r.mask).any():
            out.append(r)
    return out


def _grid(n):
    """Rows/cols for n subplots (max 4 per row)."""
    import math
    c = min(4, max(1, n))
    return math.ceil(n / c), c


def show_multib1_mtrasym(parent, datasets, rois, slice_idx=0, title="Multi-B1 MTR Asymmetry"):
    """Open the multi-B1 MTRasym + per-B1 pool-fit dialog. Returns the dialog."""
    from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                 QComboBox, QPushButton, QTabWidget, QWidget,
                                 QApplication, QFileDialog, QMessageBox)
    from PyQt6.QtCore import Qt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    rois = _rois_for_plot(rois)
    if not datasets:
        QMessageBox.information(parent, title, "Add multi-B1 scans first.")
        return None
    if not rois:
        QMessageBox.information(parent, title,
                               "Draw at least one ROI (tube) in the ROI Manager first.")
        return None

    # ── Read every B1 dataset once; cache per-ROI Z-spectra ──────────────────
    b1s, ppms, zstacks = [], [], []
    prog = QApplication.instance()
    for ds in sorted(datasets, key=lambda d: float(d.get('b1_ut', 0))):
        try:
            ppm, zst = read_dataset_zstack(ds, slice_idx)
        except Exception as exc:
            print(f"[multi-B1] skip {ds.get('path')}: {exc}")
            continue
        b1s.append(float(ds['b1_ut'])); ppms.append(ppm); zstacks.append(zst)
        if prog:
            prog.processEvents()
    if not b1s:
        QMessageBox.warning(parent, title, "None of the scans could be read.")
        return None

    import matplotlib.cm as cm
    colors = cm.get_cmap('turbo', max(len(b1s), 2))

    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(1180, 760)
    root = QVBoxLayout(dlg)

    tabs = QTabWidget()
    root.addWidget(tabs, 1)

    # ── Tab 1: MTR asymmetry (one subplot per ROI, one curve per B1) ─────────
    w_asym = QWidget(); la = QVBoxLayout(w_asym)
    fig_a = Figure(figsize=(11, 6.5)); fig_a.set_facecolor('white')
    cv_a = FigureCanvas(fig_a); la.addWidget(cv_a, 1)
    nr, nc = _grid(len(rois))
    for i, roi in enumerate(rois):
        ax = fig_a.add_subplot(nr, nc, i + 1)
        for j, (b1, ppm, zst) in enumerate(zip(b1s, ppms, zstacks)):
            m = np.asarray(roi.mask, bool)
            if m.shape != zst.shape[:2]:
                continue
            Z = roi_zspectrum(zst, m)
            pos, asym = mtr_asym(ppm, Z)
            if pos.size:
                ax.plot(pos, asym, '-', lw=1.3, color=colors(j),
                        label=b1_legend_label(b1))
        ax.set_title(getattr(roi, 'name', f'ROI {i+1}'), fontsize=10, fontweight='bold')
        ax.set_xlabel('Δω (ppm)', fontsize=9); ax.set_ylabel('MTR asymmetry', fontsize=9)
        ax.invert_xaxis(); ax.axhline(0, color='0.7', lw=0.6)
        if i == 0:
            ax.legend(fontsize=6, loc='upper right')
    fig_a.tight_layout()
    tabs.addTab(w_asym, "MTR Asymmetry")

    # ── Tab 2: per-B1 pool-fit amplitude vs B1 ───────────────────────────────
    w_fit = QWidget(); lf = QVBoxLayout(w_fit)
    ctrl = QHBoxLayout()
    ctrl.addWidget(QLabel("Fit method:"))
    combo_m = QComboBox(); combo_m.addItems(["Pseudo-Voigt", "MPLF", "Gaussian", "Lorentzian"])
    ctrl.addWidget(combo_m)
    btn_fit = QPushButton("Fit per B1"); ctrl.addWidget(btn_fit)
    lbl_fit = QLabel(""); ctrl.addWidget(lbl_fit); ctrl.addStretch()
    lf.addLayout(ctrl)
    fig_f = Figure(figsize=(11, 6.0)); fig_f.set_facecolor('white')
    cv_f = FigureCanvas(fig_f); lf.addWidget(cv_f, 1)
    tabs.addTab(w_fit, "Pool Fit vs B1")

    def _run_fit():
        method = combo_m.currentText()
        lbl_fit.setText("Fitting…"); QApplication.instance().processEvents()
        fig_f.clear()
        nr2, nc2 = _grid(len(rois))
        pool_colors = cm.get_cmap('tab10', 10)
        for i, roi in enumerate(rois):
            ax = fig_f.add_subplot(nr2, nc2, i + 1)
            series = {}   # pool -> list of (b1, amp)
            for b1, ppm, zst in zip(b1s, ppms, zstacks):
                m = np.asarray(roi.mask, bool)
                if m.shape != zst.shape[:2]:
                    continue
                Z = roi_zspectrum(zst, m)
                amps = pool_amplitudes(ppm, Z, method)
                for p, a in amps.items():
                    series.setdefault(p, []).append((b1, a))
            for k, (p, pts) in enumerate(sorted(series.items())):
                pts = sorted(pts)
                xs = [x for x, _ in pts]; ys = [y for _, y in pts]
                ax.plot(xs, ys, 'o-', ms=4, lw=1.3, color=pool_colors(k % 10), label=p)
            ax.set_title(getattr(roi, 'name', f'ROI {i+1}'), fontsize=10, fontweight='bold')
            ax.set_xlabel('B1 (µT)', fontsize=9)
            ax.set_ylabel('Pool amplitude (1−Z)', fontsize=9)
            if i == 0 and series:
                ax.legend(fontsize=7, loc='best')
        fig_f.tight_layout(); cv_f.draw()
        lbl_fit.setText(f"Done ({method}).")

    btn_fit.clicked.connect(_run_fit)

    # ── Export ───────────────────────────────────────────────────────────────
    bottom = QHBoxLayout(); bottom.addStretch()
    btn_exp = QPushButton("Export figure…")
    def _export():
        fig = fig_a if tabs.currentIndex() == 0 else fig_f
        p, _ = QFileDialog.getSaveFileName(dlg, "Export figure", "",
                                           "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if p:
            fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    btn_exp.clicked.connect(_export)
    bottom.addWidget(btn_exp)
    btn_close = QPushButton("Close"); btn_close.clicked.connect(dlg.close)
    bottom.addWidget(btn_close)
    root.addLayout(bottom)

    cv_a.draw()
    dlg.show()
    return dlg
