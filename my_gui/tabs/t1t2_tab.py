from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QLineEdit, QGroupBox,
    QCheckBox, QDoubleSpinBox, QSpinBox, QComboBox,
    QProgressBar, QFileDialog, QSizePolicy, QScrollArea,
    QSlider, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from my_gui.roi_tools import ROICanvas, roi_union_mask
from my_gui.roi_manager import apply_analysis_mask
from my_gui.plot_custom_bar import PlotCustomBar, WindowLevelToolButton


# ---------------------------------------------------------------------------
# Module-level fitting helpers
# ---------------------------------------------------------------------------

def _fit_t1(signal, TRs, full=False, bruker=True):
    """
    T1 recovery fit:  S(TR) = M0*(1 - exp(-TR/T1)) + c

    full=False → returns T1 (ms). full=True → returns (M0, T1, c) so the
    plotted curve uses the *actual* fitted amplitude/offset (not data max−min).
    bruker=True keeps the original M0 upper bound (1.5×max); GE/Siemens uses
    3×max so the asymptote can exceed a not-fully-recovered TR_max sample.

    Bounds (all in ms):
        M0  : [0,   1.5 × max(signal)]
        T1  : [50,  8 000]     — 50 ms floor avoids divide-by-zero on noise
        c   : [0,   max(signal)] — offset ≥ 0 for magnitude images

    Initial guess  (matches MATLAB t1fitting_VTR_new.m):
        M0_0 = max − min,  T1_0 = 1 000 ms,  c_0 = min

    Returns T1 in ms, or 0.0 on failure.
    """
    from scipy.optimize import curve_fit

    def model(TR, M0, T1, c):
        return M0 * (1.0 - np.exp(-TR / T1)) + c

    signal = np.asarray(signal, dtype=float)
    mx = float(signal.max())
    mn = float(signal.min())
    if mx <= 0:
        return (0.0, 0.0, 0.0) if full else 0.0

    # GE/Siemens: allow M0 up to 3× span so the asymptote can exceed the
    # longest-TR sample (data not fully recovered when TR_max ≈ T1).
    # Bruker: keep the original 1.5× bound (unchanged behaviour).
    _m0_ub = (1.5 if bruker else 3.0) * mx
    p0     = [mx - mn, 1000.0, mn]
    bounds = ([0.0, 50.0, 0.0], [_m0_ub, 8000.0, mx])

    try:
        popt, _ = curve_fit(
            model, TRs, signal, p0=p0, bounds=bounds,
            ftol=1e-8, xtol=1e-8, max_nfev=10000,
        )
        M0, T1, c = float(popt[0]), float(np.clip(popt[1], 0.0, 8000.0)), float(popt[2])
        return (M0, T1, c) if full else T1
    except Exception:
        return (0.0, 0.0, 0.0) if full else 0.0


def _fit_t2(signal, TEs, full=False, bruker=True):
    """
    Mono-exponential T2 decay:  S(TE) = M0 * exp(-TE/T2)

    full=False → returns T2 (ms). full=True → returns (M0, T2).
    bruker=True keeps the original two-point seed + 1.5×max bound; GE/Siemens
    uses a weighted log-linear seed + 3×max bound (more robust for few echoes).

    Matches MATLAB t2fitting_new.m exactly:
        - c is constrained to 0 (lb = ub = 0 in MATLAB)
        - 2-parameter fit: [M0, T2]

    Bounds (all in ms):
        M0  : [0,   1.5 × max(signal)]
        T2  : [1,   3 000]

    Initial guess (data-adaptive):
        M0_0 = max(signal)
        T2_0 = estimated from signal ratio at two echo points
               Falls back to TE_mid / ln(2) ≈ half-life estimate.
               Clamped to [5, 2000] ms to stay inside bounds.

    Returns T2 in ms, or 0.0 on failure.
    """
    from scipy.optimize import least_squares

    signal = np.asarray(signal, dtype=float)
    TEs    = np.asarray(TEs,    dtype=float)
    mx = float(signal.max())
    if mx <= 0:
        return (0.0, 0.0) if full else 0.0

    if bruker:
        # ── Original Bruker behaviour (unchanged) ───────────────────────
        try:
            mid = max(1, len(TEs) // 2)
            ratio = signal[mid] / (signal[0] + 1e-9)
            ratio = np.clip(ratio, 1e-6, 1.0 - 1e-6)
            t2_init = float(-(TEs[mid] - TEs[0]) / np.log(ratio))
        except Exception:
            t2_init = float(TEs[len(TEs) // 2] / np.log(2.0) + 1e-9)
        t2_init = float(np.clip(t2_init, 5.0, 2000.0))
        m0_init = mx
        _m0_ub  = 1.5 * mx
    else:
        # ── GE/Siemens: weighted log-linear seed, wider M0 bound ────────
        # ln(S) = ln(M0) − TE/T2 → slope = −1/T2 (weighted by S → high-SNR).
        try:
            pos = signal > 0
            # Need ≥2 points AND at least two *distinct* TE values, else the
            # log-linear seed is degenerate (poorly-conditioned polyfit → warning).
            # A constant DICOM TE (Siemens CEST-MRF) hits this until the real
            # prep-time schedule is entered — fall back to a plain seed instead.
            if pos.sum() >= 2 and np.unique(TEs[pos]).size >= 2:
                import warnings as _w
                with _w.catch_warnings():
                    _w.simplefilter("ignore")          # silence RankWarning
                    coef = np.polyfit(TEs[pos], np.log(signal[pos]), 1, w=signal[pos])
                slope = coef[0]
                t2_init = float(-1.0 / slope) if slope < 0 else 2000.0
                m0_init = float(np.exp(coef[1]))
            else:
                t2_init, m0_init = 500.0, mx
        except Exception:
            t2_init, m0_init = 500.0, mx
        t2_init = float(np.clip(t2_init, 5.0, 2900.0))
        m0_init = float(np.clip(m0_init, 1e-6, 3.0 * mx))
        _m0_ub  = 3.0 * mx

    def residuals(p):
        M0, T2 = p
        return M0 * np.exp(-TEs / T2) - signal

    x0     = np.array([m0_init, t2_init])
    bounds = ([0.0, 1.0], [_m0_ub, 3000.0])

    try:
        res = least_squares(residuals, x0, bounds=bounds,
                            ftol=1e-8, xtol=1e-8, max_nfev=10000)
        M0_val, T2_val = float(res.x[0]), float(np.clip(res.x[1], 0.0, 3000.0))
        return (M0_val, T2_val) if full else T2_val
    except Exception:
        return (0.0, 0.0) if full else 0.0


def _t2_reliability_note(x_label: str, x_vals, t2_val: float) -> str:
    """Flag T2 fits that extrapolate well beyond the acquired echo range.

    When TE_max << T2 the signal barely decays over the measured echoes, so the
    mono-exponential fit is extrapolating and the T2 value is ill-conditioned
    (small noise → large T2 swings). Returns a short warning string, or ''.
    """
    if "TE" not in x_label.upper():
        return ""
    te_max = float(np.max(x_vals))
    if t2_val <= 0:
        return ""
    # Fraction of magnetisation actually decayed by the last echo
    decayed = 1.0 - np.exp(-te_max / t2_val)
    if te_max < 0.5 * t2_val or decayed < 0.20:
        return (f"⚠ TE_max={te_max:.0f}ms ≪ T2 → extrapolated,\n"
                f"  only {decayed*100:.0f}% decay measured (low confidence)")
    return ""


def calc_b1_map(img_fa1, img_fa2, fa_nominal_deg):
    """B1 map (%) from two flip-angle RARE images.

    Standard double-angle method: α = arccos(|S_2α / (2·S_α)|).  The absolute
    value (rather than clipping the ratio to −1..1) folds noise-driven negative
    ratios into a valid arccos domain and matches the Equations-tab formula.
    """
    eps = 1e-9
    ratio = np.abs(img_fa2.astype(float) / (2.0 * img_fa1.astype(float) + eps))
    ratio = np.clip(ratio, 0.0, 1.0 - eps)   # |ratio| ≥ 0; only clamp the top for arccos
    b1_rad = np.arccos(ratio)
    fa_rad = fa_nominal_deg * np.pi / 180.0
    return (b1_rad / fa_rad) * 100.0


def calc_b1_ratio(img_fa1, img_fa2):
    """Raw double-angle B1 map = S_2α / (2·S_α) = cos(α_actual).

    No arccos / no percent conversion — the parametric image shown in the panel
    the user selected.  ~0.5 for a perfect nominal α/2α pair; higher B1 (larger
    actual flip) reads *lower*.  Kept as-is (no masking) apart from a divide
    guard; the global analysis mask restricts it to the phantom for display.
    """
    eps = 1e-9
    return img_fa2.astype(float) / (2.0 * img_fa1.astype(float) + eps)


# ---------------------------------------------------------------------------
# QThread workers
# ---------------------------------------------------------------------------

class T1FitWorker(QThread):
    finished = pyqtSignal(object)  # ndarray
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, img_2d_ntr, trs_ms, bruker=True):
        # img_2d_ntr: (Y, X, nTR)
        # trs_ms: (nTR,) TR values in ms
        super().__init__()
        self._img = img_2d_ntr
        self._trs = trs_ms
        self._bruker = bruker
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            Y, X, nTR = self._img.shape
            t1_map = np.zeros((Y, X))
            total = Y * X
            done = 0
            for i in range(Y):
                if self._stop:
                    break
                for j in range(X):
                    sig = self._img[i, j, :].astype(float)
                    if sig.max() > 0:
                        t1_map[i, j] = _fit_t1(sig, self._trs, bruker=self._bruker)
                    done += 1
                    if done % max(1, total // 100) == 0:
                        self.progress.emit(int(100 * done / total))
            self.finished.emit(t1_map)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class T2FitWorker(QThread):
    finished = pyqtSignal(object)  # ndarray
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, img_2d_nte, tes_ms, bruker=True):
        # img_2d_nte: (Y, X, nTE)
        # tes_ms: (nTE,) TE values in ms
        super().__init__()
        self._img = img_2d_nte
        self._tes = tes_ms
        self._bruker = bruker
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            Y, X, nTE = self._img.shape
            t2_map = np.zeros((Y, X))
            total = Y * X
            done = 0
            for i in range(Y):
                if self._stop:
                    break
                for j in range(X):
                    sig = self._img[i, j, :].astype(float)
                    if sig.max() > 0:
                        t2_map[i, j] = _fit_t2(sig, self._tes, bruker=self._bruker)
                    done += 1
                    if done % max(1, total // 100) == 0:
                        self.progress.emit(int(100 * done / total))
            self.finished.emit(t2_map)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class WasabiWorker(QThread):
    """Per-voxel WASABI B0/B1 fit over a masked 2-D slice.

    Emits finished(dict) with 2-D maps: b1_rel (%), db0 (ppm), b1_ut (µT),
    c, af, rmse."""
    finished = pyqtSignal(object)   # dict of 2-D maps
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, z3d, ppm, freq_mhz, tp_s, b1_nom, mask, model="4param"):
        super().__init__()
        self._z = np.asarray(z3d, dtype=float)     # (Y, X, n_off)  normalised Z
        self._ppm = np.asarray(ppm, dtype=float).ravel()
        self._freq = float(freq_mhz)
        self._tp = float(tp_s)
        self._b1n = float(b1_nom)
        self._mask = np.asarray(mask, dtype=bool)
        self._model = model
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            from my_gui.wasabi_fit import fit_wasabi_voxel, build_wasabi_lookup
            Y, X, _ = self._z.shape
            b1 = np.full((Y, X), np.nan)
            db0 = np.full((Y, X), np.nan)
            cmap = np.full((Y, X), np.nan)
            afmap = np.full((Y, X), np.nan)
            rmse = np.full((Y, X), np.nan)
            lookup = build_wasabi_lookup(self._ppm, self._freq, self._tp,
                                         self._b1n, self._model)
            idx = np.argwhere(self._mask)
            total = max(1, len(idx))
            for n, (i, j) in enumerate(idx):
                if self._stop:
                    break
                B1v, dB0v, cv, afv, rm = fit_wasabi_voxel(
                    self._z[i, j, :], self._ppm, self._freq, self._tp,
                    self._b1n, model=self._model, lookup=lookup)
                b1[i, j], db0[i, j] = B1v, dB0v
                cmap[i, j], afmap[i, j], rmse[i, j] = cv, afv, rm
                if n % max(1, total // 100) == 0:
                    self.progress.emit(int(100 * n / total))
            b1_rel = b1 / self._b1n * 100.0 if self._b1n > 0 else b1
            self.finished.emit(dict(b1_rel=b1_rel, db0=db0, b1_ut=b1,
                                    c=cmap, af=afmap, rmse=rmse,
                                    stopped=self._stop))
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Main tab widget
# ---------------------------------------------------------------------------

class T1T2Tab(QWidget):
    """Tab for T1/T2/B1 parametric mapping from Bruker MRI data."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- data storage ----
        self._t1_img: np.ndarray | None = None       # (Y, X, slices, nTR)
        self._t2_img: np.ndarray | None = None       # (Y, X, slices, nTE)
        self._b1_fa1_img: np.ndarray | None = None
        self._b1_fa2_img: np.ndarray | None = None
        self._t1_trs: np.ndarray | None = None       # ms
        self._t2_tes: np.ndarray | None = None       # ms
        self._t1_map: np.ndarray | None = None       # (Y, X)
        self._t2_map: np.ndarray | None = None       # (Y, X)
        self._b1_map: np.ndarray | None = None       # (Y, X)
        self._slice_idx: int = 0
        self._t1_dir: str = ""
        self._t2_dir: str = ""
        self._b1_fa1_dir: str = ""
        self._b1_fa2_dir: str = ""
        # Pre-computed / Bloch-Siegert B1 sources (single images)
        self._b1_pre_path: str = ""
        self._b1_bspos_path: str = ""
        self._b1_bsneg_path: str = ""
        self._b1_pre_img: np.ndarray | None = None
        self._b1_bspos_img: np.ndarray | None = None
        self._b1_bsneg_img: np.ndarray | None = None
        self._worker = None
        self._dc_annot = None
        self._dc_cid: int | None = None

        # "ROIs + Bkg" custom background image + scan-paths getter (set by app.py)
        self._roi_bg_img: np.ndarray | None = None
        self._scan_paths_getter = None

        # MTRasym overlay state
        self._mtr_overlay_file: np.ndarray | None = None   # map loaded from file
        self._mtr_source_getter = None                     # callback → CEST MRI tab map
        self._cest_image_getter = None                     # callback → CEST MRI tab image

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        # ---- left panel (scrollable) ----
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(6)
        left_layout.setContentsMargins(6, 6, 6, 6)

        left_scroll = QScrollArea()
        left_scroll.setWidget(left_widget)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMaximumWidth(390)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_layout.addWidget(self._build_t1_group())
        left_layout.addWidget(self._build_t2_group())
        left_layout.addWidget(self._build_b1_group())
        left_layout.addWidget(self._build_wasabi_scan_group())
        left_layout.addWidget(self._build_processing_group())

        # ── Global Cancel — stops a running T1 / T2 / B1 / WASABI operation ────
        self.btn_cancel_all = QPushButton("Cancel")
        self.btn_cancel_all.setToolTip(
            "Cancel a running T1, T2, B1 or WASABI operation.")
        self.btn_cancel_all.setStyleSheet(
            "QPushButton { background:#8a2f2f; color:white; border-radius:5px; "
            "font-weight:bold; padding:6px; }"
            "QPushButton:hover { background:#a83a3a; }"
            "QPushButton:disabled { background:#333; color:#777; }")
        self.btn_cancel_all.clicked.connect(self._cancel_all_scans)
        # Match the width of a "Run …" button (≈ half the panel), same position.
        _cancel_row = QHBoxLayout()
        _cancel_row.addStretch(1)
        _cancel_row.addWidget(self.btn_cancel_all, 2)
        _cancel_row.addStretch(1)
        left_layout.addLayout(_cancel_row)

        left_layout.addStretch()

        # ---- right panel ----
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(4)
        right_layout.setContentsMargins(6, 6, 6, 6)

        # display selector row
        disp_row = QHBoxLayout()
        disp_row.addWidget(QLabel("Display:"))
        self.combo_display = QComboBox()
        self.combo_display.addItems([
            "Reference Image",
            "T1 Images",
            "T2 Images",
            "α₁ Images (B1 FA1)",
            "α₂ Images (B1 FA2)",
            "T1 Map (ms)",
            "T2 Map (ms)",
            "B1 Map",
            "ΔB0 Map (ppm)",
            "MTRasym Overlay → T1 image",
            "MTRasym Overlay → T2 image",
            "MTRasym Overlay → CEST image",
        ])
        self.combo_display.currentIndexChanged.connect(self._refresh_display)
        disp_row.addWidget(self.combo_display, stretch=1)
        self.btn_hide_rois = QPushButton("Hide ROIs")
        self.btn_hide_rois.setCheckable(True)
        self.btn_hide_rois.setToolTip("Toggle ROI overlay visibility")
        self.btn_hide_rois.clicked.connect(self._toggle_hide_rois)
        # Window/Level (brightness–contrast) drag tool — OsiriX-style.
        self.btn_contrast = WindowLevelToolButton()
        self.btn_contrast.toggled.connect(
            lambda checked: self.canvas.set_wl_active(
                checked, on_change=lambda a, b: self.plot_bar.set_clim(a, b)))
        disp_row.addWidget(self.btn_contrast)
        disp_row.addWidget(self.btn_hide_rois)
        # ROIs-only mask: show the parametric map only inside the drawn ROIs
        self.chk_roi_only = QCheckBox("ROIs only")
        self.chk_roi_only.toggled.connect(self._refresh_display)
        disp_row.addWidget(self.chk_roi_only)
        # ROIs + Bg: colored map only inside the ROIs, over the 1st raw frame (gray)
        self.chk_roi_bg = QCheckBox("ROIs + Bkg")
        self.chk_roi_bg.setToolTip(
            "Show the colored map only inside the ROIs, over the 1st raw image "
            "as a gray background."
        )
        self.chk_roi_bg.toggled.connect(self._refresh_display)
        disp_row.addWidget(self.chk_roi_bg)
        # Pick a custom grayscale background for the "ROIs + Bkg" overlay
        self.btn_roi_bg = QPushButton("Bkg…")
        self.btn_roi_bg.setToolTip(
            "Pick the grayscale background image (from the Scan Directory) "
            "for the 'ROIs + Bkg' overlay."
        )
        self.btn_roi_bg.clicked.connect(self._pick_roi_bg)
        disp_row.addWidget(self.btn_roi_bg)
        btn_export = QPushButton("Export figure…")
        btn_export.clicked.connect(self._export_figure)
        disp_row.addWidget(btn_export)
        right_layout.addLayout(disp_row)

        # ── MTRasym overlay controls (hidden until an overlay view is chosen) ──
        self._overlay_frame = QFrame()
        ov_row = QHBoxLayout(self._overlay_frame)
        ov_row.setContentsMargins(0, 0, 0, 0)
        ov_row.addWidget(QLabel("MTRasym:"))
        self.combo_mtr_source = QComboBox()
        self.combo_mtr_source.addItems(["From CEST MRI tab", "From file…"])
        self.combo_mtr_source.setToolTip(
            "From CEST MRI tab — use the MTR-asymmetry map computed in the "
            "CEST MRI tab.\nFrom file… — load a saved map (.npy/.mat/.csv/.txt)."
        )
        self.combo_mtr_source.currentIndexChanged.connect(self._on_mtr_source_changed)
        ov_row.addWidget(self.combo_mtr_source)
        self.btn_load_mtr = QPushButton("Load file…")
        self.btn_load_mtr.setToolTip("Load an MTR-asymmetry map from .npy / .mat / .csv / .txt")
        self.btn_load_mtr.clicked.connect(self._load_mtr_file)
        ov_row.addWidget(self.btn_load_mtr)
        ov_row.addWidget(QLabel("Opacity:"))
        self.slider_overlay_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_overlay_alpha.setRange(0, 100)
        self.slider_overlay_alpha.setValue(50)
        self.slider_overlay_alpha.setFixedWidth(120)
        self.slider_overlay_alpha.valueChanged.connect(self._refresh_display)
        ov_row.addWidget(self.slider_overlay_alpha)
        self.lbl_overlay_alpha = QLabel("50%")
        self.lbl_overlay_alpha.setFixedWidth(36)
        ov_row.addWidget(self.lbl_overlay_alpha)
        ov_row.addStretch()
        self._overlay_frame.setVisible(False)
        right_layout.addWidget(self._overlay_frame)

        # TR/TE image scroll slider (hidden until needed)
        self._img_scroll_frame = QFrame()
        scroll_layout = QHBoxLayout(self._img_scroll_frame)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._img_slider = QSlider(Qt.Orientation.Horizontal)
        self._img_slider.setMinimum(0)
        self._img_slider.setMaximum(0)
        self._img_slider.valueChanged.connect(self._on_img_slider)
        self._img_slider_label = QLabel("\u2014")
        scroll_layout.addWidget(self._img_slider, stretch=1)
        scroll_layout.addWidget(self._img_slider_label)
        # Exclude-from-fit checkbox \u2014 lets the user drop the current TR/TE image
        self.chk_exclude_img = QCheckBox("Exclude from fit")
        self.chk_exclude_img.setToolTip(
            "Exclude the currently displayed T1 (TR) or T2 (TE) image from the\n"
            "T1/T2 fit \u2014 useful for dropping corrupted or motion-affected frames.")
        self.chk_exclude_img.toggled.connect(self._on_exclude_img_toggled)
        scroll_layout.addWidget(self.chk_exclude_img)
        self._img_scroll_frame.setVisible(False)
        right_layout.addWidget(self._img_scroll_frame)
        # Per-frame exclusion sets (indices into the TR/TE dimension)
        self._t1_excluded: set = set()
        self._t2_excluded: set = set()

        # ── Figure Customization (collapsible — mirrors the MRF Viewer) ───────
        self.grp_fig_custom = QGroupBox("Figure Customization")
        _gfc = QVBoxLayout(self.grp_fig_custom); _gfc.setContentsMargins(8, 6, 8, 6)
        self.chk_fig_custom = QCheckBox("Enable Figure Customization")
        self.chk_fig_custom.setToolTip(
            "Show the title, colormap, colour-bar limit, slice and font controls "
            "(including Bg and Log map).")
        _gfc.addWidget(self.chk_fig_custom)
        self._fig_custom_panel = QWidget()
        self._fcp_lay = QVBoxLayout(self._fig_custom_panel)
        self._fcp_lay.setContentsMargins(0, 0, 0, 0)
        _gfc.addWidget(self._fig_custom_panel)
        self._fig_custom_panel.setVisible(False)
        self.chk_fig_custom.toggled.connect(self._fig_custom_panel.setVisible)
        right_layout.addWidget(self.grp_fig_custom)

        # Custom title row
        _tt_title_row = QHBoxLayout()
        _tt_title_row.addWidget(QLabel("Title:"))
        self.edit_map_title = QLineEdit()
        self.edit_map_title.setPlaceholderText("Custom map title (leave blank for default)")
        _tt_title_row.addWidget(self.edit_map_title, stretch=1)
        self._fcp_lay.addLayout(_tt_title_row)

        from my_gui.format_bar import add_title_format_bar, connect_title_debounced
        connect_title_debounced(self.edit_map_title, self._refresh_display)

        # Plot customisation bar — fonts line on top (with the B/I/x²/x₂ title
        # formatting buttons on its right), colour-bar line below (with the
        # Slice selector on its top-right).  Everything sits under the title box.
        self.plot_bar = PlotCustomBar(default_cmap="gray", fonts_first=True)
        self.plot_bar.applied.connect(self._on_plot_bar_applied)
        # Title B / I / x² / x₂ buttons → right side of the fonts line.
        add_title_format_bar(self.edit_map_title, None,
                             target_row=self.plot_bar.font_row(),
                             default_getter=lambda: getattr(self.canvas, "_last_title", ""))
        # Slice selector → top-right corner of the colour-bar line.
        _lbl_slice = QLabel("Slice:")
        _lbl_slice.setStyleSheet("color:#aaa; font-size:11px;")
        self.plot_bar.add_to_clim_row(_lbl_slice)
        self.plot_bar.add_to_clim_row(self.spin_slice)
        self._fcp_lay.addWidget(self.plot_bar)

        # "Bg" (black background) toggle for the map figure — inserted into the
        # font row immediately before the "B" (bold title) button.  Only the
        # white surround + labels flip; the image/colormap stay identical.
        self.chk_dark_bg = QCheckBox("Bg")
        self.chk_dark_bg.setToolTip("Black background for the figure (for slides). Only the white surround and labels flip - the maps stay identical.")
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

        # ROICanvas
        self.canvas = ROICanvas()
        self.canvas.mpl_connect('scroll_event', self._on_canvas_scroll)
        right_layout.addWidget(self.canvas, stretch=1)

        # ROI Spectra + ROI Stats Table — side by side (matches the MRF viewer)
        _roi_btn_row = QHBoxLayout()
        self.btn_roi_spectra = QPushButton("ROI Spectra")
        self.btn_roi_spectra.clicked.connect(self._show_roi_fit_curves)
        _roi_btn_row.addWidget(self.btn_roi_spectra)

        self.btn_roi_table = QPushButton("ROI Statistics")
        self.btn_roi_table.clicked.connect(self._show_roi_table)
        _roi_btn_row.addWidget(self.btn_roi_table)
        right_layout.addLayout(_roi_btn_row)

        # Data cursor checkbox
        from my_gui.plot_custom_bar import DataCursorToolButton
        self.chk_datacursor = DataCursorToolButton()
        self.chk_datacursor.toggled.connect(self._toggle_datacursor)
        disp_row.insertWidget(disp_row.indexOf(self.btn_contrast) + 1, self.chk_datacursor)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_widget)
        splitter.setSizes([380, 620])

    # ------------------------------------------------------------------
    # Group builders
    # ------------------------------------------------------------------

    def _build_t1_group(self) -> QGroupBox:
        grp = QGroupBox("T1 Scan")
        layout = QVBoxLayout(grp)
        layout.setSpacing(4)

        # ── Format / vendor selector ────────────────────────────────────────
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Platform:"))
        self.combo_t1_fmt = QComboBox()
        self.combo_t1_fmt.addItems(["Bruker", "GE/Siemens"])
        self.combo_t1_fmt.currentIndexChanged.connect(self._on_t1_fmt_changed)
        fmt_row.addWidget(self.combo_t1_fmt, stretch=1)
        layout.addLayout(fmt_row)

        # ── Folder browser row ──────────────────────────────────────────────
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Scan folder:"))
        self.le_t1_dir = QLineEdit()
        self.le_t1_dir.setReadOnly(True)
        self.le_t1_dir.setPlaceholderText("Select the scan folder…")
        row1.addWidget(self.le_t1_dir, stretch=1)
        btn_browse_t1 = QPushButton("Browse…")
        btn_browse_t1.clicked.connect(self._browse_t1)
        row1.addWidget(btn_browse_t1)
        layout.addLayout(row1)

        # ── Options row ─────────────────────────────────────────────────────
        # Bruker version selector is kept in the backend (auto/PV360) but hidden.
        self.lbl_t1_pv = QLabel("Bruker version:"); self.lbl_t1_pv.setVisible(False)
        self.combo_t1_pv = QComboBox()
        self.combo_t1_pv.addItems(["PV360", "PV6 / PV7"])
        self.combo_t1_pv.setVisible(False)
        # Reverse frame order — hidden holder (the visible button was removed;
        # may be re-added later).  Kept unchecked so the fit uses the loaded
        # order; add it back to the layout to re-enable the control.
        self.chk_t1_reverse = QCheckBox("Reverse frame order (last → first)")

        # ── Recovery-time schedule override (GE/Siemens only) ───────────────
        # Siemens CEST-MRF T1 stores a *constant* DICOM TR (e.g. 6000 ms) — the
        # varying recovery/saturation time lives in the external Pulseq .seq file,
        # so the plain S(TR) fit collapses to a vertical line.  Let the user paste
        # the recovery-time schedule (one value per image) to use as the x-axis.
        self._t1_sched_frame = QFrame()
        _t1s = QHBoxLayout(self._t1_sched_frame)
        _t1s.setContentsMargins(0, 0, 0, 0); _t1s.setSpacing(4)
        _t1s.addWidget(QLabel("Recovery times (ms):"))
        self.le_t1_sched = QLineEdit()
        self.le_t1_sched.setPlaceholderText("e.g. 100, 200, 400, 800, …  (one per image)")
        self.le_t1_sched.setToolTip(
            "Recovery / saturation-recovery times for the T1 fit  S = M0·(1−e^(−t/T1)) + c.\n"
            "Required for Siemens CEST-MRF T1: the DICOM TR is constant, so the true\n"
            "schedule (from the Pulseq .seq file) must be entered here — one value per\n"
            "image, comma/space separated.  Leave blank to use the DICOM TR (Bruker).")
        _t1s.addWidget(self.le_t1_sched, stretch=1)
        btn_t1_sched = QPushButton("Load…")
        btn_t1_sched.setToolTip("Load the recovery-time schedule from a .txt file (one value per line/row).")
        btn_t1_sched.clicked.connect(lambda: self._load_schedule_txt(self.le_t1_sched))
        _t1s.addWidget(btn_t1_sched)
        self._t1_sched_frame.setVisible(False)   # shown for GE/Siemens
        layout.addWidget(self._t1_sched_frame)

        # ── Load (left) + Run fit (right) on a single row ───────────────────
        btn_load_t1 = QPushButton("Load T1 Data")
        btn_load_t1.setFixedHeight(32)
        btn_load_t1.clicked.connect(self._load_t1)

        self.btn_run_t1 = QPushButton("Run T1 Fit")
        self.btn_run_t1.setFixedHeight(32)
        self.btn_run_t1.setStyleSheet(
            "QPushButton{background:#27ae60;color:white;font-weight:bold;border-radius:4px;}"
            "QPushButton:hover{background:#2ecc71;}"
            "QPushButton:disabled{background:#555;color:#999;}")
        self.btn_run_t1.clicked.connect(self._run_t1_fit)

        _r1run = QHBoxLayout()
        _r1run.setSpacing(8)
        _r1run.addWidget(btn_load_t1, 1)
        _r1run.addWidget(self.btn_run_t1, 1)
        layout.addLayout(_r1run)

        # ── Status label ────────────────────────────────────────────────────
        self.lbl_t1_status = QLabel("")
        self.lbl_t1_status.setWordWrap(True)
        self.lbl_t1_status.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.lbl_t1_status)

        return grp

    def _build_t2_group(self) -> QGroupBox:
        grp = QGroupBox("T2 Scan")
        layout = QVBoxLayout(grp)
        layout.setSpacing(4)

        # ── Format / vendor selector ────────────────────────────────────────
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Platform:"))
        self.combo_t2_fmt = QComboBox()
        self.combo_t2_fmt.addItems(["Bruker", "GE/Siemens"])
        self.combo_t2_fmt.currentIndexChanged.connect(self._on_t2_fmt_changed)
        fmt_row.addWidget(self.combo_t2_fmt, stretch=1)
        layout.addLayout(fmt_row)

        # ── Folder browser row ──────────────────────────────────────────────
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Scan folder:"))
        self.le_t2_dir = QLineEdit()
        self.le_t2_dir.setReadOnly(True)
        self.le_t2_dir.setPlaceholderText("Select the scan folder…")
        row1.addWidget(self.le_t2_dir, stretch=1)
        btn_browse_t2 = QPushButton("Browse…")
        btn_browse_t2.clicked.connect(self._browse_t2)
        row1.addWidget(btn_browse_t2)
        layout.addLayout(row1)

        # ── NIfTI T2 TE fields (hidden for Bruker / DICOM) ─────────────────
        self._t2_nifti_te_frame = QFrame()
        te_layout = QHBoxLayout(self._t2_nifti_te_frame)
        te_layout.setContentsMargins(0, 0, 0, 0)
        te_layout.setSpacing(4)
        te_layout.addWidget(QLabel("First TE:"))
        self.spin_te_first = QDoubleSpinBox()
        self.spin_te_first.setRange(0.1, 10000.0)
        self.spin_te_first.setValue(11.0)
        self.spin_te_first.setDecimals(2)
        self.spin_te_first.setSuffix(" ms")
        self.spin_te_first.setFixedWidth(90)
        self.spin_te_first.setToolTip("Echo time of the first echo (ms)")
        te_layout.addWidget(self.spin_te_first)
        te_layout.addWidget(QLabel("TE step:"))
        self.spin_te_step = QDoubleSpinBox()
        self.spin_te_step.setRange(0.1, 10000.0)
        self.spin_te_step.setValue(11.0)
        self.spin_te_step.setDecimals(2)
        self.spin_te_step.setSuffix(" ms")
        self.spin_te_step.setFixedWidth(90)
        self.spin_te_step.setToolTip("Echo spacing — TE of echo N = First TE + (N-1) × TE step")
        te_layout.addWidget(self.spin_te_step)
        te_layout.addStretch()
        self._t2_nifti_te_frame.setVisible(False)
        layout.addWidget(self._t2_nifti_te_frame)

        # ── Options row ─────────────────────────────────────────────────────
        # Bruker version selector kept in the backend (auto/PV360) but hidden.
        self.lbl_t2_pv = QLabel("Bruker version:"); self.lbl_t2_pv.setVisible(False)
        self.combo_t2_pv = QComboBox()
        self.combo_t2_pv.addItems(["PV360", "PV6 / PV7"])
        self.combo_t2_pv.setVisible(False)
        # Reverse frame order — hidden holder (the visible button was removed;
        # may be re-added later).  Kept unchecked so the fit uses the loaded
        # order; add it back to the layout to re-enable the control.
        self.chk_t2_reverse = QCheckBox("Reverse frame order (last → first)")

        # ── Prep-time schedule override (GE/Siemens only) ───────────────────
        # Siemens CEST-MRF T2 stores a *constant* DICOM TE (e.g. 7.8 ms) — the
        # varying T2-prep / spin-lock duration lives in the external Pulseq .seq
        # file, so the plain S(TE) fit collapses to a vertical line.  Let the user
        # paste the prep-time schedule (one value per image) to use as the x-axis.
        self._t2_sched_frame = QFrame()
        _t2s = QHBoxLayout(self._t2_sched_frame)
        _t2s.setContentsMargins(0, 0, 0, 0); _t2s.setSpacing(4)
        _t2s.addWidget(QLabel("Prep / echo times (ms):"))
        self.le_t2_sched = QLineEdit()
        self.le_t2_sched.setPlaceholderText("e.g. 0, 10, 20, 40, 80, …  (one per image)")
        self.le_t2_sched.setToolTip(
            "T2-prep / spin-lock / echo times for the T2 fit  S = M0·e^(−t/T2).\n"
            "Required for Siemens CEST-MRF T2: the DICOM TE is constant, so the true\n"
            "schedule (from the Pulseq .seq file) must be entered here — one value per\n"
            "image, comma/space separated.  Leave blank to use the DICOM TE.")
        _t2s.addWidget(self.le_t2_sched, stretch=1)
        btn_t2_sched = QPushButton("Load…")
        btn_t2_sched.setToolTip("Load the prep-time schedule from a .txt file (one value per line/row).")
        btn_t2_sched.clicked.connect(lambda: self._load_schedule_txt(self.le_t2_sched))
        _t2s.addWidget(btn_t2_sched)
        self._t2_sched_frame.setVisible(False)   # shown for GE/Siemens
        layout.addWidget(self._t2_sched_frame)

        # ── Load (left) + Run fit (right) on a single row ───────────────────
        btn_load_t2 = QPushButton("Load T2 Data")
        btn_load_t2.setFixedHeight(32)
        btn_load_t2.clicked.connect(self._load_t2)

        self.btn_run_t2 = QPushButton("Run T2 Fit")
        self.btn_run_t2.setFixedHeight(32)
        self.btn_run_t2.setStyleSheet(
            "QPushButton{background:#2980b9;color:white;font-weight:bold;border-radius:4px;}"
            "QPushButton:hover{background:#3498db;}"
            "QPushButton:disabled{background:#555;color:#999;}")
        self.btn_run_t2.clicked.connect(self._run_t2_fit)

        _r2run = QHBoxLayout()
        _r2run.setSpacing(8)
        _r2run.addWidget(btn_load_t2, 1)
        _r2run.addWidget(self.btn_run_t2, 1)
        layout.addLayout(_r2run)

        self.lbl_t2_status = QLabel("")
        self.lbl_t2_status.setWordWrap(True)
        self.lbl_t2_status.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.lbl_t2_status)

        return grp

    def _build_b1_group(self) -> QGroupBox:
        grp = QGroupBox("B1 Scans")
        layout = QVBoxLayout(grp)
        layout.setSpacing(4)

        # ── B1 method selector ────────────────────────────────────────────
        m_row = QHBoxLayout()
        m_row.addWidget(QLabel("Method:"))
        self.combo_b1_method = QComboBox()
        self.combo_b1_method.addItems([
            "Double-angle (2 FA images)",
            "Pre-computed map (1 image)",
            "Bloch-Siegert (2 phase images)",
        ])
        self.combo_b1_method.currentIndexChanged.connect(self._on_b1_method_changed)
        m_row.addWidget(self.combo_b1_method, stretch=1)
        layout.addLayout(m_row)

        # ── Double-angle sub-widget (FA1 / FA2) ───────────────────────────
        self._b1_da_w = QWidget()
        layout.addWidget(self._b1_da_w)
        _da_layout = QVBoxLayout(self._b1_da_w)
        _da_layout.setContentsMargins(0, 0, 0, 0)
        _da_layout.setSpacing(4)
        layout = _da_layout   # FA rows below are added into the double-angle box

        # FA1 row
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("\u03b1\u2081  folder:"))
        self.le_b1_fa1 = QLineEdit()
        self.le_b1_fa1.setReadOnly(True)
        row1.addWidget(self.le_b1_fa1, stretch=1)
        btn_fa1 = QPushButton("Browse…")
        btn_fa1.clicked.connect(self._browse_b1_fa1)
        row1.addWidget(btn_fa1)
        layout.addLayout(row1)

        # FA2 row
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("\u03b1\u2082  folder:"))
        self.le_b1_fa2 = QLineEdit()
        self.le_b1_fa2.setReadOnly(True)
        row2.addWidget(self.le_b1_fa2, stretch=1)
        btn_fa2 = QPushButton("Browse…")
        btn_fa2.clicked.connect(self._browse_b1_fa2)
        row2.addWidget(btn_fa2)
        layout.addLayout(row2)

        # FA row hidden — FA is fixed at 60° internally
        self.spin_fa = QDoubleSpinBox()
        self.spin_fa.setRange(0.0, 500.0)
        self.spin_fa.setValue(60.0)
        self.spin_fa.setDecimals(1)
        self.spin_fa.setSuffix(" °")
        self.spin_fa.setEnabled(False)
        self.chk_fa_fixed = QCheckBox("Fixed (60°)")
        self.chk_fa_fixed.setChecked(True)
        self.chk_fa_fixed.toggled.connect(self._on_fa_fixed_toggled)

        # Restore the group-level layout for the remaining sub-widgets
        layout = grp.layout()

        # ── Pre-computed map sub-widget ───────────────────────────────────
        self._b1_pre_w = QWidget()
        pre = QVBoxLayout(self._b1_pre_w); pre.setContentsMargins(0, 0, 0, 0); pre.setSpacing(3)
        pr1 = QHBoxLayout()
        pr1.addWidget(QLabel("B1 map:"))
        self.le_b1_pre = QLineEdit(); self.le_b1_pre.setReadOnly(True)
        self.le_b1_pre.setPlaceholderText(".dcm / .nii  OR  a DICOM folder")
        pr1.addWidget(self.le_b1_pre, stretch=1)
        _bpf = QPushButton("File..."); _bpf.clicked.connect(lambda: self._browse_b1_single("pre", folder=False))
        pr1.addWidget(_bpf)
        _bpd = QPushButton("Folder..."); _bpd.clicked.connect(lambda: self._browse_b1_single("pre", folder=True))
        pr1.addWidget(_bpd)
        pre.addLayout(pr1)
        # Image/volume selector — GE often stores the B1 map in the 2nd image
        pr2 = QHBoxLayout()
        pr2.addWidget(QLabel("Image #:"))
        self.spin_b1_pre_frame = QSpinBox(); self.spin_b1_pre_frame.setRange(1, 9999)
        self.spin_b1_pre_frame.setValue(1)
        self.spin_b1_pre_frame.setToolTip(
            "Which image/volume in the series is the B1 map (1-based).\n"
            "GE often stores the B1 map as the 2nd image — set this to 2 if the\n"
            "first frame is an anatomical/reference image.")
        self.spin_b1_pre_frame.valueChanged.connect(self._reload_b1_pre)
        pr2.addWidget(self.spin_b1_pre_frame)
        self.lbl_b1_pre_n = QLabel("")
        self.lbl_b1_pre_n.setStyleSheet("font-size:10px;color:#888;")
        pr2.addWidget(self.lbl_b1_pre_n); pr2.addStretch()
        pre.addLayout(pr2)
        # Calibration mode: relative (normalise to phantom mean) vs absolute (×scale)
        pr3 = QHBoxLayout()
        self.chk_b1_pre_norm = QCheckBox("Normalise to phantom mean (=100%)")
        self.chk_b1_pre_norm.setChecked(True)
        self.chk_b1_pre_norm.setToolTip(
            "Checked  → relative: B1% = map / phantom-mean × 100 (mean pinned to 100%).\n"
            "Unchecked → absolute: B1% = map × scale, for maps already stored as a\n"
            "scaled percentage. GE 2db1map stores value/2 = % → use scale 0.5.")
        pr3.addWidget(self.chk_b1_pre_norm)
        pr3.addWidget(QLabel("scale ×"))
        self.spin_b1_pre_scale = QDoubleSpinBox()
        self.spin_b1_pre_scale.setRange(0.0001, 1000.0)
        self.spin_b1_pre_scale.setDecimals(3); self.spin_b1_pre_scale.setValue(0.5)
        self.spin_b1_pre_scale.setToolTip("Absolute-mode scale factor (GE 2db1map = 0.5).")
        self.spin_b1_pre_scale.setFixedWidth(80)
        pr3.addWidget(self.spin_b1_pre_scale)
        pr3.addStretch()
        pre.addLayout(pr3)
        self._b1_pre_w.setVisible(False); layout.addWidget(self._b1_pre_w)

        # ── Bloch-Siegert sub-widget ──────────────────────────────────────
        self._b1_bs_w = QWidget()
        bs = QVBoxLayout(self._b1_bs_w); bs.setContentsMargins(0, 0, 0, 0); bs.setSpacing(3)
        for lbl_txt, attr in [("BS +phase:", "bspos"), ("BS -phase:", "bsneg")]:
            r = QHBoxLayout(); r.addWidget(QLabel(lbl_txt))
            le = QLineEdit(); le.setReadOnly(True)
            setattr(self, f"le_b1_{attr}", le)
            r.addWidget(le, stretch=1)
            bf = QPushButton("File..."); bf.clicked.connect(lambda _=0, a=attr: self._browse_b1_single(a, folder=False))
            r.addWidget(bf)
            bd = QPushButton("Folder..."); bd.clicked.connect(lambda _=0, a=attr: self._browse_b1_single(a, folder=True))
            r.addWidget(bd)
            bs.addLayout(r)
        pr = QHBoxLayout()
        pr.addWidget(QLabel("Pulse:"))
        self.combo_bs_pulse = QComboBox(); self.combo_bs_pulse.addItems(["fermi", "gauss"])
        self.combo_bs_pulse.setToolTip("Bloch-Siegert pulse shape (sets Kbs / AmpInt constants).")
        pr.addWidget(self.combo_bs_pulse)
        pr.addWidget(QLabel("Dur (ms):"))
        self.spin_bs_dur = QDoubleSpinBox(); self.spin_bs_dur.setRange(0.01, 100.0)
        self.spin_bs_dur.setDecimals(3); self.spin_bs_dur.setValue(8.0)
        pr.addWidget(self.spin_bs_dur)
        pr.addWidget(QLabel("FA:"))
        self.spin_bs_fa = QDoubleSpinBox(); self.spin_bs_fa.setRange(1.0, 360.0)
        self.spin_bs_fa.setDecimals(1); self.spin_bs_fa.setValue(60.0)
        pr.addWidget(self.spin_bs_fa)
        pr.addStretch()
        bs.addLayout(pr)
        self._b1_bs_w.setVisible(False); layout.addWidget(self._b1_bs_w)

        # ── Load (left) + Run calibration (right) on a single row ───────────
        btn_load_b1 = QPushButton("Load B1 Data")
        btn_load_b1.setFixedHeight(32)
        btn_load_b1.clicked.connect(self._load_b1)

        self.btn_run_b1 = QPushButton("Run B1 Calibration")
        self.btn_run_b1.setFixedHeight(32)
        self.btn_run_b1.setStyleSheet(
            "QPushButton{background:#e67e22;color:white;font-weight:bold;border-radius:4px;}"
            "QPushButton:hover{background:#f39c12;}"
            "QPushButton:disabled{background:#555;color:#999;}")
        self.btn_run_b1.clicked.connect(self._run_b1_calc)

        _b1run = QHBoxLayout()
        _b1run.setSpacing(8)
        _b1run.addWidget(btn_load_b1, 1)
        _b1run.addWidget(self.btn_run_b1, 1)
        layout.addLayout(_b1run)

        return grp

    def _build_wasabi_scan_group(self) -> QGroupBox:
        """Compact WASABI scan entry — mirrors the T1/T2 groups (Platform +
        Scan folder) with an *Optimization* button that opens the full WASABI
        B0/B1 fitting panel in its own window."""
        grp = QGroupBox("WASABI Scan")
        layout = QVBoxLayout(grp)
        layout.setSpacing(4)

        # ── Platform selector (same as T1 / T2) ─────────────────────────────
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Platform:"))
        self.combo_wasabi_fmt = QComboBox()
        self.combo_wasabi_fmt.addItems(["Bruker", "GE/Siemens"])
        fmt_row.addWidget(self.combo_wasabi_fmt, stretch=1)
        layout.addLayout(fmt_row)

        # ── Scan folder browser (same as T1 / T2) ───────────────────────────
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Scan folder:"))
        self.le_wasabi_dir = QLineEdit()
        self.le_wasabi_dir.setReadOnly(True)
        self.le_wasabi_dir.setPlaceholderText("Select the WASABI scan folder…")
        row1.addWidget(self.le_wasabi_dir, stretch=1)
        btn_browse_w = QPushButton("Browse…")
        btn_browse_w.clicked.connect(self._browse_wasabi_scan)
        row1.addWidget(btn_browse_w)
        layout.addLayout(row1)

        # ── Optimization (left) + Run WASABI (right) on one row ─────────────
        self.btn_wasabi_opt = QPushButton("Optimization")
        self.btn_wasabi_opt.setFixedHeight(32)
        self.btn_wasabi_opt.clicked.connect(self._open_wasabi_dialog)

        self.btn_run_wasabi = QPushButton("Run WASABI")
        self.btn_run_wasabi.setFixedHeight(32)
        self.btn_run_wasabi.setStyleSheet(
            "QPushButton{background:#6c5ce7;color:white;font-weight:bold;border-radius:4px;}"
            "QPushButton:hover{background:#8577e8;}"
            "QPushButton:disabled{background:#555;color:#999;}")
        self.btn_run_wasabi.setToolTip(
            "Load the WASABI series from the folder above and run the per-voxel "
            "B0/B1 fit using the parameters set under Optimization.")
        self.btn_run_wasabi.clicked.connect(self._run_wasabi_from_main)

        _wrow = QHBoxLayout()
        _wrow.setSpacing(8)
        _wrow.addWidget(self.btn_wasabi_opt, 1)
        _wrow.addWidget(self.btn_run_wasabi, 1)
        layout.addLayout(_wrow)

        # Hidden Cancel — kept for the fit worker's enable/disable logic even
        # though the button is no longer exposed in the UI.
        self.btn_cancel_wasabi = QPushButton("Cancel")
        self.btn_cancel_wasabi.setEnabled(False)
        self.btn_cancel_wasabi.setVisible(False)
        self.btn_cancel_wasabi.clicked.connect(self._cancel_wasabi)

        # Status line for WASABI load / fit messages (used by the backend).
        self.lbl_wasabi_status = QLabel("")
        self.lbl_wasabi_status.setWordWrap(True)
        self.lbl_wasabi_status.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.lbl_wasabi_status)
        # Back-compat alias (older references used lbl_wasabi_scan_status)
        self.lbl_wasabi_scan_status = self.lbl_wasabi_status
        return grp

    def _browse_wasabi_scan(self):
        """Pick the WASABI scan folder from the compact scan group."""
        from PyQt6.QtWidgets import QFileDialog
        p = QFileDialog.getExistingDirectory(self, "Select WASABI scan folder", "")
        if p:
            self._wasabi_data_path = p
            self.le_wasabi_dir.setText(p)
            self.lbl_wasabi_scan_status.setText(
                "Folder set — click Optimization to load & fit.")

    def _sync_wasabi_scan_to_panel(self):
        """Push the main-tab Platform selection into the (hidden) source combo
        the load backend branches on."""
        self._ensure_wasabi_panel()
        fmt = self.combo_wasabi_fmt.currentText() if hasattr(self, "combo_wasabi_fmt") else ""
        if hasattr(self, "combo_wasabi_src"):
            self.combo_wasabi_src.setCurrentText("Bruker" if fmt == "Bruker" else "DICOM folder")
        path = getattr(self, "_wasabi_data_path", "")
        if path and hasattr(self, "le_wasabi_data"):
            self.le_wasabi_data.setText(path)

    def _ensure_wasabi_panel(self):
        """Build the WASABI parameter panel once (lazily) and cache it, so the
        parameter widgets exist even before the Optimization dialog is opened
        (Run WASABI reads them directly)."""
        if getattr(self, "_wasabi_panel", None) is None:
            self._wasabi_panel = self._build_wasabi_group()
        return self._wasabi_panel

    def _run_wasabi_from_main(self):
        """Load the WASABI series (Platform + folder from the main scan group)
        and run the per-voxel B0/B1 fit using the Optimization parameters."""
        self._ensure_wasabi_panel()
        self._sync_wasabi_scan_to_panel()
        if not getattr(self, "_wasabi_data_path", ""):
            self.lbl_wasabi_status.setText("Select a WASABI scan folder first.")
            self.lbl_wasabi_status.setStyleSheet("font-size:10px;color:#e74c3c;")
            return
        self._load_wasabi_data()
        if self._wasabi_z is not None:
            self._run_wasabi_fit()

    def _open_wasabi_dialog(self):
        """Show the WASABI parameter panel in a dedicated resizable dialog."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout
        panel = self._ensure_wasabi_panel()
        dlg = getattr(self, "_wasabi_dialog", None)
        if dlg is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("WASABI  B0 / B1 optimization")
            dlg.setMinimumWidth(600)
            _dl = QVBoxLayout(dlg)
            _dl.addWidget(panel)
            self._wasabi_dialog = dlg
        # Keep the (hidden) source selector in sync with the main-tab Platform.
        self._sync_wasabi_scan_to_panel()
        dlg.show(); dlg.raise_(); dlg.activateWindow()

    def _build_wasabi_group(self):
        """WASABI (Water Shift And B1) parameter panel — Offsets, Larmor, t_p,
        nominal B1 and the fit model. Platform + scan folder live in the main
        WASABI Scan group; loading/fitting are launched from there. Ported from
        CEST_EVAL (Schuenke 2017)."""
        wg = QGroupBox("WASABI  B0 / B1 fit")
        wg.setStyleSheet(
            "QGroupBox{border:1px solid #ffffff;border-radius:6px;margin-top:8px;"
            "padding-top:26px;font-weight:bold;}"
            "QGroupBox::title{subcontrol-origin:padding;subcontrol-position:top left;"
            "left:10px;top:5px;padding:0 6px;color:#eaeaea;}")
        wl = QVBoxLayout(wg); wl.setSpacing(4)

        # (The "Fits Z(Δω)=|c−af·sin²θ·sin²φ| … Load a WASABI offset series." hint
        #  line was removed at user request.)

        # Hidden source/version selectors — driven by the main-tab Platform combo.
        # Kept as attributes because the load backend branches on them.
        self.combo_wasabi_src = QComboBox()
        self.combo_wasabi_src.addItems(["Bruker", "DICOM folder", "NIfTI file"])
        self.combo_wasabi_src.setVisible(False)
        self.combo_wasabi_src.currentIndexChanged.connect(self._on_wasabi_src_changed)
        self.combo_wasabi_pv = QComboBox(); self.combo_wasabi_pv.addItems(["PV360", "PV6 / PV7"])
        self.combo_wasabi_pv.setVisible(False)
        # le_wasabi_data retained (hidden) — the load backend uses _wasabi_data_path,
        # but a few helpers still reference this line-edit for display.
        self.le_wasabi_data = QLineEdit(); self.le_wasabi_data.setReadOnly(True)
        self.le_wasabi_data.setVisible(False)

        _or = QHBoxLayout()
        _or.addWidget(QLabel("Offsets (.txt, ppm):"))
        self.le_wasabi_off = QLineEdit(); self.le_wasabi_off.setReadOnly(True)
        self.le_wasabi_off.setPlaceholderText("auto (Bruker) or browse a ppm list")
        _or.addWidget(self.le_wasabi_off, stretch=1)
        _wof = QPushButton("…"); _wof.setFixedWidth(30)
        _wof.clicked.connect(self._browse_wasabi_off)
        _or.addWidget(_wof)
        wl.addLayout(_or)

        pr = QHBoxLayout()
        pr.addWidget(QLabel("Larmor:"))
        self.spin_wasabi_freq = QDoubleSpinBox(); self.spin_wasabi_freq.setRange(1.0, 2000.0)
        self.spin_wasabi_freq.setDecimals(3); self.spin_wasabi_freq.setValue(300.0)
        self.spin_wasabi_freq.setSuffix(" MHz"); self.spin_wasabi_freq.setFixedWidth(120)
        pr.addWidget(self.spin_wasabi_freq)
        pr.addWidget(QLabel("tp:"))
        self.spin_wasabi_tp = QDoubleSpinBox(); self.spin_wasabi_tp.setRange(0.01, 1000.0)
        self.spin_wasabi_tp.setDecimals(3); self.spin_wasabi_tp.setValue(5.0)
        self.spin_wasabi_tp.setSuffix(" ms"); self.spin_wasabi_tp.setFixedWidth(100)
        pr.addWidget(self.spin_wasabi_tp)
        pr.addStretch()
        wl.addLayout(pr)

        pr2 = QHBoxLayout()
        pr2.addWidget(QLabel("Nominal B1:"))
        self.spin_wasabi_b1 = QDoubleSpinBox(); self.spin_wasabi_b1.setRange(0.01, 100.0)
        self.spin_wasabi_b1.setDecimals(3); self.spin_wasabi_b1.setValue(3.7)
        self.spin_wasabi_b1.setSuffix(" µT"); self.spin_wasabi_b1.setFixedWidth(110)
        pr2.addWidget(self.spin_wasabi_b1)
        pr2.addWidget(QLabel("Model:"))
        self.combo_wasabi_model = QComboBox()
        self.combo_wasabi_model.addItems(["4-param", "3-param"])
        self.combo_wasabi_model.setToolTip(
            "Number of free parameters fitted to Z(Δω)=|c − af·sin²θ·sin²φ| per voxel:\n"
            " 4-param — fits B0, B1, the baseline c and the amplitude (af). \n"
            " 3-param — fits B0, B1 and a single combined amplitude (c and af). \n ")
        pr2.addWidget(self.combo_wasabi_model)
        pr2.addStretch()
        wl.addLayout(pr2)

        # WASABI internal state
        self._wasabi_z: np.ndarray | None = None      # (Y, X, slices, offsets) Z
        self._wasabi_ppm: np.ndarray | None = None
        self._wasabi_m0: np.ndarray | None = None
        self._db0_map: np.ndarray | None = None       # (Y, X) ppm
        self._wasabi_worker = None
        self._on_wasabi_src_changed(0)
        return wg

    def _on_b1_method_changed(self, _idx: int):
        m = self.combo_b1_method.currentText()
        self._b1_da_w.setVisible(m.startswith("Double-angle"))
        self._b1_pre_w.setVisible(m.startswith("Pre-computed"))
        self._b1_bs_w.setVisible(m.startswith("Bloch-Siegert"))

    def _build_processing_group(self) -> QWidget:
        """Create the processing widgets. The old 'Processing' group box was
        removed at user request: the Slice selector now lives next to the
        colour-bar (Cbar) controls (see the plot-bar section of the right
        panel), and only a borderless progress-bar + status strip remains
        here — invisible when idle, so no box is shown."""
        # Slice selector — created here, placed beside the Cbar controls.
        self.spin_slice = QSpinBox()
        self.spin_slice.setRange(1, 99)
        self.spin_slice.setValue(1)
        self.spin_slice.valueChanged.connect(self._update_slice)

        # Hidden Cancel — kept only for the fit worker's enable/disable wiring.
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._cancel_fit)

        # Borderless status strip (no title box): progress bar + status label.
        strip = QWidget()
        layout = QVBoxLayout(strip)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        self.lbl_proc_status = QLabel("")
        self.lbl_proc_status.setWordWrap(True)
        layout.addWidget(self.lbl_proc_status)
        return strip

    # ------------------------------------------------------------------
    # Browse slots
    # ------------------------------------------------------------------

    # ── Format change handlers ────────────────────────────────────────────────

    def _on_t1_fmt_changed(self, idx: int):
        """Show the recovery-time schedule field for GE/Siemens (idx == 1)."""
        if hasattr(self, "_t1_sched_frame"):
            self._t1_sched_frame.setVisible(idx == 1)

    def _on_t2_fmt_changed(self, idx: int):
        """Show the NIfTI TE fields + prep-time schedule for GE/Siemens."""
        self._t2_nifti_te_frame.setVisible(idx == 1)
        if hasattr(self, "_t2_sched_frame"):
            self._t2_sched_frame.setVisible(idx == 1)

    @staticmethod
    def _parse_schedule(text: str):
        """Parse a comma/space/newline-separated list of numbers → float ndarray
        (empty → None).  Used for the GE/Siemens T1/T2 schedule override."""
        import re as _re
        toks = [t for t in _re.split(r"[,\s]+", str(text).strip()) if t]
        if not toks:
            return None
        return np.asarray([float(t) for t in toks], dtype=float)

    def _load_schedule_txt(self, line_edit):
        """Load a schedule (recovery/prep times) from a .txt file into a QLineEdit."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load schedule (recovery / prep times)", "",
            "Text files (*.txt *.csv *.dat);;All files (*)")
        if not path:
            return
        try:
            vals = self._parse_schedule(open(path, "r", encoding="utf-8", errors="ignore").read())
            if vals is None or vals.size == 0:
                raise ValueError("no numeric values found")
            line_edit.setText(", ".join(f"{v:g}" for v in vals))
        except Exception as exc:
            QMessageBox.warning(self, "Schedule load failed", f"Could not read schedule:\n{exc}")

    @staticmethod
    def _folder_is_nifti(folder: str) -> bool:
        """True if the folder contains NIfTI files (→ NIfTI), else DICOM."""
        import glob, os
        return bool(glob.glob(os.path.join(folder, "*.nii"))
                    or glob.glob(os.path.join(folder, "*.nii.gz")))

    # ── Browse slots ──────────────────────────────────────────────────────────

    def _browse_t1(self):
        path = QFileDialog.getExistingDirectory(self, "Select T1 scan folder")
        if path:
            self._t1_dir = path
            self.le_t1_dir.setText(path)

    def _browse_t2(self):
        path = QFileDialog.getExistingDirectory(self, "Select T2 scan folder")
        if path:
            self._t2_dir = path
            self.le_t2_dir.setText(path)

    def _on_fa_fixed_toggled(self, checked: bool):
        """Lock the nominal FA spin box at 60° when Fixed is checked."""
        self.spin_fa.setEnabled(not checked)
        if checked:
            self.spin_fa.setValue(60.0)
        self.chk_fa_fixed.setText("Fixed (60°)" if checked else "Custom")

    def _browse_b1_fa1(self):
        path = QFileDialog.getExistingDirectory(self, "Select FA\u2081 folder")
        if path:
            self._b1_fa1_dir = path
            self.le_b1_fa1.setText(path)

    def _browse_b1_fa2(self):
        path = QFileDialog.getExistingDirectory(self, "Select FA\u2082 folder")
        if path:
            self._b1_fa2_dir = path
            self.le_b1_fa2.setText(path)

    def _browse_b1_single(self, which: str, folder: bool):
        """Browse a single B1 image (file or DICOM folder) for the pre-computed
        / Bloch-Siegert methods. `which` \u2208 {'pre','bspos','bsneg'}."""
        if folder:
            path = QFileDialog.getExistingDirectory(self, "Select DICOM folder")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select image", "",
                "Images (*.dcm *.IMA *.nii *.nii.gz);;All files (*)")
        if not path:
            return
        setattr(self, f"_b1_{which}_path", path)
        getattr(self, f"le_b1_{which}").setText(path)

    @staticmethod
    def _load_single_2d(path: str, frame: int = 1):
        """Load one 2-D image (1-based `frame`) from a DICOM file/folder or NIfTI.

        Returns (image_2d, n_total). GE B1 series often store the B1 map as the
        2nd image, so the caller can select which frame/volume to use.
        """
        from my_gui.tabs.image_viewer_tab import load_image_stack
        stack, _labels = load_image_stack(path, log_fn=lambda *_a, **_k: None)
        n = stack.shape[2]
        idx = int(np.clip(frame - 1, 0, n - 1))
        return stack[:, :, idx].astype(float), n

    def _reload_b1_pre(self):
        """Re-load the pre-computed B1 map when the image-index spinbox changes."""
        if getattr(self, "_b1_pre_path", ""):
            self._load_b1()

    # ------------------------------------------------------------------
    # Load slots
    # ------------------------------------------------------------------

    def _load_t1(self):
        if not self._t1_dir:
            self.lbl_t1_status.setText("No folder selected.")
            self.lbl_t1_status.setStyleSheet("color: red; font-size: 10px;")
            return
        try:
            fmt_idx = self.combo_t1_fmt.currentIndex()
            if fmt_idx == 0:
                # ── Bruker ──────────────────────────────────────────────────
                from my_gui.bruker_reader import read_2dseq_t1_rarevtr
                pv360 = self.combo_t1_pv.currentText() == "PV360"
                self._t1_img, self._t1_trs = read_2dseq_t1_rarevtr(self._t1_dir, pv360=pv360)
            elif self._folder_is_nifti(self._t1_dir):
                # ── GE / Siemens — NIfTI (auto-detected) ────────────────────
                from my_gui.ge_siemens_reader import read_nifti_t1_vtr
                self._t1_img, self._t1_trs = read_nifti_t1_vtr(self._t1_dir)
            else:
                # ── GE / Siemens — DICOM (auto-detected) ────────────────────
                from my_gui.ge_siemens_reader import read_dicom_t1_vtr
                self._t1_img, self._t1_trs = read_dicom_t1_vtr(self._t1_dir)

            # Reverse the measurement (TR) axis if requested — Siemens CEST-MRF
            # T1 series need the last image first.
            if self.chk_t1_reverse.isChecked():
                self._t1_img = np.ascontiguousarray(self._t1_img[..., ::-1])
                self._t1_trs = np.ascontiguousarray(np.asarray(self._t1_trs)[::-1])

            # Override the TR x-axis with the user-entered recovery schedule
            # (GE/Siemens CEST-MRF: the DICOM TR is constant, so the fit needs the
            # real recovery times).  Applied AFTER reversal so the schedule maps
            # positionally to the displayed image order.
            sched_applied = False
            if fmt_idx == 1:
                sched = self._parse_schedule(self.le_t1_sched.text())
                if sched is not None:
                    n_meas = self._t1_img.shape[3]
                    if sched.size != n_meas:
                        raise ValueError(
                            f"Recovery-time schedule has {sched.size} values but "
                            f"{n_meas} T1 images were loaded — they must match.")
                    self._t1_trs = sched
                    sched_applied = True

            self._t1_excluded.clear()   # reset exclusions for new data
            Y, X, n_sl, nTR = self._t1_img.shape
            trs = np.asarray(self._t1_trs, dtype=float)
            axis_name = "recovery" if sched_applied else "TR"
            tr_range = f"{trs.min():.0f}–{trs.max():.0f} ms"
            msg = f"✔  Loaded: {Y}×{X}  slices={n_sl}  n={nTR}  {axis_name}={tr_range}"
            self.lbl_t1_status.setText(msg)
            self.lbl_t1_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")
            self._update_slice_spinbox()
            self._show_reference()
        except Exception as exc:
            import traceback
            self.lbl_t1_status.setText(f"Error: {exc}")
            self.lbl_t1_status.setStyleSheet("color: red; font-size: 10px;")

    def _load_t2(self):
        if not self._t2_dir:
            self.lbl_t2_status.setText("No folder selected.")
            self.lbl_t2_status.setStyleSheet("color: red; font-size: 10px;")
            return
        try:
            fmt_idx = self.combo_t2_fmt.currentIndex()
            if fmt_idx == 0:
                # ── Bruker ──────────────────────────────────────────────────
                from my_gui.bruker_reader import read_2dseq_t2_msme
                pv360 = self.combo_t2_pv.currentText() == "PV360"
                self._t2_img, self._t2_tes = read_2dseq_t2_msme(self._t2_dir, pv360=pv360)
            elif self._folder_is_nifti(self._t2_dir):
                # ── GE / Siemens — NIfTI (auto-detected) ────────────────────
                from my_gui.ge_siemens_reader import read_nifti_t2_msme
                self._t2_img, self._t2_tes = read_nifti_t2_msme(
                    self._t2_dir, te_first_ms=self.spin_te_first.value(),
                    te_step_ms=self.spin_te_step.value())
            else:
                # ── GE / Siemens — DICOM (auto-detected) ────────────────────
                from my_gui.ge_siemens_reader import read_dicom_t2_msme
                self._t2_img, self._t2_tes = read_dicom_t2_msme(self._t2_dir)

            # Reverse the echo (TE) axis if requested — Siemens CEST-MRF T2
            # series need the last image first.
            if self.chk_t2_reverse.isChecked():
                self._t2_img = np.ascontiguousarray(self._t2_img[..., ::-1])
                self._t2_tes = np.ascontiguousarray(np.asarray(self._t2_tes)[::-1])

            # Override the TE x-axis with the user-entered prep-time schedule
            # (GE/Siemens CEST-MRF: the DICOM TE is constant, so the fit needs the
            # real prep / spin-lock times).  Applied AFTER reversal.
            sched_applied = False
            if fmt_idx == 1:
                sched = self._parse_schedule(self.le_t2_sched.text())
                if sched is not None:
                    n_meas = self._t2_img.shape[3]
                    if sched.size != n_meas:
                        raise ValueError(
                            f"Prep-time schedule has {sched.size} values but "
                            f"{n_meas} T2 images were loaded — they must match.")
                    self._t2_tes = sched
                    sched_applied = True

            self._t2_excluded.clear()   # reset exclusions for new data
            Y, X, n_sl, nTE = self._t2_img.shape
            tes = np.asarray(self._t2_tes, dtype=float)
            axis_name = "prep" if sched_applied else "TE"
            te_range = f"{tes.min():.1f}–{tes.max():.1f} ms"
            msg = f"✔  Loaded: {Y}×{X}  slices={n_sl}  n={nTE}  {axis_name}={te_range}"
            self.lbl_t2_status.setText(msg)
            self.lbl_t2_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")
            self._update_slice_spinbox()
            self._show_reference()
        except Exception as exc:
            self.lbl_t2_status.setText(f"Error: {exc}")
            self.lbl_t2_status.setStyleSheet("color: red; font-size: 10px;")

    def _load_b1(self):
        m = self.combo_b1_method.currentText()
        try:
            if m.startswith("Double-angle"):
                if not self._b1_fa1_dir or not self._b1_fa2_dir:
                    self.lbl_proc_status.setText("Select both FA folders first.")
                    return
                from my_gui.bruker_reader import read_2dseq_b1_rare
                self._b1_fa1_img = read_2dseq_b1_rare(self._b1_fa1_dir)
                self._b1_fa2_img = read_2dseq_b1_rare(self._b1_fa2_dir)
                self.lbl_proc_status.setText("B1 data loaded (double-angle).")
            elif m.startswith("Pre-computed"):
                if not self._b1_pre_path:
                    self.lbl_proc_status.setText("Select a B1 map first.")
                    return
                frame = self.spin_b1_pre_frame.value()
                self._b1_pre_img, n_tot = self._load_single_2d(self._b1_pre_path, frame)
                # Keep the spinbox range in sync and show the total count
                self.spin_b1_pre_frame.blockSignals(True)
                self.spin_b1_pre_frame.setMaximum(max(1, n_tot))
                self.spin_b1_pre_frame.blockSignals(False)
                self.lbl_b1_pre_n.setText(f"of {n_tot}")
                self.lbl_proc_status.setText(
                    f"Pre-computed B1 map loaded — image {min(frame, n_tot)}/{n_tot} "
                    f"({self._b1_pre_img.shape[0]}×{self._b1_pre_img.shape[1]}).")
            else:  # Bloch-Siegert
                if not self._b1_bspos_path or not self._b1_bsneg_path:
                    self.lbl_proc_status.setText("Select both BS phase images first.")
                    return
                self._b1_bspos_img, _ = self._load_single_2d(self._b1_bspos_path)
                self._b1_bsneg_img, _ = self._load_single_2d(self._b1_bsneg_path)
                self.lbl_proc_status.setText("Bloch-Siegert phase images loaded.")
        except Exception as exc:
            self.lbl_proc_status.setText(f"B1 load error: {exc}")

    # ------------------------------------------------------------------
    # Slice helpers
    # ------------------------------------------------------------------

    def _update_slice_spinbox(self):
        # One shared slice index drives both T1 and T2, so cap it at the SMALLER
        # of the loaded volumes' slice counts — otherwise a slice valid for a
        # de-tiled 88-slice mosaic would index out of a single-slice partner.
        counts = [v.shape[2] for v in (self._t1_img, self._t2_img) if v is not None]
        if counts:
            self.spin_slice.setMaximum(max(1, min(counts)))

    def _update_slice(self):
        self._slice_idx = self.spin_slice.value() - 1
        # A fitted T1/T2 map is a single 2-D slice; once the user moves to a
        # different slice it no longer applies, so drop it rather than show a
        # stale map (and mis-registered overlay) for the wrong slice.
        if (getattr(self, "_t1_map", None) is not None
                and self._slice_idx != getattr(self, "_t1_fit_slice", None)):
            self._t1_map = None
        if (getattr(self, "_t2_map", None) is not None
                and self._slice_idx != getattr(self, "_t2_fit_slice", None)):
            self._t2_map = None
        self._refresh_display()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _show_reference(self):
        ref_img = None
        if self._t1_img is not None:
            ref_img = self._t1_img[:, :, self._slice_idx, 0]
        elif self._t2_img is not None:
            ref_img = self._t2_img[:, :, self._slice_idx, 0]
        elif self._b1_fa1_img is not None:
            if self._b1_fa1_img.ndim == 3:
                ref_img = self._b1_fa1_img[:, :, self._slice_idx]
            else:
                ref_img = self._b1_fa1_img[:, :]
        if ref_img is None:
            return
        self.canvas.show_map(ref_img, "Reference Image", cmap="gray")
        self.canvas._img_data = ref_img

    def set_scan_paths_getter(self, fn):
        self._scan_paths_getter = fn

    def _pick_roi_bg(self):
        from my_gui.roi_tools import choose_background_image
        sp = {}
        _g = getattr(self, "_scan_paths_getter", None)
        if callable(_g):
            try:
                sp = _g() or {}
            except Exception:
                sp = {}
        mode, img = choose_background_image(self, sp)
        if mode == "set":
            self._roi_bg_img = img
        elif mode == "default":
            self._roi_bg_img = None
        else:
            return
        self.chk_roi_bg.setChecked(True)
        self._refresh_display()

    def _refresh_display(self):
        if hasattr(self, "chk_dark_bg"):
            self.canvas._dark_bg = self.chk_dark_bg.isChecked()
        if hasattr(self, "chk_logmap"):
            self.canvas._log_map = self.chk_logmap.isChecked()
        self._dc_annot = None   # reset data cursor (axes rebuilt on show_map)
        choice = self.combo_display.currentText()
        cmap = self.plot_bar.get_cmap()
        vmin, vmax = self.plot_bar.get_clim()

        is_overlay = choice.startswith("MTRasym Overlay")
        # Show / hide the TR/TE/B1 image slider — overlays also scroll the base frame
        is_scrollable = choice in ("T1 Images", "T2 Images", "α₁ Images (B1 FA1)", "α₂ Images (B1 FA2)")
        if is_overlay:
            is_scrollable = ("→ T1" in choice and self._t1_img is not None) or \
                            ("→ T2" in choice and self._t2_img is not None)
        self._img_scroll_frame.setVisible(is_scrollable)
        self._overlay_frame.setVisible(is_overlay)
        self.lbl_overlay_alpha.setText(f"{self.slider_overlay_alpha.value()}%")

        if is_overlay and "→ T1" in choice and self._t1_img is not None:
            n = self._t1_img.shape[3]
            self._img_slider.setMaximum(max(n - 1, 0))
            idx = self._img_slider.value()
            self._img_slider_label.setText(f"TR = {self._t1_trs[idx]:.0f} ms  ({idx + 1}/{n})")
        elif is_overlay and "→ T2" in choice and self._t2_img is not None:
            n = self._t2_img.shape[3]
            self._img_slider.setMaximum(max(n - 1, 0))
            idx = self._img_slider.value()
            self._img_slider_label.setText(f"TE = {self._t2_tes[idx]:.0f} ms  ({idx + 1}/{n})")

        # Exclude checkbox only applies to T1/T2 image views
        _is_t1t2_img = choice in ("T1 Images", "T2 Images")
        self.chk_exclude_img.setVisible(_is_t1t2_img)

        if choice == "T1 Images" and self._t1_img is not None:
            n = self._t1_img.shape[3]
            self._img_slider.setMaximum(max(n - 1, 0))
            idx = self._img_slider.value()
            _exc = idx in self._t1_excluded
            self._img_slider_label.setText(
                f"TR = {self._t1_trs[idx]:.0f} ms  ({idx + 1}/{n})"
                + ("  [EXCLUDED]" if _exc else "")
            )
            self.chk_exclude_img.blockSignals(True)
            self.chk_exclude_img.setChecked(_exc)
            self.chk_exclude_img.blockSignals(False)
        elif choice == "T2 Images" and self._t2_img is not None:
            n = self._t2_img.shape[3]
            self._img_slider.setMaximum(max(n - 1, 0))
            idx = self._img_slider.value()
            _exc = idx in self._t2_excluded
            self._img_slider_label.setText(
                f"TE = {self._t2_tes[idx]:.0f} ms  ({idx + 1}/{n})"
                + ("  [EXCLUDED]" if _exc else "")
            )
            self.chk_exclude_img.blockSignals(True)
            self.chk_exclude_img.setChecked(_exc)
            self.chk_exclude_img.blockSignals(False)
        elif choice in ("α₁ Images (B1 FA1)", "α₂ Images (B1 FA2)"):
            imgs = (self._b1_fa1_img if choice == "α₁ Images (B1 FA1)"
                    else self._b1_fa2_img)
            if imgs is not None:
                n = imgs.shape[2] if imgs.ndim >= 3 else 1
                self._img_slider.setMaximum(max(n - 1, 0))
                idx = self._img_slider.value()
                lbl = "FA1" if choice == "α₁ Images (B1 FA1)" else "FA2"
                self._img_slider_label.setText(f"B1 {lbl}  ({idx + 1}/{n})")

        # Apply Phantom_outline mask if available (suppress background noise)
        phantom = next(
            (r for r in getattr(self, '_last_rois', []) if r.name == "Phantom_outline"),
            None,
        )

        def _rescale_mask(m: np.ndarray, shape) -> "np.ndarray | None":
            m = np.asarray(m)
            if m.shape[:2] == shape[:2]:
                return m.astype(bool)
            try:
                from scipy.ndimage import zoom as _zoom
                zy = shape[0] / max(m.shape[0], 1)
                zx = shape[1] / max(m.shape[1], 1)
                return _zoom(m.astype(float), (zy, zx), order=0) > 0.5
            except Exception:
                return None

        def _mask(arr: np.ndarray) -> np.ndarray:
            if arr is None:
                return arr
            # "ROIs only" → blank everything outside the union of drawn ROIs
            if getattr(self, "chk_roi_only", None) is not None and self.chk_roi_only.isChecked():
                union = None
                for r in getattr(self, "_last_rois", []):
                    if getattr(r, "name", "") == "Phantom_outline":
                        continue
                    m = _rescale_mask(r.mask, arr.shape)
                    if m is None:
                        continue
                    union = m if union is None else (union | m)
                if union is not None:
                    return np.where(union, arr, np.nan)
                return arr   # no ROIs drawn → show full map
            # Default: Phantom_outline mask (suppress background)
            if phantom is None:
                return arr
            msk = _rescale_mask(phantom.mask, arr.shape)
            if msk is None:
                return arr
            return arr * msk

        fs = self.plot_bar.get_font_sizes()
        _ct = getattr(self, 'edit_map_title', None)
        _ctitle = _ct.text().strip() if _ct else ""

        if choice == "Reference Image":
            self._show_reference()
        elif choice == "T1 Images":
            if self._t1_img is not None:
                tr_idx = self._img_slider.value()
                img = self._t1_img[:, :, self._slice_idx, tr_idx]
                title = _ctitle or f"T1 Image \u2014 TR = {self._t1_trs[tr_idx]:.0f} ms"
                self.canvas.show_map(_mask(img), title,
                                     cmap=cmap, vmin=vmin, vmax=vmax, **fs)
                self.canvas._img_data = img
        elif choice == "T2 Images":
            if self._t2_img is not None:
                te_idx = self._img_slider.value()
                img = self._t2_img[:, :, self._slice_idx, te_idx]
                title = _ctitle or f"T2 Image \u2014 TE = {self._t2_tes[te_idx]:.0f} ms"
                self.canvas.show_map(_mask(img), title,
                                     cmap=cmap, vmin=vmin, vmax=vmax, **fs)
                self.canvas._img_data = img
        elif choice == "T1 Map (ms)":
            if self._t1_map is not None:
                if getattr(self, "chk_roi_bg", None) is not None and self.chk_roi_bg.isChecked():
                    _base = (self._t1_img[:, :, self._slice_idx, 0]
                             if self._t1_img is not None else None)
                    if getattr(self, "_roi_bg_img", None) is not None:
                        _base = self._roi_bg_img
                    _union = roi_union_mask(getattr(self, "_last_rois", []),
                                            self._t1_map.shape)
                    if _base is not None and _union is not None:
                        _ovc = cmap if str(cmap).lower() not in ("gray", "greys", "greys_r") else "jet"
                        self.canvas.show_map_over_raw(
                            self._t1_map, _base, _union, _ctitle or "T1 map (ms)",
                            cmap=_ovc, vmin=vmin, vmax=vmax, **fs)
                        return
                roi_only = (getattr(self, "chk_roi_only", None) is not None
                            and self.chk_roi_only.isChecked())
                if roi_only and self._t1_img is not None:
                    # Overlay ROI-restricted T1 map on the last-TR image
                    base = self._t1_img[:, :, self._slice_idx, -1]
                    ov_cmap = cmap if cmap not in ("gray", "Greys") else "jet"
                    self.canvas.show_overlay(
                        base, _mask(self._t1_map),
                        _ctitle or f"T1 map (ms) on last-TR image "
                                   f"({self._t1_trs[-1]:.0f} ms)",
                        base_cmap="gray", overlay_cmap=ov_cmap, alpha=0.9,
                        vmin=vmin, vmax=vmax, **fs)
                else:
                    self.canvas.show_map(_mask(self._t1_map), _ctitle or "T1 map (ms)",
                                         cmap=cmap, vmin=vmin, vmax=vmax, **fs)
        elif choice == "T2 Map (ms)":
            if self._t2_map is not None:
                if getattr(self, "chk_roi_bg", None) is not None and self.chk_roi_bg.isChecked():
                    _base = (self._t2_img[:, :, self._slice_idx, 0]
                             if self._t2_img is not None else None)
                    if getattr(self, "_roi_bg_img", None) is not None:
                        _base = self._roi_bg_img
                    _union = roi_union_mask(getattr(self, "_last_rois", []),
                                            self._t2_map.shape)
                    if _base is not None and _union is not None:
                        _ovc = cmap if str(cmap).lower() not in ("gray", "greys", "greys_r") else "jet"
                        self.canvas.show_map_over_raw(
                            self._t2_map, _base, _union, _ctitle or "T2 map (ms)",
                            cmap=_ovc, vmin=vmin, vmax=vmax, **fs)
                        return
                roi_only = (getattr(self, "chk_roi_only", None) is not None
                            and self.chk_roi_only.isChecked())
                if roi_only and self._t2_img is not None:
                    # Overlay ROI-restricted T2 map on the last-TE image
                    base = self._t2_img[:, :, self._slice_idx, -1]
                    ov_cmap = cmap if cmap not in ("gray", "Greys") else "jet"
                    self.canvas.show_overlay(
                        base, _mask(self._t2_map),
                        _ctitle or f"T2 map (ms) on last-TE image "
                                   f"({self._t2_tes[-1]:.0f} ms)",
                        base_cmap="gray", overlay_cmap=ov_cmap, alpha=0.9,
                        vmin=vmin, vmax=vmax, **fs)
                else:
                    self.canvas.show_map(_mask(self._t2_map), _ctitle or "T2 map (ms)",
                                         cmap=cmap, vmin=vmin, vmax=vmax, **fs)
        elif choice == "α₁ Images (B1 FA1)":
            if self._b1_fa1_img is not None:
                imgs = self._b1_fa1_img
                idx  = self._img_slider.value()
                img  = (imgs[:, :, idx] if imgs.ndim == 3
                        else imgs[:, :, self._slice_idx, idx] if imgs.ndim == 4
                        else imgs[:, :])
                title = _ctitle or f"B1 FA1 Image  ({idx + 1}/{max(imgs.shape[2] if imgs.ndim >= 3 else 1, 1)})"
                self.canvas.show_map(_mask(img), title,
                                     cmap=cmap, vmin=vmin, vmax=vmax, **fs)
                self.canvas._img_data = img
        elif choice == "α₂ Images (B1 FA2)":
            if self._b1_fa2_img is not None:
                imgs = self._b1_fa2_img
                idx  = self._img_slider.value()
                img  = (imgs[:, :, idx] if imgs.ndim == 3
                        else imgs[:, :, self._slice_idx, idx] if imgs.ndim == 4
                        else imgs[:, :])
                title = _ctitle or f"B1 FA2 Image  ({idx + 1}/{max(imgs.shape[2] if imgs.ndim >= 3 else 1, 1)})"
                self.canvas.show_map(_mask(img), title,
                                     cmap=cmap, vmin=vmin, vmax=vmax, **fs)
                self.canvas._img_data = img
        elif choice == "B1 Map":
            if self._b1_map is not None:
                if getattr(self, "chk_roi_bg", None) is not None and self.chk_roi_bg.isChecked():
                    _raw = self._b1_fa1_img
                    _base = None
                    if _raw is not None:
                        if _raw.ndim == 4:
                            _base = _raw[:, :, self._slice_idx, 0]
                        elif _raw.ndim == 3:
                            _base = _raw[:, :, 0]
                    if getattr(self, "_roi_bg_img", None) is not None:
                        _base = self._roi_bg_img
                    _union = roi_union_mask(getattr(self, "_last_rois", []),
                                            self._b1_map.shape)
                    if _base is not None and _union is not None:
                        _ovc = cmap if str(cmap).lower() not in ("gray", "greys", "greys_r") else "jet"
                        self.canvas.show_map_over_raw(
                            self._b1_map, _base, _union, _ctitle or "B1 map",
                            cmap=_ovc, vmin=vmin, vmax=vmax, **fs)
                        return
                self.canvas.show_map(_mask(self._b1_map), _ctitle or "B1 map",
                                     cmap=cmap, vmin=vmin, vmax=vmax, **fs)
        elif choice == "ΔB0 Map (ppm)":
            db0 = getattr(self, "_db0_map", None)
            if db0 is not None:
                if getattr(self, "chk_roi_bg", None) is not None and self.chk_roi_bg.isChecked():
                    _raw = self._b1_fa1_img
                    _base = None
                    if _raw is not None:
                        if _raw.ndim == 4:
                            _base = _raw[:, :, self._slice_idx, 0]
                        elif _raw.ndim == 3:
                            _base = _raw[:, :, 0]
                    if getattr(self, "_roi_bg_img", None) is not None:
                        _base = self._roi_bg_img
                    _union = roi_union_mask(getattr(self, "_last_rois", []), db0.shape)
                    if _base is not None and _union is not None:
                        _vmin, _vmax = vmin, vmax
                        if _vmin is None and _vmax is None:
                            _sel = db0[_union] if db0.shape[:2] == _union.shape[:2] else db0
                            _fin = np.asarray(_sel)[np.isfinite(_sel)]
                            _lim = float(np.nanpercentile(np.abs(_fin), 99)) if _fin.size else 1.0
                            _vmin, _vmax = -_lim, _lim
                        _dcmap = cmap if cmap != "viridis" else "bwr"
                        _ovc = _dcmap if str(_dcmap).lower() not in ("gray", "greys", "greys_r") else "jet"
                        self.canvas.show_map_over_raw(
                            db0, _base, _union, _ctitle or "ΔB0 map (ppm)  [WASABI]",
                            cmap=_ovc, vmin=_vmin, vmax=_vmax, **fs)
                        return
                mm = _mask(db0)
                if vmin is None and vmax is None:
                    fin = mm[np.isfinite(mm)] if mm is not None else np.array([])
                    lim = float(np.nanpercentile(np.abs(fin), 99)) if fin.size else 1.0
                    vmin, vmax = -lim, lim
                self.canvas.show_map(mm, _ctitle or "ΔB0 map (ppm)  [WASABI]",
                                     cmap=cmap if cmap != "viridis" else "bwr",
                                     vmin=vmin, vmax=vmax, **fs)
        elif is_overlay:
            self._show_mtr_overlay(choice, cmap, vmin, vmax, fs, _ctitle, _mask)

    def _show_mtr_overlay(self, choice, cmap, vmin, vmax, fs, ctitle, mask_fn):
        """Overlay the CEST MTR-asymmetry map on a T1 / T2 / CEST *image* frame."""
        on_cest = "→ CEST" in choice
        on_t1   = "→ T1" in choice

        if on_cest:
            base = None
            getter = getattr(self, "_cest_image_getter", None)
            if callable(getter):
                try:
                    base = getter()
                except Exception:
                    base = None
            if base is None:
                self._overlay_notice(
                    "No CEST image available.\n\n"
                    "Load and analyze data in the CEST MRI tab to see the overlay."
                )
                return
            base_label, idx_label = "CEST", ""
        else:
            img_stack = self._t1_img if on_t1 else self._t2_img
            if img_stack is None:
                _w = "T1" if on_t1 else "T2"
                self._overlay_notice(
                    f"No {_w} images loaded.\n\nLoad {_w} data first to see the overlay."
                )
                return
            idx = min(self._img_slider.value(), img_stack.shape[3] - 1)
            base = img_stack[:, :, self._slice_idx, idx]
            trs  = self._t1_trs if on_t1 else self._t2_tes
            unit = "TR" if on_t1 else "TE"
            base_label = "T1" if on_t1 else "T2"
            idx_label  = f" — {unit} = {trs[idx]:.0f} ms"

        mtr = self._get_mtr_overlay()
        if mtr is None:
            self._overlay_notice(
                "No MTR-asymmetry map available.\n\n"
                "Load one from file (set source to 'From file…'), or analyze the "
                "CEST MRI tab to compute it, to see the overlay."
            )
            return

        # A real overlay will render — reset the notice guard
        self._last_overlay_notice = None

        overlay_cmap = cmap if cmap not in ("gray", "Greys") else "jet"
        alpha = self.slider_overlay_alpha.value() / 100.0
        ppm   = self._mtr_overlay_ppm()
        default_t = f"MTRasym (±{ppm:.2f} ppm) on {base_label} image{idx_label}"
        self.canvas.show_overlay(
            mask_fn(base), mask_fn(mtr), ctitle or default_t,
            base_cmap="gray", overlay_cmap=overlay_cmap,
            alpha=alpha, vmin=vmin, vmax=vmax, **fs,
        )

    def _overlay_notice(self, msg: str):
        """Pop an info dialog about a missing overlay prerequisite (guarded so it
        does not re-pop on every refresh, e.g. while dragging the opacity slider)."""
        if getattr(self, "_last_overlay_notice", None) == msg:
            return
        self._last_overlay_notice = msg
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(self, "MTR-asymmetry Overlay", msg)

    def _get_mtr_overlay(self):
        """Return the active MTRasym map (file or CEST MRI tab) or None."""
        if self.combo_mtr_source.currentText() == "From file…":
            return self._mtr_overlay_file
        if callable(self._mtr_source_getter):
            try:
                return self._mtr_source_getter()
            except Exception:
                return None
        return None

    def _mtr_overlay_ppm(self) -> float:
        getter = getattr(self, "_mtr_ppm_getter", None)
        if callable(getter):
            try:
                return float(getter())
            except Exception:
                pass
        return 3.5

    def _on_mtr_source_changed(self, _idx: int):
        self.btn_load_mtr.setEnabled(self.combo_mtr_source.currentText() == "From file…")
        self._refresh_display()

    def _load_mtr_file(self):
        """Load an MTR-asymmetry map from .npy / .mat / .csv / .txt."""
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Load MTRasym map", "",
            "Map files (*.npy *.mat *.csv *.txt);;All files (*)"
        )
        if not path:
            return
        try:
            arr = self._read_map_file(path)
            self._mtr_overlay_file = arr
            self._last_overlay_notice = None
            self.combo_mtr_source.setCurrentText("From file…")
            self.lbl_t1_status.setText(f"Loaded MTRasym map: {arr.shape}")
            self.lbl_t1_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")
            self._refresh_display()
        except Exception as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "MTR-asymmetry Overlay",
                                f"Could not load MTRasym map:\n{exc}")

    @staticmethod
    def _read_map_file(path: str) -> np.ndarray:
        import os
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            arr = np.load(path)
        elif ext == ".mat":
            from scipy.io import loadmat
            md = loadmat(path)
            cands = {k: v for k, v in md.items()
                     if not k.startswith("__") and hasattr(v, "ndim") and v.ndim >= 2}
            if not cands:
                raise ValueError("No 2-D array found in .mat file.")
            # pick the largest 2-D array
            arr = max(cands.values(), key=lambda a: a.size)
        else:  # .csv / .txt
            delim = "," if ext == ".csv" else None
            arr = np.loadtxt(path, delimiter=delim)
        arr = np.asarray(arr, dtype=float)
        return arr[..., 0] if arr.ndim == 3 else arr

    def set_mtr_source(self, map_getter, ppm_getter=None):
        """Register callbacks the CEST MRI tab uses to supply its MTRasym map."""
        self._mtr_source_getter = map_getter
        self._mtr_ppm_getter = ppm_getter

    def set_cest_image_source(self, image_getter):
        """Register a callback returning the current CEST image (overlay base)."""
        self._cest_image_getter = image_getter

    def _toggle_hide_rois(self, checked: bool):
        self.canvas.toggle_rois_visible()
        self.btn_hide_rois.setText("Show ROIs" if checked else "Hide ROIs")

    def _on_img_slider(self):
        """Called when the TR/TE image slider value changes."""
        self._refresh_display()

    def _on_exclude_img_toggled(self, checked: bool):
        """Add/remove the current TR/TE frame from the exclusion set."""
        choice = self.combo_display.currentText()
        idx = self._img_slider.value()
        if choice == "T1 Images":
            s = self._t1_excluded
        elif choice == "T2 Images":
            s = self._t2_excluded
        else:
            return
        if checked:
            s.add(idx)
        else:
            s.discard(idx)
        self._refresh_display()

    def _on_canvas_scroll(self, event):
        """Mouse wheel over canvas scrolls through TR/TE/B1 images."""
        choice = self.combo_display.currentText()
        if (choice not in ("T1 Images", "T2 Images", "α₁ Images (B1 FA1)", "α₂ Images (B1 FA2)")
                and not choice.startswith("MTRasym Overlay")):
            return
        step = 1 if event.step > 0 else -1
        new_val = max(0, min(self._img_slider.maximum(),
                             self._img_slider.value() + step))
        self._img_slider.setValue(new_val)

    # ------------------------------------------------------------------
    # Fitting / calculation slots
    # ------------------------------------------------------------------

    def _run_t1_fit(self):
        if self._t1_img is None or self._t1_trs is None:
            self.lbl_proc_status.setText("Load T1 data first.")
            return
        sl = self._slice_idx
        nTR = self._t1_img.shape[3]
        keep = [i for i in range(nTR) if i not in self._t1_excluded]
        if len(keep) < 3:
            self.lbl_proc_status.setText(
                f"Need ≥3 TR images for T1 fit ({len(keep)} kept after exclusions).")
            return
        if len(keep) < nTR:
            self.lbl_proc_status.setText(
                f"T1 fit using {len(keep)}/{nTR} TR images "
                f"(excluded: {sorted(self._t1_excluded)}).")
        img_sl = self._t1_img[:, :, sl, :][:, :, keep]  # (Y, X, nKeep)
        trs    = self._t1_trs[keep]
        self._worker = T1FitWorker(img_sl, trs,
                                   bruker=(self.combo_t1_fmt.currentIndex() == 0))
        self._worker.finished.connect(self._on_t1_done)
        self._worker.progress.connect(self._on_progress)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()
        self._set_buttons_enabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.lbl_proc_status.setText("Fitting T1…")

    def _run_t2_fit(self):
        if self._t2_img is None or self._t2_tes is None:
            self.lbl_proc_status.setText("Load T2 data first.")
            return
        sl = self._slice_idx
        nTE = self._t2_img.shape[3]
        keep = [i for i in range(nTE) if i not in self._t2_excluded]
        if len(keep) < 3:
            self.lbl_proc_status.setText(
                f"Need ≥3 TE images for T2 fit ({len(keep)} kept after exclusions).")
            return
        if len(keep) < nTE:
            self.lbl_proc_status.setText(
                f"T2 fit using {len(keep)}/{nTE} TE images "
                f"(excluded: {sorted(self._t2_excluded)}).")
        img_sl = self._t2_img[:, :, sl, :][:, :, keep]  # (Y, X, nKeep)
        tes    = self._t2_tes[keep]
        self._worker = T2FitWorker(img_sl, tes,
                                   bruker=(self.combo_t2_fmt.currentIndex() == 0))
        self._worker.finished.connect(self._on_t2_done)
        self._worker.progress.connect(self._on_progress)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()
        self._set_buttons_enabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.lbl_proc_status.setText("Fitting T2…")

    def _run_b1_calc(self):
        m = self.combo_b1_method.currentText()
        try:
            if m.startswith("Double-angle"):
                if self._b1_fa1_img is None or self._b1_fa2_img is None:
                    self.lbl_proc_status.setText("Load B1 data first.")
                    return
                sl = self._slice_idx
                if self._b1_fa1_img.ndim == 3:
                    fa1_sl = self._b1_fa1_img[:, :, sl]
                    fa2_sl = self._b1_fa2_img[:, :, sl]
                else:
                    fa1_sl = self._b1_fa1_img[:, :]
                    fa2_sl = self._b1_fa2_img[:, :]
                self._b1_map = calc_b1_ratio(fa1_sl, fa2_sl)
                self.lbl_proc_status.setText(
                    "B1 map calculated (double-angle, S₂α/2S₁ = cos α).")
            elif m.startswith("Pre-computed"):
                if self._b1_pre_img is None:
                    self.lbl_proc_status.setText("Load a B1 map first.")
                    return
                from my_gui.b1_processing import calc_b1_precomputed
                _norm = self.chk_b1_pre_norm.isChecked()
                self._b1_map, _msk = calc_b1_precomputed(
                    self._b1_pre_img, normalize=_norm,
                    scale=self.spin_b1_pre_scale.value())
                _in = self._b1_map[np.isfinite(self._b1_map)]
                _mode = ("normalised to phantom mean" if _norm
                         else f"absolute (×{self.spin_b1_pre_scale.value():g})")
                self.lbl_proc_status.setText(
                    f"B1 map (pre-computed, {_mode}) — "
                    f"mean {np.nanmean(_in):.1f}% ± {np.nanstd(_in):.1f}%."
                    if _in.size else "Pre-computed B1 map: empty phantom mask.")
            else:  # Bloch-Siegert
                if self._b1_bspos_img is None or self._b1_bsneg_img is None:
                    self.lbl_proc_status.setText("Load both BS phase images first.")
                    return
                from my_gui.b1_processing import calc_b1_bloch_siegert
                self._b1_map = calc_b1_bloch_siegert(
                    self._b1_bspos_img, self._b1_bsneg_img,
                    flip_angle=self.spin_bs_fa.value(),
                    duration=self.spin_bs_dur.value(),
                    pulse_type=self.combo_bs_pulse.currentText())
                self.lbl_proc_status.setText("B1 map calculated (Bloch-Siegert).")
            # Restrict the B1 map to the global analysis mask (brain / phantom).
            self._b1_map = apply_analysis_mask(self._b1_map)
            self.combo_display.setCurrentText("B1 Map")
            self._refresh_display()
        except Exception as exc:
            self.lbl_proc_status.setText(f"B1 error: {exc}")

    # ------------------------------------------------------------------
    # WASABI B0 / B1 fit
    # ------------------------------------------------------------------

    def _on_wasabi_src_changed(self, _idx: int):
        # Source/PV selectors are hidden (driven by the main-tab Platform); the
        # offsets file is always accepted (Bruker fills from method; DICOM/NIfTI
        # need the ppm list here).
        self.le_wasabi_off.setEnabled(True)

    def _browse_wasabi(self):
        from PyQt6.QtWidgets import QFileDialog
        src = self.combo_wasabi_src.currentText()
        if src == "NIfTI file":
            p, _ = QFileDialog.getOpenFileName(
                self, "Select WASABI NIfTI", "", "NIfTI (*.nii *.nii.gz);;All files (*)")
        else:
            p = QFileDialog.getExistingDirectory(self, f"Select WASABI {src}", "")
        if p:
            self._wasabi_data_path = p
            self.le_wasabi_data.setText(p)

    def _browse_wasabi_off(self):
        from PyQt6.QtWidgets import QFileDialog
        p, _ = QFileDialog.getOpenFileName(
            self, "Select WASABI offsets (ppm)", "", "Text (*.txt *.csv);;All files (*)")
        if p:
            self._wasabi_off_path = p
            self.le_wasabi_off.setText(p)

    def _load_wasabi_data(self):
        from my_gui.wasabi_fit import prep_wasabi
        src = self.combo_wasabi_src.currentText()
        path = getattr(self, "_wasabi_data_path", "")
        if not path:
            self.lbl_wasabi_status.setText("Browse to WASABI data first.")
            return
        try:
            off_path = getattr(self, "_wasabi_off_path", "")
            off_txt = np.loadtxt(off_path).ravel() if off_path else None

            if src == "Bruker":
                image, ppm = self._load_bruker_wasabi(path, off_txt)
            elif src == "DICOM folder":
                from my_gui.tabs.zspec_tab import _load_dicom_4d
                image, ppm_auto = _load_dicom_4d(path, off_path, lambda *_: None)
                ppm = off_txt if off_txt is not None else ppm_auto
                freq = self._peek_dicom_freq(path)
                if freq:
                    self.spin_wasabi_freq.setValue(freq)
            else:  # NIfTI
                import nibabel as nib
                data = np.asarray(nib.load(path).dataobj, dtype=float)
                if data.ndim == 3:
                    data = data[:, :, np.newaxis, :]
                image = data
                if off_txt is None:
                    raise ValueError("NIfTI WASABI needs an offsets (.txt, ppm) file.")
                ppm = off_txt

            if ppm is None or len(ppm) != image.shape[-1]:
                raise ValueError(
                    f"Offsets ({0 if ppm is None else len(ppm)}) must match the "
                    f"{image.shape[-1]} acquired frames — provide a ppm .txt file "
                    f"in the 'Offsets' box (one value per frame).")

            # Split off M0 (|offset| > 30 ppm, e.g. the −300 ppm Pulseq M0) and
            # M0-normalise → Z spectrum on the remaining WASABI offsets.
            z4, ppm_fit, m0img = prep_wasabi(image, ppm, m0=None)

            self._wasabi_z = np.asarray(z4, dtype=float)
            self._wasabi_ppm = np.asarray(ppm_fit, dtype=float).ravel()
            self._wasabi_m0 = m0img
            self.spin_slice.setMaximum(max(self.spin_slice.maximum(),
                                           self._wasabi_z.shape[2]))
            n_m0 = image.shape[-1] - self._wasabi_z.shape[-1]
            self.lbl_wasabi_status.setText(
                f"Loaded WASABI: {self._wasabi_z.shape[0]}×{self._wasabi_z.shape[1]} "
                f"px, {self._wasabi_z.shape[2]} slice(s), {len(self._wasabi_ppm)} fit "
                f"offsets [{self._wasabi_ppm.min():.2f}..{self._wasabi_ppm.max():.2f} ppm]"
                + (f"  (+{n_m0} M0 frame(s))" if n_m0 else "")
                + ".  Set t_p / nominal B1 before fitting.")
            self.lbl_wasabi_status.setStyleSheet("font-size:10px;color:#4ec9b0;")
        except Exception as exc:
            self.lbl_wasabi_status.setText(f"WASABI load error: {exc}")
            self.lbl_wasabi_status.setStyleSheet("font-size:10px;color:#e74c3c;")

    def _load_bruker_wasabi(self, path, off_txt):
        """Load a Bruker WASABI 2dseq data directly (robust to Pulseq-CEST scans
        whose offsets/frame-count are NOT in the method file). Returns
        (image (Y,X,slices,frames), ppm)."""
        import os
        from pathlib import Path
        from my_gui.bruker_reader import (_read_geometry, _read_raw,
                                          read_2dseq_cest, read_bruker_params)
        # Resolve to the pdata/1 directory (accept either the scan folder or pdata/1)
        p = path
        if not os.path.isfile(os.path.join(p, "2dseq")):
            cand = os.path.join(p, "pdata", "1")
            if os.path.isfile(os.path.join(cand, "2dseq")):
                p = cand
        pdata = Path(p)

        # Try the standard CEST reader first (handles normal Bruker CEST cleanly);
        # fall back to a raw load when it can't (Pulseq-CEST offset/frame mismatch).
        try:
            image, _m0, info = read_2dseq_cest(str(pdata),
                                               pv360=(self.combo_wasabi_pv.currentText() == "PV360"))
            ppm = (off_txt if off_txt is not None
                   else np.asarray(info.get("ppm_all"), dtype=float).ravel())
            if info.get("omega_0"):
                self.spin_wasabi_freq.setValue(float(info["omega_0"]))
            # read_2dseq_cest returns the M0-removed, ppm-sorted subset; for WASABI
            # we want the ORIGINAL frame order to match the offsets list, so prefer
            # img_all when present.
            if info.get("img_all") is not None:
                image = np.asarray(info["img_all"], dtype=float)
            return image, ppm
        except Exception:
            pass

        # Raw fallback: derive the frame count from the binary, not the method.
        nx, ny, nsl, niter = _read_geometry(pdata)
        raw = _read_raw(pdata, nx, ny, nsl, niter)      # (nx, ny, nsl, n_frames)
        image = np.transpose(raw, (1, 0, 2, 3)).astype(float)   # (Y, X, slices, frames)
        # Larmor from the method (PVM_FrqWork), if available
        try:
            d = read_bruker_params(pdata, "method", ["##$PVM_FrqWork"])
            from my_gui.bruker_reader import _parse_numbers
            fr = _parse_numbers(d.get("##$PVM_FrqWork"))
            if fr.size and fr[0] > 1.0:
                self.spin_wasabi_freq.setValue(float(fr[0]))
        except Exception:
            pass
        ppm = off_txt   # method has no usable offsets → require the .txt list
        return image, ppm

    @staticmethod
    def _read_bruker_tp(path):
        """WASABI saturation pulse length (ms) from the Bruker method, or None."""
        try:
            from my_gui.bruker_reader import read_bruker_params
            d = read_bruker_params(path, "method",
                                   ["##$PVM_MagTransPulse1", "##$Fp_SatDur",
                                    "##$PVM_SatTransPulse"])
            for v in d.values():
                if v:
                    tok = str(v).replace(",", " ").split()
                    for t in tok:
                        try:
                            f = float(t)
                            if 0.05 < f < 1000:     # plausible ms
                                return f
                        except ValueError:
                            continue
        except Exception:
            pass
        return None

    @staticmethod
    def _peek_dicom_freq(path):
        """ImagingFrequency (MHz) from the first DICOM in a folder, or None."""
        try:
            import glob as _g, os as _os, pydicom
            files = [f for f in _g.glob(_os.path.join(path, "*"))
                     if _os.path.isfile(f)]
            for f in files[:20]:
                try:
                    ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
                    v = float(getattr(ds, "ImagingFrequency", 0.0))
                    if v > 1.0:
                        return v
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def _run_wasabi_fit(self):
        if self._wasabi_z is None or self._wasabi_ppm is None:
            self.lbl_wasabi_status.setText("Load WASABI data first.")
            return
        z = self._wasabi_z
        sl = max(0, min(self._slice_idx, z.shape[2] - 1))
        z_slice = z[:, :, sl, :]                       # (Y, X, off)
        # Fit mask: threshold the unsaturated M0 signal where available (more
        # reliable than the normalised Z), else the mean |Z| (∩ phantom ROI).
        m0 = getattr(self, "_wasabi_m0", None)
        if m0 is not None and np.asarray(m0).ndim == 3 and m0.shape[2] > sl:
            sig = np.abs(np.asarray(m0, dtype=float)[:, :, sl])
        else:
            sig = np.nanmean(np.abs(z_slice), axis=-1)
        finite = sig[np.isfinite(sig)]
        thr = 0.1 * (float(np.nanmax(finite)) if finite.size else 1.0)
        mask = np.isfinite(sig) & (sig > thr)
        phantom = next((r for r in getattr(self, "_last_rois", [])
                        if getattr(r, "name", "") == "Phantom_outline"), None)
        if phantom is not None:
            pm = np.asarray(phantom.mask, dtype=bool)
            if pm.shape == mask.shape:
                mask &= pm
        if not mask.any():
            self.lbl_wasabi_status.setText("WASABI: empty fit mask (no signal).")
            return
        model = "3param" if self.combo_wasabi_model.currentIndex() == 1 else "4param"
        self._wasabi_worker = WasabiWorker(
            z_slice, self._wasabi_ppm, self.spin_wasabi_freq.value(),
            self.spin_wasabi_tp.value() * 1e-3,            # ms → s
            self.spin_wasabi_b1.value(), mask, model=model)
        self._wasabi_worker.progress.connect(self.progress_bar.setValue)
        self._wasabi_worker.finished.connect(self._on_wasabi_done)
        self._wasabi_worker.error.connect(self._on_wasabi_error)
        self.btn_run_wasabi.setEnabled(False)
        self.btn_cancel_wasabi.setEnabled(True)
        self.progress_bar.setValue(0); self.progress_bar.show()
        self.lbl_wasabi_status.setText(f"Fitting WASABI on slice {sl + 1} "
                                       f"({int(mask.sum())} voxels)…")
        self.lbl_wasabi_status.setStyleSheet("font-size:10px;color:#888;")
        self._wasabi_worker.start()

    def _cancel_wasabi(self):
        if self._wasabi_worker is not None and self._wasabi_worker.isRunning():
            self._wasabi_worker.stop()
        self.btn_cancel_wasabi.setEnabled(False)

    def _on_wasabi_done(self, res: dict):
        self.progress_bar.hide()
        self.btn_run_wasabi.setEnabled(True)
        self.btn_cancel_wasabi.setEnabled(False)
        self._b1_map = apply_analysis_mask(res["b1_rel"])   # relative B1 (%) — reuses "B1 Map"
        self._db0_map = apply_analysis_mask(res["db0"])     # ΔB0 (ppm)
        b1v = self._b1_map[np.isfinite(self._b1_map)]
        d0v = self._db0_map[np.isfinite(self._db0_map)]
        note = " (cancelled)" if res.get("stopped") else ""
        self.lbl_wasabi_status.setText(
            (f"WASABI fit done{note}: B1 = {np.nanmean(b1v):.1f}% ± {np.nanstd(b1v):.1f}%, "
             f"ΔB0 = {np.nanmean(d0v):+.3f} ± {np.nanstd(d0v):.3f} ppm."
             if b1v.size else f"WASABI fit produced no valid voxels{note}."))
        self.lbl_wasabi_status.setStyleSheet("font-size:10px;color:#4ec9b0;")
        if "ΔB0 Map (ppm)" == self.combo_display.currentText():
            self._refresh_display()
        else:
            self.combo_display.setCurrentText("B1 Map")
            self._refresh_display()

    def _on_wasabi_error(self, msg: str):
        self.progress_bar.hide()
        self.btn_run_wasabi.setEnabled(True)
        self.btn_cancel_wasabi.setEnabled(False)
        self.lbl_wasabi_status.setText(f"WASABI fit error: {msg.splitlines()[0]}")
        self.lbl_wasabi_status.setStyleSheet("font-size:10px;color:#e74c3c;")

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _on_t1_done(self, t1_map):
        # Restrict to the global analysis mask (brain / phantom outline), if any.
        self._t1_map = apply_analysis_mask(t1_map)
        self._t1_fit_slice = self._slice_idx     # map belongs to this slice
        self._dc_annot = None
        self._set_buttons_enabled(True)
        self.progress_bar.hide()
        self.lbl_proc_status.setText("T1 fit complete.")
        self.combo_display.setCurrentText("T1 Map (ms)")
        self._refresh_display()

    def _on_t2_done(self, t2_map):
        self._t2_map = apply_analysis_mask(t2_map)
        self._t2_fit_slice = self._slice_idx     # map belongs to this slice
        self._dc_annot = None
        self._set_buttons_enabled(True)
        self.progress_bar.hide()
        self.lbl_proc_status.setText("T2 fit complete.")
        self.combo_display.setCurrentText("T2 Map (ms)")
        self._refresh_display()

    def _on_progress(self, pct: int):
        self.progress_bar.setValue(pct)

    def _on_worker_error(self, msg: str):
        self._set_buttons_enabled(True)
        self.progress_bar.hide()
        self.lbl_proc_status.setText(f"Error: {msg[:120]}")

    def _set_buttons_enabled(self, enabled: bool):
        self.btn_run_t1.setEnabled(enabled)
        self.btn_run_t2.setEnabled(enabled)
        self.btn_run_b1.setEnabled(enabled)
        # Cancel is the inverse — only active while a fit is running
        if hasattr(self, "btn_cancel"):
            self.btn_cancel.setEnabled(not enabled)

    def _cancel_fit(self):
        """Request the running T1/T2 fit worker to stop."""
        w = getattr(self, "_worker", None)
        if w is not None and hasattr(w, "stop"):
            w.stop()
            self.lbl_proc_status.setText("Cancelling…")
            self.btn_cancel.setEnabled(False)

    def _cancel_all_scans(self):
        """Cancel whichever scan operation (T1 fit, T2 fit or WASABI) is running.
        (B1 calibration is an instantaneous inline computation — nothing to stop.)"""
        cancelled = False
        for attr in ("_worker", "_wasabi_worker"):
            w = getattr(self, attr, None)
            if (w is not None and hasattr(w, "isRunning") and w.isRunning()
                    and hasattr(w, "stop")):
                w.stop()
                cancelled = True
        if hasattr(self, "btn_cancel_wasabi"):
            self.btn_cancel_wasabi.setEnabled(False)
        if hasattr(self, "btn_cancel"):
            self.btn_cancel.setEnabled(False)
        if hasattr(self, "lbl_proc_status"):
            self.lbl_proc_status.setText(
                "Cancelling…" if cancelled else "No scan is currently running.")

    # ------------------------------------------------------------------
    # Fitting curve display
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ROI Manager integration
    # ------------------------------------------------------------------

    def connect_roi_manager(self, roi_manager):
        """Subscribe canvas to ROI updates from the shared ROI Manager tab."""
        roi_manager.connect_canvas(self.canvas)
        roi_manager.rois_changed.connect(self._update_roi_stats)
        self._last_rois: list = []

    def _update_roi_stats(self, rois: list):
        self._last_rois = list(rois)
        # Stats shown on demand via the ROI Stats Table button

    def _show_roi_table(self):
        from my_gui.roi_table_dialog import show_roi_table
        rois = getattr(self, '_last_rois', [])
        map_items: list[tuple] = []
        if self._t1_map is not None:
            map_items.append(("T1 (ms)", self._t1_map))
        if self._t2_map is not None:
            map_items.append(("T2 (ms)", self._t2_map))
        if self._b1_map is not None:
            map_items.append(("B1", self._b1_map))
        img = self.canvas._img_data
        if img is not None and not map_items:
            map_items.append(("Current", img))
        show_roi_table(self, rois, map_items, title="T1 / T2 / B1 — ROI Statistics")

    def _on_plot_bar_applied(self):
        self._refresh_display()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_figure(self):
        from my_gui.fig_export import save_figure, FIG_EXPORT_FILTER
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export figure",
            "figure.png",
            FIG_EXPORT_FILTER,
        )
        if not path:
            return
        try:
            save_figure(self.canvas._fig, path, dpi=300)
            self.lbl_proc_status.setText(f"Saved: {path}")
        except Exception as exc:
            self.lbl_proc_status.setText(f"Export error: {exc}")

    # ------------------------------------------------------------------
    # Per-ROI T1 / T2 fit curves dialog
    # ------------------------------------------------------------------

    def _show_roi_fit_curves(self):
        """Open a dialog with per-ROI T1 and T2 relaxation fit curves
        and a live font/style control bar."""
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
            QWidget, QMessageBox, QSpinBox, QComboBox, QLabel as _QL,
            QPushButton as _QPB,
        )
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.font_manager import FontProperties
        from my_gui.fig_theme import apply_fig_dark_theme

        rois = getattr(self, '_last_rois', [])
        rois = [r for r in rois if r.name != "Phantom_outline"]
        has_t1 = self._t1_img is not None and self._t1_trs is not None
        has_t2 = self._t2_img is not None and self._t2_tes is not None

        if not has_t1 and not has_t2:
            QMessageBox.information(self, "No Data", "Load T1 or T2 data and run fits first.")
            return

        sl = self._slice_idx

        dlg = QDialog(self)
        dlg.setWindowTitle("Per-ROI T1/T2 Fit Curves")
        dlg.resize(980, 700)
        vl = QVBoxLayout(dlg)

        # ── Font / style control bar ──────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        ctrl.addWidget(_QL("Font:"))
        combo_ff = QComboBox()
        combo_ff.setFixedWidth(148)
        for _f in ("Default", "Arial", "Times New Roman",
                   "Helvetica", "DejaVu Sans", "DejaVu Serif"):
            combo_ff.addItem(_f)
        ctrl.addWidget(combo_ff)

        def _fspin(label, default, tip):
            ctrl.addWidget(_QL(label))
            sp = QSpinBox(); sp.setRange(5, 32); sp.setValue(default)
            sp.setFixedWidth(50); sp.setToolTip(tip)
            ctrl.addWidget(sp)
            return sp

        sp_title  = _fspin("Title:",  11, "Subplot title font size")
        sp_suptitle = _fspin("Main:", 13, "Overall figure title font size")
        sp_axes   = _fspin("Axes:",    9, "X/Y axis label font size")
        sp_ticks  = _fspin("Ticks:",   8, "Tick label font size")
        sp_legend = _fspin("Legend:",  8, "Legend font size")

        chk_dark_bg = QCheckBox("Bg")
        chk_dark_bg.setToolTip("Black background for the figure (for slides). Only the white surround and labels flip - the maps stay identical.")
        ctrl.addWidget(chk_dark_bg)

        # ── Custom single colour for every ROI curve (instead of random) ──────
        _color_override = [None]
        _picked = ["#1f77b4"]
        chk_color = QCheckBox("Custom color")
        chk_color.setToolTip(
            "Plot every ROI curve in one chosen colour instead of each ROI's "
            "own (random) colour.")
        ctrl.addWidget(chk_color)
        btn_color = QPushButton(); btn_color.setFixedWidth(30)
        btn_color.setToolTip("Pick the curve colour")

        def _swatch():
            btn_color.setStyleSheet(
                f"background:{_picked[0]}; border:1px solid #888; border-radius:3px;")
        _swatch()

        def _pick_color():
            from PyQt6.QtWidgets import QColorDialog
            from PyQt6.QtGui import QColor
            c = QColorDialog.getColor(QColor(_picked[0]), dlg, "Curve colour")
            if c.isValid():
                _picked[0] = c.name(); _swatch()
                if chk_color.isChecked():
                    _color_override[0] = _picked[0]; _sched_font()

        def _on_color_toggle(on):
            _color_override[0] = _picked[0] if on else None
            _sched_font()

        btn_color.clicked.connect(_pick_color)
        chk_color.toggled.connect(_on_color_toggle)
        ctrl.addWidget(btn_color)

        ctrl.addStretch()
        vl.addLayout(ctrl)

        # ── Tab widget ────────────────────────────────────────────────────────
        tabs = QTabWidget()
        vl.addWidget(tabs, stretch=1)

        def _make_fit_panel(img_sl, x_vals, x_label, kind, bruker, panel_title,
                            rois_subset, title_fs, suptitle_fs, axes_fs, ticks_fs,
                            legend_fs, font_fam):
            """Build one QWidget with per-ROI subplots (≤8 ROIs) and fonts.

            kind ∈ {'t1','t2'}; bruker selects the original (Bruker) curve model
            (M0 = max−min for T1, M0 = max for T2) vs the GE/Siemens model that
            uses the actual fitted M0/c. rois_subset is the page of ROIs to plot.
            """
            fit_fn = _fit_t1 if kind == "t1" else _fit_t2

            def _fit_and_curve(sig):
                """Return (val_ms, curve_y, warn_str) for one mean spectrum."""
                if bruker:
                    val = fit_fn(sig, x_vals, bruker=True)        # original bounds
                    if val <= 0:
                        return 0.0, None, ""
                    if kind == "t1":
                        M0 = float(sig.max() - sig.min()); c = float(sig.min())
                        curve = M0 * (1.0 - np.exp(-x_fine / val)) + c   # original model
                    else:
                        curve = float(sig.max()) * np.exp(-x_fine / val)  # original model
                    return val, curve, ""
                # GE/Siemens — use the actual fitted amplitude/offset
                params = fit_fn(sig, x_vals, full=True, bruker=False)
                val = params[1]
                if val <= 0:
                    return 0.0, None, ""
                if kind == "t1":
                    M0, T1, c = params; curve = M0 * (1.0 - np.exp(-x_fine / T1)) + c
                else:
                    M0, T2 = params; curve = M0 * np.exp(-x_fine / T2)
                return val, curve, ""      # no extrapolation warning text

            H, W, nX = img_sl.shape
            n_rois = len(rois_subset)
            use_fp = font_fam and font_fam.lower() not in ("default", "")

            def _fp(size):
                return FontProperties(family=font_fam, size=size) if use_fp \
                       else FontProperties(size=size)

            if n_rois == 0:
                effective_rois = None
                nCols, nRows = 1, 1
            else:
                effective_rois = rois_subset
                nCols = min(4, n_rois)
                nRows = max(1, (n_rois + nCols - 1) // nCols)

            fig = Figure(figsize=(nCols * 4.5, nRows * 3.5), facecolor='white')

            # Main figure title
            st_kwargs = {"fontsize": suptitle_fs, "fontweight": "bold"}
            if use_fp:
                st_kwargs["fontproperties"] = _fp(suptitle_fs)
                del st_kwargs["fontsize"]
            fig.suptitle(panel_title, **st_kwargs)

            x_fine = np.linspace(x_vals[0], x_vals[-1], 300)

            def _style_ax(ax, sub_title):
                ax.set_xlabel(x_label, fontproperties=_fp(axes_fs))
                ax.set_ylabel("Signal (a.u.)", fontproperties=_fp(axes_fs))
                ax.tick_params(labelsize=ticks_fs)
                if use_fp:
                    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                        lbl.set_fontproperties(_fp(ticks_fs))
                t_kw = {"fontweight": "bold"}
                if use_fp:
                    t_kw["fontproperties"] = _fp(title_fs)
                else:
                    t_kw["fontsize"] = title_fs
                ax.set_title(sub_title, **t_kw)

            if effective_rois is None:
                ax = fig.add_subplot(1, 1, 1)
                sig = img_sl.reshape(-1, nX).mean(axis=0).astype(float)
                ax.plot(x_vals, sig, 'ko', markersize=5, label='Mean (all pixels)')
                val, curve, _warn = _fit_and_curve(sig)
                if val > 0:
                    ax.plot(x_fine, curve, 'r-', linewidth=2, label=f'Fit: {val:.1f} ms')
                    if _warn:
                        ax.text(0.03, 0.04, _warn, transform=ax.transAxes,
                                fontsize=max(7, ticks_fs - 1), color='#b00',
                                va='bottom', ha='left')
                _style_ax(ax, 'All pixels (no ROIs defined)')
                ax.legend(prop=_fp(legend_fs))
            else:
                for i, roi in enumerate(effective_rois):
                    ax = fig.add_subplot(nRows, nCols, i + 1)
                    ax.set_facecolor('#f8f8f8')

                    msk = roi.mask
                    if msk.shape != (H, W):
                        from scipy.ndimage import zoom
                        zy = H / max(msk.shape[0], 1)
                        zx = W / max(msk.shape[1], 1)
                        msk = zoom(msk.astype(float), (zy, zx), order=1) > 0.5

                    if not msk.any():
                        ax.text(0.5, 0.5, 'No pixels', ha='center', va='center',
                                transform=ax.transAxes, fontsize=ticks_fs)
                        _style_ax(ax, roi.name)
                        continue

                    flat     = img_sl.reshape(-1, nX)
                    sigs     = flat[msk.ravel()].astype(float)
                    sig_mean = sigs.mean(axis=0)
                    sig_std  = sigs.std(axis=0)

                    if _color_override[0] is not None:
                        color = _color_override[0]
                    else:
                        color = roi.color if hasattr(roi, 'color') else f'C{i % 10}'
                    ax.fill_between(x_vals, sig_mean - sig_std, sig_mean + sig_std,
                                    color=color, alpha=0.2)
                    ax.plot(x_vals, sig_mean, 'o', markersize=5, color=color,
                            label=roi.name)
                    val, curve, _warn = _fit_and_curve(sig_mean)
                    if val > 0:
                        ax.plot(x_fine, curve, '-', color=color, linewidth=2,
                                label=f'Fit: {val:.1f} ms')
                        if _warn:
                            ax.text(0.03, 0.04, _warn, transform=ax.transAxes,
                                    fontsize=max(7, ticks_fs - 1), color='#b00',
                                    va='bottom', ha='left')

                    _style_ax(ax, roi.name)
                    ax.legend(prop=_fp(legend_fs), frameon=False)

            fig.tight_layout()

            w = QWidget()
            wl = QVBoxLayout(w)
            apply_fig_dark_theme(fig, chk_dark_bg.isChecked())
            fc = FigureCanvas(fig)
            wl.addWidget(fc, stretch=1)
            btn_save = _QPB(f"Save {panel_title} figure…")
            def _save(checked=False, _fig=fig, _t=panel_title):
                from my_gui.fig_export import save_figure, FIG_EXPORT_FILTER
                p, _ = QFileDialog.getSaveFileName(
                    dlg, f"Save {_t}", f"{_t}.png",
                    FIG_EXPORT_FILTER)
                if p:
                    save_figure(_fig, p, dpi=300)
            btn_save.clicked.connect(_save)
            wl.addWidget(btn_save)
            return w

        def _get_font_params():
            return dict(
                title_fs    = sp_title.value(),
                suptitle_fs = sp_suptitle.value(),
                axes_fs     = sp_axes.value(),
                ticks_fs    = sp_ticks.value(),
                legend_fs   = sp_legend.value(),
                font_fam    = combo_ff.currentText(),
            )

        _PER_TAB = 8   # max ROIs per sub-tab (≤4 cols × ≤2 rows)

        def _add_paginated(img_sl, x_vals, x_label, kind, bruker, base_title,
                           tab_label, fp):
            """Split ROIs into pages of ≤8 and add one sub-tab per page."""
            _rois = rois if rois else []
            if not _rois:
                tabs.addTab(
                    _make_fit_panel(img_sl, x_vals, x_label, kind, bruker,
                                    base_title, [], **fp),
                    tab_label)
                return
            n_pages = (len(_rois) + _PER_TAB - 1) // _PER_TAB
            for p in range(n_pages):
                chunk = _rois[p * _PER_TAB:(p + 1) * _PER_TAB]
                suffix = "" if n_pages == 1 else f"  (page {p + 1}/{n_pages})"
                lbl    = tab_label if n_pages == 1 else f"{tab_label} ({p + 1})"
                tabs.addTab(
                    _make_fit_panel(img_sl, x_vals, x_label, kind, bruker,
                                    base_title + suffix, chunk, **fp),
                    lbl)

        def _rebuild_tabs():
            """Rebuild all tab panels with current font settings."""
            while tabs.count():
                tabs.removeTab(0)
            fp = _get_font_params()
            t1_bruker = (self.combo_t1_fmt.currentIndex() == 0)
            t2_bruker = (self.combo_t2_fmt.currentIndex() == 0)
            # Label the x-axis "Recovery/Prep time" when a GE/Siemens schedule is set.
            t1_sched = (not t1_bruker) and self._parse_schedule(self.le_t1_sched.text()) is not None
            t2_sched = (not t2_bruker) and self._parse_schedule(self.le_t2_sched.text()) is not None
            t1_xlabel = "Recovery time (ms)" if t1_sched else "TR (ms)"
            t2_xlabel = "Prep time (ms)" if t2_sched else "TE (ms)"
            if has_t1:
                _add_paginated(self._t1_img[:, :, sl, :], self._t1_trs, t1_xlabel,
                               "t1", t1_bruker, "T1 Recovery Curves", "T1 Fits", fp)
            if has_t2:
                _add_paginated(self._t2_img[:, :, sl, :], self._t2_tes, t2_xlabel,
                               "t2", t2_bruker, "T2 Decay Curves", "T2 Fits", fp)

        # Auto-apply: any font change rebuilds the figures (debounced so holding
        # a spinbox arrow does not trigger a redraw storm).  No Apply button.
        from PyQt6.QtCore import QTimer as _QTimer
        _font_timer = _QTimer(dlg)
        _font_timer.setSingleShot(True)
        _font_timer.setInterval(180)
        _font_timer.timeout.connect(_rebuild_tabs)
        def _sched_font(*_):
            _font_timer.start()
        combo_ff.currentIndexChanged.connect(_sched_font)
        for _sp in (sp_title, sp_suptitle, sp_axes, sp_ticks, sp_legend):
            _sp.valueChanged.connect(_sched_font)
        chk_dark_bg.toggled.connect(_sched_font)
        _rebuild_tabs()   # initial render with defaults
        dlg.show()

    # ── Data cursor ───────────────────────────────────────────────────────────

    def _toggle_datacursor(self, enabled: bool):
        if enabled:
            if self._dc_cid is None:
                self._dc_cid = self.canvas.mpl_connect(
                    "motion_notify_event", self._on_dc_hover
                )
            self._dc_annot = None
        else:
            if self._dc_cid is not None:
                self.canvas.mpl_disconnect(self._dc_cid)
                self._dc_cid = None
            if self._dc_annot is not None:
                try:
                    self._dc_annot.set_visible(False)
                    self.canvas.draw_idle()
                except Exception:
                    pass
                self._dc_annot = None

    def _on_dc_hover(self, event):
        ax = self.canvas._ax
        if event.inaxes is not ax or self.canvas._img_data is None:
            if self._dc_annot is not None:
                self._dc_annot.set_visible(False)
                self.canvas.draw_idle()
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        xi, yi = int(round(x)), int(round(y))
        img = self.canvas._img_data
        H, W = img.shape[:2]
        if 0 <= yi < H and 0 <= xi < W:
            val = img[yi, xi]
            if self._dc_annot is None:
                self._dc_annot = ax.annotate(
                    "", xy=(0, 0), xytext=(14, 14),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", ec="#4a9eff", alpha=0.9),
                    fontsize=8, color="white",
                    arrowprops=dict(arrowstyle="->", color="#4a9eff"),
                    visible=False,
                )
            self._dc_annot.xy = (x, y)
            self._dc_annot.set_text(f"x={xi}  y={yi}\nval = {val:.4g}")
            self._dc_annot.set_visible(True)
        else:
            if self._dc_annot is not None:
                self._dc_annot.set_visible(False)
        self.canvas.draw_idle()
