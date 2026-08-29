"""
quesp_tab.py
QUESP (Quantitative Exchange Saturation Transfer using Pulsed and CW irradiation)
analysis tab.

Ports the MATLAB pipeline from:
  QUESP_load_proc.m  — data loading, pairing pos/neg, thresholding, voxelwise fit
  QUESPfcn.m         — three fitting models: Regular, Inverse, OmegaPlot
  QUESPfitting.m     — parameter organisation wrapper

Physics (CW approximation):
  MTRRex = fs·ksw·w1² / (w1² + ksw²) / R1A

Inverse model (linear in 1/w1²):
  1/MTRRex = (R1A·ksw/fs)·(1/w1²) + R1A/(fs·ksw)
  slope a = R1A·ksw/fs,  intercept b = R1A/(fs·ksw)
  → ksw = √(a/b),   fs = R1A / √(a·b)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QLineEdit, QGroupBox,
    QCheckBox, QDoubleSpinBox, QSpinBox, QComboBox,
    QProgressBar, QFileDialog, QSizePolicy, QScrollArea,
    QRadioButton, QButtonGroup, QTextEdit,
    QSlider, QFrame, QDialog, QFormLayout, QDialogButtonBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from my_gui.pool_dialog import PoolSelectionWidget, POOL_CATALOG, default_pools

from my_gui.roi_tools import ROICanvas, roi_union_mask
from my_gui.roi_manager import apply_analysis_mask
from my_gui.plot_custom_bar import PlotCustomBar, WindowLevelToolButton


# ─────────────────────────────────────────────────────────────────────────────
# Physics helpers
# ─────────────────────────────────────────────────────────────────────────────

_GAMMA_RAD_PER_UT = 267.522  # rad s⁻¹ μT⁻¹


def _w1_rad_per_s(b1_uT: float) -> float:
    """Convert B1 peak amplitude in μT to angular frequency in rad/s."""
    return _GAMMA_RAD_PER_UT * b1_uT


def _mtr_rex_cw(w1: np.ndarray, fs: float, ksw: float, R1A: float) -> np.ndarray:
    """CW MTRRex model.  w1 in rad/s, ksw in s⁻¹, R1A in s⁻¹."""
    return fs * ksw * w1**2 / (w1**2 + ksw**2) / R1A


def _mtr_asym_regular(w1: np.ndarray, fs: float, ksw: float,
                       R1A: float, tp_s: float, rd_s: float,
                       zi_override: float | None = None) -> np.ndarray:
    """
    Full MATLAB 'Regular' QUESP model (QUESPfcn.m, case 'Regular').
    Uses MTR_asym = Zref - Zlab with temporal decay.

      Reff = R1A + fs·ksw·w1² / (w1² + ksw²)
      Zi   = 1 - exp(-R1A · rd)   ← magnetisation at start of saturation,
                                     after recovering from Z=0 for time rd.
             General form: Zi = 1 - (1-Z_prev) * exp(-R1A * rd)
             (Z_prev = 0 for fully saturated prior state, = 1 for full recovery)
      MTR  = fs·ksw·w1²/(w1²+ksw²)/Reff
             - (Zi - R1A/Reff)·exp(-Reff·tp)
             + (Zi - 1)·exp(-R1A·tp)

    w1 in rad/s, ksw in s⁻¹, R1A in s⁻¹, tp_s/rd_s in seconds.
    zi_override: if provided, use this value directly instead of computing from rd_s.
    """
    import math
    if zi_override is not None:
        Zi = float(zi_override)
    else:
        # Standard: assume spins start from Z=0 (fully saturated/readout) and
        # recover for rd_s before the next saturation. Matches MATLAB P.Zi when
        # P.Zi = 1 - exp(-R1A * Trec) and the readout leaves Z≈0.
        Zi = 1.0 - math.exp(-R1A * rd_s)
    Reff = R1A + fs * ksw * w1**2 / (w1**2 + ksw**2 + 1e-30)
    ss   = fs * ksw * w1**2 / (w1**2 + ksw**2 + 1e-30) / (Reff + 1e-30)
    mtr  = (ss
            - (Zi - R1A / (Reff + 1e-30)) * np.exp(-Reff * tp_s)
            + (Zi - 1.0) * np.exp(-R1A * tp_s))
    return np.clip(mtr, 0, None)


def _mtr_rex_pulsed(w1: np.ndarray, fs: float, ksw: float, R1A: float,
                    tp_s: float, rd_s: float) -> np.ndarray:
    """
    Pulsed MTRRex — matches MATLAB QUESPfcn.m pulsed inverse model exactly.

    MATLAB formula:
        c1  = sqrt(2π) / (2×2.92)
        c22 = (c1 × √(√2))²
        MTRRex = DC × c1 × fs×ksw×w1² / (w1² + ksw²×c22) / R1A
    """
    import math
    c1  = math.sqrt(2.0 * math.pi) / (2.0 * 2.92)
    c22 = (c1 * math.sqrt(math.sqrt(2.0))) ** 2
    DC  = tp_s / (tp_s + rd_s)
    return DC * c1 * fs * ksw * w1**2 / (w1**2 + ksw**2 * c22) / R1A


def _super_lorentzian(dw_rad: float, T2_s: float,
                      B0_MHz: float = 400.0) -> float:
    """
    Super-Lorentzian lineshape for MT pool.
    Faithful Python port of SimulationParameters.cpp
    (InterpolateSuperLorentzianShape + CubicHermiteSplineInterpolation).

    Uses 101-point powder average (u = i×0.01, i = 0…100).
    Pole region |dw| < 1 ppm (= 2π·B0_MHz rad/s) uses the C++
    Cubic Hermite Spline with tangentWeight = 30.

    Parameters
    ----------
    dw_rad  : frequency offset between RF and MT pool (rad/s)
    T2_s    : MT pool T2 (s)
    B0_MHz  : Larmor frequency (MHz) — defines 1-ppm pole cutoff
    Returns lineshape G [s].
    """
    u    = np.arange(101) * 0.01          # u = 0, 0.01, …, 1.0
    pcu2 = np.abs(3.0 * u**2 - 1.0)
    pcu2 = np.where(pcu2 < 1e-10, 1e-10, pcu2)

    def _intg(dw: float) -> float:
        return float(
            np.sum(np.sqrt(2.0 / np.pi) * T2_s / pcu2
                   * np.exp(-2.0 * (dw * T2_s / pcu2)**2))
            * np.pi * 0.01
        )

    omega0 = 2.0 * np.pi * B0_MHz    # 1 ppm in rad/s at this field
    if abs(dw_rad) >= omega0:
        return _intg(dw_rad)

    # Cubic Hermite Spline (C++ CubicHermiteSplineInterpolation, tangentWeight=30)
    px = np.array([-300.0 - omega0, -100.0 - omega0,
                    100.0 + omega0,  300.0 + omega0])
    py = np.array([_intg(p) for p in px])
    p0y, p1y = py[1], py[2]
    d0y = 30.0 * (p0y - py[0])
    d1y = 30.0 * (py[3] - p1y)
    cs  = abs((dw_rad - px[1] + 1.0) / (px[2] - px[1] + 1.0))
    c3, c2 = cs**3, cs**2
    return ((2*c3 - 3*c2 + 1)*p0y + (-2*c3 + 3*c2)*p1y
            + (c3 - 2*c2 + cs)*d0y + (c3 - c2)*d1y)


def _fit_inverse(mtr_rex: np.ndarray, w1_vals: np.ndarray,
                  R1A: float, pulsed: bool = False,
                  tp_s: float = 0.1, rd_s: float = 3.0) -> tuple[float, float, float]:
    """
    Inverse QUESP fit — nonlinear CW/pulsed model fit to MTRRex data.
    MTRRex = fb*kb*w1^2/(w1^2+kb^2)/R1A  (matches MATLAB QUESPfcn 'Inverse').
    Returns (fs, ksw, R²). Returns (0, 0, 0) on failure.
    """
    valid = (mtr_rex > 0) & np.isfinite(mtr_rex) & (w1_vals > 0)
    if valid.sum() < 2:
        return 0.0, 0.0, 0.0
    w1v = w1_vals[valid]
    yv  = mtr_rex[valid]

    if pulsed:
        def model(w1, fs, ksw):
            return _mtr_rex_pulsed(w1, fs, ksw, R1A, tp_s, rd_s)
    else:
        def model(w1, fs, ksw):
            return _mtr_rex_cw(w1, fs, ksw, R1A)

    # Try multiple starting points — bounds match MATLAB QUESPfcn.m defaults
    # fb: [1.35e-5, 1.35e-2],  ksw: [0, 150 000 s⁻¹]
    best = (0.0, 0.0, 0.0)
    for fs0, ksw0 in [(1.35e-4, 4000.0), (1e-4, 2000.0), (5e-4, 8000.0), (1e-3, 1000.0)]:
        try:
            popt, _ = curve_fit(
                model, w1v, yv,
                p0=[fs0, ksw0],
                bounds=([1.35e-5, 0.0], [1.35e-2, 150000.0]),
                maxfev=5000,
            )
            fs_fit, ksw_fit = float(popt[0]), float(popt[1])
            y_pred = model(w1v, fs_fit, ksw_fit)
            ss_res = float(np.sum((yv - y_pred) ** 2))
            ss_tot = float(np.sum((yv - yv.mean()) ** 2))
            rsq = float(np.clip(1.0 - ss_res / (ss_tot + 1e-12), 0, 1))
            if rsq > best[2]:
                best = (max(fs_fit, 0.0), max(ksw_fit, 0.0), rsq)
        except Exception:
            continue
    return best


def _fit_regular(mtr_asym: np.ndarray, w1_vals: np.ndarray,
                  R1A: float, pulsed: bool = False,
                  tp_s: float = 0.1, rd_s: float = 3.0) -> tuple[float, float, float]:
    """
    Regular QUESP fit — ports MATLAB QUESPfcn.m 'Regular' case.
    Input: mtr_asym = Zref − Zlab  (MTR asymmetry, NOT MTRRex).
    Model: full temporal decay with finite Zi = 1 − exp(−R1A·rd).
    Returns (fs, ksw, R²).
    """
    valid = np.isfinite(mtr_asym) & (mtr_asym > 0) & (w1_vals > 0)
    if valid.sum() < 2:
        return 0.0, 0.0, 0.0
    w1v = w1_vals[valid]
    yv  = mtr_asym[valid]

    # Always use the full temporal model (includes pulsed via Zi)
    def model(w1, fs, ksw):
        return _mtr_asym_regular(w1, fs, ksw, R1A, tp_s, rd_s)

    best = (0.0, 0.0, 0.0)
    for fs0, ksw0 in [(1.35e-4, 4000.0), (1e-4, 2000.0), (5e-4, 8000.0), (1e-3, 1000.0)]:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    model, w1v, yv,
                    p0=[fs0, ksw0],
                    bounds=([1.35e-5, 0.0], [1.35e-2, 150000.0]),
                    maxfev=5000,
                )
            fs_fit, ksw_fit = float(popt[0]), float(popt[1])
            y_pred = model(w1v, fs_fit, ksw_fit)
            ss_res = float(np.sum((yv - y_pred)**2))
            ss_tot = float(np.sum((yv - yv.mean())**2))
            rsq = float(np.clip(1.0 - ss_res / (ss_tot + 1e-12), 0, 1))
            if rsq > best[2]:
                best = (max(fs_fit, 0.0), max(ksw_fit, 0.0), rsq)
        except Exception:
            continue
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Multi-B1 add-scan dialog
# ─────────────────────────────────────────────────────────────────────────────

class _MultiB1AddDialog(QDialog):
    """Dialog for adding a single scan (one B1 power level) to the multi-B1 list."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Scan — Multi-B1 Mode")
        self.setMinimumWidth(480)

        vl = QVBoxLayout(self)

        # Path row
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Scan path:"))
        self._le_path = QLineEdit()
        self._le_path.setReadOnly(True)
        self._le_path.setPlaceholderText("Select the scan folder or .mat file…")
        path_row.addWidget(self._le_path, stretch=1)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        path_row.addWidget(btn_browse)
        vl.addLayout(path_row)

        # Auto-detected B1 label
        self._lbl_detected = QLabel("Detected B1: —")
        self._lbl_detected.setStyleSheet("color: gray; font-size: 10px;")
        vl.addWidget(self._lbl_detected)

        # B1 spinbox row
        b1_row = QHBoxLayout()
        b1_row.addWidget(QLabel("B1 power (µT):"))
        self._spin_b1 = QDoubleSpinBox()
        self._spin_b1.setRange(0.01, 50.0)
        self._spin_b1.setValue(1.0)
        self._spin_b1.setDecimals(2)
        self._spin_b1.setSuffix(" µT")
        b1_row.addWidget(self._spin_b1)
        b1_row.addStretch()
        vl.addLayout(b1_row)

        # Bruker version kept in the backend (auto/PV360) but not shown.
        self._combo_pv = QComboBox()
        self._combo_pv.addItems(["PV360", "PV6 / PV7"])
        self._combo_pv.setVisible(False)

        # Standard buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        vl.addWidget(btns)

    def _browse(self):
        # Try directory first (Bruker pdata/1/ folder)
        d = QFileDialog.getExistingDirectory(self, "Select Bruker pdata/1/ folder", "")
        if d:
            self._le_path.setText(d)
            self._try_autodetect(d)
            return
        # Fallback: open file dialog for .mat/.npz
        fn, _ = QFileDialog.getOpenFileName(
            self, "Select .mat file", "",
            "Data files (*.mat);;All files (*)"
        )
        if fn:
            self._le_path.setText(fn)
            self._lbl_detected.setText("Detected B1: — (file, set manually)")
            self._lbl_detected.setStyleSheet("color: gray; font-size: 10px;")

    def _try_autodetect(self, path: str):
        try:
            from my_gui.bruker_reader import read_2dseq_cest
            pv360 = self._combo_pv.currentText() == "PV360"
            _, _, info = read_2dseq_cest(path, pv360=pv360)
            b1 = float(info.get('satpwr_uT', 0.0))
            if b1 > 0:
                self._spin_b1.setValue(b1)
                self._lbl_detected.setText(f"Detected B1: {b1:.4g} µT (pre-filled)")
                self._lbl_detected.setStyleSheet("color: #4ec9b0; font-size: 10px;")
            else:
                self._lbl_detected.setText("Detected B1: not found — set manually")
                self._lbl_detected.setStyleSheet("color: gray; font-size: 10px;")
        except Exception as exc:
            self._lbl_detected.setText(f"Auto-detect failed: {exc}")
            self._lbl_detected.setStyleSheet("color: orange; font-size: 10px;")

    def get_values(self) -> tuple:
        """Returns (path: str, b1_ut: float, pv360: bool)."""
        return (
            self._le_path.text(),
            self._spin_b1.value(),
            self._combo_pv.currentText() == "PV360",
        )


# ─────────────────────────────────────────────────────────────────────────────
# QThread worker
# ─────────────────────────────────────────────────────────────────────────────

class QUESPFitWorker(QThread):
    """Voxelwise QUESP fitting on a background thread."""

    finished = pyqtSignal(object, object, object)   # fs_map, ksw_map, rsq_map
    progress  = pyqtSignal(int)
    error     = pyqtSignal(str)

    def __init__(
        self,
        quesp_eff:  np.ndarray,          # (H, W, N)  MTRRex = 1/Zlab − 1/Zref
        t1_map:     np.ndarray,          # (H, W)     T1 in ms
        w1_vals:    np.ndarray,          # (N,)       w1 in rad/s
        mask:       np.ndarray,          # (H, W) bool
        model:      str,                 # 'inverse' | 'regular'
        pulsed:     bool = False,
        tp_s:       float = 0.1,
        rd_s:       float = 3.0,
        quesp_asym: np.ndarray | None = None,  # (H, W, N) MTR_asym for Regular
    ):
        super().__init__()
        self._eff   = quesp_eff
        self._asym  = quesp_asym  # may be None — falls back to _eff
        self._t1    = t1_map
        self._w1    = w1_vals
        self._mask  = mask
        self._model = model
        self._pulsed = pulsed
        self._tp    = tp_s
        self._rd    = rd_s
        self._stop  = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            H, W, N = self._eff.shape
            fs_map  = np.zeros((H, W))
            ksw_map = np.zeros((H, W))
            rsq_map = np.zeros((H, W))

            ys, xs = np.where(self._mask)
            total   = len(ys)
            if total == 0:
                self.finished.emit(fs_map, ksw_map, rsq_map)
                return

            for idx, (i, j) in enumerate(zip(ys, xs)):
                if self._stop:
                    break
                t1_ms = float(self._t1[i, j])
                if t1_ms <= 0:
                    continue
                R1A = 1000.0 / t1_ms  # s⁻¹

                if self._model == 'inverse':
                    mtr = self._eff[i, j, :]   # MTRRex for Inverse
                    fs, ksw, rsq = _fit_inverse(
                        mtr, self._w1, R1A,
                        pulsed=self._pulsed,
                        tp_s=self._tp, rd_s=self._rd,
                    )
                else:
                    # Regular model uses MTR_asym = Zref − Zlab
                    mtr = (self._asym[i, j, :] if self._asym is not None
                           else self._eff[i, j, :])
                    fs, ksw, rsq = _fit_regular(
                        mtr, self._w1, R1A,
                        pulsed=self._pulsed,
                        tp_s=self._tp, rd_s=self._rd,
                    )

                fs_map[i, j]  = fs            # pool fraction (dimensionless)
                ksw_map[i, j] = ksw
                rsq_map[i, j] = rsq

                if (idx + 1) % max(1, total // 100) == 0:
                    self.progress.emit(int(100 * (idx + 1) / total))

            self.progress.emit(100)
            self.finished.emit(fs_map, ksw_map, rsq_map)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Tab
# ─────────────────────────────────────────────────────────────────────────────

class QUESPTab(QWidget):
    """QUESP analysis: load data → compute MTRRex → voxelwise fit → maps."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── data storage ──────────────────────────────────────────────────
        self._quesp_raw: np.ndarray | None = None    # (H, W, n_sl, n_sat) sat images only
        self._quesp_M0:  np.ndarray | None = None    # (H, W[, n_M0]) M0 reference images
        self._quesp_info: dict | None = None         # sat_amplitudes, sat_offsets, etc.
        self._quesp_eff:  np.ndarray | None = None   # (H, W, N) MTRRex  = 1/Zlab − 1/Zref
        self._quesp_asym: np.ndarray | None = None   # (H, W, N) MTR_asym = Zref − Zlab
        self._t1_map:    np.ndarray | None = None    # (H, W) ms
        self._b1_powers: list[float] = []            # μT
        self._w1_vals:   np.ndarray | None = None    # rad/s
        self._slice_idx: int = 0
        self._fs_map:    np.ndarray | None = None
        self._ksw_map:   np.ndarray | None = None
        self._rsq_map:   np.ndarray | None = None
        self._quesp_dir: str = ""
        self._worker = None
        self._dc_annot = None
        self._dc_cid: int | None = None
        self._datasets: list[dict] = []  # each: {'path': str, 'b1_ut': float, 'pv360': bool}

        # User-picked grayscale background for the "ROIs + Bkg" overlay.
        self._roi_bg_img = None
        self._scan_paths_getter = None

        # Pool & fit options (stored as hidden widgets so existing logic reads them)
        self._quesp_pools: list[str] = ["amide", "amine", "OH", "Trp", "MT"]
        # Hidden widgets that hold fit option state (dialog reads/writes these)
        self.rb_pseudovoigt = QRadioButton("Pseudo-Voigt")
        self.rb_lorentzian  = QRadioButton("Lorentzian")
        self.rb_pseudovoigt.setChecked(True)
        self._peak_fit_group = QButtonGroup(self)
        self._peak_fit_group.addButton(self.rb_pseudovoigt)
        self._peak_fit_group.addButton(self.rb_lorentzian)
        self.chk_keep_gl = QCheckBox("Keep GL character same across peaks")
        self.chk_keep_gl.setChecked(True)
        self.chk_mt_superlorentz = QCheckBox("Super-Lorentzian lineshape for MT pool")
        self.chk_mt_superlorentz.setChecked(True)
        self.chk_multipool = QCheckBox("Use multi-pool lineshape fitting")
        self.chk_multipool.setChecked(True)
        self.combo_model = QComboBox()
        self.combo_model.addItems(["Inverse (linear)", "Regular (nonlinear)"])
        self.chk_pulsed = QCheckBox("Apply pulsed correction (uses tp, td)")
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(0.5, 2.0)
        self.spin_thresh.setValue(1.05)
        self.spin_thresh.setSingleStep(0.01)
        self.spin_thresh.setDecimals(2)
        self.spin_t1_thresh = QDoubleSpinBox()
        self.spin_t1_thresh.setRange(0.0, 5000.0)
        self.spin_t1_thresh.setValue(100.0)
        self.spin_t1_thresh.setSingleStep(50.0)

        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────
    # ROIs + Bkg background picker
    # ─────────────────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── left panel ────────────────────────────────────────────────────
        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.setSpacing(6)
        left_l.setContentsMargins(6, 6, 6, 6)

        scroll = QScrollArea()
        scroll.setWidget(left_w)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(400)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_l.addWidget(self._build_data_group())
        # Multi-B1 scans get their own box (built after data group so btn_multi_b1
        # already exists).
        left_l.addWidget(self._build_multib1_group())
        # T1 Map (R1A) group is now placed under the "QUESP Processing & Fitting"
        # button, inside _build_processing_group().
        left_l.addWidget(self._build_processing_group())

        left_l.addStretch()

        # ── right panel ───────────────────────────────────────────────────
        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setSpacing(4)
        right_l.setContentsMargins(6, 6, 6, 6)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Display:"))
        self.combo_display = QComboBox()
        self.combo_display.addItems([
            "Reference Image",
            "QUESP Images",
            "fs Map (mM)",
            "ksw Map (s⁻¹)",
            "R² Map",
            "CEST Peak Range",
        ])
        self.combo_display.currentIndexChanged.connect(self._refresh_display)
        ctrl_row.addWidget(self.combo_display, stretch=1)

        # "ROIs + Bg" — colored map only inside the drawn ROIs, over the 1st
        # raw image shown in grayscale outside the ROIs.
        self.chk_roi_bg = QCheckBox("ROIs + Bkg")
        self.chk_roi_bg.setToolTip(
            "Show the colored map only inside the ROIs, over the 1st raw image "
            "as a gray background.")
        self.chk_roi_bg.toggled.connect(self._refresh_display)
        ctrl_row.addWidget(self.chk_roi_bg)

        # "Bkg…" — pick ANY image (from the Scan Directory or browse) as the
        # grayscale background for the "ROIs + Bkg" overlay.
        self.btn_roi_bg = QPushButton("Bkg…")
        self.btn_roi_bg.setToolTip(
            "Pick the grayscale background image (from the Scan Directory) for "
            "the 'ROIs + Bkg' overlay.")
        self.btn_roi_bg.clicked.connect(self._pick_roi_bg)
        ctrl_row.addWidget(self.btn_roi_bg)

        btn_export = QPushButton("Export figure…")
        btn_export.clicked.connect(self._export_figure)
        ctrl_row.addWidget(btn_export)
        self.btn_hide_rois = QPushButton("Hide ROIs")
        self.btn_hide_rois.setCheckable(True)
        self.btn_hide_rois.setToolTip("Toggle ROI overlay visibility on the map")
        self.btn_hide_rois.clicked.connect(self._toggle_hide_rois)
        # Window/Level (brightness–contrast) drag tool — OsiriX-style.
        self.btn_contrast = WindowLevelToolButton()
        self.btn_contrast.toggled.connect(
            lambda checked: self.canvas.set_wl_active(
                checked, on_change=lambda a, b: self.plot_bar.set_clim(a, b)))
        ctrl_row.addWidget(self.btn_contrast)
        ctrl_row.addWidget(self.btn_hide_rois)
        right_l.addLayout(ctrl_row)

        # ── B1 image slider frame (visible only for "QUESP Images") ──
        self._img_slider_frame = QFrame()
        _sf_lay = QHBoxLayout(self._img_slider_frame)
        _sf_lay.setContentsMargins(4, 2, 4, 2)
        _sf_lay.addWidget(QLabel("B1 index:"))
        self._img_slider = QSlider(Qt.Orientation.Horizontal)
        self._img_slider.setMinimum(0)
        self._img_slider.setMaximum(0)
        self._img_slider.setValue(0)
        self._img_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._img_slider.setTickInterval(1)
        self._img_slider.valueChanged.connect(self._on_img_slider)
        _sf_lay.addWidget(self._img_slider, stretch=1)
        self._img_slider_label = QLabel("B1 = — µT  (0/0)")
        self._img_slider_label.setMinimumWidth(180)
        _sf_lay.addWidget(self._img_slider_label)
        self._img_slider_frame.hide()
        right_l.addWidget(self._img_slider_frame)

        # ── Figure Customization (collapsible — mirrors the MRF Viewer) ───────
        from PyQt6.QtWidgets import QGroupBox as _QGBQ, QWidget as _QWQ
        self.grp_fig_custom = _QGBQ("Figure Customization")
        _gfc = QVBoxLayout(self.grp_fig_custom); _gfc.setContentsMargins(8, 6, 8, 6)
        self.chk_fig_custom = QCheckBox("Enable Figure Customization")
        self.chk_fig_custom.setToolTip(
            "Show the title, colormap, colour-bar limit and font controls "
            "(including Bg and Log map).")
        _gfc.addWidget(self.chk_fig_custom)
        self._fig_custom_panel = _QWQ()
        self._fcp_lay = QVBoxLayout(self._fig_custom_panel)
        self._fcp_lay.setContentsMargins(0, 0, 0, 0)
        _gfc.addWidget(self._fig_custom_panel)
        self._fig_custom_panel.setVisible(False)
        self.chk_fig_custom.toggled.connect(self._fig_custom_panel.setVisible)
        right_l.addWidget(self.grp_fig_custom)

        # Custom title row
        from PyQt6.QtWidgets import QLineEdit as _QLEQ
        _qtitle_row = QHBoxLayout()
        _qtitle_row.addWidget(QLabel("Title:"))
        self.edit_map_title = _QLEQ()
        self.edit_map_title.setPlaceholderText("Custom map title (leave blank for default)")
        _qtitle_row.addWidget(self.edit_map_title, stretch=1)
        self._fcp_lay.addLayout(_qtitle_row)

        from my_gui.format_bar import add_title_format_bar, connect_title_debounced
        connect_title_debounced(self.edit_map_title, self._refresh_display)

        # Colormap + color-limits bar — fonts line on top (with the B/I/x²/x₂
        # title buttons on its right), colour-bar line below.
        self.plot_bar = PlotCustomBar(default_cmap="viridis", fonts_first=True)
        self.plot_bar.applied.connect(self._refresh_display)
        add_title_format_bar(self.edit_map_title, None,
                             target_row=self.plot_bar.font_row(),
                             default_getter=lambda: getattr(self.canvas, "_last_title", ""))
        self._fcp_lay.addWidget(self.plot_bar)

        # "Bg" — black figure background for slides (maps/curves stay identical).
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
                _b_idx = _i
                break
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
        self.canvas.mpl_connect('scroll_event', self._on_canvas_scroll)
        right_l.addWidget(self.canvas, stretch=1)

        # ROI Spectra + ROI Statistics — side by side below the image (matches the T1/T2 tab)
        _roi_btn_row = QHBoxLayout()
        self.btn_roi_spectra = QPushButton("ROI Spectra")
        self.btn_roi_spectra.clicked.connect(self._show_roi_spectra)
        _roi_btn_row.addWidget(self.btn_roi_spectra)

        self.btn_roi_table = QPushButton("ROI Statistics")
        self.btn_roi_table.clicked.connect(self._show_roi_table)
        _roi_btn_row.addWidget(self.btn_roi_table)
        right_l.addLayout(_roi_btn_row)

        # Data cursor checkbox
        from my_gui.plot_custom_bar import DataCursorToolButton
        self.chk_datacursor = DataCursorToolButton()
        self.chk_datacursor.toggled.connect(self._toggle_datacursor)
        ctrl_row.insertWidget(ctrl_row.indexOf(self.btn_contrast) + 1, self.chk_datacursor)

        splitter.addWidget(scroll)
        splitter.addWidget(right_w)
        splitter.setSizes([390, 610])

    # ─────────────────────────────────────────────────────────────────────
    # Group builders
    # ─────────────────────────────────────────────────────────────────────

    def _build_data_group(self) -> QGroupBox:
        grp = QGroupBox("QUESP Data")
        v = QVBoxLayout(grp)
        v.setSpacing(4)

        # ── Scan-source config lives in a "Scan Parameters" dialog ─────────
        # (platform, scan folder, QUESP acquisition order, power levels, slice).
        from PyQt6.QtWidgets import QDialog as _QDlg, QDialogButtonBox as _QDBB
        self._scan_params_dialog = _QDlg(self)
        self._scan_params_dialog.setWindowTitle("QUESP Parameters")
        self._scan_params_dialog.setMinimumWidth(500)
        _dl = QVBoxLayout(self._scan_params_dialog)

        # Platform / vendor selector — shown in the main panel.
        _plat_row = QHBoxLayout()
        _plat_row.addWidget(QLabel("Platform:"))
        self.combo_quesp_vendor = QComboBox()
        self.combo_quesp_vendor.addItems(["Bruker", "GE / Siemens"])
        self.combo_quesp_vendor.setFixedWidth(150)
        self.combo_quesp_vendor.currentIndexChanged.connect(self._on_quesp_vendor_changed)
        _plat_row.addWidget(self.combo_quesp_vendor)
        _plat_row.addStretch()
        v.addLayout(_plat_row)

        # ── Stacked data source widget ────────────────────────────────────
        from PyQt6.QtWidgets import QStackedWidget as _QSW
        self._quesp_data_stack = _QSW()

        # ── Page 0: Bruker ────────────────────────────────────────────────
        bruker_page = QWidget()
        bruker_lay  = QVBoxLayout(bruker_page)
        bruker_lay.setContentsMargins(0, 0, 0, 0)
        bruker_lay.setSpacing(4)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Folder:"))
        self.le_quesp_dir = QLineEdit()
        self.le_quesp_dir.setReadOnly(True)
        self.le_quesp_dir.setPlaceholderText("Select the scan folder…")
        row1.addWidget(self.le_quesp_dir, stretch=1)
        btn_browse_bruker = QPushButton("Browse…")
        btn_browse_bruker.clicked.connect(self._browse_quesp)
        row1.addWidget(btn_browse_bruker)
        bruker_lay.addLayout(row1)

        # Bruker version kept in the backend (auto/PV360) but not shown.
        self.combo_pv = QComboBox()
        self.combo_pv.addItems(["PV360", "PV6 / PV7"])
        self.combo_pv.setVisible(False)

        self._quesp_data_stack.addWidget(bruker_page)   # index 0

        # ── Page 1: GE / Siemens ──────────────────────────────────────────
        gs_page = QWidget()
        gs_lay  = QVBoxLayout(gs_page)
        gs_lay.setContentsMargins(0, 0, 0, 0)
        gs_lay.setSpacing(4)

        # Study folder row
        gs_dir_row = QHBoxLayout()
        gs_dir_row.addWidget(QLabel("Data folder:"))
        self.le_quesp_gs_dir = QLineEdit()
        self.le_quesp_gs_dir.setReadOnly(True)
        self.le_quesp_gs_dir.setPlaceholderText("Select folder with DICOM / NIfTI…")
        gs_dir_row.addWidget(self.le_quesp_gs_dir, stretch=1)
        btn_browse_gs = QPushButton("Browse…")
        btn_browse_gs.clicked.connect(self._browse_quesp_gs)
        gs_dir_row.addWidget(btn_browse_gs)
        gs_lay.addLayout(gs_dir_row)

        # Format + Detect row
        gs_fmt_row = QHBoxLayout()
        gs_fmt_row.addWidget(QLabel("Format:"))
        self.combo_quesp_gs_format = QComboBox()
        self.combo_quesp_gs_format.addItems(["DICOM (.dcm)", "NIfTI (.nii)"])
        self.combo_quesp_gs_format.setFixedWidth(130)
        self.combo_quesp_gs_format.setToolTip(
            "DICOM: folder or subfolders contain .dcm files.\n"
            "NIfTI: folder contains .nii / .nii.gz files."
        )
        gs_fmt_row.addWidget(self.combo_quesp_gs_format)
        btn_gs_detect = QPushButton("↻  Detect")
        btn_gs_detect.setFixedHeight(24)
        btn_gs_detect.setStyleSheet(
            "QPushButton { background: #333; color: #aaa; border: none; "
            "border-radius: 4px; padding: 2px 8px; font-size: 11px; }"
            "QPushButton:hover { background: #555; color: white; }"
        )
        btn_gs_detect.clicked.connect(self._refresh_quesp_gs_info)
        gs_fmt_row.addWidget(btn_gs_detect)
        gs_fmt_row.addStretch()
        gs_lay.addLayout(gs_fmt_row)

        # Auto-detected scanner info
        self.lbl_quesp_gs_info = QLabel("Browse a folder to auto-detect scanner info.")
        self.lbl_quesp_gs_info.setStyleSheet("font-size: 10px; color: #888;")
        self.lbl_quesp_gs_info.setWordWrap(True)
        gs_lay.addWidget(self.lbl_quesp_gs_info)

        # DICOM / NIfTI series list
        from PyQt6.QtWidgets import QListWidget as _LW2
        self.lst_quesp_gs_scans = _LW2()
        self.lst_quesp_gs_scans.setMaximumHeight(100)
        self.lst_quesp_gs_scans.setStyleSheet("""
            QListWidget { background: #1a1a1a; color: #ddd;
                          border: 1px solid #333; border-radius: 4px;
                          padding: 3px; font-family: Arial; font-size: 11px; }
            QListWidget::item { padding: 2px 5px; border-radius: 2px; }
            QListWidget::item:selected { background: #1565c0; color: white; }
            QListWidget::item:hover:!selected { background: #2a2a2a; }
        """)
        self.lst_quesp_gs_scans.setToolTip(
            "Select the series that contains the QUESP data.\n"
            "Its folder becomes the active path for 'Load QUESP Data'."
        )
        self.lst_quesp_gs_scans.currentRowChanged.connect(self._on_quesp_gs_row_changed)
        gs_lay.addWidget(self.lst_quesp_gs_scans)

        # Series detail label
        self.lbl_quesp_gs_detail = QLabel("")
        self.lbl_quesp_gs_detail.setStyleSheet("font-size: 10px; color: #888;")
        self.lbl_quesp_gs_detail.setWordWrap(True)
        gs_lay.addWidget(self.lbl_quesp_gs_detail)

        # Internal GE/Siemens state
        self._gs_quesp_study_dir: str = ""
        self._gs_quesp_scan_dirs: list[str] = []

        self._quesp_data_stack.addWidget(gs_page)   # index 1

        v.addWidget(self._quesp_data_stack)

        # Size the data-source stack to its CURRENT page — the Bruker page is a
        # single "Folder:" row while the GE/Siemens page is tall; without this the
        # short Bruker page is stretched to the tall page's height, leaving large
        # empty gaps above and below "Folder:".
        def _fit_quesp_stack(_i=None):
            _pg = self._quesp_data_stack.currentWidget()
            if _pg is not None:
                self._quesp_data_stack.setMaximumHeight(_pg.sizeHint().height() + 6)
        self._quesp_data_stack.currentChanged.connect(_fit_quesp_stack)
        _fit_quesp_stack()

        # ── QUESP Parameters dialog contents: acquisition + N + slice ──────
        _acq_row = QHBoxLayout()
        _acq_row.addWidget(QLabel("QUESP acquisition:"))
        self.combo_pair = QComboBox()
        self.combo_pair.addItems([
            "Alternating offsets (+/-/+/-/…)",
            "Sequential offsets (+/+/-/-/…)",
        ])
        _acq_row.addWidget(self.combo_pair, stretch=1)
        _dl.addLayout(_acq_row)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("# power levels N:"))
        self.spin_n_powers = QSpinBox()
        self.spin_n_powers.setRange(1, 50)
        self.spin_n_powers.setValue(6)
        row3.addWidget(self.spin_n_powers)
        row3.addWidget(QLabel("Slice:"))
        self.spin_slice = QSpinBox()
        self.spin_slice.setRange(1, 99)
        self.spin_slice.setValue(1)
        # Re-render when the slice changes (multi-slice de-tiled mosaic stacks)
        self.spin_slice.valueChanged.connect(self._refresh_display)
        row3.addWidget(self.spin_slice)
        row3.addStretch()
        _dl.addLayout(row3)

        _sp_close = _QDBB(_QDBB.StandardButton.Close)
        _sp_close.rejected.connect(self._scan_params_dialog.reject)
        _dl.addWidget(_sp_close)

        # ── Main panel: QUESP Parameters + Load QUESP Data (side by side) ───
        btn_row = QHBoxLayout()
        self.btn_scan_params = QPushButton("QUESP Parameters")
        self.btn_scan_params.setFixedHeight(32)
        self.btn_scan_params.setToolTip(
            "QUESP acquisition order, # power levels and slice.")
        self.btn_scan_params.clicked.connect(
            lambda: (self._scan_params_dialog.show(), self._scan_params_dialog.raise_()))
        btn_row.addWidget(self.btn_scan_params)

        self.btn_load = QPushButton("Load QUESP Data")
        self.btn_load.setFixedHeight(32)
        self.btn_load.clicked.connect(self._load_quesp)
        btn_row.addWidget(self.btn_load)
        v.addLayout(btn_row)

        # ── Multi-B1 mode — collapsed into a dialog behind one button ──────
        # (one scan per B1 power).  Keeps the panel uncluttered; the controls
        # live in the "Multi-B1 scan Parameters" dialog.
        from PyQt6.QtWidgets import QListWidget, QDialog, QDialogButtonBox
        self._multi_b1_dialog = QDialog(self)
        self._multi_b1_dialog.setWindowTitle("Multi-B1 Scan Parameters")
        self._multi_b1_dialog.setMinimumWidth(470)
        _mbl = QVBoxLayout(self._multi_b1_dialog)
        _mbl.addWidget(QLabel("Multi-B1 mode — one scan per B1 power."))

        self.lw_datasets = QListWidget()
        self.lw_datasets.setMinimumHeight(140)
        self.lw_datasets.setToolTip("Each entry: B1 (µT) — scan path")
        _mbl.addWidget(self.lw_datasets)

        _ds_btn_row = QHBoxLayout()
        self.btn_add_ds = QPushButton("Add Scan…")
        self.btn_add_ds.clicked.connect(self._add_dataset)
        _ds_btn_row.addWidget(self.btn_add_ds)
        self.btn_remove_ds = QPushButton("Remove Selected")
        self.btn_remove_ds.clicked.connect(self._remove_dataset)
        _ds_btn_row.addWidget(self.btn_remove_ds)
        _ds_btn_row.addStretch()
        _mbl.addLayout(_ds_btn_row)

        _offset_row = QHBoxLayout()
        _offset_row.addWidget(QLabel("Sat. offset ±"))
        self.spin_multi_offset = QDoubleSpinBox()
        self.spin_multi_offset.setRange(0.1, 20.0)
        self.spin_multi_offset.setValue(3.5)
        self.spin_multi_offset.setSingleStep(0.1)
        self.spin_multi_offset.setDecimals(2)
        _offset_row.addWidget(self.spin_multi_offset)
        _offset_row.addWidget(QLabel("ppm"))
        _offset_row.addStretch()
        _mbl.addLayout(_offset_row)

        self.btn_load_multi = QPushButton("Load Multi-B1 Power Scans")
        self.btn_load_multi.setFixedHeight(32)
        self.btn_load_multi.clicked.connect(self._load_multi_quesp)
        _mbl.addWidget(self.btn_load_multi)

        _mb_close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        _mb_close.rejected.connect(self._multi_b1_dialog.reject)
        _mbl.addWidget(_mb_close)

        # Button that opens the Multi-B1 dialog — lives in its OWN
        # "Multi-B1 Scans" box (built by _build_multib1_group), not here.
        self.btn_multi_b1 = QPushButton("Multi-B1 scan Parameters")
        self.btn_multi_b1.setFixedHeight(32)
        self.btn_multi_b1.clicked.connect(
            lambda: (self._multi_b1_dialog.show(), self._multi_b1_dialog.raise_()))

        self.lbl_data_status = QLabel("No data loaded.")
        self.lbl_data_status.setWordWrap(True)
        self.lbl_data_status.setStyleSheet("color: gray; font-size: 10px;")
        v.addWidget(self.lbl_data_status)

        return grp

    def _build_multib1_group(self) -> QGroupBox:
        """Separate 'Multi-B1 Scans' box (same style as 'QUESP Data') holding the
        button that opens the Multi-B1 scan-parameters dialog."""
        grp = QGroupBox("Multi-B1 Scans")
        v = QVBoxLayout(grp)
        v.setSpacing(4)
        v.addWidget(QLabel("One scan per B1 power — load a set to fit fs / ksw."))
        v.addWidget(self.btn_multi_b1)
        return grp

    def _build_t1_group(self) -> QGroupBox:
        grp = QGroupBox("T₁ Map (R₁A)")
        v = QVBoxLayout(grp)
        v.setSpacing(4)

        info = QLabel(
            "T<sub>1</sub> map is used to compute R<sub>1</sub>A = 1/T<sub>1</sub>. It is automatically\n"
            "populated from the T1/T2/B1 tab if a T1 fit has been run."
        )
        info.setStyleSheet("font-size: 10px; color: gray;")
        info.setWordWrap(True)
        v.addWidget(info)

        self.lbl_t1_status = QLabel("No T1 map loaded.")
        self.lbl_t1_status.setStyleSheet("color: gray; font-size: 10px;")
        v.addWidget(self.lbl_t1_status)

        row = QHBoxLayout()
        btn_t1_browse = QPushButton("Browse T1 map (.mat)…")
        btn_t1_browse.clicked.connect(self._browse_t1)
        row.addWidget(btn_t1_browse)
        row.addStretch()
        v.addLayout(row)

        row2 = QHBoxLayout()
        btn_t1_manual = QPushButton("Enter T1 value manually…")
        btn_t1_manual.setToolTip(
            "Enter a single T1 value (ms) to use as a uniform T1 map.\n"
            "Useful when no T1 map is available."
        )
        btn_t1_manual.clicked.connect(self._enter_t1_manually)
        row2.addWidget(btn_t1_manual)
        row2.addStretch()
        v.addLayout(row2)

        return grp

    def _build_params_group(self) -> QGroupBox:
        grp = QGroupBox("Measurement Parameters")
        v = QVBoxLayout(grp)
        v.setSpacing(4)

        # B1 powers
        row_b1 = QHBoxLayout()
        row_b1.addWidget(QLabel("B1 powers (μT):"))
        self.le_b1_powers = QLineEdit("0.5, 1.0, 1.5, 2.0, 2.5, 3.0")
        self.le_b1_powers.setToolTip("Peak B1 amplitude in μT for each power level, use (,) comma separated")
        row_b1.addWidget(self.le_b1_powers, stretch=1)
        v.addLayout(row_b1)

        # Saturation offset + peak range
        row_off = QHBoxLayout()
        row_off.addWidget(QLabel("Sat. offset (ppm):"))
        self.spin_offset = QDoubleSpinBox()
        self.spin_offset.setRange(0.1, 10.0)
        self.spin_offset.setValue(3.5)
        self.spin_offset.setSingleStep(0.1)
        self.spin_offset.setDecimals(2)
        self.spin_offset.setToolTip("Centre saturation frequency offset for CEST extraction (ppm)")
        row_off.addWidget(self.spin_offset)
        row_off.addWidget(QLabel("±"))
        self.spin_peak_half_bw = QDoubleSpinBox()
        self.spin_peak_half_bw.setRange(0.0, 3.0)
        self.spin_peak_half_bw.setValue(0.0)
        self.spin_peak_half_bw.setSingleStep(0.1)
        self.spin_peak_half_bw.setDecimals(2)
        self.spin_peak_half_bw.setFixedWidth(68)
        self.spin_peak_half_bw.setToolTip(
            "Half-bandwidth around the saturation offset for CEST peak extraction (ppm).\n"
            "0.0 = single point (default).\n"
            "E.g. 0.5 → average over [offset−0.5, offset+0.5] ppm.\n"
            "Applies when 'Load Multi-B1 Power Scans' is used."
        )
        row_off.addWidget(self.spin_peak_half_bw)
        row_off.addWidget(QLabel("ppm"))
        row_off.addStretch()
        v.addLayout(row_off)

        # Pool quick-set: picking a pool auto-fills spin_offset
        row_pool_off = QHBoxLayout()
        row_pool_off.addWidget(QLabel("Quick-set offset:"))
        self._combo_pool_offset = QComboBox()
        self._combo_pool_offset.setToolTip(
            "Select a pool to automatically set the saturation offset.\n"
            "The Regular / Inverse / Omega Plot analyses then target that pool."
        )
        _POOL_PPM = [
            ("— select pool —", None),
            ("OH (0.8 ppm)",        0.8),
            ("Creatine (1.9 ppm)",  1.9),
            ("NOE (−3.5 ppm)",      3.5),
            ("Amine / guanidinium (2.0 ppm)", 2.0),
            ("Amine (3.0 ppm)",     3.0),
            ("Amide (3.5 ppm)",     3.5),
            ("4.4 ppm",             4.4),
            ("Trp (5.4 ppm)",       5.4),
            ("7.3 ppm",             7.3),
            ("9.8 ppm",             9.8),
        ]
        for label, _ in _POOL_PPM:
            self._combo_pool_offset.addItem(label)
        def _on_pool_offset_combo(idx):
            _, ppm_val = _POOL_PPM[idx]
            if ppm_val is not None:
                self.spin_offset.setValue(abs(ppm_val))
                self._combo_pool_offset.setCurrentIndex(0)  # reset to placeholder
        self._combo_pool_offset.currentIndexChanged.connect(_on_pool_offset_combo)
        row_pool_off.addWidget(self._combo_pool_offset, stretch=1)
        row_pool_off.addStretch()
        v.addLayout(row_pool_off)

        # Refresh CEST Peak Range display when offset/BW changes
        def _maybe_refresh_range():
            if self.combo_display.currentText() == "CEST Peak Range":
                self._refresh_display()
        self.spin_offset.valueChanged.connect(lambda _: _maybe_refresh_range())
        self.spin_peak_half_bw.valueChanged.connect(lambda _: _maybe_refresh_range())

        # tp / rd / B0
        row_tp = QHBoxLayout()
        row_tp.addWidget(QLabel("tp (ms):"))
        self.spin_tp = QDoubleSpinBox()
        self.spin_tp.setRange(0.0, 50000.0)
        self.spin_tp.setValue(100.0)
        self.spin_tp.setSingleStep(10.0)
        row_tp.addWidget(self.spin_tp)
        row_tp.addWidget(QLabel("td (ms):"))
        self.spin_rd = QDoubleSpinBox()
        self.spin_rd.setRange(100.0, 60000.0)
        self.spin_rd.setValue(3000.0)
        self.spin_rd.setSingleStep(100.0)
        row_tp.addWidget(self.spin_rd)
        v.addLayout(row_tp)

        # Continuous-wave (default) vs pulsed saturation.  Surfaced here next to
        # tp/td so it is obvious which model the fit uses.
        self.chk_pulsed.setToolTip(
            "Unchecked (default) → continuous-wave saturation:\n"
            "  • MTRasym uses the finite-duration temporal model (tp, td)\n"
            "  • MTRRex / Ω-plot use the CW steady-state model\n"
            "Checked → pulsed saturation: MTRRex uses a duty-cycle correction\n"
            "DC = tp/(tp+td).  Use only for pulse-train sequences (NPulses > 1);\n"
            "leave OFF for a single continuous saturation block.")
        v.addWidget(self.chk_pulsed)

        row_b0 = QHBoxLayout()
        row_b0.addWidget(QLabel("B0 (T):"))
        self.spin_b0 = QDoubleSpinBox()
        self.spin_b0.setRange(1.0, 21.0)
        self.spin_b0.setValue(9.4)
        self.spin_b0.setSingleStep(0.1)
        row_b0.addWidget(self.spin_b0)
        row_b0.addStretch()
        v.addLayout(row_b0)

        # No. of exchangeable protons (for fs → mM conversion)
        row_np = QHBoxLayout()
        row_np.addWidget(QLabel("No. of protons:"))
        self.spin_n_protons = QSpinBox()
        self.spin_n_protons.setRange(1, 20)
        self.spin_n_protons.setValue(3)
        self.spin_n_protons.setToolTip(
            "Number of exchangeable protons per molecule.\n"
            "fs (mM) = fs (fraction) × 110 000 / no. of protons\n"
            "E.g. 3 for Amine, 2 for Amide etc."
        )
        self.spin_n_protons.valueChanged.connect(self._refresh_display)
        row_np.addWidget(self.spin_n_protons)
        row_np.addStretch()
        v.addLayout(row_np)

        return grp

    def _build_processing_group(self) -> QGroupBox:
        grp = QGroupBox("Processing && Fitting")
        v = QVBoxLayout(grp)
        v.setSpacing(6)

        # ── Persistent "QUESP Processing & Fitting" dialog ────────────────
        # Holds: Measurement Parameters + Pool & Fit Options button
        from PyQt6.QtWidgets import QDialog as _QDlg, QFrame as _QFr
        _proc_dlg = _QDlg(self)
        _proc_dlg.setWindowTitle("QUESP Processing & Fitting")
        _proc_dlg.setMinimumWidth(460)
        _dlg_v = QVBoxLayout(_proc_dlg)
        _dlg_v.setSpacing(8)

        # ── Section 1: Measurement Parameters ────────────────────────────
        _proc_dlg.params_grp = self._build_params_group()
        _dlg_v.addWidget(_proc_dlg.params_grp)

        # ── Divider ───────────────────────────────────────────────────────
        _sep = _QFr()
        _sep.setFrameShape(_QFr.Shape.HLine)
        _sep.setStyleSheet("color: #444;")
        _dlg_v.addWidget(_sep)

        # ── Section 2: Pool & Fit Options ─────────────────────────────────
        _pool_grp = QGroupBox("Select Pools for QUESP Fitting")
        _pool_v   = QVBoxLayout(_pool_grp)
        _pool_v.setSpacing(4)

        self.lbl_quesp_pools = QLabel(f"{len(self._quesp_pools)} pools selected")
        self.lbl_quesp_pools.setStyleSheet("font-size: 11px; color: #aaa;")
        _pool_v.addWidget(self.lbl_quesp_pools)

        btn_quesp_opts = QPushButton("Pools")
        btn_quesp_opts.setStyleSheet(
            "background:#37474f;color:white;border:none;"
            "border-radius:4px;padding:5px 14px;"
        )
        btn_quesp_opts.clicked.connect(self._open_quesp_options)
        _pool_v.addWidget(btn_quesp_opts)
        _dlg_v.addWidget(_pool_grp)

        # ── Section 3: T1 Map (R1A) — inside this dialog ──────────────────
        _sep_t1 = _QFr(); _sep_t1.setFrameShape(_QFr.Shape.HLine)
        _sep_t1.setStyleSheet("color: #444;")
        _dlg_v.addWidget(_sep_t1)
        _dlg_v.addWidget(self._build_t1_group())

        # ── Close button ──────────────────────────────────────────────────
        _close_row = QHBoxLayout()
        _close_row.addStretch()
        _btn_close_dlg = QPushButton("Close")
        _btn_close_dlg.clicked.connect(_proc_dlg.hide)
        _close_row.addWidget(_btn_close_dlg)
        _dlg_v.addLayout(_close_row)

        # ── Main open button (in the left panel group box) ────────────────
        btn_open = QPushButton("QUESP Processing && Fitting")
        btn_open.setFixedHeight(34)
        btn_open.setStyleSheet(
            "QPushButton { background: #1565c0; color: white; font-weight: bold;"
            "  border: none; border-radius: 5px; padding: 4px 14px; }"
            "QPushButton:hover { background: #1976d2; }"
        )
        btn_open.clicked.connect(_proc_dlg.show)
        btn_open.clicked.connect(_proc_dlg.raise_)
        v.addWidget(btn_open)

        # ── Separator ─────────────────────────────────────────────────────
        _sep2 = _QFr()
        _sep2.setFrameShape(_QFr.Shape.HLine)
        _sep2.setStyleSheet("color: #333;")
        v.addWidget(_sep2)

        # ── Action buttons (always visible in left panel) ─────────────────
        self.btn_run_fit = QPushButton("Run QUESP Fit")
        self.btn_run_fit.setFixedHeight(38)
        self.btn_run_fit.setStyleSheet(
            "QPushButton { background: #1e8f3e; color: white; font-size: 13px; "
            "font-weight: bold; border: none; border-radius: 6px; padding: 4px 12px; }"
            "QPushButton:hover { background: #27ae60; }"
            "QPushButton:disabled { background: #444; color: #888; }"
        )
        self.btn_run_fit.clicked.connect(self._run_fit)
        v.addWidget(self.btn_run_fit)

        self.btn_cancel_fit = QPushButton("Cancel")
        self.btn_cancel_fit.setEnabled(False)
        self.btn_cancel_fit.setStyleSheet(
            "QPushButton { background: #c0392b; color: white; border: none; "
            "border-radius: 5px; padding: 4px 12px; }"
            "QPushButton:disabled { background: #555; color: #999; }"
        )
        self.btn_cancel_fit.clicked.connect(self._cancel_fit)
        v.addWidget(self.btn_cancel_fit)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()
        v.addWidget(self.progress_bar)

        self.lbl_proc_status = QLabel("")
        self.lbl_proc_status.setWordWrap(True)
        self.lbl_proc_status.setStyleSheet("font-size: 10px; color: gray;")
        v.addWidget(self.lbl_proc_status)

        return grp

    # ─────────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────────

    # ── Vendor switching ──────────────────────────────────────────────────────

    def _on_quesp_vendor_changed(self, idx: int):
        """Switch between Bruker and GE/Siemens data source panels."""
        self._quesp_data_stack.setCurrentIndex(idx)

    # ── Bruker browse ─────────────────────────────────────────────────────────

    def _browse_quesp(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select QUESP scan folder (pdata/1/)", ""
        )
        if d:
            self._quesp_dir = d
            self.le_quesp_dir.setText(d)
            self.lbl_data_status.setText("Directory selected — click 'Load QUESP Data'.")

    # ── GE / Siemens browse & detect ─────────────────────────────────────────

    def _browse_quesp_gs(self):
        """Browse a GE / Siemens data folder."""
        import os as _os
        d = QFileDialog.getExistingDirectory(
            self, "Select GE / Siemens QUESP data folder (DICOM or NIfTI)", ""
        )
        if not d:
            return
        self._gs_quesp_study_dir = d
        self.le_quesp_gs_dir.setText(d)
        self._refresh_quesp_gs_info()

    def _refresh_quesp_gs_info(self):
        """Auto-detect scanner info and series list from the GE/Siemens folder."""
        import os as _os, glob as _glob
        path = self._gs_quesp_study_dir
        if not path:
            self.lbl_quesp_gs_info.setText("No folder selected.")
            return

        fmt = "dicom" if self.combo_quesp_gs_format.currentIndex() == 0 else "nifti"

        if fmt == "nifti":
            nii_files = (
                _glob.glob(_os.path.join(path, "*.nii")) +
                _glob.glob(_os.path.join(path, "*.nii.gz")) +
                _glob.glob(_os.path.join(path, "**/*.nii"), recursive=True)
            )
            nii_files = sorted(set(nii_files))
            if nii_files:
                self.lbl_quesp_gs_info.setText(
                    f"NIfTI mode — {len(nii_files)} .nii file(s) found."
                )
                self.lbl_quesp_gs_info.setStyleSheet("font-size: 10px; color: #4ec9b0;")
                self.lst_quesp_gs_scans.clear()
                self._gs_quesp_scan_dirs = []
                from PyQt6.QtWidgets import QListWidgetItem as _QLWI
                from PyQt6.QtGui import QFont as _QF
                for f in nii_files:
                    item = _QLWI(f"  {_os.path.basename(f)}")
                    item.setFont(_QF("Arial", 11))
                    self.lst_quesp_gs_scans.addItem(item)
                    self._gs_quesp_scan_dirs.append(f)
            else:
                self.lbl_quesp_gs_info.setText("No .nii files found in this folder.")
                self.lbl_quesp_gs_info.setStyleSheet("font-size: 10px; color: orange;")
                self.lst_quesp_gs_scans.clear()
            return

        # DICOM mode
        dcm_files = (
            _glob.glob(_os.path.join(path, "*.dcm")) or
            _glob.glob(_os.path.join(path, "**/*.dcm"), recursive=True)
        )
        if not dcm_files:
            self.lbl_quesp_gs_info.setText(
                "No .dcm files found. Check the folder or switch to NIfTI."
            )
            self.lbl_quesp_gs_info.setStyleSheet("font-size: 10px; color: orange;")
            return

        try:
            import pydicom
        except ImportError:
            self.lbl_quesp_gs_info.setText(
                "pydicom not installed — run:  pip install pydicom"
            )
            self.lbl_quesp_gs_info.setStyleSheet("font-size: 10px; color: orange;")
            return

        try:
            ds           = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
            manufacturer = str(getattr(ds, 'Manufacturer', '')).strip()
            model        = str(getattr(ds, 'ManufacturerModelName', '')).strip()
            field_T      = getattr(ds, 'MagneticFieldStrength', None)
            freq_MHz     = getattr(ds, 'ImagingFrequency', None)

            mfg_up = manufacturer.upper()
            if 'GE' in mfg_up or 'GEMS' in mfg_up:
                vendor_label = 'GE Medical Systems'
            elif 'SIEMENS' in mfg_up:
                vendor_label = 'Siemens'
            else:
                vendor_label = manufacturer or 'Unknown vendor'

            parts = [f"Platform: {vendor_label}"]
            if model:     parts.append(f"Model: {model}")
            if field_T is not None:  parts.append(f"Field: {float(field_T):.2g} T")
            if freq_MHz is not None: parts.append(f"Freq: {float(freq_MHz):.1f} MHz")
            self.lbl_quesp_gs_info.setText("   |   ".join(parts))
            self.lbl_quesp_gs_info.setStyleSheet("font-size: 10px; color: #4ec9b0;")
            self._build_quesp_gs_scan_list(path, vendor_label, pydicom)
        except Exception as exc:
            self.lbl_quesp_gs_info.setText(f"Detection error: {exc}")
            self.lbl_quesp_gs_info.setStyleSheet("font-size: 10px; color: red;")

    def _build_quesp_gs_scan_list(self, root_path: str, vendor_label: str, pydicom_mod):
        """Scan subdirectories for DICOM series and populate the series list."""
        import os as _os, glob as _glob
        from PyQt6.QtWidgets import QListWidgetItem as _QLWI
        from PyQt6.QtGui    import QFont as _QF, QColor as _QC

        self.lst_quesp_gs_scans.clear()
        self._gs_quesp_scan_dirs = []

        try:
            subdirs = sorted([
                _os.path.join(root_path, d) for d in _os.listdir(root_path)
                if _os.path.isdir(_os.path.join(root_path, d))
            ])
        except OSError:
            subdirs = []
        if not subdirs:
            subdirs = [root_path]

        is_ge       = 'GE' in vendor_label.upper()
        series_seen: set[str] = set()

        for subdir in subdirs:
            dcms = _glob.glob(_os.path.join(subdir, "*.dcm"))
            if not dcms:
                dcms = _glob.glob(_os.path.join(subdir, "**/*.dcm"), recursive=True)
            if not dcms:
                continue
            try:
                ds  = pydicom_mod.dcmread(dcms[0], stop_before_pixels=True)
                uid = str(getattr(ds, 'SeriesInstanceUID', subdir))
                if uid in series_seen:
                    continue
                series_seen.add(uid)

                series_num = str(getattr(ds, 'SeriesNumber', '?')).strip()
                if is_ge:
                    desc = (str(getattr(ds, 'SeriesDescription', '')) or
                            str(getattr(ds, 'SequenceName', _os.path.basename(subdir))))
                else:
                    desc = (str(getattr(ds, 'SequenceName', '')) or
                            str(getattr(ds, 'SeriesDescription', _os.path.basename(subdir))))

                display = f"  {series_num:>4}   ·   {desc.strip()}"
                item = _QLWI(display)
                item.setFont(_QF("Arial", 11))
                self.lst_quesp_gs_scans.addItem(item)
                self._gs_quesp_scan_dirs.append(subdir)
            except Exception:
                continue

        if self.lst_quesp_gs_scans.count() == 0:
            item = _QLWI("  (no DICOM series found in subdirectories)")
            item.setForeground(_QC("#666"))
            self.lst_quesp_gs_scans.addItem(item)

    def _on_quesp_gs_row_changed(self, row: int):
        """Set active path and show series details when user selects a series."""
        import os as _os, glob as _glob
        if row < 0 or row >= len(self._gs_quesp_scan_dirs):
            self.lbl_quesp_gs_detail.setText("")
            return
        selected = self._gs_quesp_scan_dirs[row]

        # NIfTI — path is a file
        if _os.path.isfile(selected):
            self._quesp_dir = selected
            self.lbl_quesp_gs_detail.setText(f"File: {_os.path.basename(selected)}")
            self.lbl_data_status.setText("Series selected — click 'Load QUESP Data'.")
            return

        # DICOM folder
        self._quesp_dir = selected
        dcms = _glob.glob(_os.path.join(selected, "*.dcm"))
        if not dcms:
            self.lbl_quesp_gs_detail.setText(f"Path: {selected}")
            self.lbl_data_status.setText("Series selected — click 'Load QUESP Data'.")
            return
        try:
            import pydicom
            ds          = pydicom.dcmread(dcms[0], stop_before_pixels=True)
            n_dcm       = len(dcms)
            series_desc = str(getattr(ds, 'SeriesDescription', ''))
            seq_name    = str(getattr(ds, 'SequenceName', ''))
            rows_px     = getattr(ds, 'Rows', '?')
            cols_px     = getattr(ds, 'Columns', '?')
            detail = f"{n_dcm} files  |  {rows_px}×{cols_px}"
            if series_desc: detail += f"  |  {series_desc}"
            if seq_name:    detail += f"  |  {seq_name}"
            self.lbl_quesp_gs_detail.setText(detail)
            self.lbl_data_status.setText("Series selected — click 'Load QUESP Data'.")
        except Exception:
            self.lbl_quesp_gs_detail.setText(f"Path: {selected}")

    # ── Load dispatcher ───────────────────────────────────────────────────────

    def _load_quesp(self):
        """Route to the correct loader based on the active vendor selection."""
        vendor_idx = self.combo_quesp_vendor.currentIndex()
        if vendor_idx == 0:
            self._load_quesp_bruker()
        else:
            fmt = "dicom" if self.combo_quesp_gs_format.currentIndex() == 0 else "nifti"
            if fmt == "nifti":
                self._load_quesp_nifti()
            else:
                self._load_quesp_dicom()

    def _load_quesp_bruker(self):
        if not self._quesp_dir:
            self.lbl_data_status.setText("Browse to a pdata/1/ folder first.")
            return
        try:
            from my_gui.bruker_reader import read_2dseq_quesp
            pv360 = self.combo_pv.currentText() == "PV360"
            # read_2dseq_quesp separates M0 (B1=0) images from saturation images
            imgs, M0imgs, info = read_2dseq_quesp(self._quesp_dir, pv360=pv360)
            # imgs:   (H, W, n_sl, n_sat) — saturation images only (B1 > 0)
            # M0imgs: (H, W[, n_M0])       — unsaturated reference(s)
            if imgs.ndim == 3:
                imgs = imgs[:, :, np.newaxis, :]  # ensure 4-D
            if imgs.ndim < 4:
                imgs = imgs[..., np.newaxis]
            H, W, n_sl, n_sat = imgs.shape
            self._quesp_raw  = imgs
            self._quesp_M0   = M0imgs
            self._quesp_info = info

            # Auto-detect pairing mode and N from sat_amplitudes + sat_offsets
            sat_amps = np.array(info.get('sat_amplitudes', []))
            sat_offs = np.array(info.get('sat_offsets', []))
            if len(sat_amps) == n_sat and len(sat_offs) == n_sat:
                # Detect unique positive amplitudes (one per power level)
                pos_mask = sat_offs > 0
                if not np.any(pos_mask):
                    pos_mask = np.ones(n_sat, dtype=bool)
                unique_amps = np.unique(np.round(sat_amps[pos_mask], 4))
                N_detected = len(unique_amps)
                self.spin_n_powers.setValue(N_detected)
                # Set pairing mode: if all pos come before all neg → Sequential
                if np.all(sat_offs[:n_sat//2] > 0) and np.all(sat_offs[n_sat//2:] < 0):
                    self.combo_pair.setCurrentIndex(1)  # Sequential
                else:
                    self.combo_pair.setCurrentIndex(0)  # Alternating
                # Auto-fill B1 powers (positive-offset amplitudes, sorted)
                pos_amps = sorted(set(np.round(sat_amps[pos_mask], 4).tolist()))
                self.le_b1_powers.setText(", ".join(f"{v:.4g}" for v in pos_amps))
            else:
                N_detected = n_sat // 2

            # Auto-fill timing from info
            sat_dur = info.get('sat_duration', np.array([0.0]))
            trec    = info.get('Trec', np.array([0.0]))
            if hasattr(sat_dur, '__len__') and len(sat_dur) > 0:
                self.spin_tp.setValue(float(np.unique(sat_dur)[0]))
            elif isinstance(sat_dur, (int, float)):
                self.spin_tp.setValue(float(sat_dur))
            if hasattr(trec, '__len__') and len(trec) > 0:
                self.spin_rd.setValue(float(np.unique(trec)[0]))
            elif isinstance(trec, (int, float)):
                self.spin_rd.setValue(float(trec))

            self.spin_slice.setMaximum(n_sl)
            self.lbl_data_status.setText(
                f"Loaded {H}×{W}  slices={n_sl}  sat.images={n_sat}  "
                f"(N={N_detected} power levels per side)"
            )
            self.lbl_data_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")
            self._show_reference()
        except Exception as exc:
            import traceback
            self.lbl_data_status.setText(f"Error: {exc}")
            self.lbl_data_status.setStyleSheet("color: red; font-size: 10px;")

    def _load_quesp_dicom(self):
        """
        Load QUESP data from a DICOM series folder.
        Images are sorted by InstanceNumber and stacked as (H, W, 1, N).
        ImageComments are parsed for saturation offset / power info when present.
        """
        import os as _os, glob as _glob
        path = self._quesp_dir
        if not path or not _os.path.isdir(path):
            self.lbl_data_status.setText(
                "Select a DICOM series from the list first."
            )
            return
        try:
            import pydicom
        except ImportError:
            self.lbl_data_status.setText("pydicom not installed — run: pip install pydicom")
            self.lbl_data_status.setStyleSheet("color: red; font-size: 10px;")
            return
        try:
            dcm_files = sorted(
                _glob.glob(_os.path.join(path, "*.dcm")) or
                _glob.glob(_os.path.join(path, "**/*.dcm"), recursive=True)
            )
            if not dcm_files:
                self.lbl_data_status.setText("No .dcm files found in the selected folder.")
                self.lbl_data_status.setStyleSheet("color: red; font-size: 10px;")
                return

            # Read all slices, sort by InstanceNumber
            slices = []
            for f in dcm_files:
                ds = pydicom.dcmread(f)
                slices.append(ds)
            slices.sort(key=lambda d: int(getattr(d, 'InstanceNumber', 0)))

            # Stack pixel arrays → (H, W, n_slices, n_sat); de-tile Siemens mosaics
            from my_gui.dicom_mosaic import is_mosaic, mosaic_to_volume
            vols = []
            for s in slices:
                a = s.pixel_array.astype(float)
                if a.ndim == 2 and is_mosaic(s):
                    v = mosaic_to_volume(a, s)                 # (H, W, n_slices)
                    if v.ndim == 2:
                        v = v[:, :, np.newaxis]
                elif a.ndim == 2:
                    v = a[:, :, np.newaxis]                    # (H, W, 1)
                else:                                          # multiframe/colour → first plane
                    v = np.asarray(a).reshape(a.shape[-2], a.shape[-1])[:, :, np.newaxis]
                vols.append(v)
            n_sat = len(vols)
            imgs  = np.stack(vols, axis=-1)                    # (H, W, n_slices, n_sat)
            H, W  = imgs.shape[:2]

            # Try to parse saturation offsets / powers from ImageComments
            import re as _re
            sat_offs, sat_amps = [], []
            for s in slices:
                comment = str(getattr(s, 'ImageComments', '')).lower()
                m_off = _re.search(r'offset[=:\s]*([-\d.]+)\s*ppm', comment)
                m_amp = _re.search(r'b1[=:\s]*([\d.]+)\s*[uμ]t', comment)
                sat_offs.append(float(m_off.group(1)) if m_off else np.nan)
                sat_amps.append(float(m_amp.group(1)) if m_amp else np.nan)

            info: dict = {}
            if not np.all(np.isnan(sat_offs)):
                info['sat_offsets']    = np.array(sat_offs)
                info['sat_amplitudes'] = np.array(sat_amps)
                # Auto-fill pairing and N
                valid_amps = [a for a in sat_amps if not np.isnan(a) and a > 0]
                if valid_amps:
                    unique_amps = np.unique(np.round(valid_amps, 3))
                    self.spin_n_powers.setValue(len(unique_amps))
                    # Auto-fill B1 field from unique amplitudes
                    self.le_b1_powers.setText(
                        ", ".join(f"{v:.4g}" for v in sorted(unique_amps))
                    )

            self._quesp_raw  = imgs
            self._quesp_M0   = None
            self._quesp_info = info
            H2, W2, n_sl2, n_sat2 = imgs.shape
            self.spin_slice.setMaximum(n_sl2)
            self.lbl_data_status.setText(
                f"DICOM loaded: {H2}×{W2}  {n_sat2} images from {len(dcm_files)} files"
            )
            self.lbl_data_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")
            self._show_reference()
        except Exception as exc:
            import traceback
            self.lbl_data_status.setText(f"DICOM load error: {exc}")
            self.lbl_data_status.setStyleSheet("color: red; font-size: 10px;")

    def _load_quesp_nifti(self):
        """
        Load QUESP data from a NIfTI file.
        Expects a 4-D volume (H, W, slices, volumes) where each volume = one
        saturation measurement.  A 3-D file is treated as a single saturation image.
        """
        import os as _os
        path = self._quesp_dir   # set by _on_quesp_gs_row_changed
        if not path or not _os.path.isfile(path):
            self.lbl_data_status.setText(
                "Select a NIfTI file from the list first."
            )
            return
        try:
            import nibabel as nib
        except ImportError:
            self.lbl_data_status.setText("nibabel not installed — run: pip install nibabel")
            self.lbl_data_status.setStyleSheet("color: red; font-size: 10px;")
            return
        try:
            img  = nib.load(path)
            data = np.asarray(img.dataobj, dtype=float)

            # Normalise to (H, W, slices, n_sat)
            if data.ndim == 2:
                data = data[:, :, np.newaxis, np.newaxis]
            elif data.ndim == 3:
                data = data[:, :, :, np.newaxis]
            elif data.ndim == 4:
                pass
            else:
                data = data[:, :, :data.shape[2], :]   # take first 3 spatial dims

            H, W, n_sl, n_sat = data.shape
            self._quesp_raw  = data
            self._quesp_M0   = None
            self._quesp_info = {}
            self.spin_slice.setMaximum(n_sl)
            self.spin_n_powers.setValue(max(1, n_sat // 2))
            self.lbl_data_status.setText(
                f"NIfTI loaded: {H}×{W}  slices={n_sl}  volumes={n_sat}  "
                f"({_os.path.basename(path)})"
            )
            self.lbl_data_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")
            self._show_reference()
        except Exception as exc:
            import traceback
            self.lbl_data_status.setText(f"NIfTI load error: {exc}")
            self.lbl_data_status.setStyleSheet("color: red; font-size: 10px;")

    def _load_mat(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Load QUESP data", "",
            "Data files (*.mat);;All files (*)"
        )
        if not fn:
            return
        try:
            if fn.endswith(".npz"):
                d = dict(np.load(fn, allow_pickle=True))
            else:
                import scipy.io as sio
                d = sio.loadmat(fn)

            if "quesp_eff" in d:
                eff = np.array(d["quesp_eff"]).squeeze()
                if eff.ndim == 2:
                    eff = eff[..., np.newaxis]
                self._quesp_eff = eff
                self._quesp_raw = None
                H, W, N = eff.shape
                self.lbl_data_status.setText(
                    f"Loaded quesp_eff {H}×{W}×{N} from {fn}"
                )
            if "b1_powers" in d:
                bp = np.array(d["b1_powers"]).ravel()
                self.le_b1_powers.setText(", ".join(f"{v:.4g}" for v in bp))
            if "t1_map" in d:
                self._t1_map = np.array(d["t1_map"]).squeeze()
                H2, W2 = self._t1_map.shape[:2]
                self.lbl_t1_status.setText(f"T1 map from file: {H2}×{W2}")

            self._show_reference()
            self.lbl_data_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")
        except Exception as exc:
            self.lbl_data_status.setText(f"Error: {exc}")
            self.lbl_data_status.setStyleSheet("color: red; font-size: 10px;")

    # ─────────────────────────────────────────────────────────────────────
    # Multi-B1 dataset management
    # ─────────────────────────────────────────────────────────────────────

    def _add_dataset(self):
        dlg = _MultiB1AddDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            path, b1, pv360 = dlg.get_values()
            if not path:
                self.lbl_data_status.setText("No path selected — scan not added.")
                return
            self._datasets.append({'path': path, 'b1_ut': b1, 'pv360': pv360})
            self._refresh_dataset_list()

    def _remove_dataset(self):
        row = self.lw_datasets.currentRow()
        if row < 0 or row >= len(self._datasets):
            return
        del self._datasets[row]
        self._refresh_dataset_list()

    def _refresh_dataset_list(self):
        self.lw_datasets.clear()
        for ds in self._datasets:
            self.lw_datasets.addItem(f"{ds['b1_ut']:.2f} µT — {ds['path']}")

    def _load_multi_quesp(self):
        if not self._datasets:
            self.lbl_data_status.setText("No scans in the list — add scans first.")
            return

        offset_ppm = self.spin_multi_offset.value()
        sl = self.spin_slice.value() - 1

        eff_list = []
        asym_list = []
        b1_list = []
        H_ref = W_ref = None

        for ds in self._datasets:
            path = ds['path']
            b1 = ds['b1_ut']
            pv360 = ds['pv360']
            try:
                if path.endswith('.mat') or path.endswith('.npz'):
                    # Load .mat/.npz: try CEST keys first, then quesp_eff fallback
                    if path.endswith('.npz'):
                        d = dict(np.load(path, allow_pickle=True))
                    else:
                        import scipy.io as sio
                        d = sio.loadmat(path)

                    if 'image' in d and 'w_offsetPPM' in d:
                        img = np.array(d['image'])      # (H, W, slices, n_off) or (H, W, n_off)
                        ppm = np.array(d['w_offsetPPM']).ravel()
                        M0_key = next((k for k in ('M0image', 'M0', 'm0') if k in d), None)
                        M0 = np.array(d[M0_key]) if M0_key else None

                        if img.ndim == 3:
                            imgs_2d = img          # (H, W, n_off)
                            M0_2d = M0[:, :] if M0 is not None and M0.ndim >= 2 else None
                        else:
                            sl_idx = min(sl, img.shape[2] - 1)
                            imgs_2d = img[:, :, sl_idx, :]
                            M0_2d = (M0[:, :, sl_idx] if M0 is not None and M0.ndim >= 3
                                     else (M0 if M0 is not None else None))

                        pos_i = int(np.argmin(np.abs(ppm - offset_ppm)))
                        neg_i = int(np.argmin(np.abs(ppm + offset_ppm)))
                        M0_ref = (M0_2d.astype(float) if M0_2d is not None
                                  else np.ones(imgs_2d.shape[:2], dtype=float))
                        M0_ref = M0_ref + 1e-9
                        Zpos = np.clip(imgs_2d[:, :, pos_i].astype(float) / M0_ref, 1e-6, None)
                        Zneg = np.clip(imgs_2d[:, :, neg_i].astype(float) / M0_ref, 1e-6, None)
                        rex  = np.clip(1.0 / Zpos - 1.0 / Zneg, 0, None)
                        asym = np.clip(Zneg - Zpos, 0, None)
                    elif 'quesp_eff' in d:
                        arr = np.array(d['quesp_eff']).squeeze()
                        if arr.ndim == 2:
                            rex = arr
                        elif arr.ndim == 3:
                            rex = arr[:, :, min(sl, arr.shape[2] - 1)]
                        else:
                            rex = arr[:, :]
                        asym = rex.copy()
                    else:
                        self.lbl_data_status.setText(
                            f"Skipped {path}: no recognised keys (image/quesp_eff)."
                        )
                        continue
                else:
                    # Bruker folder
                    from my_gui.bruker_reader import read_2dseq_cest
                    img, M0, info = read_2dseq_cest(path, pv360=pv360)
                    # img: (H, W, slices, n_offsets) sorted descending ppm (pos first)
                    ppm = np.array(info['w_offsetPPM'])
                    sl_idx = min(sl, img.shape[2] - 1)
                    imgs_2d = img[:, :, sl_idx, :].astype(float)    # (H, W, n_off)
                    if M0.ndim >= 3:
                        M0_2d = M0[:, :, min(sl, M0.shape[2] - 1)].astype(float)
                    else:
                        M0_2d = M0.astype(float)

                    pos_i = int(np.argmin(np.abs(ppm - offset_ppm)))
                    neg_i = int(np.argmin(np.abs(ppm + offset_ppm)))

                    Zpos = np.clip(imgs_2d[:, :, pos_i] / (M0_2d + 1e-9), 1e-6, None)
                    Zneg = np.clip(imgs_2d[:, :, neg_i] / (M0_2d + 1e-9), 1e-6, None)

                    rex  = np.clip(1.0 / Zpos - 1.0 / Zneg, 0, None)
                    asym = np.clip(Zneg - Zpos, 0, None)

                H, W = rex.shape[:2]
                if H_ref is None:
                    H_ref, W_ref = H, W
                elif (H, W) != (H_ref, W_ref):
                    self.lbl_data_status.setText(
                        f"Skipped {path}: size {H}x{W} != expected {H_ref}x{W_ref}."
                    )
                    continue

                eff_list.append(rex)
                asym_list.append(asym)
                b1_list.append(b1)

            except Exception as exc:
                import traceback
                self.lbl_data_status.setText(
                    f"Error loading {path}: {exc}"
                )
                self.lbl_data_status.setStyleSheet("color: orange; font-size: 10px;")
                continue

        if not eff_list:
            self.lbl_data_status.setText("Multi-B1 load failed: no valid scans loaded.")
            self.lbl_data_status.setStyleSheet("color: red; font-size: 10px;")
            return

        # Sort by B1 power ascending
        order = np.argsort(b1_list)
        eff_list  = [eff_list[i]  for i in order]
        asym_list = [asym_list[i] for i in order]
        b1_list   = [b1_list[i]   for i in order]

        self._quesp_eff  = np.stack(eff_list,  axis=-1)   # (H, W, N)
        self._quesp_asym = np.stack(asym_list, axis=-1)   # (H, W, N)
        self._b1_powers  = b1_list
        self._w1_vals    = np.array([_w1_rad_per_s(v) for v in b1_list])
        self._quesp_raw  = None
        self._quesp_M0   = None

        # Update B1 powers display
        self.le_b1_powers.setText(", ".join(f"{v:.4g}" for v in b1_list))
        self.spin_n_powers.setValue(len(b1_list))

        self.lbl_data_status.setText(
            f"Multi-B1 loaded: {len(b1_list)} scans × {H_ref}×{W_ref}, "
            f"B1=[{b1_list[0]:.2g}–{b1_list[-1]:.2g}] µT, "
            f"offset=±{offset_ppm:.2f} ppm"
        )
        self.lbl_data_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")
        self._show_quesp_ref()

    def _browse_t1(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Load T1 map", "",
            "Data files (*.mat);;All files (*)"
        )
        if not fn:
            return
        try:
            if fn.endswith(".npz"):
                d = dict(np.load(fn, allow_pickle=True))
            else:
                import scipy.io as sio
                d = sio.loadmat(fn)
            for key in ("t1_map", "T1", "T1map", "t1"):
                if key in d:
                    self._t1_map = np.array(d[key]).squeeze()
                    break
            if self._t1_map is None:
                # Try first array
                arrs = [v for k, v in d.items()
                        if not k.startswith("_") and isinstance(v, np.ndarray)]
                if arrs:
                    self._t1_map = arrs[0].squeeze()
            if self._t1_map is not None:
                H, W = self._t1_map.shape[:2]
                self.lbl_t1_status.setText(f"Loaded: {H}×{W}  ms")
                self.lbl_t1_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")
        except Exception as exc:
            self.lbl_t1_status.setText(f"Error: {exc}")
            self.lbl_t1_status.setStyleSheet("color: red; font-size: 10px;")

    def _enter_t1_manually(self):
        """Let the user type a single T1 value (ms) to use as a uniform T1 map."""
        from PyQt6.QtWidgets import QInputDialog
        val, ok = QInputDialog.getDouble(
            self, "Enter T1 value",
            "T1 (ms):",
            value=2000.0, min=1.0, max=20000.0, decimals=1,
        )
        if ok and val > 0:
            # Build a 1×1 uniform T1 map — will be tiled/broadcast to image size at fit time
            import numpy as np
            if self._quesp_eff is not None:
                H, W = self._quesp_eff.shape[:2]
                self._t1_map = np.full((H, W), val, dtype=float)
            else:
                self._t1_map = np.array([[val]], dtype=float)
            self.lbl_t1_status.setText(
                f"T1 = {val:.1f} ms (manual, uniform)"
            )
            self.lbl_t1_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")

    def set_t1_map(self, t1_map: np.ndarray):
        """Called from app.py / T1T2 tab when a T1 map is available."""
        self._t1_map = np.array(t1_map).squeeze()
        H, W = self._t1_map.shape[:2]
        self.lbl_t1_status.setText(f"T1 map from T1/T2/B1 tab: {H}×{W}  ms")
        self.lbl_t1_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")

    def load_from_path(self, path: str, pv360: bool = False):
        """Auto-load from scan directory path (called by app.py on scan assignment)."""
        self._quesp_dir = path
        self.le_quesp_dir.setText(path)
        self.combo_pv.setCurrentText("PV360" if pv360 else "PV6 / PV7")
        self._load_quesp()

    # ─────────────────────────────────────────────────────────────────────
    # Processing
    # ─────────────────────────────────────────────────────────────────────

    def _parse_b1_powers(self) -> list[float]:
        raw = self.le_b1_powers.text()
        vals = []
        for tok in raw.replace(";", ",").split(","):
            tok = tok.strip()
            if tok:
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
        return vals

    def _compute_quesp_effect(self):
        """
        Compute per-power MTRRex from raw paired measurements.
        Ports QUESP_load_proc.m: pairs positive and negative offset images
        by matching amplitude, normalises by M0 (B1=0) reference.
        MTRRex = 1/Zlab - 1/Zref  (Inverse method, matches MATLAB QUESPfcn.m)
        """
        if self._quesp_raw is None:
            self.lbl_proc_status.setText("Load raw QUESP data first.")
            return

        sl = self.spin_slice.value() - 1
        H, W, n_sl, n_sat = self._quesp_raw.shape
        imgs = self._quesp_raw[:, :, min(sl, n_sl - 1), :].astype(float)  # (H, W, n_sat)

        # ── M0 reference images ───────────────────────────────────────────
        M0img = self._quesp_M0
        if M0img is not None:
            M0img = np.array(M0img, dtype=float)
            if M0img.ndim == 2:
                M0img = M0img[:, :, np.newaxis]    # (H, W, 1)
            elif M0img.ndim == 3 and M0img.shape[2] > 2:
                # 3-D (H,W,slices) — pick the requested slice
                M0img = M0img[:, :, min(sl, M0img.shape[2] - 1), np.newaxis]
        else:
            M0img = None

        # ── Pairing: MATLAB-style ─────────────────────────────────────────
        info = self._quesp_info or {}
        sat_amps = np.array(info.get('sat_amplitudes', []), dtype=float)
        sat_offs = np.array(info.get('sat_offsets',    []), dtype=float)

        # Determine offset unit: if values > 100 → assume Hz; else ppm
        # Tolerance for "opposite offset" pairing
        if len(sat_offs) == n_sat and len(sat_amps) == n_sat and n_sat > 0:
            off_tol = 0.5 if np.max(np.abs(sat_offs)) < 100 else 50.0  # ppm or Hz

            # Find pairs: |off_i + off_j| < tol AND |amp_i - amp_j| < 1e-3
            pos_idx_list, neg_idx_list = [], []
            used = [False] * n_sat
            for i in range(n_sat):
                if used[i]:
                    continue
                for j in range(i + 1, n_sat):
                    if used[j]:
                        continue
                    if (abs(sat_offs[i] + sat_offs[j]) < off_tol and
                            abs(sat_amps[i] - sat_amps[j]) < 1e-3):
                        if sat_offs[i] > 0:
                            pos_idx_list.append(i)
                            neg_idx_list.append(j)
                        else:
                            pos_idx_list.append(j)
                            neg_idx_list.append(i)
                        used[i] = used[j] = True
                        break

            if pos_idx_list:
                # Sort pairs by amplitude (ascending)
                order = np.argsort([sat_amps[i] for i in pos_idx_list])
                pos_idx = [pos_idx_list[k] for k in order]
                neg_idx = [neg_idx_list[k] for k in order]
                b1_auto = [float(sat_amps[i]) for i in pos_idx]
                N = len(pos_idx)
            else:
                # Fallback: manual pairing from combo box
                pos_idx, neg_idx, N, b1_auto = self._manual_pairing(n_sat)
        else:
            pos_idx, neg_idx, N, b1_auto = self._manual_pairing(n_sat)

        pos_imgs = imgs[:, :, pos_idx]  # (H, W, N)  Zlab (labeled, +ppm)
        neg_imgs = imgs[:, :, neg_idx]  # (H, W, N)  Zref (reference, -ppm)

        # ── M0 normalisation ─────────────────────────────────────────────
        # Use B1=0 M0 images if available (preferred); otherwise use first
        # image of each group as approximate M0.
        if M0img is not None and M0img.shape[2] >= 2:
            M0_pos = M0img[:, :, 0:1] + 1e-9   # (H, W, 1)
            M0_neg = M0img[:, :, 1:2] + 1e-9
        elif M0img is not None and M0img.shape[2] == 1:
            M0_pos = M0_neg = M0img[:, :, 0:1] + 1e-9
        else:
            # No M0 image — use mean of lowest-power images as proxy
            M0_pos = pos_imgs[:, :, 0:1] + 1e-9
            M0_neg = neg_imgs[:, :, 0:1] + 1e-9

        Zpos = np.clip(pos_imgs / M0_pos, 1e-6, None)   # (H, W, N)  Zlab/M0
        Zneg = np.clip(neg_imgs / M0_neg, 1e-6, None)   # (H, W, N)  Zref/M0

        # ── MTRRex = 1/Zlab - 1/Zref  (Inverse / OmegaPlot QUESP) ──────
        quesp_eff = 1.0 / Zpos - 1.0 / Zneg             # (H, W, N)
        quesp_eff = np.clip(quesp_eff, 0, None)

        # ── MTR_asym = Zref - Zlab  (Regular QUESP model input) ──────────
        quesp_asym = np.clip(Zneg - Zpos, 0, None)      # (H, W, N)

        self._quesp_eff  = quesp_eff
        self._quesp_asym = quesp_asym

        # ── B1 powers ────────────────────────────────────────────────────
        b1 = self._parse_b1_powers()
        if not b1 or len(b1) != N:
            b1 = b1_auto if b1_auto else [float(i + 1) for i in range(N)]
        self._b1_powers = b1
        self._w1_vals   = np.array([_w1_rad_per_s(v) for v in b1])

        self.spin_n_powers.setValue(N)
        if b1_auto:
            self.le_b1_powers.setText(", ".join(f"{v:.4g}" for v in b1_auto))

        self.lbl_proc_status.setText(
            f"QUESP effect computed: {N} power levels, B1={b1[0]:.2g}–{b1[-1]:.2g} µT, "
            f"MTRRex range [{quesp_eff.min():.3g}, {quesp_eff.max():.3g}]"
        )
        self.lbl_proc_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")
        self._show_quesp_ref()

    def _manual_pairing(self, n_sat: int):
        """Fallback manual pairing from combo box and N spinbox."""
        N = self.spin_n_powers.value()
        pair_mode = self.combo_pair.currentIndex()
        if pair_mode == 0:
            pos_idx = list(range(0, min(2 * N, n_sat), 2))
            neg_idx = list(range(1, min(2 * N, n_sat), 2))
        else:
            pos_idx = list(range(0, min(N, n_sat)))
            neg_idx = list(range(min(N, n_sat), min(2 * N, n_sat)))
        N = min(len(pos_idx), len(neg_idx))
        pos_idx, neg_idx = pos_idx[:N], neg_idx[:N]
        b1_auto = [float(i + 1) for i in range(N)]
        return pos_idx, neg_idx, N, b1_auto

    def _quesp_from_1z(self):
        """Load a 1/Z QUESP stack (.mat from the 1/Z tab) and fit fs/ksw per
        ROI & pool with the linear inverse-QUESP model."""
        from PyQt6.QtWidgets import (QFileDialog, QMessageBox, QDialog,
                                     QVBoxLayout, QTableWidget, QTableWidgetItem,
                                     QPushButton, QHBoxLayout)
        import scipy.io as sio
        from my_gui.quesp_from_invz import linear_quesp, fs_to_concentration

        path, _ = QFileDialog.getOpenFileName(
            self, "Load 1/Z stack  or  figure/data (.mat/.npz)", "",
            "Data (*.mat *.npz);;MATLAB (*.mat);;NumPy (*.npz);;All files (*)")
        if not path:
            return
        import os as _os
        # A 1/Z QUESP stack is a .mat carrying satpwr_uT + mtrrex__<pool> keys.
        # Anything else (external .mat/.npz holding exported figures or image
        # arrays) is simply displayed in this tab via _display_loaded_figures.
        _probe = None
        if _os.path.splitext(path)[1].lower() == '.mat':
            try:
                _probe = sio.loadmat(path, squeeze_me=True)
            except Exception:
                _probe = None
        if not (_probe is not None and 'satpwr_uT' in _probe
                and any(str(k).startswith('mtrrex__') for k in _probe)):
            self._display_loaded_figures(path)
            return
        try:
            m = _probe
            b1 = np.atleast_1d(np.asarray(m["satpwr_uT"], dtype=float)).ravel()
            pools = [str(x) for x in np.atleast_1d(m["pools"]).ravel()]
            rnames = [str(x) for x in np.atleast_1d(m["roi_names"]).ravel()]
            R1 = float(np.asarray(m.get("R1", 1.0)).ravel()[0])
            nH = int(self._nh.value()) if hasattr(self, "_nh") else 1
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", f"Could not read stack:\n{exc}")
            return

        rows = []   # (roi, pool, fs, ksw, conc_mM, r2, n)
        for pool in pools:
            key = f"mtrrex__{pool}"
            if key not in m:
                continue
            M = np.atleast_2d(np.asarray(m[key], dtype=float))
            if M.shape[0] != len(rnames):
                M = M.reshape(len(rnames), -1)
            for ri, rname in enumerate(rnames):
                r = linear_quesp(b1, M[ri])
                rows.append((rname, pool, r["fs"], r["ksw"],
                             fs_to_concentration(r["fs"], nH), r["r2"], r["n"]))
        if not rows:
            QMessageBox.information(self, "No data",
                                    "No fittable pools found in the stack.")
            return

        dlg = QDialog(self); dlg.setWindowTitle("QUESP from 1/Z — fs / ksw")
        dlg.resize(640, 420); dl = QVBoxLayout(dlg)
        hdr = ["ROI", "Pool", "fs", "ksw (s⁻¹)", f"conc (mM, nH={nH})", "R²", "n(B1)"]
        tbl = QTableWidget(len(rows), len(hdr))
        tbl.setHorizontalHeaderLabels(hdr)
        for i, (rn, pl, fs, ksw, conc, r2, n) in enumerate(rows):
            vals = [rn, pl,
                    f"{fs:.4g}" if np.isfinite(fs) else "—",
                    f"{ksw:.1f}" if np.isfinite(ksw) else "—",
                    f"{conc:.1f}" if np.isfinite(conc) else "—",
                    f"{r2:.3f}" if np.isfinite(r2) else "—", str(n)]
            for j, vv in enumerate(vals):
                tbl.setItem(i, j, QTableWidgetItem(vv))
        tbl.resizeColumnsToContents()
        dl.addWidget(tbl)
        self._quesp_1z_rows = rows   # keep for CSV copy
        br = QHBoxLayout()
        btn_csv = QPushButton("📋 Copy (CSV)")
        def _copy():
            from PyQt6.QtWidgets import QApplication
            lines = ["\t".join(hdr)]
            for rn, pl, fs, ksw, conc, r2, n in rows:
                lines.append("\t".join([rn, pl, f"{fs:.6g}", f"{ksw:.4g}",
                                        f"{conc:.4g}", f"{r2:.4g}", str(n)]))
            QApplication.clipboard().setText("\n".join(lines))
        btn_csv.clicked.connect(_copy); br.addWidget(btn_csv); br.addStretch()
        _bc = QPushButton("Close"); _bc.clicked.connect(dlg.accept); br.addWidget(_bc)
        dl.addLayout(br)
        dlg.show()

    def _run_fit(self):
        # Always (re-)compute the QUESP effect from raw data first
        self._compute_quesp_effect()
        if self._quesp_eff is None:
            return   # _compute_quesp_effect already set the status label

        # T1 map — required
        if self._t1_map is None:
            self.lbl_proc_status.setText("Load a T1 map first.")
            return

        b1 = self._parse_b1_powers()
        N  = self._quesp_eff.shape[-1]
        if not b1 or len(b1) != N:
            b1 = [float(i + 1) * 0.5 for i in range(N)]
        self._b1_powers = b1
        self._w1_vals   = np.array([_w1_rad_per_s(v) for v in b1])

        # Align T1 map spatial dims to quesp_eff
        H, W = self._quesp_eff.shape[:2]
        t1 = self._t1_map
        if t1.shape[:2] != (H, W):
            try:
                from scipy.ndimage import zoom
                zy = H / t1.shape[0]
                zx = W / t1.shape[1]
                t1 = zoom(t1[:H, :W] if zy == 1 else t1, (zy, zx), order=1)
            except Exception:
                t1 = t1[:H, :W]

        # Build mask
        t1_min = self.spin_t1_thresh.value()
        mask = (t1 > t1_min)
        # Also exclude pixels where mean quesp_eff is near zero
        mean_eff = self._quesp_eff.mean(axis=-1)
        mask &= (mean_eff > 0.001)

        model_txt = self.combo_model.currentText()
        model = 'inverse' if 'Inverse' in model_txt else 'regular'
        pulsed = self.chk_pulsed.isChecked()

        self._worker = QUESPFitWorker(
            quesp_eff  = self._quesp_eff,
            t1_map     = t1,
            w1_vals    = self._w1_vals,
            mask       = mask,
            model      = model,
            pulsed     = pulsed,
            tp_s       = self.spin_tp.value() / 1000.0,
            rd_s       = self.spin_rd.value() / 1000.0,
            quesp_asym = self._quesp_asym,   # MTR_asym for Regular model
        )
        self._worker.finished.connect(self._on_fit_done)
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.error.connect(self._on_fit_error)

        self.btn_run_fit.setEnabled(False)
        self.btn_cancel_fit.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.lbl_proc_status.setText("Fitting in progress…")
        self.lbl_proc_status.setStyleSheet("color: gray; font-size: 10px;")
        self._worker.start()

    def _cancel_fit(self):
        if self._worker is not None:
            self._worker.stop()
        self.btn_cancel_fit.setEnabled(False)
        self.lbl_proc_status.setText("Cancelling…")

    def _on_fit_done(self, fs_map, ksw_map, rsq_map):
        # Restrict to the global analysis mask (brain / phantom outline), if any.
        fs_map  = apply_analysis_mask(fs_map)
        ksw_map = apply_analysis_mask(ksw_map)
        rsq_map = apply_analysis_mask(rsq_map)
        self._fs_map  = fs_map
        self._ksw_map = ksw_map
        self._rsq_map = rsq_map
        self.btn_run_fit.setEnabled(True)
        self.btn_cancel_fit.setEnabled(False)
        self.progress_bar.hide()
        n_fit = int((fs_map > 0).sum())
        n_p = self.spin_n_protons.value()
        if n_fit > 0:
            fs_v = fs_map[fs_map > 0]
            ksw_v = ksw_map[ksw_map > 0]
            self.lbl_proc_status.setText(
                f"Fit complete — {n_fit} voxels fitted.\n"
                f"fs: [{fs_v.min():.2e}–{fs_v.max():.2e}]  "
                f"({fs_v.min()*110000/n_p:.2g}–{fs_v.max()*110000/n_p:.2g} mM)\n"
                f"ksw: [{ksw_v.min():.0f}–{ksw_v.max():.0f}] s⁻¹"
            )
        else:
            self.lbl_proc_status.setText("Fit complete — no voxels fitted.")
        self.lbl_proc_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")
        self.combo_display.setCurrentText("fs Map (mM)")
        self._refresh_display()

    def _on_fit_error(self, msg: str):
        self.btn_run_fit.setEnabled(True)
        self.btn_cancel_fit.setEnabled(False)
        self.progress_bar.hide()
        self.lbl_proc_status.setText(f"Error: {msg[:200]}")
        self.lbl_proc_status.setStyleSheet("color: red; font-size: 10px;")

    # ─────────────────────────────────────────────────────────────────────
    # Display
    # ─────────────────────────────────────────────────────────────────────

    def _toggle_hide_rois(self, checked: bool):
        self.canvas.toggle_rois_visible()   # mirrors Z-spec tab logic exactly
        self.btn_hide_rois.setText("Show ROIs" if checked else "Hide ROIs")

    def _open_quesp_options(self):
        dlg = _QUESPOptionsDialog(
            selected_pools=self._quesp_pools,
            rb_pv_checked=self.rb_pseudovoigt.isChecked(),
            keep_gl=self.chk_keep_gl.isChecked(),
            super_lorentz=self.chk_mt_superlorentz.isChecked(),
            multipool=self.chk_multipool.isChecked(),
            model_idx=self.combo_model.currentIndex(),
            thresh=self.spin_thresh.value(),
            t1_thresh=self.spin_t1_thresh.value(),
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            s = dlg.get_settings()
            self._quesp_pools = s["pools"]
            self.rb_pseudovoigt.setChecked(s["rb_pv_checked"])
            self.rb_lorentzian.setChecked(not s["rb_pv_checked"])
            self.chk_keep_gl.setChecked(s["keep_gl"])
            self.chk_mt_superlorentz.setChecked(s["super_lorentz"])
            self.chk_multipool.setChecked(s["multipool"])
            self.combo_model.setCurrentIndex(s["model_idx"])
            self.spin_thresh.setValue(s["thresh"])
            self.spin_t1_thresh.setValue(s["t1_thresh"])
            n = len(self._quesp_pools)
            self.lbl_quesp_pools.setText(f"{n} pool{'s' if n != 1 else ''} selected")

    def _display_loaded_figures(self, path: str):
        """Show the contents of an external .mat/.npz in this QUESP tab.

        2-D numeric arrays (parametric maps such as this ksw map) are shown as
        LIVE maps via canvas.show_map, so the colormap / clim / title / font
        controls all work like any other map.  Line-plot figures (our exported
        spectra/fingerprints) are reconstructed and shown as a rendered image.
        Each item is added to the Display selector as "Loaded — <name>"."""
        from PyQt6.QtWidgets import QMessageBox
        import os, re
        from my_gui.fig_export import _load_any, load_figures, render_figure_rgba
        try:
            data = _load_any(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", f"Could not read file:\n{exc}")
            return
        base = os.path.basename(path)

        # 2-D numeric arrays → live maps (image keys like ax0_image0, or any 2-D)
        map_items = []
        for k, v in data.items():
            a = np.asarray(v)
            if a.ndim == 2 and a.size > 4 and a.dtype.kind in 'fiu':
                map_items.append((str(k), a.astype(float)))
        has_lines = any(re.search(r'ax\d+_.+_(x|y)$', str(k)) for k in data)

        self._loaded_maps    = getattr(self, '_loaded_maps', {})
        self._loaded_display = getattr(self, '_loaded_display', [])
        labels = []

        for k, arr in map_items:
            nm = base if (len(map_items) == 1 and not has_lines) else f"{base} : {k}"
            self._loaded_maps[nm] = arr
            labels.append(f"Loaded — {nm}")

        # Only reconstruct line-plot figures when there are no maps to show live.
        if has_lines and not map_items:
            try:
                figs = load_figures(path)
            except Exception:
                figs = []
            for i, fg in enumerate(figs):
                nm = base if len(figs) == 1 else f"{base} — fig {i + 1}"
                self._loaded_display.append((nm, render_figure_rgba(fg)))
                labels.append(f"Loaded — {nm}")

        if not labels:
            QMessageBox.information(
                self, "Nothing to display",
                "No 2-D image/map arrays or plotted-figure data were found in:\n"
                f"{base}")
            return

        # De-dup the rendered-figure store (reloading replaces).
        seen, dedup = set(), []
        for nm, im in reversed(self._loaded_display):
            if nm in seen:
                continue
            seen.add(nm); dedup.append((nm, im))
        self._loaded_display = list(reversed(dedup))

        self.combo_display.blockSignals(True)
        for lbl in labels:
            if self.combo_display.findText(lbl) < 0:
                self.combo_display.addItem(lbl)
        self.combo_display.blockSignals(False)
        self.combo_display.setCurrentText(labels[0])

    def _show_reference(self):
        img = None
        sl = self.spin_slice.value() - 1
        if self._quesp_raw is not None:
            img = self._quesp_raw[:, :, sl, 0].astype(float)
        elif self._quesp_eff is not None:
            img = self._quesp_eff[:, :, 0]
        if img is not None:
            self.canvas.show_map(img, "Reference (first image)", cmap="gray")

    def _show_quesp_ref(self):
        if self._quesp_eff is not None:
            mean_eff = self._quesp_eff.mean(axis=-1)
            cmap = self.plot_bar.get_cmap()
            vmin, vmax = self.plot_bar.get_clim()
            self.canvas.show_map(mean_eff, "Mean QUESP Effect",
                                 cmap=cmap, vmin=vmin, vmax=vmax)

    def _on_img_slider(self, value: int):
        """Called when the B1 image slider changes."""
        self._show_quesp_b1_image()

    def _on_canvas_scroll(self, event):
        """Mouse wheel over canvas scrolls through QUESP B1 images."""
        if self.combo_display.currentText() != "QUESP Images":
            return
        step = 1 if event.step > 0 else -1
        new_val = max(0, min(self._img_slider.maximum(),
                             self._img_slider.value() + step))
        self._img_slider.setValue(new_val)

    def _show_quesp_b1_image(self):
        """Display a single raw QUESP saturation image for the selected B1 index."""
        if self._quesp_raw is None:
            return
        H, W, n_sl, n_sat = self._quesp_raw.shape
        sl = self.spin_slice.value() - 1
        sl = min(sl, n_sl - 1)

        # Determine B1 amplitude labels from info or parsed powers
        info = self._quesp_info or {}
        sat_amps = np.array(info.get('sat_amplitudes', []), dtype=float)
        if len(sat_amps) == n_sat:
            b1_vals = sat_amps
        else:
            # Fall back to parsed B1 powers (may not cover all n_sat)
            b1_vals = np.array(self._parse_b1_powers(), dtype=float)
            if len(b1_vals) != n_sat:
                b1_vals = np.arange(n_sat, dtype=float)

        # Update slider range
        self._img_slider.setMaximum(n_sat - 1)
        sat_idx = self._img_slider.value()
        sat_idx = min(sat_idx, n_sat - 1)

        amp_val = float(b1_vals[sat_idx]) if sat_idx < len(b1_vals) else 0.0
        self._img_slider_label.setText(
            f"B1 = {amp_val:.1f} \u00b5T  ({sat_idx + 1}/{n_sat})"
        )

        img = self._quesp_raw[:, :, sl, sat_idx].astype(float)
        cmap = self.plot_bar.get_cmap()
        vmin, vmax = self.plot_bar.get_clim()

        _ct = getattr(self, 'edit_map_title', None)
        _ctitle = _ct.text().strip() if _ct else ""
        title = _ctitle or f"QUESP Image \u2014 B1 = {amp_val:.1f} \u00b5T  ({sat_idx + 1}/{n_sat})"

        qfs = self.plot_bar.get_font_sizes()
        self.canvas.show_map(img, title, cmap=cmap, vmin=vmin, vmax=vmax, **qfs)

    def _refresh_display(self):
        if hasattr(self, "chk_dark_bg"):
            self.canvas._dark_bg = self.chk_dark_bg.isChecked()
        if hasattr(self, "chk_logmap"):
            self.canvas._log_map = self.chk_logmap.isChecked()
        self._dc_annot = None   # reset data cursor (axes rebuilt on show_map)
        choice = self.combo_display.currentText()
        cmap   = self.plot_bar.get_cmap()
        vmin, vmax = self.plot_bar.get_clim()

        # Custom title
        _ct = getattr(self, 'edit_map_title', None)
        _ctitle = _ct.text().strip() if _ct else ""

        # Apply Phantom_outline mask if available
        phantom = next(
            (r for r in getattr(self, '_last_rois', []) if r.name == "Phantom_outline"),
            None,
        )

        def _mask(arr: np.ndarray) -> np.ndarray:
            if phantom is None or arr is None:
                return arr
            msk = phantom.mask
            if msk.shape[:2] == arr.shape[:2]:
                return np.where(msk, arr, np.nan)
            return arr

        qfs = self.plot_bar.get_font_sizes()
        n_p = self.spin_n_protons.value()

        # Grayscale underlay (1st raw frame of this tab) for "ROIs + Bg" mode.
        _base = None
        if getattr(self, "_quesp_raw", None) is not None:
            _si = self.spin_slice.value() - 1
            _base = np.asarray(self._quesp_raw)[:, :, _si, 0]
        elif getattr(self, "_quesp_eff", None) is not None:
            _base = np.asarray(self._quesp_eff)[:, :, 0]

        # User-picked background overrides the default raw underlay.
        if getattr(self, "_roi_bg_img", None) is not None:
            _base = self._roi_bg_img

        # Show/hide B1 slider frame
        _show_b1_slider = (choice == "QUESP Images")
        self._img_slider_frame.setVisible(_show_b1_slider)

        # Loaded external .mat/.npz content.
        if choice.startswith("Loaded — "):
            nm = choice[len("Loaded — "):]
            arr = getattr(self, '_loaded_maps', {}).get(nm)
            if arr is not None:
                # LIVE map — colormap / clim / title / font all controllable,
                # exactly like the built-in maps.
                a = _mask(arr)
                fin = a[np.isfinite(a)] if a is not None else np.array([])
                _vmx = vmax if vmax is not None else (
                    float(np.nanpercentile(fin, 99)) if fin.size else None)
                if self.chk_roi_bg.isChecked():
                    _union = roi_union_mask(getattr(self, "_last_rois", []), arr.shape)
                    if _base is not None and _union is not None:
                        _ovc = cmap if str(cmap).lower() not in ("gray", "greys", "greys_r") else "jet"
                        self.canvas.show_map_over_raw(
                            arr, _base, _union, _ctitle or nm,
                            cmap=_ovc, vmin=(vmin if vmin is not None else 0),
                            vmax=_vmx, **qfs)
                        return
                self.canvas.show_map(a, _ctitle or nm, cmap=cmap,
                                     vmin=(vmin if vmin is not None else 0),
                                     vmax=_vmx, **qfs)
                return
            rgba = dict(getattr(self, '_loaded_display', [])).get(nm)
            if rgba is not None:
                fig = self.canvas._fig
                fig.clf()
                fig.patch.set_facecolor(
                    'black' if getattr(self.canvas, '_dark_bg', False) else 'white')
                ax = fig.add_subplot(111)
                ax.imshow(rgba)
                ax.axis('off')
                if _ctitle:
                    ax.set_title(_ctitle, fontsize=qfs.get('title_fs', 12))
                self.canvas.draw()
            return

        if choice == "Reference Image":
            self._show_reference()
        elif choice == "QUESP Images":
            self._show_quesp_b1_image()
        elif choice == "fs Map (mM)":
            if self._fs_map is not None:
                fs_mM = _mask(self._fs_map) * (110000.0 / n_p)
                fin = fs_mM[np.isfinite(fs_mM) & (fs_mM > 0)] if fs_mM is not None else np.array([])
                vmax_mM = vmax if vmax is not None else (float(np.nanpercentile(fin, 99)) if fin.size else 10.0)
                if self.chk_roi_bg.isChecked():
                    _union = roi_union_mask(getattr(self, "_last_rois", []), fs_mM.shape)
                    if _base is not None and _union is not None:
                        _ovc = cmap if str(cmap).lower() not in ("gray", "greys", "greys_r") else "jet"
                        self.canvas.show_map_over_raw(
                            fs_mM, _base, _union, _ctitle or "fs  (mM)",
                            cmap=_ovc, vmin=vmin or 0, vmax=vmax_mM, **qfs)
                        return
                self.canvas.show_map(fs_mM,
                                     _ctitle or "fs  (mM)",
                                     cmap=cmap, vmin=vmin or 0, vmax=vmax_mM, **qfs)
        elif choice == "ksw Map (s⁻¹)":
            if self._ksw_map is not None:
                if self.chk_roi_bg.isChecked():
                    _union = roi_union_mask(getattr(self, "_last_rois", []), self._ksw_map.shape)
                    if _base is not None and _union is not None:
                        _ovc = cmap if str(cmap).lower() not in ("gray", "greys", "greys_r") else "jet"
                        self.canvas.show_map_over_raw(
                            self._ksw_map, _base, _union, _ctitle or "ksw map (s⁻¹)",
                            cmap=_ovc, vmin=vmin, vmax=vmax, **qfs)
                        return
                self.canvas.show_map(_mask(self._ksw_map), _ctitle or "ksw map (s⁻¹)",
                                     cmap=cmap, vmin=vmin, vmax=vmax, **qfs)
        elif choice == "R² Map":
            if self._rsq_map is not None:
                _vmin = vmin if vmin is not None else 0
                _vmax = vmax if vmax is not None else 1
                if self.chk_roi_bg.isChecked():
                    _union = roi_union_mask(getattr(self, "_last_rois", []), self._rsq_map.shape)
                    if _base is not None and _union is not None:
                        _ovc = cmap if str(cmap).lower() not in ("gray", "greys", "greys_r") else "jet"
                        self.canvas.show_map_over_raw(
                            self._rsq_map, _base, _union, _ctitle or "R² map",
                            cmap=_ovc, vmin=_vmin, vmax=_vmax, **qfs)
                        return
                self.canvas.show_map(_mask(self._rsq_map), _ctitle or "R² map",
                                     cmap=cmap, vmin=_vmin, vmax=_vmax, **qfs)
        elif choice == "CEST Peak Range":
            self._show_cest_peak_range()

    def _show_cest_peak_range(self):
        """
        Display a schematic CEST peak-range diagram directly on the canvas.

        Shows a representative Lorentzian z-spectrum with:
          • A vertical dashed line at the saturation offset (ppm)
          • A shaded region of ±half-bandwidth around the offset
          • Annotation showing the exact range
        """
        center  = self.spin_offset.value()
        half_bw = self.spin_peak_half_bw.value()

        # Build ppm axis
        ppm = np.linspace(-(center + 3.0), (center + 3.0), 500)

        # Schematic z-spectrum: water Lorentzian + CEST dip at ±center ppm
        def _lor(x, amp, w, x0):
            return amp * (w / 2) ** 2 / ((w / 2) ** 2 + (x - x0) ** 2)

        z_schematic = 1.0 - _lor(ppm, 0.85, 1.6, 0.0)
        z_schematic -= _lor(ppm, 0.04, 0.8,  center)
        z_schematic -= _lor(ppm, 0.04, 0.8, -center)
        z_schematic  = np.clip(z_schematic, 0.0, 1.0)

        # Draw directly into the ROICanvas figure
        self.canvas._fig.clf()
        self.canvas._ax = self.canvas._fig.add_subplot(111)
        ax = self.canvas._ax
        ax.set_facecolor('#f8f8f8')

        ax.plot(ppm, z_schematic, '-', color='#1f77b4', lw=2.0,
                label='Z(Δω)  [schematic]')

        # Mark ±center
        ax.axvline( center, color='#d62728', lw=1.8, ls='--',
                    label=f'+{center:.2f} ppm')
        ax.axvline(-center, color='#2ca02c', lw=1.8, ls='--',
                    label=f'−{center:.2f} ppm')

        if half_bw > 0.0:
            ax.axvspan( center - half_bw,  center + half_bw,
                        color='#d62728', alpha=0.15,
                        label=f'+{center:.2f} ± {half_bw:.2f} ppm')
            ax.axvspan(-center - half_bw, -center + half_bw,
                        color='#2ca02c', alpha=0.15,
                        label=f'−{center:.2f} ± {half_bw:.2f} ppm')
            _desc = (f"Extraction range:  +{center:.2f} ± {half_bw:.2f} ppm  "
                     f"(+{center - half_bw:.2f} to +{center + half_bw:.2f})   "
                     f"and  −{center:.2f} ± {half_bw:.2f} ppm")
        else:
            _desc = f"Single-point extraction at ±{center:.2f} ppm"

        ax.set_xlabel('Δω (ppm)', fontsize=10)
        ax.set_ylabel('Z(Δω)',    fontsize=10)
        ax.set_title('CEST Peak Range', fontsize=12, fontweight='bold')
        ax.set_xlim(ppm[-1], ppm[0])   # positive ppm on left (CEST convention)
        ax.set_ylim(-0.05, 1.10)
        ax.legend(fontsize=8, loc='lower right', framealpha=0.85)
        ax.grid(True, alpha=0.25)
        ax.text(0.5, -0.14, _desc,
                transform=ax.transAxes, ha='center', fontsize=9, color='#555',
                wrap=True)
        self.canvas._fig.tight_layout(rect=[0, 0.06, 1, 1])
        self.canvas._img_data  = None   # no spatial image
        self.canvas._last_title = 'CEST Peak Range'
        self.canvas._apply_bg_theme()   # last: honour the "Bg" toggle here too
        self.canvas.draw()

    def _plot_quesp_all_models(self):
        """
        Three-panel QUESP plot per ROI: [Regular | Inverse | OmegaPlot]
        Two rows if both Pseudo-Voigt and Lorentzian are selected.
        Matches MATLAB QUESP_fit_display.m 3-panel layout.
        """
        if self._quesp_eff is None or self._w1_vals is None:
            return
        import matplotlib.pyplot as plt

        rois = self.canvas.get_rois()
        H, W, N = self._quesp_eff.shape
        w1_vals = self._w1_vals        # shape (N,)
        inv_w1sq = np.where(w1_vals > 0, 1.0 / w1_vals**2, np.nan)

        # Peak fit types to show
        fit_types = []
        if self.rb_pseudovoigt.isChecked():
            fit_types.append("Pseudo-Voigt")
        if self.rb_lorentzian.isChecked():
            fit_types.append("Lorentzian")
        if not fit_types:
            fit_types = ["Pseudo-Voigt"]
        n_rows = len(fit_types)

        # ── standalone omega fit returning (fs, ksw, rsq) ───────────────────
        def _fit_omega_standalone(xx, y, R1A_val):
            def _omega_model(xx_, fb, kb):
                return R1A_val * (1.0 / (fb * kb + 1e-30) + kb / (fb + 1e-30) * xx_)
            fin = np.isfinite(xx) & np.isfinite(y) & (y > 0)
            if fin.sum() < 3:
                return 0.0, 0.0, 0.0
            xf, yf = xx[fin], y[fin]
            sigma_w = np.sqrt(np.clip(yf, 1e-20, None))
            best = None
            for fb0, kb0 in [(0.01, 500.0), (0.001, 2000.0), (0.05, 200.0)]:
                try:
                    popt, pcov = curve_fit(
                        _omega_model, xf, yf,
                        p0=[fb0, kb0],
                        bounds=([1e-6, 1.0], [1.0, 1e5]),
                        sigma=sigma_w, absolute_sigma=True, maxfev=3000,
                    )
                    fb_fit, kb_fit = float(popt[0]), float(popt[1])
                    y_pred = _omega_model(xf, fb_fit, kb_fit)
                    ss_res = float(np.sum((yf - y_pred)**2))
                    ss_tot = float(np.sum((yf - yf.mean())**2))
                    rsq = float(np.clip(1.0 - ss_res / (ss_tot + 1e-12), 0, 1))
                    if best is None or rsq > best[2]:
                        best = (fb_fit, kb_fit, rsq)
                except Exception:
                    continue
            return best if best else (0.0, 0.0, 0.0)

        # ── plot one ROI into a figure ───────────────────────────────────────
        def _make_roi_figure(roi_name, curve_mean, mask_roi):
            fig, axes = plt.subplots(n_rows, 3,
                                     figsize=(14, 4.5 * n_rows),
                                     squeeze=False)
            fig.suptitle(f"ROI: {roi_name}", fontsize=14, fontweight='bold')

            # T1/R1A for this ROI
            if self._t1_map is not None and mask_roi is not None and mask_roi.any():
                t1_ms = float(np.nanmedian(self._t1_map[mask_roi]))
            elif self._t1_map is not None:
                t1_ms = float(np.nanmedian(self._t1_map[self._t1_map > 0])) if np.any(self._t1_map > 0) else 2000.0
            else:
                t1_ms = 2000.0
            R1A = 1000.0 / max(t1_ms, 1.0)

            DATA_COLOR = '#1f77b4'   # matplotlib tab:blue
            FIT_COLOR  = '#d62728'   # matplotlib tab:red

            for row_idx, fit_type in enumerate(fit_types):
                ax_reg = axes[row_idx][0]
                ax_inv = axes[row_idx][1]
                ax_omg = axes[row_idx][2]

                # Row label
                for ax_, title in [(ax_reg, "Regular"), (ax_inv, "Inverse"), (ax_omg, "Omega Plot")]:
                    ax_.set_title(f"{fit_type} — {title}", fontsize=10, fontweight='bold')
                    ax_.grid(True, alpha=0.25, linestyle='--')

                # ── Regular ────────────────────────────────────────────────────
                ax_reg.plot(w1_vals, curve_mean, 'o', color=DATA_COLOR,
                            markersize=7, markerfacecolor='none', markeredgewidth=1.8,
                            label='Data')
                ax_reg.set_xlabel("ω₁  (rad/s)", fontsize=9)
                ax_reg.set_ylabel("MTRRex", fontsize=9)
                ax_reg.set_xlim(left=0); ax_reg.set_ylim(bottom=0)

                fs_r, ksw_r, rsq_r = _fit_regular(curve_mean, w1_vals, R1A,
                                                   self.chk_pulsed.isChecked(),
                                                   self.spin_tp.value()*1e-3,
                                                   self.spin_rd.value()*1e-3)
                if ksw_r > 0:
                    w1_fine = np.linspace(0, w1_vals[-1]*1.1, 300)
                    ax_reg.plot(w1_fine, _mtr_rex_cw(w1_fine, fs_r, ksw_r, R1A),
                                '-', color=FIT_COLOR, linewidth=2,
                                label=(f"QUESP fit (ROI fit),\n"
                                       f"$f_s$={fs_r:.3e}\n"
                                       f"$k_{{sw}}$={ksw_r:.1f} s⁻¹\n"
                                       f"R²={rsq_r:.3f}"))
                ax_reg.legend(fontsize=8, frameon=True, framealpha=0.9,
                              edgecolor='gray', loc='upper left')

                # ── Inverse ────────────────────────────────────────────────────
                ax_inv.plot(w1_vals, curve_mean, 'o', color=DATA_COLOR,
                            markersize=7, markerfacecolor='none', markeredgewidth=1.8,
                            label='Data')
                ax_inv.set_xlabel("ω₁  (rad/s)", fontsize=9)
                ax_inv.set_ylabel("MTR$_{Rex}$", fontsize=9)
                ax_inv.set_xlim(left=0); ax_inv.set_ylim(bottom=0)

                fs_i, ksw_i, rsq_i = _fit_inverse(curve_mean, w1_vals, R1A,
                                                   self.chk_pulsed.isChecked(),
                                                   self.spin_tp.value()*1e-3,
                                                   self.spin_rd.value()*1e-3)
                if ksw_i > 0:
                    w1_fine = np.linspace(0, w1_vals[-1]*1.1, 300)
                    ax_inv.plot(w1_fine, _mtr_rex_cw(w1_fine, fs_i, ksw_i, R1A),
                                '-', color=FIT_COLOR, linewidth=2,
                                label=(f"QUESP fit (ROI fit),\n"
                                       f"$f_s$={fs_i:.3e}\n"
                                       f"$k_{{sw}}$={ksw_i:.1f} s⁻¹\n"
                                       f"R²={rsq_i:.3f}"))
                ax_inv.legend(fontsize=8, frameon=True, framealpha=0.9,
                              edgecolor='gray', loc='upper left')

                # ── OmegaPlot ──────────────────────────────────────────────────
                valid = (curve_mean > 1e-10) & np.isfinite(curve_mean)
                y_inv_data = np.where(valid, 1.0 / curve_mean, np.nan)
                keep = valid & (y_inv_data > 0) & np.isfinite(inv_w1sq)
                if np.any(keep):
                    ax_omg.plot(inv_w1sq[keep], y_inv_data[keep], 'o', color=DATA_COLOR,
                                markersize=7, label='Data', zorder=3)
                ax_omg.set_xlabel("1/ω₁²  (s²/rad²)", fontsize=9)
                ax_omg.set_ylabel("1/MTR$_{Rex}$", fontsize=9)

                fb_o, kb_o, rsq_o = _fit_omega_standalone(inv_w1sq, y_inv_data, R1A)
                if kb_o > 0 and np.any(keep):
                    def _omega_model_local(xx_, fb_, kb_):
                        return R1A * (1.0 / (fb_ * kb_ + 1e-30) + kb_ / (fb_ + 1e-30) * xx_)
                    x_line = np.linspace(float(np.nanmin(inv_w1sq[keep])),
                                         float(np.nanmax(inv_w1sq[keep])), 200)
                    ax_omg.plot(x_line, _omega_model_local(x_line, fb_o, kb_o),
                                '--', color=FIT_COLOR, linewidth=2,
                                label=(f"Omega-plot fit (ROI fit),\n"
                                       f"$f_s$={fb_o:.3e}\n"
                                       f"$k_{{sw}}$={kb_o:.1f} s⁻¹\n"
                                       f"R²={rsq_o:.3f}"))
                ax_omg.legend(fontsize=8, frameon=True, framealpha=0.9,
                              edgecolor='gray', loc='upper right')
                ax_omg.grid(True, alpha=0.25, linestyle='--')

            fig.tight_layout()
            plt.show(block=False)
            return fig

        # ── Iterate ROIs ─────────────────────────────────────────────────────
        if rois:
            for idx, roi in enumerate(rois):
                mask = roi.mask[:H, :W]
                if not mask.any():
                    continue
                curve = self._quesp_eff[mask].mean(axis=0)
                _make_roi_figure(roi.name, curve, mask)
        else:
            curve = self._quesp_eff.reshape(-1, N).mean(axis=0)
            _make_roi_figure("All voxels", curve, None)

    # ─────────────────────────────────────────────────────────────────────
    # ROI Manager integration
    # ─────────────────────────────────────────────────────────────────────

    def connect_roi_manager(self, roi_manager):
        roi_manager.connect_canvas(self.canvas)
        roi_manager.rois_changed.connect(self._update_roi_stats)
        self._last_rois: list = []

    def _update_roi_stats(self, rois: list):
        self._last_rois = list(rois)
        # Stats shown on demand via the ROI Stats Table button

    def _show_roi_table(self):
        """ROI statistics = the ROI-MEAN QUESP fit, reported SEPARATELY for each
        variant (MTR_asym / MTR_Rex / Ω-plot) — the same numbers shown in the
        ROI-Spectra legend, not the noisier per-voxel map average."""
        import warnings
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                     QTableWidget, QTableWidgetItem, QPushButton,
                                     QMessageBox)
        from PyQt6.QtCore import Qt
        if self._quesp_eff is None or self._w1_vals is None:
            QMessageBox.information(self, "No Data",
                "Run QUESP analysis first (need MTR_Rex vs B1).")
            return
        rois = [r for r in getattr(self, '_last_rois', [])
                if r.name != 'Phantom_outline']
        H, W, N = self._quesp_eff.shape
        w1 = self._w1_vals
        w1sq = w1 ** 2
        inv_w1sq = np.where(w1sq > 0, 1.0 / np.where(w1sq > 0, w1sq, 1.0), np.nan)
        has_asym = self._quesp_asym is not None
        n_p    = self.spin_n_protons.value()
        pulsed = self.chk_pulsed.isChecked()
        tp_s   = self.spin_tp.value() * 1e-3
        rd_s   = self.spin_rd.value()
        mM     = 110000.0 / max(n_p, 1)

        def _R1A(msk):
            if self._t1_map is not None:
                arr = self._t1_map[:H, :W][msk] if msk is not None else self._t1_map.ravel()
                pos = arr[arr > 0]
                t1 = float(np.nanmedian(pos)) if pos.size else 2000.0
            else:
                t1 = 2000.0
            return 1000.0 / max(t1, 1.0)

        def _omega(mtr_rex, R1A):
            good = (mtr_rex > 0) & np.isfinite(mtr_rex) & np.isfinite(inv_w1sq)
            if good.sum() < 2:
                return (0.0, 0.0, 0.0)
            xf, yf = inv_w1sq[good], 1.0 / mtr_rex[good]
            def _om(xx, fb, kb):
                return R1A * (1.0 / (fb * kb + 1e-30) + kb / (fb + 1e-30) * xx)
            best = (0.0, 0.0, 0.0)
            for fb0, kb0 in [(0.01, 500.0), (0.001, 2000.0), (0.05, 200.0)]:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        popt, _ = curve_fit(_om, xf, yf, p0=[fb0, kb0],
                                            bounds=([1e-6, 1.0], [1.0, 1e5]),
                                            maxfev=3000)
                    yp = _om(xf, *popt)
                    ssr = float(np.sum((yf - yp) ** 2))
                    sst = float(np.sum((yf - yf.mean()) ** 2))
                    r2 = float(np.clip(1.0 - ssr / (sst + 1e-12), 0, 1))
                    if r2 > best[2]:
                        best = (float(popt[0]), float(popt[1]), r2)
                except Exception:
                    continue
            return best

        rows = []
        for roi in (rois if rois else [None]):
            if roi is not None:
                msk = roi.mask[:H, :W]
                if not msk.any():
                    continue
                name, npx = roi.name, int(msk.sum())
                c_rex  = self._quesp_eff[msk].mean(axis=0)
                c_asym = (self._quesp_asym[msk].mean(axis=0) if has_asym else c_rex)
            else:
                name, npx, msk = "All pixels", H * W, None
                c_rex  = self._quesp_eff.reshape(-1, N).mean(axis=0)
                c_asym = (self._quesp_asym.reshape(-1, N).mean(axis=0) if has_asym else c_rex)
            R1A = _R1A(msk)
            rows.append((name, npx, [
                ("MTR_asym (Regular)", _fit_regular(c_asym, w1, R1A, pulsed, tp_s, rd_s)),
                ("MTR_Rex (Inverse)",  _fit_inverse(c_rex,  w1, R1A, pulsed, tp_s, rd_s)),
                ("Ω-plot",             _omega(c_rex, R1A)),
            ]))
        if not rows:
            QMessageBox.information(self, "No Data", "No valid ROIs found.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("QUESP — ROI Statistics (per variant)")
        dlg.setMinimumSize(760, 420)
        vl = QVBoxLayout(dlg)
        vl.addWidget(QLabel("<b>ROI-mean QUESP fit</b> — f<sub>s</sub>, k<sub>sw</sub> and R² "
                            "reported separately for each variant (matches the ROI-Spectra legend)."))
        cols = ["ROI", "Variant", "fs", f"fs (mM, nH={n_p})", "ksw (s⁻¹)", "R²", "n (px)"]
        tbl = QTableWidget(sum(len(r[2]) for r in rows), len(cols))
        tbl.setHorizontalHeaderLabels(cols)
        tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        ri = 0
        for name, npx, variants in rows:
            for vi, (vname, (fs, ksw, r2)) in enumerate(variants):
                def _it(txt, bold=False, right=False):
                    it = QTableWidgetItem(txt)
                    if bold:
                        f = it.font(); f.setBold(True); it.setFont(f)
                    if right:
                        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    return it
                tbl.setItem(ri, 0, _it(name if vi == 0 else "", bold=True))
                tbl.setItem(ri, 1, _it(vname))
                tbl.setItem(ri, 2, _it(f"{fs:.3e}", right=True))
                tbl.setItem(ri, 3, _it(f"{fs * mM:.3g}", right=True))
                tbl.setItem(ri, 4, _it(f"{ksw:.1f}", right=True))
                tbl.setItem(ri, 5, _it(f"{r2:.3f}", right=True))
                tbl.setItem(ri, 6, _it(str(npx) if vi == 0 else "", right=True))
                ri += 1
        tbl.resizeColumnsToContents()
        tbl.horizontalHeader().setStretchLastSection(True)
        vl.addWidget(tbl)
        brow = QHBoxLayout(); brow.addStretch()
        _b = QPushButton("Close"); _b.clicked.connect(dlg.accept); brow.addWidget(_b)
        vl.addLayout(brow)
        dlg.exec()

    def _show_roi_spectra(self):
        """
        Per-ROI QUESP dialog — one tab per ROI.
        Each tab shows a 3-column panel per fit type (Pseudo-Voigt / Lorentzian):
          Col 0: Regular QUESP    — MTRRex vs ω₁
          Col 1: Inverse QUESP    — MTRRex vs ω₁
          Col 2: Omega Plot       — 1/MTRRex vs 1/ω₁²
        Legend shows mean ± std for fs, ksw, and R².
        """
        import warnings
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QPushButton as _QPB,
            QTabWidget, QWidget as _QW, QScrollArea,
        )

        if self._quesp_eff is None or self._w1_vals is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "No Data",
                                    "Load QUESP data and run analysis first.")
            return

        rois = [r for r in getattr(self, '_last_rois', [])
                if r.name != "Phantom_outline"]
        H, W, N = self._quesp_eff.shape
        w1_vals  = self._w1_vals
        w1_fine  = np.linspace(0, w1_vals[-1] * 1.1, 300)
        w1sq     = w1_vals ** 2
        # Safe reciprocal — avoids divide-by-zero RuntimeWarning
        inv_w1sq = np.where(w1sq > 0, 1.0 / np.where(w1sq > 0, w1sq, 1.0), np.nan)

        # Build series list: (label, curve_mtrex, curve_asym, color, mask or None)
        # curve_mtrex = MTRRex  (Inverse / OmegaPlot)
        # curve_asym  = MTR_asym (Regular)
        has_asym = self._quesp_asym is not None
        if rois:
            series = []
            for idx, roi in enumerate(rois):
                msk = roi.mask[:H, :W]
                if not msk.any():
                    continue
                c_rex  = self._quesp_eff[msk].mean(axis=0)
                c_asym = (self._quesp_asym[msk].mean(axis=0)
                          if has_asym else c_rex)
                clr = getattr(roi, 'color', f'C{idx % 10}')
                series.append((roi.name, c_rex, c_asym, clr, msk))
        else:
            c_rex  = self._quesp_eff.reshape(-1, N).mean(axis=0)
            c_asym = (self._quesp_asym.reshape(-1, N).mean(axis=0)
                      if has_asym else c_rex)
            series = [("All pixels", c_rex, c_asym, "tab:blue", None)]

        if not series:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "No Data", "No valid ROIs found.")
            return

        # Which fit-type rows to show
        fit_types = []
        if hasattr(self, 'rb_pseudovoigt') and self.rb_pseudovoigt.isChecked():
            fit_types.append("Pseudo-Voigt")
        if hasattr(self, 'rb_lorentzian') and self.rb_lorentzian.isChecked():
            fit_types.append("Lorentzian")
        if not fit_types:
            fit_types = ["Pseudo-Voigt"]
        n_rows = len(fit_types)

        n_p      = self.spin_n_protons.value()
        pulsed   = self.chk_pulsed.isChecked()
        tp_s     = self.spin_tp.value() * 1e-3
        rd_s     = self.spin_rd.value()
        DATA_CLR = '#1f77b4'   # blue for data circles
        FIT_CLR  = '#d62728'   # red for fit lines

        def _get_R1A(msk):
            if self._t1_map is not None:
                arr = (self._t1_map[:H, :W][msk] if msk is not None
                       else self._t1_map.ravel())
                pos = arr[arr > 0]
                t1_ms = float(np.nanmedian(pos)) if pos.size else 2000.0
            else:
                t1_ms = 2000.0
            return 1000.0 / max(t1_ms, 1.0)

        def _omega_fit(xf, yf, R1A_val):
            """Weighted nonlinear OmegaPlot fit. Returns (fb, kb, rsq) or None."""
            def _om(xx_, fb, kb):
                return R1A_val * (1.0 / (fb * kb + 1e-30) + kb / (fb + 1e-30) * xx_)
            sigma_w = np.sqrt(np.clip(yf, 1e-20, None))
            best = None
            for fb0, kb0 in [(0.01, 500.0), (0.001, 2000.0), (0.05, 200.0)]:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        popt, _ = curve_fit(
                            _om, xf, yf, p0=[fb0, kb0],
                            bounds=([1e-6, 1.0], [1.0, 1e5]),
                            sigma=sigma_w, absolute_sigma=True, maxfev=3000,
                        )
                    fb_fit, kb_fit = float(popt[0]), float(popt[1])
                    y_pred = _om(xf, fb_fit, kb_fit)
                    ss_res = float(np.sum((yf - y_pred)**2))
                    ss_tot = float(np.sum((yf - yf.mean())**2))
                    rsq = float(np.clip(1.0 - ss_res / (ss_tot + 1e-12), 0, 1))
                    if best is None or rsq > best[2]:
                        best = (fb_fit, kb_fit, rsq, _om)
                except Exception:
                    continue
            return best  # (fb, kb, rsq, model_fn) or None

        dark_bg = [False]   # "Bg" toggle state (black slide background)

        def _make_roi_tab(label, curve, curve_asym, color, msk,
                          in_mM: bool = False) -> _QW:
            tab   = _QW()
            vl_   = QVBoxLayout(tab)
            R1A   = _get_R1A(msk)
            mM_scale = 110000.0 / max(n_p, 1)
            # Current font sizes (spinbox values) — live-updated on rebuild.
            title_fs  = spin_title.value()
            main_fs   = spin_main.value()
            axes_fs   = spin_axes.value()
            ticks_fs  = spin_ticks.value()
            legend_fs = spin_legend.value()

            fig = Figure(figsize=(13, 4.2 * n_rows), facecolor='white')
            fig.suptitle(f"ROI: {label}", fontproperties=_fp(main_fs, bold=True), y=1.01)

            for row_idx, fit_type in enumerate(fit_types):
                base = row_idx * 3 + 1   # subplot index (1-based)

                ax_reg = fig.add_subplot(n_rows, 3, base)
                ax_inv = fig.add_subplot(n_rows, 3, base + 1)
                ax_omg = fig.add_subplot(n_rows, 3, base + 2)

                # ── Row label as subtitle (omit fit-type prefix when only one) ─
                _pref = f"{fit_type}  —  " if n_rows > 1 else ""
                ax_reg.set_title(f"{_pref}Regular", fontproperties=_fp(title_fs, bold=True))
                ax_inv.set_title(f"{_pref}Inverse", fontproperties=_fp(title_fs, bold=True))
                ax_omg.set_title(f"{_pref}Omega Plot", fontproperties=_fp(title_fs, bold=True))

                # ── Regular QUESP (MTR_asym = Zref − Zlab) ───────────────
                ax_reg.plot(w1_vals, curve_asym, 'o', color=DATA_CLR, markersize=10,
                            markerfacecolor='white', markeredgewidth=2.2,
                            markeredgecolor=DATA_CLR, zorder=4, label='Data')
                ax_reg.set_xlabel("ω₁  (rad/s)", fontproperties=_fp(axes_fs))
                ax_reg.set_ylabel("MTR$_{asym}$", fontproperties=_fp(axes_fs))
                ax_reg.tick_params(labelsize=ticks_fs)
                ax_reg.set_xlim(left=0)
                _y_lo = float(np.nanmin(curve_asym)) if curve_asym.size else 0.0
                _y_hi = float(np.nanmax(curve_asym)) if curve_asym.size else 0.1
                _pad  = (_y_hi - _y_lo) * 0.15 + 1e-6
                ax_reg.set_ylim(_y_lo - _pad, _y_hi + _pad)
                ax_reg.grid(True, alpha=0.25, ls='--')

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fs_r, ksw_r, rsq_r = _fit_regular(
                        curve_asym, w1_vals, R1A, pulsed, tp_s, rd_s)
                if ksw_r > 0:
                    # Show this model's own ROI-fit values (not the voxelwise map)
                    if in_mM:
                        _fs_lbl = f"$f_s$={fs_r * mM_scale:.3g} mM (ROI fit)"
                    else:
                        _fs_lbl = f"$f_s$={fs_r:.3e} (ROI fit)"
                    ax_reg.plot(w1_fine,
                                _mtr_asym_regular(w1_fine, fs_r, ksw_r, R1A, tp_s, rd_s),
                                '-', color=FIT_CLR, lw=2,
                                label=(f"QUESP fit,\n"
                                       f"{_fs_lbl}\n"
                                       f"$k_{{sw}}$={ksw_r:.1f} s⁻¹\n"
                                       f"R²={rsq_r:.3f}"))
                ax_reg.legend(prop=_fp(legend_fs), frameon=True, framealpha=0.9, loc='upper left')

                # ── Inverse QUESP ─────────────────────────────────────────
                ax_inv.plot(w1_vals, curve, 'o', color=DATA_CLR, markersize=10,
                            markerfacecolor='white', markeredgewidth=2.2,
                            markeredgecolor=DATA_CLR, zorder=4, label='Data')
                ax_inv.set_xlabel("ω₁  (rad/s)", fontproperties=_fp(axes_fs))
                ax_inv.set_ylabel("MTR$_{Rex}$", fontproperties=_fp(axes_fs))
                ax_inv.tick_params(labelsize=ticks_fs)
                ax_inv.set_xlim(left=0)
                _y_lo_i = float(np.nanmin(curve)) if curve.size else 0.0
                _y_hi_i = float(np.nanmax(curve)) if curve.size else 0.1
                _pad_i  = (_y_hi_i - _y_lo_i) * 0.15 + 1e-6
                ax_inv.set_ylim(_y_lo_i - _pad_i, _y_hi_i + _pad_i)
                ax_inv.grid(True, alpha=0.25, ls='--')

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fs_i, ksw_i, rsq_i = _fit_inverse(curve, w1_vals, R1A, pulsed, tp_s, rd_s)
                if ksw_i > 0:
                    # Show this model's own ROI-fit values (not the voxelwise map)
                    if in_mM:
                        _fs_lbl = f"$f_s$={fs_i * mM_scale:.3g} mM (ROI fit)"
                    else:
                        _fs_lbl = f"$f_s$={fs_i:.3e} (ROI fit)"
                    ax_inv.plot(w1_fine, _mtr_rex_cw(w1_fine, fs_i, ksw_i, R1A),
                                '-', color=FIT_CLR, lw=2,
                                label=(f"QUESP fit,\n"
                                       f"{_fs_lbl}\n"
                                       f"$k_{{sw}}$={ksw_i:.1f} s⁻¹\n"
                                       f"R²={rsq_i:.3f}"))
                ax_inv.legend(prop=_fp(legend_fs), frameon=True, framealpha=0.9, loc='upper left')

                # ── Omega Plot ────────────────────────────────────────────
                ax_omg.set_xlabel("1/ω₁²  (s²/rad²)", fontproperties=_fp(axes_fs))
                ax_omg.set_ylabel("1/MTR$_{Rex}$", fontproperties=_fp(axes_fs))
                ax_omg.tick_params(labelsize=ticks_fs)
                ax_omg.grid(True, alpha=0.25, ls='--')

                safe_c = np.where(curve > 1e-10, curve, 1.0)
                y_inv  = np.where(curve > 1e-10, 1.0 / safe_c, np.nan)
                keep   = (curve > 1e-10) & np.isfinite(y_inv) & np.isfinite(inv_w1sq)
                if np.any(keep):
                    ax_omg.plot(inv_w1sq[keep], y_inv[keep], 'o', color=DATA_CLR,
                                markersize=10, markerfacecolor='white',
                                markeredgewidth=2.2, markeredgecolor=DATA_CLR,
                                label='Data', zorder=4)
                    result = _omega_fit(inv_w1sq[keep], y_inv[keep], R1A)
                    if result is not None:
                        fb_o, kb_o, rsq_o, om_fn = result
                        x_line = np.linspace(float(inv_w1sq[keep].min()),
                                             float(inv_w1sq[keep].max()), 200)
                        # Show this model's own ROI-fit values (not the voxelwise map)
                        if in_mM:
                            _fs_lbl = f"$f_s$={fb_o * mM_scale:.3g} mM (ROI fit)"
                        else:
                            _fs_lbl = f"$f_s$={fb_o:.3e} (ROI fit)"
                        ax_omg.plot(x_line, om_fn(x_line, fb_o, kb_o),
                                    '--', color=FIT_CLR, lw=2,
                                    label=(f"Omega-plot fit,\n"
                                           f"{_fs_lbl}\n"
                                           f"$k_{{sw}}$={kb_o:.1f} s⁻¹\n"
                                           f"R²={rsq_o:.3f}"))
                ax_omg.legend(prop=_fp(legend_fs), frameon=True, framealpha=0.9, loc='upper right')

            fig.tight_layout()
            from my_gui.fig_theme import apply_fig_dark_theme
            apply_fig_dark_theme(fig, dark_bg[0])   # last step before canvas
            fc = FigureCanvas(fig)
            fc.setMinimumHeight(int(4.2 * n_rows * 90))
            from matplotlib.backends.backend_qt import NavigationToolbar2QT as _NTBA
            vl_.addWidget(_NTBA(fc, tab))
            vl_.addWidget(fc, stretch=1)

            # Save button
            btn_sv = _QPB("Save figure…")
            def _save_tab(checked=False, _fig=fig, _lbl=label):
                from my_gui.fig_export import save_figure, FIG_EXPORT_FILTER
                p, _ = QFileDialog.getSaveFileName(
                    dlg, f"Save {_lbl}", f"quesp_roi_{_lbl}.png",
                    FIG_EXPORT_FILTER)
                if p:
                    save_figure(_fig, p, dpi=300)
            btn_sv.clicked.connect(_save_tab)
            sv_row = QHBoxLayout()
            sv_row.addWidget(btn_sv)
            sv_row.addStretch()
            vl_.addLayout(sv_row)
            return tab

        dlg = QDialog(self)
        dlg.setWindowTitle("ROI QUESP Spectra")
        dlg.resize(1100, int(420 * n_rows + 120))
        vl = QVBoxLayout(dlg)

        # ── fₛ → mM toggle ────────────────────────────────────────────────
        show_mM   = [False]
        _mM_scale = 110000.0 / max(n_p, 1)
        top_row   = QHBoxLayout()
        from PyQt6.QtWidgets import QCheckBox as _QCB
        chk_mM = _QCB(
            f"Show fₛ in mM  (× 110 000 ÷ {n_p} protons  =  ×{_mM_scale:.1f})"
        )
        chk_mM.setToolTip(
            "Convert solute fraction fₛ to millimolar concentration:\n"
            f"  [solute] (mM) = fₛ × 110 000 / n_protons\n"
            f"  with n_protons = {n_p}  →  scale factor = {_mM_scale:.2f}"
        )
        top_row.addWidget(chk_mM)
        chk_bg = _QCB("Bg")
        chk_bg.setToolTip(
            "Black background for the figure (for slides). Only the surround "
            "and labels flip - the plotted curves stay identical.")
        top_row.addWidget(chk_bg)

        # ── Font-customisation controls (family + Title / Main / Axes / Ticks /
        #    Legend sizes) — live-update every ROI figure. ─────────────────────
        from PyQt6.QtWidgets import (
            QComboBox as _QCombo, QSpinBox as _QSB, QLabel as _QL,
        )
        top_row.addWidget(_QL("Font:"))
        combo_ff = _QCombo(); combo_ff.setFixedWidth(140)
        combo_ff.setToolTip("Font family for titles, axis labels and legends")
        for _ff in ("Default", "Arial", "Times New Roman",
                    "Helvetica", "DejaVu Sans", "DejaVu Serif"):
            combo_ff.addItem(_ff)
        top_row.addWidget(combo_ff)

        def _mkfs(label, default, lo, hi, tip):
            top_row.addWidget(_QL(label))
            sp = _QSB(); sp.setRange(lo, hi); sp.setValue(default)
            sp.setFixedWidth(50); sp.setToolTip(tip)
            top_row.addWidget(sp)
            return sp

        spin_title  = _mkfs("Title:",   9, 5, 32, "Subplot title font size")
        spin_main   = _mkfs("Main:",   12, 5, 32, "Overall figure title font size")
        spin_axes   = _mkfs("Axes:",    9, 5, 28, "X / Y axis label font size")
        spin_ticks  = _mkfs("Ticks:",   9, 5, 24, "Tick label font size")
        spin_legend = _mkfs("Legend:",  8, 5, 24, "Legend font size")

        # Shared font-properties helper — honours the chosen family + size
        # (+ bold for titles). Used by every font site in _make_roi_tab.
        from matplotlib.font_manager import FontProperties as _FP
        def _fp(size, bold=False):
            ff = combo_ff.currentText()
            kw = {'size': size}
            if ff and ff.lower() != "default":
                kw['family'] = ff
            if bold:
                kw['weight'] = 'bold'
            return _FP(**kw)

        top_row.addStretch()
        vl.addLayout(top_row)

        tab_wgt = QTabWidget()

        def _rebuild_tabs():
            cur = tab_wgt.currentIndex()
            tab_wgt.clear()
            for _lbl, _cv, _ca, _clr, _msk in series:
                tab_wgt.addTab(
                    _make_roi_tab(_lbl, _cv, _ca, _clr, _msk, show_mM[0]),
                    _lbl,
                )
            tab_wgt.setCurrentIndex(max(0, min(cur, tab_wgt.count() - 1)))

        def _on_mM_toggle(state):
            show_mM[0] = bool(state)
            _rebuild_tabs()

        def _on_bg_toggle(_state):
            dark_bg[0] = chk_bg.isChecked()
            _rebuild_tabs()

        chk_mM.stateChanged.connect(_on_mM_toggle)
        chk_bg.toggled.connect(_on_bg_toggle)

        # Font family / size changes rebuild the ROI tabs, debounced so holding a
        # spin arrow (or scrubbing the combo) coalesces into a single rebuild
        # rather than one heavy rebuild per intermediate value.
        from PyQt6.QtCore import QTimer as _QTimer
        _font_timer = _QTimer(dlg)
        _font_timer.setSingleShot(True)
        _font_timer.setInterval(180)
        _font_timer.timeout.connect(_rebuild_tabs)
        def _sched_font(*_):
            _font_timer.start()
        for _sp in (spin_title, spin_main, spin_axes, spin_ticks, spin_legend):
            _sp.valueChanged.connect(_sched_font)
        combo_ff.currentIndexChanged.connect(_sched_font)

        _rebuild_tabs()
        vl.addWidget(tab_wgt, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_close = _QPB("Close")
        btn_close.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_close)
        vl.addLayout(btn_row)
        dlg.show()

    # ─────────────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────────────

    def _export_figure(self):
        from my_gui.fig_export import save_figure, FIG_EXPORT_FILTER
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", "quesp_figure.png",
            FIG_EXPORT_FILTER,
        )
        if path:
            try:
                save_figure(self.canvas._fig, path, dpi=300)
                self.lbl_proc_status.setText(f"Saved: {path}")
            except Exception as exc:
                self.lbl_proc_status.setText(f"Export error: {exc}")

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


# ─────────────────────────────────────────────────────────────────────────────
# QUESP pool & fit options dialog
# ─────────────────────────────────────────────────────────────────────────────

class _QUESPOptionsDialog(QDialog):
    """Dialog for configuring QUESP pool selection and fitting settings."""

    def __init__(self, selected_pools: list, rb_pv_checked: bool,
                 keep_gl: bool, super_lorentz: bool, multipool: bool,
                 model_idx: int, thresh: float,
                 t1_thresh: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("QUESP Pool & Fit Options")
        self.setMinimumWidth(500)

        vl = QVBoxLayout(self)

        # Pool selection
        self._pool_widget = PoolSelectionWidget(initial_selection=selected_pools, parent=self)
        vl.addWidget(self._pool_widget)

        form = QFormLayout()

        # Multi-pool fitting toggle
        self._chk_multipool = QCheckBox("Use multi-pool lineshape fitting")
        self._chk_multipool.setChecked(multipool)
        form.addRow("Multi-pool fitting:", self._chk_multipool)

        # Fit type
        fit_row = QHBoxLayout()
        self._rb_pv  = QRadioButton("Pseudo-Voigt")
        self._rb_lor = QRadioButton("Lorentzian")
        self._rb_pv.setChecked(rb_pv_checked)
        self._rb_lor.setChecked(not rb_pv_checked)
        _grp = QButtonGroup(self)
        _grp.addButton(self._rb_pv)
        _grp.addButton(self._rb_lor)
        fit_row.addWidget(self._rb_pv)
        fit_row.addWidget(self._rb_lor)
        fit_row.addStretch()
        form.addRow("Peak fit type:", fit_row)

        # Keep GL character
        self._chk_keep_gl = QCheckBox("Keep GL character same across peaks")
        self._chk_keep_gl.setChecked(keep_gl)
        self._chk_keep_gl.setToolTip(
            "Keep Gaussian-Lorentzian character the same across all peaks\n"
            "(except water + MT) during multi-pool lineshape fitting."
        )
        form.addRow("", self._chk_keep_gl)

        # MT super-Lorentzian
        self._chk_superlorentz = QCheckBox("Super-Lorentzian lineshape for MT pool")
        self._chk_superlorentz.setChecked(super_lorentz)
        self._chk_superlorentz.setToolTip(
            "Use super-Lorentzian lineshape for the MT pool."
        )
        form.addRow("", self._chk_superlorentz)

        # Model
        self._combo_model = QComboBox()
        self._combo_model.addItems(["Inverse (linear)", "Regular (nonlinear)"])
        self._combo_model.setCurrentIndex(model_idx)
        form.addRow("Model:", self._combo_model)

        # Pulsed correction

        # QUESP pos/neg threshold
        self._spin_thresh = QDoubleSpinBox()
        self._spin_thresh.setRange(0.5, 2.0)
        self._spin_thresh.setValue(thresh)
        self._spin_thresh.setSingleStep(0.01)
        self._spin_thresh.setDecimals(2)
        form.addRow("QUESP pos/neg thresh:", self._spin_thresh)

        # Min T1
        self._spin_t1_thresh = QDoubleSpinBox()
        self._spin_t1_thresh.setRange(0.0, 5000.0)
        self._spin_t1_thresh.setValue(t1_thresh)
        self._spin_t1_thresh.setSingleStep(50.0)
        form.addRow("Min T1 (ms):", self._spin_t1_thresh)

        vl.addLayout(form)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        vl.addWidget(btns)

    def get_settings(self) -> dict:
        return {
            "pools":        self._pool_widget.selected_pools(),
            "rb_pv_checked": self._rb_pv.isChecked(),
            "keep_gl":      self._chk_keep_gl.isChecked(),
            "super_lorentz": self._chk_superlorentz.isChecked(),
            "multipool":    self._chk_multipool.isChecked(),
            "model_idx":    self._combo_model.currentIndex(),
            "thresh":       self._spin_thresh.value(),
            "t1_thresh":    self._spin_t1_thresh.value(),
        }
