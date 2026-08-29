"""
results_tab.py

Results tab — load acquired data + dictionary, run dot-product matching,
display quantitative parametric maps with full plot customisation controls.

Pipeline used:
  Dictionary simulation : cest_mrf/dictionary/generation.py
                          → generate_mrf_cest_dictionary()
  Dot-product matching  : cest_mrf/metrics/dot_product.py
                          → dot_prod_matching()

Map keys returned by dot_prod_matching:
  dp, t1w, t2w, fs, ksw          (always)
  fs2, ksw2                       (2-pool CEST only)

Per-map display defaults (clim, cmap, scale, unit) match the original
visualize_and_save_results() function.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QComboBox, QCheckBox,
    QDoubleSpinBox, QSpinBox, QLineEdit, QGroupBox,
    QSplitter, QSizePolicy, QFrame, QSlider,
    QDialog, QFormLayout, QDialogButtonBox,
)
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from my_gui.fig_theme import apply_fig_dark_theme


# ── Standard matplotlib colormaps ─────────────────────────────────────────
# List of (internal_name, display_label)
_COLORMAPS_STD: list[tuple[str, str]] = [
    # Sequential
    ("viridis",  "viridis"),
    ("plasma",   "plasma"),
    ("inferno",  "inferno"),
    ("magma",    "magma"),
    ("cividis",  "cividis"),
    ("hot",      "hot"),
    ("afmhot",   "afmhot"),
    ("gray",     "gray"),
    ("bone",     "bone"),
    ("copper",   "copper"),
    ("YlOrRd",   "YlOrRd"),
    ("YlGnBu",   "YlGnBu"),
    ("Blues",    "Blues"),
    ("Greens",   "Greens"),
    # Diverging
    ("bwr",      "bwr"),
    ("coolwarm", "coolwarm"),
    ("RdBu",     "RdBu"),
    ("seismic",  "seismic"),
    # Other
    ("turbo",    "turbo"),
    ("jet",      "jet"),
    ("rainbow",  "rainbow"),
    # Black-background standard (from colormaps_dk.py)
    ("b_viridis", "b_viridis (black bg)"),
    ("b_winter",  "b_winter  (black bg)"),
    ("b_hot",     "b_hot     (black bg)"),
    ("b_plasma",  "b_plasma  (black bg)"),
    ("b_gray",    "b_gray    (black bg)"),
]

def _build_cmap_list() -> list[tuple[str, str]]:
    """
    Combine standard colormaps with any custom MRF colormaps loaded
    from T1cm.mat / T2cm.mat / differenceMaps.mat.
    Returns list of (internal_name, display_label).
    """
    combined = list(_COLORMAPS_STD)
    try:
        from my_gui.colormaps_gui import get_cmap_list
        custom = get_cmap_list()
        if custom:
            combined = [("── Custom MRF ──", "── Custom MRF ──")] + custom + \
                       [("── Standard ──", "── Standard ──")] + _COLORMAPS_STD
    except Exception:
        pass
    return combined


# ── Per-map display defaults (mirrors visualize_and_save_results) ────────────
# key → (display_label, cmap_name, vmin, vmax, scale_factor, unit_note)
_MAP_DEFAULTS: dict[str, tuple] = {
    "fs":   ("fₛ (mM)",                       "T1cm",    0,      40,    110e3/3, "×110e3/3 scale"),
    "fs_raw": ("fₛ (proton fraction)",       "T1cm",    0,    None,    1,       "no mM scale"),
    "ksw":  ("kₛw (s⁻¹)",                     "T2cm",    0,    5000,    1,       ""),
    "fs2":  ("fₛ₂ — Pool-2 (mM)",             "T1cm",    0,      30,    110e3/3, "×110e3/3 scale"),
    "fs2_raw": ("fₛ₂ (proton fraction)",     "T1cm",    0,    None,    1,       "no mM scale"),
    "ksw2": ("kₛw₂ — Pool 2 (s⁻¹)",          "T2cm",    0,    3000,    1,       ""),
    "dp":   ("Dot product",                  "magma",   0.99,  0.995,   1,       ""),
    "t1w":  ("T₁ water (ms)",                "T1cm",    0,   4000,    1000,    "×1000"),
    "t2w":  ("T₂ water (ms)",                "T2cm",    0,   2500,    1000,    "×1000"),
    "t1s":  ("T₁ solute (ms)",               "T1cm",    0,   3000,    1,       ""),
    "t2s":  ("T₂ solute (ms)",               "T2cm",    0,    100,    1,       ""),
    # MT pool keys (if dot_product_mt variant used)
    "fm":   ("fₘ — MT fraction (%)",         "T1cm",    0,     15,    100,     "×100"),
    "t1m":  ("T₁ MT (ms)",                    "T1cm",    0,      2200,   100,       ""),
    "t2m":  ("T₂ MT (µs)",                    "T2cm",    0,     20,    1e6,     "×1e6"),
    "kss":  ("kₛₛ (s⁻¹)",                     "T2cm",    0,     100,    1,       ""),
    "dmw":  ("Δω_MT (ppm)",                   "b_winter", -1,    1,    1,       ""),
}

# Rendered-figure title overrides — used ONLY for the plot title (which is
# mathtext-capable via ROICanvas._apply_title), so a real subscript can be shown
# where Unicode has no subscript glyph (there is no subscript M/T for "MT").
# The dropdown keeps the plain-text label from _MAP_DEFAULTS.
_TITLE_MATHTEXT: dict[str, str] = {
    "dmw": r"$\Delta\omega_{\mathrm{MT}}$ (ppm)",
}

# dp threshold for masking (from original script)
_DP_MASK_THRESH = 0.99


def _smart_scale(key: str, data: np.ndarray) -> float:
    """
    Return the effective display scale factor for *key*.

    For maps with a ×1000 factor (t1w, t2w …):
      • Dot-product matching stores values in SECONDS → ×1000 gives ms.
      • External MATLAB files often store values already in MILLISECONDS.

    Heuristic: if the median absolute finite value > 10 the data is already
    in ms, so return 1.0 (no re-scaling).  T1/T2 in seconds never exceed
    ~5 s, so the 10-unit threshold is unambiguous.

    All other scale factors (×110e3/3 for fs, ×100 for fm, etc.) are
    returned as-is — they are not subject to the units ambiguity.
    """
    if key not in _MAP_DEFAULTS:
        return 1.0
    nominal = _MAP_DEFAULTS[key][4]          # scale_factor column
    if nominal == 1000:
        finite = data[np.isfinite(data.astype(float))]
        if finite.size > 0 and float(np.nanmedian(np.abs(finite))) > 10.0:
            return 1.0                       # already in ms — skip ×1000
    return float(nominal)


class _NomConcDialog(QDialog):
    """Dialog to input nominal (ground truth) concentration per ROI."""
    def __init__(self, rois, existing: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Nominal Concentrations")
        self.setMinimumWidth(350)
        vl = QVBoxLayout(self)
        vl.addWidget(QLabel("<b>Enter ground truth concentration (mM) for each ROI:</b>"))
        form = QFormLayout()
        self._spins: dict[str, QDoubleSpinBox] = {}
        for roi in rois:
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 1000.0)
            sp.setDecimals(2)
            sp.setSuffix(" mM")
            sp.setValue(existing.get(roi.name, 0.0))
            self._spins[roi.name] = sp
            form.addRow(roi.name + ":", sp)
        vl.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        vl.addWidget(btns)

    def get_values(self) -> dict[str, float]:
        return {name: sp.value() for name, sp in self._spins.items()}


class ResultsTab(QWidget):
    """
    Parametric map viewer with interactive plot-customisation panel.

    Left panel  — file loading, matching controls, plot settings
    Right panel — matplotlib canvas
    """

    def __init__(self):
        super().__init__()
        self._quant_maps: dict | None = None
        self._data_fn: str | None = None
        self._dict_fn: str | None = None
        self._acquired_data: np.ndarray | None = None  # (n_meas, n_slices, Y, X) raw images
        self._dp_mask: np.ndarray | None = None   # boolean mask from dp > threshold
        self._dc_annot = None
        self._dc_cid: int | None = None
        self._dc_img_data: np.ndarray | None = None  # currently displayed 2D array
        self._current_map_key: str | None = None
        self._nom_conc: dict[str, float] = {}   # roi_name → nominal concentration (mM)
        self._rois: list = []                    # ROIs from ROI Manager
        self._rois_visible: bool = True          # overlay show/hide toggle
        self._picked_pixel: tuple[int, int] | None = None  # (row, col) from pixel-pick mode
        self._pp_cid: int | None = None          # mpl event connection for pixel pick
        self._pp_marker = None                   # axes artist for the crosshair marker

        # Window/Level (brightness–contrast) drag tool state — OsiriX-style
        self._im = None                          # current AxesImage (for live clim)
        self._wl_active: bool = False
        self._wl_cids: list = []                 # mpl event connections
        self._wl_drag: dict | None = None        # in-progress drag anchor/state
        # Per-map dragged window, remembered across redraws: (map_key, vmin, vmax).
        # Kept separate from the custom-clim spinboxes so a WL drag never changes
        # the colormap / fonts — it only re-scales the display window.
        self._wl_override: tuple | None = None

        # "ROIs + Bkg" overlay: optional user-picked grayscale background image
        # and a getter (installed by app.py) returning {modality: scan_path}.
        self._roi_bg_img = None
        self._scan_paths_getter = None

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer = QHBoxLayout(self)
        outer.addWidget(splitter)

        # ── Left panel — wrapped in a scroll area so Plot Customisation
        #    controls are always reachable even on small screens ──────────────
        from PyQt6.QtWidgets import QScrollArea as _QSA
        _left_scroll = _QSA()
        _left_scroll.setWidgetResizable(True)
        _left_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        _left_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        _left_scroll.setMinimumWidth(380)
        _left_scroll.setMaximumWidth(520)

        left = QWidget()
        left.setMinimumWidth(370)
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(8)
        _left_scroll.setWidget(left)

        # ═══════════════════════════════════════════════════════════════════
        # PATH 1 — Load inputs → Run matching
        # ═══════════════════════════════════════════════════════════════════
        grp_match = QGroupBox("Run Dot-Product Matching")
        grp_match.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #555; "
            "border-radius: 5px; margin-top: 6px; padding-top: 22px; }"
            "QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; left: 8px; top: 4px; font-size: 14px;"
            "padding: 0 4px; }"
        )
        ml = QVBoxLayout(grp_match)
        ml.setSpacing(6)

        # ── Side-by-side load buttons ──────────────────────────────────────
        load_row = QHBoxLayout()
        load_row.setSpacing(6)

        # Left cell — acquired data
        acq_cell = QVBoxLayout()
        acq_cell.setSpacing(2)
        self.btn_data = QPushButton("Acquired Data")
        self.btn_data.setFixedHeight(34)
        self.btn_data.setStyleSheet(
            "QPushButton { background: #2a4a6b; border-radius: 4px; "
            "font-weight: bold; padding: 4px; }"
            "QPushButton:hover { background: #3a6a9b; }"
        )
        self.btn_data.clicked.connect(self._pick_data)
        self.lbl_data = QLabel("No file selected.")
        self.lbl_data.setStyleSheet(
            "font-size: 10px; color: #888; qproperty-alignment: AlignCenter;"
        )
        self.lbl_data.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_data.setWordWrap(True)
        self.lbl_data.setFixedHeight(32)
        acq_cell.addWidget(self.btn_data)
        acq_cell.addWidget(self.lbl_data)

        # Right cell — dictionary
        dict_cell = QVBoxLayout()
        dict_cell.setSpacing(2)
        self.btn_dict = QPushButton("Dictionary")
        self.btn_dict.setFixedHeight(34)
        self.btn_dict.setStyleSheet(
            "QPushButton { background: #2a4a6b; border-radius: 4px; "
            "font-weight: bold; padding: 4px; }"
            "QPushButton:hover { background: #3a6a9b; }"
        )
        self.btn_dict.clicked.connect(self._pick_dict)
        self.lbl_dict = QLabel("No file selected.")
        self.lbl_dict.setStyleSheet(
            "font-size: 10px; color: #888; qproperty-alignment: AlignCenter;"
        )
        self.lbl_dict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_dict.setWordWrap(True)
        self.lbl_dict.setFixedHeight(32)
        dict_cell.addWidget(self.btn_dict)
        dict_cell.addWidget(self.lbl_dict)

        load_row.addLayout(acq_cell, stretch=1)
        load_row.addLayout(dict_cell, stretch=1)
        ml.addLayout(load_row)

        # ── Converging arrows ──────────────────────────────────────────────
        arrow_lbl = QLabel("↓                    ↓")
        arrow_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arrow_lbl.setStyleSheet("color: #666; font-size: 13px; letter-spacing: 4px;")
        ml.addWidget(arrow_lbl)

        # ── Motion correction (rigid, itk-elastix) ─────────────────────────
        from my_gui.motion_correction import (elastix_available as _elastix_ok,
                                              REFERENCE_MODES as _MOCO_REFS)
        moco_row = QHBoxLayout()
        self.chk_moco = QCheckBox("Motion correction")
        self.chk_moco.setToolTip(
            "Rigidly register every acquired schedule frame to a reference frame\n"
            "(Elastix) before dot-product matching. Applied to the loaded scan;\n"
            "uncheck to restore the original images.")
        moco_row.addWidget(self.chk_moco)
        self.combo_moco_ref = QComboBox(); self.combo_moco_ref.addItems(_MOCO_REFS)
        moco_row.addWidget(self.combo_moco_ref, stretch=1)
        if not _elastix_ok():
            self.chk_moco.setEnabled(False); self.combo_moco_ref.setEnabled(False)
            self.chk_moco.setText("Motion correction  (pip install itk-elastix)")
        self.chk_moco.toggled.connect(self._on_moco_toggled)
        ml.addLayout(moco_row)

        # ── Run button ─────────────────────────────────────────────────────
        self.btn_match = QPushButton("Run Dot-Product Matching")
        self.btn_match.setFixedHeight(38)
        self.btn_match.setStyleSheet(
            "QPushButton { background: #1d6b2e; border-radius: 5px; "
            "font-weight: bold; font-size: 13px; }"
            "QPushButton:hover { background: #27963f; }"
            "QPushButton:disabled { background: #333; color: #666; }"
        )
        self.btn_match.clicked.connect(self._run_matching)
        ml.addWidget(self.btn_match)

        # ── Status ─────────────────────────────────────────────────────────
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("font-size: 11px;")
        self.lbl_status.setWordWrap(True)
        ml.addWidget(self.lbl_status)

        left_layout.addWidget(grp_match)

        # ── Divider with "or" ──────────────────────────────────────────────
        div_row = QHBoxLayout()
        div_row.setSpacing(6)
        _line1 = QFrame(); _line1.setFrameShape(QFrame.Shape.HLine)
        _line1.setStyleSheet("color: #444;")
        _line2 = QFrame(); _line2.setFrameShape(QFrame.Shape.HLine)
        _line2.setStyleSheet("color: #444;")
        _or_lbl = QLabel("or")
        _or_lbl.setStyleSheet("color: #888; font-size: 11px;")
        _or_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        div_row.addWidget(_line1, stretch=1)
        div_row.addWidget(_or_lbl)
        div_row.addWidget(_line2, stretch=1)
        left_layout.addLayout(div_row)

        # ═══════════════════════════════════════════════════════════════════
        # PATH 2 — Load pre-computed quant_maps.mat directly
        # ═══════════════════════════════════════════════════════════════════
        grp_qm = QGroupBox("Pre-computed Maps")
        grp_qm.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #555; "
            "border-radius: 5px; margin-top: 6px; padding-top: 22px; }"
            "QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; left: 8px; top: 4px; font-size: 14px;"
            "padding: 0 4px; }"
        )
        qm_lay = QVBoxLayout(grp_qm)
        qm_lay.setSpacing(4)

        self.btn_load_qm = QPushButton("Load Quantified Maps")
        self.btn_load_qm.setFixedHeight(34)
        self.btn_load_qm.setStyleSheet(
            "QPushButton { background: #4a3a1a; border-radius: 4px; "
            "font-weight: bold; padding: 4px; }"
            "QPushButton:hover { background: #6a561f; }"
        )
        self.btn_load_qm.clicked.connect(self._load_quant_maps)
        self.lbl_qm = QLabel("No file selected.")
        self.lbl_qm.setStyleSheet("font-size: 10px; color: #888;")
        self.lbl_qm.setWordWrap(True)
        qm_lay.addWidget(self.btn_load_qm)
        qm_lay.addWidget(self.lbl_qm)

        left_layout.addWidget(grp_qm)

        # ── Map selector ──────────────────────────────────────────────────
        # Map selection & dp-mask.  The map selector ("Display:") and the
        # acquired-mode sliders are placed in the Display row above the image
        # (added with the right panel below).  The dp-mask control is kept as a
        # hidden data holder — quality masking still applies with its default
        # threshold, there is just no on-screen control for it.
        self.chk_mask = QCheckBox("dp mask")
        self.chk_mask.setChecked(True)
        self.chk_mask.toggled.connect(self._on_mask_changed)
        self.spin_dp_thresh = QDoubleSpinBox()
        self.spin_dp_thresh.setRange(0.90, 1.0)
        self.spin_dp_thresh.setValue(_DP_MASK_THRESH)
        self.spin_dp_thresh.setDecimals(5)
        self.spin_dp_thresh.setSingleStep(0.0001)
        self.spin_dp_thresh.valueChanged.connect(self._on_mask_changed)

        self.map_selector = QComboBox()
        # Will be populated dynamically after matching; defaults shown here
        for k, (label, *_) in _MAP_DEFAULTS.items():
            self.map_selector.addItem(label, k)
        self.map_selector.currentIndexChanged.connect(self._refresh_plot)

        # ── Acquired-images scrollers (hidden until acquired mode active) ──
        # Measurement slider
        self._meas_frame = QFrame()
        meas_lay = QHBoxLayout(self._meas_frame)
        meas_lay.setContentsMargins(4, 2, 4, 2)
        meas_lay.addWidget(QLabel("Meas:"))
        self._meas_slider = QSlider(Qt.Orientation.Horizontal)
        self._meas_slider.setMinimum(0)
        self._meas_slider.setMaximum(0)
        self._meas_slider.setValue(0)
        self._meas_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._meas_slider.setTickInterval(1)
        self._meas_slider.valueChanged.connect(self._refresh_plot)
        meas_lay.addWidget(self._meas_slider, stretch=1)
        self._meas_label = QLabel("1 / 1")
        self._meas_label.setFixedWidth(60)
        meas_lay.addWidget(self._meas_label)
        self._meas_frame.setVisible(False)

        # Slice slider (only shown when n_slices > 1)
        self._acq_slice_frame = QFrame()
        acq_sl_lay = QHBoxLayout(self._acq_slice_frame)
        acq_sl_lay.setContentsMargins(4, 2, 4, 2)
        acq_sl_lay.addWidget(QLabel("Slice:"))
        self._acq_slice_slider = QSlider(Qt.Orientation.Horizontal)
        self._acq_slice_slider.setMinimum(0)
        self._acq_slice_slider.setMaximum(0)
        self._acq_slice_slider.setValue(0)
        self._acq_slice_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._acq_slice_slider.setTickInterval(1)
        self._acq_slice_slider.valueChanged.connect(self._refresh_plot)
        acq_sl_lay.addWidget(self._acq_slice_slider, stretch=1)
        self._acq_slice_label = QLabel("1 / 1")
        self._acq_slice_label.setFixedWidth(60)
        acq_sl_lay.addWidget(self._acq_slice_label)
        self._acq_slice_frame.setVisible(False)

        # Hint label
        self._acq_hint = QLabel(
            "🖱 Scroll wheel: step through measurements"
        )
        self._acq_hint.setStyleSheet("font-size: 10px; color: #888; padding: 1px 4px;")
        self._acq_hint.setVisible(False)

        # ── Error Maps group ──────────────────────────────────────────────
        grp_err = QGroupBox("Error Maps (Ground Truth)")
        grp_err.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #555; "
            "border-radius: 5px; margin-top: 6px; padding-top: 22px; }"
            "QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; left: 8px; top: 4px; font-size: 14px;"
            "padding: 0 4px; }"
        )
        err_lay = QVBoxLayout(grp_err)
        err_lay.setSpacing(4)

        err_hint = QLabel("Compare measured maps to nominal (ground truth) concentrations per ROI.")
        err_hint.setWordWrap(True)
        err_hint.setStyleSheet("font-size: 10px; color: #888;")
        err_lay.addWidget(err_hint)

        err_btn_row = QHBoxLayout()
        self.btn_set_nom = QPushButton("Set Nominal Conc…")
        self.btn_set_nom.setToolTip("Enter ground truth concentration (mM) per ROI")
        self.btn_set_nom.clicked.connect(self._set_nominal_conc)
        self.btn_gen_err = QPushButton("Generate Error Maps")
        self.btn_gen_err.setToolTip("Compute absolute and % error maps vs nominal concentration")
        self.btn_gen_err.clicked.connect(self._generate_error_maps)
        err_btn_row.addWidget(self.btn_set_nom)
        err_btn_row.addWidget(self.btn_gen_err)
        err_lay.addLayout(err_btn_row)

        self.lbl_err_status = QLabel("")
        self.lbl_err_status.setStyleSheet("font-size: 10px; color: #888;")
        self.lbl_err_status.setWordWrap(True)
        err_lay.addWidget(self.lbl_err_status)

        left_layout.addWidget(grp_err)

        # ── Plot customisation group ──────────────────────────────────────
        grp_plot = QGroupBox("Figure Customization")
        grp_plot.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #555; "
            "border-radius: 5px; margin-top: 6px; padding-top: 22px; }"
            "QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; left: 8px; top: 4px; font-size: 14px;padding: 0 4px; }"
        )
        self.chk_custom = QCheckBox("Enable Figure Customization")
        self.chk_custom.setChecked(False)
        self.chk_custom.setStyleSheet("font-weight: bold;")
        pl_outer = QVBoxLayout(grp_plot)
        pl_outer.setContentsMargins(10, 6, 10, 8)
        pl_outer.setSpacing(6)
        pl_outer.addWidget(self.chk_custom)

        # Container that shows/hides
        self._plot_ctrl = QWidget()
        pl = QVBoxLayout(self._plot_ctrl)
        pl.setContentsMargins(0, 4, 0, 0)
        pl.setSpacing(5)

        _tgl_ss = (
            "QPushButton { border:1px solid #555; border-radius:3px;"
            " padding:1px 5px; font-size:12px; min-width:24px; color:#e0e0e0; }"
            "QPushButton:hover   { background:#2a3050; border-color:#7986cb; }"
            "QPushButton:checked { background:#3a3a6a; border-color:#7aa2f7; color:#ffffff; }"
        )

        # ── Row 1: fonts (Aa family · T title · C cbar)  +  B I x² x₂ ──────────
        font_row = QHBoxLayout()
        font_row.setSpacing(5)
        _aa = QLabel("Aa"); _aa.setToolTip("Title font family")
        _aa.setStyleSheet("color:#aaa; font-size:12px; font-weight:bold;")
        font_row.addWidget(_aa)

        from PyQt6.QtWidgets import QComboBox as _QCBR
        self.combo_title_font = _QCBR()
        self.combo_title_font.setFixedWidth(140)
        self.combo_title_font.setToolTip("Title font family")
        for _ff in ("Default", "Arial", "Times New Roman",
                    "Helvetica", "DejaVu Sans", "DejaVu Serif"):
            self.combo_title_font.addItem(_ff)
        self.combo_title_font.currentIndexChanged.connect(self._refresh_plot)
        font_row.addWidget(self.combo_title_font)

        def _make_fspin(default: int) -> QSpinBox:
            sp = QSpinBox(); sp.setRange(4, 40); sp.setValue(default)
            sp.setFixedWidth(46)
            sp.valueChanged.connect(self._refresh_plot)
            return sp

        def _fs_lbl(sym: str, sp: QSpinBox, tip: str):
            _l = QLabel(sym); _l.setToolTip(tip)
            _l.setStyleSheet("color:#ccc; font-size:12px; font-weight:bold;")
            font_row.addWidget(_l); font_row.addWidget(sp)

        # "T" = title font size;  "C" = colour-bar tick font size.
        self.spin_fs_title = _make_fspin(11)
        self.spin_fs_axes  = _make_fspin(9)    # kept for internal use (not shown)
        self.spin_fs_ticks = _make_fspin(8)    # kept for internal use (not shown)
        self.spin_fs_cbar  = _make_fspin(8)
        _fs_lbl("T", self.spin_fs_title, "Title font size")
        _fs_lbl("C", self.spin_fs_cbar, "Colour-bar tick font size")

        font_row.addStretch()

        # Bold / Italic (whole-title) — the x²/x₂ superscript/subscript buttons
        # are appended right after these (below) so the group reads B I x² x₂.
        self.btn_bold = QPushButton("B"); self.btn_bold.setCheckable(True)
        self.btn_bold.setFixedSize(26, 22); self.btn_bold.setToolTip("Bold")
        self.btn_bold.setStyleSheet(_tgl_ss + " QPushButton { font-weight: bold; }")
        self.btn_bold.toggled.connect(self._refresh_plot)
        self.btn_italic = QPushButton("I"); self.btn_italic.setCheckable(True)
        self.btn_italic.setFixedSize(26, 22); self.btn_italic.setToolTip("Italic")
        self.btn_italic.setStyleSheet(_tgl_ss + " QPushButton { font-style: italic; }")
        self.btn_italic.toggled.connect(self._refresh_plot)
        font_row.addWidget(self.btn_bold)
        font_row.addWidget(self.btn_italic)
        pl.addLayout(font_row)

        # ── Row 2: Colormap  +  clim (min – max, side by side) ────────────────
        cc_row = QHBoxLayout()
        cc_row.setSpacing(5)
        cc_row.addWidget(QLabel("Colormap:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.setFixedWidth(120)
        self._cmap_entries = _build_cmap_list()   # list of (name, label)
        for name, label in self._cmap_entries:
            self.combo_cmap.addItem(label, name)  # userData = internal name
        for i, (n, _) in enumerate(self._cmap_entries):
            if n == "viridis":
                self.combo_cmap.setCurrentIndex(i)
                break
        self.combo_cmap.setToolTip(
            "Matplotlib colormap for the image.\n"
            "Custom MRF colormaps (T1cm, T2cm, difference) appear at the top\n"
            "if T1cm.mat / T2cm.mat / differenceMaps.mat are found."
        )
        self.combo_cmap.currentIndexChanged.connect(self._refresh_plot)
        cc_row.addWidget(self.combo_cmap)

        # Fuderer perceptual "log-like" colour scaling (MRM 2025).  Data and the
        # colourbar ticks stay linear — only the colormap's colour allocation is
        # warped so low values get more contrast.
        self.chk_logmap = QCheckBox("Log map")
        self.chk_logmap.setToolTip(
            "Log-color scaling, redistribute the colormaps so equal color "
            "steps = equal % change in value.")
        self.chk_logmap.toggled.connect(self._refresh_plot)
        cc_row.addWidget(self.chk_logmap)

        self.chk_clim = QCheckBox("clim")
        self.chk_clim.setToolTip("Use custom colour-bar limits (otherwise automatic)")
        self.chk_clim.setChecked(False)
        self.chk_clim.toggled.connect(self._on_clim_toggle)
        cc_row.addWidget(self.chk_clim)

        self.spin_clim_min = QDoubleSpinBox()
        self.spin_clim_min.setRange(-1e6, 1e6); self.spin_clim_min.setValue(0.0)
        self.spin_clim_min.setDecimals(4); self.spin_clim_min.setSingleStep(0.01)
        self.spin_clim_min.setEnabled(False); self.spin_clim_min.setFixedWidth(84)
        self.spin_clim_min.valueChanged.connect(self._refresh_plot)
        cc_row.addWidget(self.spin_clim_min)
        cc_row.addWidget(QLabel("–"))
        self.spin_clim_max = QDoubleSpinBox()
        self.spin_clim_max.setRange(-1e6, 1e6); self.spin_clim_max.setValue(1.0)
        self.spin_clim_max.setDecimals(4); self.spin_clim_max.setSingleStep(0.01)
        self.spin_clim_max.setEnabled(False); self.spin_clim_max.setFixedWidth(84)
        self.spin_clim_max.valueChanged.connect(self._refresh_plot)
        cc_row.addWidget(self.spin_clim_max)

        # No. of exchangeable protons — drives the fs / fs2 → mM conversion
        # (fs [mM] = fs [fraction] × 110000 / N).  Default 3.  Wrapped in a
        # container so the label + spinbox can be shown ONLY for the fs / fs2
        # (mM) maps (they mean nothing for ksw, T1/T2, dot-product, … maps).
        cc_row.addSpacing(14)
        self._protons_box = QWidget()
        _protons_lay = QHBoxLayout(self._protons_box)
        _protons_lay.setContentsMargins(0, 0, 0, 0)
        _protons_lay.setSpacing(4)
        _protons_lay.addWidget(QLabel("Protons:"))
        self.spin_protons = QSpinBox()
        self.spin_protons.setRange(1, 50)
        self.spin_protons.setValue(3)
        self.spin_protons.setFixedWidth(58)
        self.spin_protons.setToolTip(
            "Number of exchangeable protons per solute molecule.  Used only for\n"
            "the fs / fs2 (mM) maps:  fs [mM] = fs [fraction] × 110000 / N.")
        self.spin_protons.valueChanged.connect(self._refresh_plot)
        _protons_lay.addWidget(self.spin_protons)
        cc_row.addWidget(self._protons_box)

        cc_row.addStretch()
        pl.addLayout(cc_row)

        # ── Row 3: Title text ─────────────────────────────────────────────────
        title_row = QHBoxLayout()
        _tl = QLabel("Title:"); _tl.setFixedWidth(40)
        title_row.addWidget(_tl)
        self.edit_title = QLineEdit()
        self.edit_title.setPlaceholderText("Leave blank to use default map name")
        title_row.addWidget(self.edit_title, stretch=1)
        pl.addLayout(title_row)

        from my_gui.format_bar import add_title_format_bar, connect_title_debounced
        connect_title_debounced(self.edit_title, self._refresh_plot)
        # Append x²/x₂ (superscript/subscript) into the fonts row after B/I so the
        # title-style group reads  B  I  x²  x₂  together.
        add_title_format_bar(self.edit_title, None, target_row=font_row)

        # — Export & data cursor & ROIs-only ————————————————————————————————
        # These live in the Display row above the image (added with the right
        # panel below), not inside the collapsible customization box.
        self.btn_export = QPushButton("Export figure…")
        self.btn_export.clicked.connect(self._export_figure)

        from PyQt6.QtWidgets import QCheckBox as _QCB
        from my_gui.plot_custom_bar import DataCursorToolButton
        self.chk_datacursor = DataCursorToolButton()
        self.chk_datacursor.toggled.connect(self._toggle_datacursor)

        self.chk_roi_only = _QCB("ROIs only")
        self.chk_roi_only.setToolTip("Show the map only inside the drawn ROIs")
        self.chk_roi_only.toggled.connect(self._refresh_plot)

        pl_outer.addWidget(self._plot_ctrl)
        self._plot_ctrl.setVisible(False)
        self.chk_custom.toggled.connect(self._plot_ctrl.setVisible)
        self.chk_custom.toggled.connect(self._refresh_plot)

        # Figure Customization is placed directly above the image in the right
        # panel (added below) — consistent with the T1/T2 and CEST-MRI tabs where
        # the customisation bar sits above the map — rather than in the left panel.
        left_layout.addStretch()
        splitter.addWidget(_left_scroll)

        # ── Right panel — Display row + canvas ────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Tools that sit on the Display row (created here so the row can use them).
        from my_gui.plot_custom_bar import WindowLevelToolButton
        self.btn_contrast = WindowLevelToolButton()
        self.btn_contrast.toggled.connect(self._on_wl_tool_toggled)

        self.btn_hide_rois = QPushButton("Hide ROIs")
        self.btn_hide_rois.setCheckable(True)
        self.btn_hide_rois.setToolTip("Toggle ROI overlay visibility on the parametric map")
        self.btn_hide_rois.setStyleSheet(
            "QPushButton { border-radius:4px; padding:4px 10px; }"
            "QPushButton:checked { background:#333; color:#888; }"
        )
        self.btn_hide_rois.clicked.connect(self._toggle_rois_visible)

        # Display row: map selector + pixel/contrast tools + Hide ROIs + ROIs only
        # + Export — matching the T1/T2/B1/WASABI tab, above the image.
        disp_row = QHBoxLayout()
        disp_row.addWidget(QLabel("Display:"))
        disp_row.addWidget(self.map_selector, stretch=1)
        disp_row.addWidget(self.btn_contrast)     # window/level
        disp_row.addWidget(self.chk_datacursor)   # pixel values (red cursor)
        disp_row.addWidget(self.btn_hide_rois)
        disp_row.addWidget(self.chk_roi_only)
        self.chk_roi_bg = QCheckBox("ROIs + Bkg")
        self.chk_roi_bg.setToolTip(
            "Show the colored map only inside the ROIs, over the 1st raw "
            "image as a gray anatomical background.")
        self.chk_roi_bg.toggled.connect(self._refresh_plot)
        disp_row.addWidget(self.chk_roi_bg)
        self.btn_roi_bg = QPushButton("Bkg…")
        self.btn_roi_bg.setToolTip(
            "Pick the grayscale background image (from the Scan Directory) "
            "for the 'ROIs + Bkg' overlay.")
        self.btn_roi_bg.clicked.connect(self._pick_roi_bg)
        disp_row.addWidget(self.btn_roi_bg)
        self.chk_dark_bg = QCheckBox("Bg")
        self.chk_dark_bg.setToolTip(
            "Black background for the figure (for slides). Only the white "
            "surround and labels flip — the maps stay identical."
        )
        self.chk_dark_bg.toggled.connect(self._refresh_plot)
        disp_row.addWidget(self.chk_dark_bg)
        disp_row.addWidget(self.btn_export)
        right_layout.addLayout(disp_row)

        # acquired-mode sliders (hidden unless acquired images are loaded)
        right_layout.addWidget(self._meas_frame)
        right_layout.addWidget(self._acq_slice_frame)
        right_layout.addWidget(self._acq_hint)

        right_layout.addWidget(grp_plot)      # Figure Customization — above the image

        self._fig = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self._fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._fig.canvas.mpl_connect('scroll_event', self._on_canvas_scroll)
        right_layout.addWidget(self.canvas)

        # ── Bottom row — four action buttons only ─────────────────────────
        btn_row = QHBoxLayout()
        self.btn_roi_spectra = QPushButton("ROI Spectra")
        self.btn_roi_spectra.clicked.connect(self._show_roi_spectra)
        btn_row.addWidget(self.btn_roi_spectra)

        self.btn_roi_table = QPushButton("ROI Statistics")
        self.btn_roi_table.clicked.connect(self._show_roi_table)
        btn_row.addWidget(self.btn_roi_table)

        self.btn_pick_pixel = QPushButton("Simulate")
        self.btn_pick_pixel.setToolTip(
            "Simulate the MRF fingerprint — choose drawn ROIs or a single pixel.\n"
            "The measured signal is dot-product matched against the loaded\n"
            "dictionary; the best-match fingerprint is plotted."
        )
        self.btn_pick_pixel.setStyleSheet(
            "QPushButton { border-radius:4px; padding:4px 10px; background:#7b3f00;"
            " color:#ffcc80; font-weight:bold; border:1px solid #ffaa40; }"
            "QPushButton:hover { background:#8a4a00; }"
        )
        self.btn_pick_pixel.clicked.connect(self._open_simulate_menu)
        btn_row.addWidget(self.btn_pick_pixel)

        self.lbl_picked = QPushButton("✕ Clear pixel")
        self.lbl_picked.setFlat(True)
        self.lbl_picked.setStyleSheet("color:#aaa; font-size:10px; padding:2px 6px;")
        self.lbl_picked.setVisible(False)
        self.lbl_picked.setToolTip("Clear the picked pixel")
        self.lbl_picked.clicked.connect(self._clear_picked_pixel)
        btn_row.addWidget(self.lbl_picked)

        btn_row.addStretch()
        right_layout.addLayout(btn_row)

        self.lbl_roi_stats = QLabel("")
        self.lbl_roi_stats.setStyleSheet("font-size:10px; color:#aaa; padding:2px;")
        self.lbl_roi_stats.setWordWrap(True)
        right_layout.addWidget(self.lbl_roi_stats)

        splitter.addWidget(right)
        splitter.setSizes([420, 680])

    # ─────────────────────────────────────────────────────────────────────
    # Clim / axis toggle helpers
    # ─────────────────────────────────────────────────────────────────────

    def _on_clim_toggle(self, checked: bool):
        self.spin_clim_min.setEnabled(checked)
        self.spin_clim_max.setEnabled(checked)
        self._refresh_plot()

    # ─────────────────────────────────────────────────────────────────────
    # File pickers
    # ─────────────────────────────────────────────────────────────────────

    def _pick_data(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Select acquired data", "",
            "Data files (*.mat *.npz);;All files (*)"
        )
        if fn:
            self._data_fn = fn
            self.lbl_data.setText(Path(fn).name)
            self.lbl_data.setStyleSheet("font-size: 11px; color: #ccc;")
            self._try_load_acquired(fn)

    def _pick_dict(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Select dictionary", "",
            "Data files (*.mat *.npz);;All files (*)"
        )
        if fn:
            self._dict_fn = fn
            self.lbl_dict.setText(Path(fn).name)
            self.lbl_dict.setStyleSheet("font-size: 11px; color: #ccc;")

    def _try_load_acquired(self, fn: str):
        """
        Load raw acquired images from .mat / .npz.
        Stores as (n_meas, n_slices, Y, X) so both measurements and slices
        are scrollable.  Adds 'Acquired Images' to the map selector immediately.
        """
        try:
            import scipy.io as sio
            if fn.endswith(".npz"):
                raw = dict(np.load(fn, allow_pickle=True))
            else:
                raw = sio.loadmat(fn)

            if 'acquired_data' not in raw:
                self._acquired_data = None
                return

            arr = np.array(raw['acquired_data'], dtype=float)

            # Normalise to (n_meas, n_slices, Y, X)
            # Bruker / save_acquired_data stores as (Y, X, n_slices, n_meas)
            if arr.ndim == 4:
                arr = np.transpose(arr, (3, 2, 0, 1))   # → (n_meas, n_slices, Y, X)
            elif arr.ndim == 3:
                # (Y, X, n_meas) — no slice dimension
                arr = np.transpose(arr, (2, 0, 1))       # → (n_meas, Y, X)
                arr = arr[:, np.newaxis, :, :]            # → (n_meas, 1, Y, X)
            elif arr.ndim == 2:
                arr = arr[np.newaxis, np.newaxis, :, :]  # → (1, 1, Y, X)
            else:
                self._acquired_data = None
                return

            self._acquired_data = arr
            n_meas, n_slices = arr.shape[0], arr.shape[1]

            # ── Update measurement slider ─────────────────────────────────
            self._meas_slider.blockSignals(True)
            self._meas_slider.setMaximum(n_meas - 1)
            self._meas_slider.setValue(0)
            self._meas_slider.blockSignals(False)
            self._meas_label.setText(f"1 / {n_meas}")

            # ── Update slice slider ───────────────────────────────────────
            self._acq_slice_slider.blockSignals(True)
            self._acq_slice_slider.setMaximum(n_slices - 1)
            self._acq_slice_slider.setValue(0)
            self._acq_slice_slider.blockSignals(False)
            self._acq_slice_label.setText(f"1 / {n_slices}")
            # Only show slice slider when there are multiple slices
            self._acq_slice_frame.setVisible(False)   # hidden until mode active

            # ── Add "Acquired Images" to the map selector ────────────────
            # Insert at position 0 so it's the first/top choice.
            # Guard against duplicates.
            acq_idx = next(
                (i for i in range(self.map_selector.count())
                 if self.map_selector.itemData(i) == "__acquired__"),
                None,
            )
            if acq_idx is None:
                self.map_selector.insertItem(0, "Acquired Images", "__acquired__")
                acq_idx = 0

            # Always switch to Acquired Images so the user sees the scan
            # immediately (quant maps can still be chosen from the dropdown)
            self.map_selector.setCurrentIndex(acq_idx)

            # Motion correction: fresh scan → drop raw cache; auto-apply if on.
            self._acquired_data_raw = None
            if (getattr(self, 'chk_moco', None) is not None
                    and self.chk_moco.isChecked()):
                self._apply_moco()
            self._refresh_plot()

        except Exception:
            self._acquired_data = None

    # ── Motion correction (rigid, itk-elastix) ────────────────────────────────
    def _on_moco_toggled(self, checked: bool):
        if self._acquired_data is None and self._data_fn:
            self._try_load_acquired(self._data_fn)
        if self._acquired_data is None:
            if checked:
                self.lbl_status.setText("Motion correction: load acquired data first.")
                self.chk_moco.blockSignals(True)
                self.chk_moco.setChecked(False)
                self.chk_moco.blockSignals(False)
            return
        self._apply_moco() if checked else self._restore_moco()

    def _apply_moco(self):
        """Rigidly register every schedule frame to the chosen reference."""
        from PyQt6.QtWidgets import QProgressDialog, QApplication
        from my_gui import motion_correction as mc
        if self._acquired_data is None:
            return
        if getattr(self, '_acquired_data_raw', None) is None:
            self._acquired_data_raw = np.asarray(self._acquired_data).copy()
        ref = self.combo_moco_ref.currentText()
        n = int(self._acquired_data_raw.shape[0])
        prog = QProgressDialog("Motion correction (Elastix)…", "Cancel", 0, n, self)
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setMinimumDuration(0); prog.setValue(0); prog.show()

        def _p(done, total):
            prog.setValue(done); QApplication.processEvents()
            if prog.wasCanceled():
                raise RuntimeError("cancelled")

        try:
            corr = mc.moco_4d(self._acquired_data_raw, frame_axis=0,
                              ref_mode=ref, progress=_p).astype(np.float32)
        except RuntimeError:
            prog.close()
            self.chk_moco.blockSignals(True); self.chk_moco.setChecked(False)
            self.chk_moco.blockSignals(False)
            self.lbl_status.setText("Motion correction cancelled.")
            return
        except Exception as e:          # noqa: BLE001
            prog.close()
            self.chk_moco.blockSignals(True); self.chk_moco.setChecked(False)
            self.chk_moco.blockSignals(False)
            self.lbl_status.setText(f"Motion correction failed: {e}")
            return
        prog.close()
        self._acquired_data = corr
        self.lbl_status.setText(
            f"Motion correction applied (rigid, ref='{ref}'). Run matching to use it.")
        self._refresh_plot()

    def _restore_moco(self):
        raw = getattr(self, '_acquired_data_raw', None)
        if raw is None:
            return
        self._acquired_data = raw.copy()
        self.lbl_status.setText("Motion correction removed — original images restored.")
        self._refresh_plot()

    def set_dict_fn(self, path: str):
        """Called automatically after dictionary generation completes."""
        self._dict_fn = path
        self.lbl_dict.setText(Path(path).name)
        self.lbl_dict.setStyleSheet("font-size: 11px; color: green;")

    def _load_quant_maps(self):
        """Load a previously saved quant_maps.mat directly (skip matching)."""
        fn, _ = QFileDialog.getOpenFileName(
            self, "Select quant_maps file", "",
            "MAT files (*.mat);;NumPy (*.npz);;All files (*)"
        )
        if not fn:
            return
        try:
            import scipy.io as sio
            if fn.endswith(".npz"):
                raw = dict(np.load(fn, allow_pickle=True))
            else:
                raw = sio.loadmat(fn)
            quant_maps = {
                k: np.array(v, dtype=float)
                for k, v in raw.items()
                if not k.startswith("_") and isinstance(v, np.ndarray)
            }
            self._set_quant_maps(quant_maps)
            self.lbl_qm.setText(f"Loaded: {Path(fn).name}")
            self.lbl_qm.setStyleSheet("font-size: 11px; color: green;")
            self.lbl_status.setText(f"Loaded quant_maps — keys: {', '.join(quant_maps.keys())}")
        except Exception as exc:
            self.lbl_qm.setText(f"Error: {exc}")
            self.lbl_qm.setStyleSheet("font-size: 11px; color: red;")

    def set_quant_maps_external(self, quant_maps: dict):
        """Called from DictTab 'Load quant_maps.mat' button via app.py callback."""
        self._set_quant_maps(quant_maps)
        self.lbl_status.setText(f"Maps from Dictionary tab — keys: {', '.join(quant_maps.keys())}")

    def _on_mask_changed(self):
        """Recompute the dp mask when checkbox or threshold spinner changes."""
        if self._quant_maps is not None and 'dp' in self._quant_maps:
            thresh = self.spin_dp_thresh.value()
            self._dp_mask = np.array(self._quant_maps['dp'], dtype=float) > thresh
        self._refresh_plot()

    def _set_quant_maps(self, quant_maps: dict):
        """Store quant_maps, build dp mask, update map selector, refresh plot."""
        self._quant_maps = quant_maps

        # Restrict every quant map to the global analysis mask (brain / phantom
        # outline), if one is active — affects display, ROI stats and error maps.
        from my_gui.roi_manager import apply_analysis_mask
        for _k, _v in list(quant_maps.items()):
            if isinstance(_v, np.ndarray) and _v.ndim >= 2:
                quant_maps[_k] = apply_analysis_mask(_v)

        # Add raw proton-fraction maps (same data as fs/fs2, but shown without
        # the ×110e3/3 → mM conversion) so both views are available.
        for _src, _raw in (("fs", "fs_raw"), ("fs2", "fs2_raw")):
            if _src in quant_maps and _raw not in quant_maps:
                quant_maps[_raw] = quant_maps[_src]

        # dp mask — use live threshold from spinner
        if 'dp' in quant_maps:
            thresh = self.spin_dp_thresh.value()
            self._dp_mask = np.array(quant_maps['dp'], dtype=float) > thresh
        else:
            self._dp_mask = None

        # Rebuild map selector from available keys, preserving per-map labels
        available = [k for k in quant_maps.keys() if k in _MAP_DEFAULTS or k != '_']
        self.map_selector.blockSignals(True)
        self.map_selector.clear()
        for k in available:
            label = _MAP_DEFAULTS[k][0] if k in _MAP_DEFAULTS else k
            self.map_selector.addItem(label, k)
        # Add "Acquired Images" entry if raw data is available
        if self._acquired_data is not None:
            self.map_selector.addItem("Acquired Images", "__acquired__")
        self.map_selector.blockSignals(False)

        self._refresh_plot()

    @staticmethod
    def _sci_colorbar(cb, data, cbar_fs):
        """Format a colorbar with a ×10ⁿ multiplier so small values read cleanly.

        The exponent is chosen so the tick numbers fall in the tens
        (e.g. 3.5e-4 → ticks '0 … 35' with a '×10⁻⁵' header).
        """
        from matplotlib.ticker import FuncFormatter
        finite = data[np.isfinite(data)]
        vmx = float(np.nanmax(finite)) if finite.size else 0.0
        if vmx <= 0:
            return
        exp = int(np.floor(np.log10(vmx))) - 1   # → tick numbers in [10, 100)
        factor = 10.0 ** (-exp)
        try:
            # Set the Colorbar's own formatter (the axis formatter is reset by
            # update_ticks(), so cb.formatter is the durable place).
            cb.formatter = FuncFormatter(lambda v, _p: f"{round(v * factor, 4):g}")
            cb.update_ticks()
            # "×10ⁿ" multiplier header above the colorbar
            cb.ax.set_title(f"×10$^{{{exp}}}$", fontsize=max(cbar_fs - 1, 7),
                            pad=6)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────
    # Matching
    #   cest_mrf/metrics/dot_product.py  →  dot_prod_matching()
    #   Returns: dp, t1w, t2w, fs, ksw  (+ fs2/ksw2 for 2-pool CEST)
    # ─────────────────────────────────────────────────────────────────────

    def _run_matching(self):
        if not self._data_fn or not self._dict_fn:
            self.lbl_status.setText("Select both acquired data and dictionary first.")
            return
        try:
            # Try dot_product_mt first (handles MT-only and mixed pool dicts).
            # Fall back to dot_product for legacy CEST-only dicts.
            try:
                from cest_mrf.metrics.dot_product_mt import dot_prod_matching
            except ImportError:
                from cest_mrf.metrics.dot_product import dot_prod_matching
            import time

            self.lbl_status.setText("Running dot-product matching…")
            self.btn_match.setEnabled(False)
            self._fig.clf(); self.canvas.draw()

            # Ensure acquired images are loaded for viewing
            if self._acquired_data is None and self._data_fn:
                self._try_load_acquired(self._data_fn)

            t0 = time.perf_counter()
            _moco_on = (getattr(self, 'chk_moco', None) is not None
                        and self.chk_moco.isChecked()
                        and self._acquired_data is not None)
            if _moco_on:
                # Motion-corrected images (n_meas, n_slices, Y, X) → the matcher's
                # in-memory layout (n_meas, Y, X, n_slices); skips the raw file.
                acq_in = np.transpose(self._acquired_data, (0, 2, 3, 1))
                quant_maps = dot_prod_matching(
                    dict_fn       = self._dict_fn,
                    acquired_data = acq_in,
                )
            else:
                quant_maps = dot_prod_matching(
                    dict_fn          = self._dict_fn,
                    acquired_data_fn = self._data_fn,
                )
            dt = time.perf_counter() - t0

            self._set_quant_maps(quant_maps)
            self.lbl_status.setText(
                f"Done in {dt:.1f}s — maps: {', '.join(quant_maps.keys())}"
            )
        except Exception as exc:
            import traceback
            self.lbl_status.setText(f"Error: {exc}")
        finally:
            self.btn_match.setEnabled(True)

    # ─────────────────────────────────────────────────────────────────────
    # Plot refresh
    # ─────────────────────────────────────────────────────────────────────

    def _on_canvas_scroll(self, event):
        """Mouse wheel over canvas scrolls through acquired MRF measurements."""
        idx = self.map_selector.currentIndex()
        if self.map_selector.itemData(idx) != "__acquired__":
            return
        step = 1 if event.step > 0 else -1
        new_val = max(0, min(self._meas_slider.maximum(),
                             self._meas_slider.value() + step))
        self._meas_slider.setValue(new_val)

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
        self._refresh_plot()

    def _refresh_plot(self):
        # ── Resolve key from map selector ─────────────────────────────────
        idx = self.map_selector.currentIndex()
        key = self.map_selector.itemData(idx)
        if key is None:
            # Fallback for old text-only items
            txt = self.map_selector.currentText()
            key = txt.split("—")[0].strip().split()[0]

        # Protons spinbox drives ONLY the fs / fs2 → mM conversion, so show it
        # for those two maps and hide it for every other display.
        if hasattr(self, '_protons_box'):
            self._protons_box.setVisible(key in ('fs', 'fs2'))

        # ── Acquired Images mode ──────────────────────────────────────────
        is_acquired = (key == "__acquired__")
        self._meas_frame.setVisible(is_acquired)
        if not is_acquired:
            self._acq_slice_frame.setVisible(False)
            self._acq_hint.setVisible(False)

        if is_acquired:
            if self._acquired_data is None:
                self.lbl_status.setText("No acquired data loaded.")
                return
            n_meas   = self._acquired_data.shape[0]
            n_slices = self._acquired_data.shape[1]

            meas_idx  = self._meas_slider.value()
            slice_idx = self._acq_slice_slider.value()

            # Clamp indices in case data changed
            meas_idx  = max(0, min(meas_idx,  n_meas  - 1))
            slice_idx = max(0, min(slice_idx, n_slices - 1))

            # Update labels
            self._meas_label.setText(f"{meas_idx + 1} / {n_meas}")
            self._acq_slice_label.setText(f"{slice_idx + 1} / {n_slices}")

            # Show slice slider only when data is multi-slice
            self._acq_slice_frame.setVisible(n_slices > 1)
            self._acq_hint.setVisible(True)

            data = self._acquired_data[meas_idx, slice_idx, :, :]
            self._current_map_key = "__acquired__"
            self._dc_img_data = data

            if n_slices > 1:
                title = (f"Acquired Image — Meas {meas_idx + 1}/{n_meas}  "
                         f"Slice {slice_idx + 1}/{n_slices}")
            else:
                title = f"Acquired Image — Measurement {meas_idx + 1}/{n_meas}"

            # Honour the same customisation controls as parametric maps
            use_custom = self.chk_custom.isChecked()
            title_fs = self.spin_fs_title.value() if use_custom else 11
            ticks_fs = self.spin_fs_ticks.value() if use_custom else 8
            cbar_fs  = self.spin_fs_cbar.value()  if use_custom else 8
            axes_fs  = self.spin_fs_axes.value()  if use_custom else 9

            # Colormap: custom selection (default gray) when custom enabled
            acq_cmap = "gray"
            if use_custom:
                raw_cmap = self.combo_cmap.currentData() or "gray"
                if isinstance(raw_cmap, str) and not raw_cmap.startswith("──"):
                    acq_cmap = raw_cmap
                try:
                    from my_gui.colormaps_gui import get_cmap as _gcm
                    acq_cmap = _gcm(acq_cmap)
                except Exception:
                    pass

            # Colorbar limits from spinboxes when custom + clim enabled
            acq_vmin = acq_vmax = None
            if use_custom and self.chk_clim.isChecked():
                acq_vmin = self.spin_clim_min.value()
                acq_vmax = self.spin_clim_max.value()
            elif (self._wl_override is not None
                  and self._wl_override[0] == "__acquired__"):
                # Honour a window/level the user dragged on the acquired image.
                acq_vmin, acq_vmax = self._wl_override[1], self._wl_override[2]

            # Custom title override
            custom_title = self.edit_title.text().strip() if use_custom else ""
            if custom_title:
                title = custom_title

            # Fuderer perceptual log remap (optional) — same as the parametric path.
            if (use_custom and getattr(self, "chk_logmap", None) is not None
                    and self.chk_logmap.isChecked()):
                try:
                    lo = acq_vmin if acq_vmin is not None else float(np.nanmin(data))
                    hi = acq_vmax if acq_vmax is not None else float(np.nanmax(data))
                    from my_gui.colormaps_gui import log_remap_cmap
                    acq_cmap = log_remap_cmap(acq_cmap, lo, hi)
                    acq_vmin, acq_vmax = lo, hi
                except Exception:
                    pass

            self._fig.clf()
            ax = self._fig.add_subplot(111)
            im = ax.imshow(data, cmap=acq_cmap, vmin=acq_vmin, vmax=acq_vmax, origin="upper")
            self._im = im
            cb = self._fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=cbar_fs)
            cb.ax.yaxis.label.set_size(axes_fs)
            _ff = getattr(self, 'combo_title_font', None)
            _title_font = (_ff.currentText() if _ff else "Default")
            _title_font = _title_font if _title_font != "Default" else "default"
            _tb = getattr(self, 'btn_bold', None); _ti = getattr(self, 'btn_italic', None)
            from my_gui.roi_tools import ROICanvas
            ROICanvas._apply_title(ax, title, fontsize=title_fs, title_font=_title_font,
                                   title_bold=bool(_tb and _tb.isChecked()),
                                   title_italic=bool(_ti and _ti.isChecked()))
            ax.tick_params(labelsize=ticks_fs)
            ax.axis("off")
            apply_fig_dark_theme(
                self._fig,
                self.chk_dark_bg.isChecked() if hasattr(self, "chk_dark_bg") else False,
            )
            self.canvas.draw()
            return

        # ── Normal parametric map mode ────────────────────────────────────
        if self._quant_maps is None:
            return

        if key not in self._quant_maps:
            self.lbl_status.setText(f"Key '{key}' not found in results.")
            return

        self._current_map_key = key
        data = np.array(self._quant_maps[key], dtype=float)

        # Squeeze to 2-D (take first slice if 3-D)
        if data.ndim == 3:
            data = data[:, :, 0]
        elif data.ndim > 3:
            data = data[:, :, 0, 0]

        # ── Per-map defaults from _MAP_DEFAULTS ───────────────────────────
        if key in _MAP_DEFAULTS:
            _label, def_cmap, def_vmin, def_vmax, def_scale, _unit = _MAP_DEFAULTS[key]
        else:
            def_cmap, def_vmin, def_vmax, def_scale = "viridis", None, None, 1.0

        # Apply per-map scale factor via the smart helper (handles external
        # MATLAB files where T1/T2 are already stored in ms rather than s).
        effective_scale = _smart_scale(key, data)
        # fs / fs2 → mM use the user-set proton count (110000 / N) instead of
        # the hard-wired ÷3 default.
        if key in ('fs', 'fs2') and hasattr(self, 'spin_protons'):
            effective_scale = 110e3 / max(self.spin_protons.value(), 1)
        if effective_scale != 1:
            data = data * effective_scale

        # ── Apply dp mask (threshold scrollable by user) ──────────────────
        if self.chk_mask.isChecked() and self._dp_mask is not None:
            mask = self._dp_mask
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            elif mask.ndim > 3:
                mask = mask[:, :, 0, 0]
            if mask.shape == data.shape:
                data = data.copy()
                data[~mask] = np.nan   # masked voxels → NaN (transparent/background)

        # ── Apply Phantom_outline mask (if that ROI is present) ───────────
        last_rois = getattr(self, '_last_rois', [])
        for _roi in last_rois:
            if getattr(_roi, 'name', '') == 'Phantom_outline':
                try:
                    ph_msk = _roi.mask
                    if ph_msk.shape != data.shape[:2]:
                        from scipy.ndimage import zoom as _zoom
                        zy = data.shape[0] / max(ph_msk.shape[0], 1)
                        zx = data.shape[1] / max(ph_msk.shape[1], 1)
                        ph_msk = _zoom(ph_msk.astype(float), (zy, zx), order=1) > 0.5
                    data = data.copy()
                    data = np.where(ph_msk, data, np.nan)
                except Exception:
                    pass
                break

        # ── "ROIs only" / "ROIs + Bg" — restrict colour to the drawn ROIs ──
        # "ROIs only" blanks everything outside the ROI union (white/black bg);
        # "ROIs + Bg" additionally lays the ROI colours over the 1st raw frame as
        # a gray underlay (drawn below, just before the main imshow).
        _roi_bg_on = (getattr(self, 'chk_roi_bg', None) is not None
                      and self.chk_roi_bg.isChecked())
        _roi_only_on = (getattr(self, 'chk_roi_only', None) is not None
                        and self.chk_roi_only.isChecked())
        _roi_union = None
        if _roi_bg_on or _roi_only_on:
            union = None
            for _roi in last_rois:
                if getattr(_roi, 'name', '') == 'Phantom_outline':
                    continue
                try:
                    m = _roi.mask
                    if m.shape != data.shape[:2]:
                        from scipy.ndimage import zoom as _zoom
                        zy = data.shape[0] / max(m.shape[0], 1)
                        zx = data.shape[1] / max(m.shape[1], 1)
                        m = _zoom(m.astype(float), (zy, zx), order=1) > 0.5
                    union = m if union is None else (union | m)
                except Exception:
                    pass
            if union is not None:
                _roi_union = union
                data = np.where(union, data, np.nan)

        # ── Read customisation controls ───────────────────────────────────
        use_custom = self.chk_custom.isChecked()

        # Font sizes (only meaningful when custom enabled, but always read)
        title_fs = self.spin_fs_title.value() if use_custom else 11
        axes_fs  = self.spin_fs_axes.value()  if use_custom else 9
        ticks_fs = self.spin_fs_ticks.value() if use_custom else 8
        cbar_fs  = self.spin_fs_cbar.value()  if use_custom else 8

        # Resolve colormap:
        #   • custom enabled  → use combo selection
        #   • custom disabled → use per-map default from _MAP_DEFAULTS
        def _resolve_cmap(name: str):
            """Return a colormap object, falling back gracefully."""
            try:
                from my_gui.colormaps_gui import get_cmap
                return get_cmap(name)
            except Exception:
                try:
                    return plt.get_cmap(name)
                except Exception:
                    return plt.get_cmap("viridis")

        if use_custom:
            raw_cmap = self.combo_cmap.currentData() or def_cmap
            if isinstance(raw_cmap, str) and raw_cmap.startswith("──"):
                raw_cmap = def_cmap
            cmap = _resolve_cmap(raw_cmap)
        else:
            cmap = _resolve_cmap(def_cmap)

        # Colorbar limits:
        #   • custom enabled AND chk_clim → use spinbox values
        #   • otherwise                   → per-map defaults from _MAP_DEFAULTS
        if use_custom and self.chk_clim.isChecked():
            vmin = self.spin_clim_min.value()
            vmax = self.spin_clim_max.value()
        else:
            vmin, vmax = def_vmin, def_vmax
            # Honour a window/level the user dragged on this map (colormap/fonts
            # are left untouched — only the display window changes).
            if self._wl_override is not None and self._wl_override[0] == key:
                vmin, vmax = self._wl_override[1], self._wl_override[2]

        custom_title = self.edit_title.text().strip() if use_custom else ""
        # Prefer a mathtext title override (real subscripts) for maps like Δω_MT.
        title = (custom_title if custom_title
                 else _TITLE_MATHTEXT.get(key, self.map_selector.currentText()))

        # ── Draw ──────────────────────────────────────────────────────────
        self._fig.clf()
        ax = self._fig.add_subplot(111)

        # Use 'lower' origin so anatomical images display correctly;
        # NaN values in masked data render as the colormap's bad-colour (transparent).
        current_cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

        # Fuderer perceptual log remap (optional): warp the colour allocation for
        # the current window.  Data and colour-bar ticks stay linear — only which
        # colour lands on which value changes.  When no explicit clim is set, use
        # the data range as the window.
        if (use_custom and getattr(self, "chk_logmap", None) is not None
                and self.chk_logmap.isChecked()):
            try:
                lo = vmin if vmin is not None else float(np.nanmin(data))
                hi = vmax if vmax is not None else float(np.nanmax(data))
                from my_gui.colormaps_gui import log_remap_cmap
                current_cmap = log_remap_cmap(current_cmap, lo, hi)
                vmin, vmax = lo, hi
            except Exception:
                pass

        current_cmap = current_cmap.copy() if hasattr(current_cmap, 'copy') else current_cmap
        try:
            current_cmap.set_bad(color='black', alpha=0.0)
        except Exception:
            pass

        # ── "ROIs + Bg": gray 1st-raw-image underlay behind the ROI colours ──
        # The colour map (drawn next) is transparent outside the ROI union
        # (set_bad alpha=0 above), so this base shows through everywhere else.
        if _roi_bg_on and _roi_union is not None and self._acquired_data is not None:
            try:
                _sl = 0
                _slider = getattr(self, '_acq_slice_slider', None)
                if self._acquired_data.shape[1] > 1 and _slider is not None:
                    _sl = max(0, min(_slider.value(),
                                     self._acquired_data.shape[1] - 1))
                _base = np.asarray(self._acquired_data[0, _sl, :, :], dtype=float)
                if getattr(self, "_roi_bg_img", None) is not None:
                    _base = np.asarray(self._roi_bg_img, dtype=float)
                if _base.shape != data.shape[:2]:
                    from scipy.ndimage import zoom as _zoom
                    zy = data.shape[0] / max(_base.shape[0], 1)
                    zx = data.shape[1] / max(_base.shape[1], 1)
                    _base = _zoom(_base, (zy, zx), order=1)
                ax.imshow(_base, cmap="gray", origin="upper")
            except Exception:
                pass

        im = ax.imshow(data, cmap=current_cmap, vmin=vmin, vmax=vmax, origin="upper")
        self._im = im
        cb = self._fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=cbar_fs)
        cb.ax.yaxis.label.set_size(axes_fs)

        # Raw proton-fraction maps: show a ×10ⁿ multiplier so the tiny values
        # read as clean numbers (e.g. 0–35 ×10⁻⁵ instead of 0.00000–0.00035).
        if key in ('fs_raw', 'fs2_raw'):
            self._sci_colorbar(cb, data, cbar_fs)
        # Use crash-safe title renderer with font family support
        _ff = getattr(self, 'combo_title_font', None)
        _title_font = (_ff.currentText() if _ff else "Default")
        _title_font = _title_font if _title_font != "Default" else "default"
        _tb = getattr(self, 'btn_bold', None); _ti = getattr(self, 'btn_italic', None)
        from my_gui.roi_tools import ROICanvas
        ROICanvas._apply_title(ax, title, fontsize=title_fs, title_font=_title_font,
                               title_bold=bool(_tb and _tb.isChecked()),
                               title_italic=bool(_ti and _ti.isChecked()))
        ax.tick_params(labelsize=ticks_fs)
        ax.axis("off")

        # Store for data cursor
        self._dc_img_data = data
        self._dc_annot = None  # reset annotation (axes just changed)
        self._pp_marker = None  # marker references are stale after clf()

        # Draw ROI contour overlays on top of the map
        if self._rois_visible and self._rois:
            self._draw_roi_overlays(ax, data)

        apply_fig_dark_theme(
            self._fig,
            self.chk_dark_bg.isChecked() if hasattr(self, "chk_dark_bg") else False,
        )
        self.canvas.draw()

        # Auto-fill clim spinboxes with scaled data range (reference, non-blocking)
        if use_custom and not self.chk_clim.isChecked():
            finite = data[np.isfinite(data)]
            if finite.size > 0:
                self.spin_clim_min.blockSignals(True)
                self.spin_clim_max.blockSignals(True)
                self.spin_clim_min.setValue(float(np.nanmin(finite)))
                self.spin_clim_max.setValue(float(np.nanmax(finite)))
                self.spin_clim_min.blockSignals(False)
                self.spin_clim_max.blockSignals(False)

    # ─────────────────────────────────────────────────────────────────────
    # ROI Manager integration
    # ─────────────────────────────────────────────────────────────────────

    def connect_roi_manager(self, roi_manager):
        """
        Subscribe to ROI updates.

        • Stores the current ROI list so _refresh_plot can draw overlays.
        • Redraws the map immediately whenever ROIs change.
        • Updates the per-ROI stats label.

        Note: results_tab uses a plain FigureCanvas (not ROICanvas), so
        roi_manager.connect_canvas() is a silent no-op here.  We handle
        overlay drawing ourselves inside _refresh_plot / _draw_roi_overlays.
        """
        self._roi_manager_ref = roi_manager
        roi_manager.rois_changed.connect(self._on_rois_changed)
        self._last_rois: list = []

    def _on_rois_changed(self, rois: list):
        """Receive updated ROI list, refresh overlays and stats."""
        self._rois = list(rois)
        self._last_rois = list(rois)
        self._update_roi_stats(rois)
        # Redraw map so contour overlays reflect the new ROIs
        if self._dc_img_data is not None:
            self._refresh_plot()

    def _draw_roi_overlays(self, ax, data):
        """
        Draw ROI contour outlines on *ax* using the same style as other tabs.

        Each ROI is drawn as a coloured contour border (2 px solid line).
        The mask is rescaled to match the current map shape if needed.
        """
        rois = getattr(self, '_rois', [])
        if not rois or data is None:
            return
        h, w = data.shape[:2]
        for roi in rois:
            if roi.mask is None:
                continue
            mask = roi.mask
            # Rescale mask to match current map pixel dimensions
            if mask.shape[:2] != (h, w):
                try:
                    from scipy.ndimage import zoom as _zoom
                    zy = h / max(mask.shape[0], 1)
                    zx = w / max(mask.shape[1], 1)
                    mask = _zoom(mask.astype(float), (zy, zx), order=1) > 0.5
                except Exception:
                    continue
            if not mask.any():
                continue
            try:
                ax.contour(
                    mask.astype(float),
                    levels=[0.5],
                    colors=[roi.color],
                    linewidths=2.0,
                    linestyles="solid",
                )
            except Exception:
                pass

    def _toggle_rois_visible(self, checked: bool):
        """Show or hide ROI overlays on the parametric map."""
        self._rois_visible = not checked
        btn = getattr(self, 'btn_hide_rois', None)
        if btn is not None:
            btn.setText("Show ROIs" if checked else "Hide ROIs")
        if self._dc_img_data is not None:
            self._refresh_plot()

    def _update_roi_stats(self, rois: list):
        self._last_rois = list(rois)
        if not hasattr(self, 'lbl_roi_stats'):
            return
        if not rois:
            self.lbl_roi_stats.setText("")
            return
        lines = []
        img = self._dc_img_data
        current_key = getattr(self, '_current_map_key', None)

        # Determine unit label from current map.
        # NOTE: _dc_img_data is already in display units (scale already applied
        # by _refresh_plot), so we must NOT apply scale again here — just look
        # up the unit string for the label.
        scale = 1.0   # always 1 — data is pre-scaled
        unit  = ""
        if current_key and current_key in _MAP_DEFAULTS:
            if current_key in ('fs', 'fs2'):
                unit = " mM"
            elif current_key in ('ksw', 'ksw2', 'kss'):
                unit = " s⁻¹"
            elif current_key in ('t1w', 't2w', 't1s', 't2s', 't2m'):
                unit = " ms"
            elif current_key == 't1m':
                unit = " ms"
            elif current_key == 'fm':
                unit = " %"
            elif current_key == 'dmw':
                unit = " ppm"

        for roi in rois:
            if img is None:
                lines.append(f"<b>{roi.name}</b>: —")
                continue
            try:
                msk = roi.mask
                if msk.shape != img.shape[:2]:
                    from scipy.ndimage import zoom
                    zy = img.shape[0] / max(msk.shape[0], 1)
                    zx = img.shape[1] / max(msk.shape[1], 1)
                    msk = zoom(msk.astype(float), (zy, zx), order=1) > 0.5
                vals = img[msk]
                finite = vals[np.isfinite(vals)]
                if finite.size > 0:
                    mean_v = float(np.mean(finite)) * scale
                    std_v  = float(np.std(finite))  * scale
                    lines.append(
                        f"<b>{roi.name}</b>: {mean_v:.3g} ± {std_v:.3g}{unit}  (n={finite.size})"
                    )
                else:
                    lines.append(f"<b>{roi.name}</b>: no finite values")
            except Exception as exc:
                lines.append(f"<b>{roi.name}</b>: error — {exc}")
        self.lbl_roi_stats.setText("\n".join(lines))

    # ─────────────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────────────

    def _export_figure(self):
        from my_gui.fig_export import save_figure, FIG_EXPORT_FILTER
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", "parametric_map",
            FIG_EXPORT_FILTER
        )
        if path:
            save_figure(self._fig, path, dpi=300)
            self.lbl_status.setText(f"Saved: {Path(path).name}")

    # ── Error Maps ────────────────────────────────────────────────────────────

    def _set_nominal_conc(self):
        """Open dialog to enter per-ROI nominal concentration."""
        rois = getattr(self, '_last_rois', [])
        rois = [r for r in rois if getattr(r, 'name', '') != 'Phantom_outline']
        if not rois:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "No ROIs", "Draw ROIs in the ROI Manager tab first.")
            return
        dlg = _NomConcDialog(rois, getattr(self, '_nom_conc', {}), self)
        if dlg.exec():
            self._nom_conc = dlg.get_values()
            names = ', '.join(f"{k}: {v:.1f} mM" for k, v in self._nom_conc.items())
            self.lbl_err_status.setText(f"Nominal: {names}")

    def _generate_error_maps(self):
        """Compute absolute and % error maps vs nominal concentration."""
        if self._quant_maps is None:
            self.lbl_err_status.setText("Run matching first (no maps available).")
            return
        nom = getattr(self, '_nom_conc', {})
        if not nom:
            self.lbl_err_status.setText("Set nominal concentrations first.")
            return
        rois = getattr(self, '_last_rois', [])
        rois = [r for r in rois if getattr(r, 'name', '') != 'Phantom_outline']
        if not rois:
            self.lbl_err_status.setText("No ROIs available.")
            return

        fs_raw = self._quant_maps.get('fs')
        if fs_raw is None:
            self.lbl_err_status.setText("No 'fs' map in results.")
            return

        fs_arr = np.array(fs_raw, dtype=float)
        if fs_arr.ndim == 3:
            fs_arr = fs_arr[:, :, 0]
        elif fs_arr.ndim > 3:
            fs_arr = fs_arr[:, :, 0, 0]

        SCALE = 110e3 / max(self.spin_protons.value(), 1)  # proton fraction → mM
        fs_mM = fs_arr * SCALE

        # Build nominal map: fill each ROI mask with its nominal value
        nom_map = np.full(fs_arr.shape, np.nan)
        H, W = fs_arr.shape
        for roi in rois:
            if roi.name not in nom:
                continue
            msk = roi.mask
            if msk.shape != (H, W):
                try:
                    from scipy.ndimage import zoom as _zm
                    msk = _zm(msk.astype(float), (H / msk.shape[0], W / msk.shape[1]), order=1) > 0.5
                except Exception:
                    continue
            nom_map[msk] = nom[roi.name]

        err_abs = fs_mM - nom_map              # absolute error (mM)
        with np.errstate(divide='ignore', invalid='ignore'):
            err_pct = np.where(nom_map > 0, (fs_mM - nom_map) / nom_map * 100.0, np.nan)

        # Apply dp mask if available
        dp = None
        if self._dp_mask is not None:
            dp = self._dp_mask
            if dp.ndim == 3:
                dp = dp[:, :, 0]
            err_abs = np.where(dp, err_abs, np.nan)
            err_pct = np.where(dp, err_pct, np.nan)

        # Store in quant_maps so _refresh_plot can display them
        self._quant_maps['err_fs_abs'] = err_abs
        self._quant_maps['err_fs_pct'] = err_pct

        # Do same for fs2 if available
        fs2_raw = self._quant_maps.get('fs2')
        if fs2_raw is not None:
            fs2_arr = np.array(fs2_raw, dtype=float)
            if fs2_arr.ndim == 3:
                fs2_arr = fs2_arr[:, :, 0]
            fs2_mM = fs2_arr * SCALE
            err2_abs = fs2_mM - nom_map
            with np.errstate(divide='ignore', invalid='ignore'):
                err2_pct = np.where(nom_map > 0, (fs2_mM - nom_map) / nom_map * 100.0, np.nan)
            if dp is not None:
                err2_abs = np.where(dp, err2_abs, np.nan)
                err2_pct = np.where(dp, err2_pct, np.nan)
            self._quant_maps['err_fs2_abs'] = err2_abs
            self._quant_maps['err_fs2_pct'] = err2_pct

        # Add error map entries to _MAP_DEFAULTS dynamically
        _MAP_DEFAULTS['err_fs_abs'] = ("Error Maps: fs abs (mM)",  "bwr", None, None, 1.0, "mM")
        _MAP_DEFAULTS['err_fs_pct'] = ("Error Maps: fs % error",   "bwr", None, None, 1.0, "%")
        if 'err_fs2_abs' in self._quant_maps:
            _MAP_DEFAULTS['err_fs2_abs'] = ("Error Maps: fs2 abs (mM)", "bwr", None, None, 1.0, "mM")
            _MAP_DEFAULTS['err_fs2_pct'] = ("Error Maps: fs2 % error",  "bwr", None, None, 1.0, "%")

        # Rebuild map selector to include error maps
        self._rebuild_map_selector()
        self.lbl_err_status.setText("Error maps generated — see Map Selection.")
        self._refresh_plot()

    def _rebuild_map_selector(self):
        """Rebuild the map selector combobox from current quant_maps."""
        if self._quant_maps is None:
            return
        self.map_selector.blockSignals(True)
        self.map_selector.clear()

        # Regular maps first
        regular_keys = [k for k in self._quant_maps if k in _MAP_DEFAULTS and not k.startswith('err_')]
        for k in regular_keys:
            self.map_selector.addItem(_MAP_DEFAULTS[k][0], k)

        # Error maps with separator
        err_keys = [k for k in self._quant_maps if k.startswith('err_') and k in _MAP_DEFAULTS]
        if err_keys:
            self.map_selector.addItem("── Error Maps ──", "__sep__")
            for k in err_keys:
                self.map_selector.addItem(_MAP_DEFAULTS[k][0], k)

        # Acquired images
        if self._acquired_data is not None:
            self.map_selector.addItem("Acquired Images", "__acquired__")

        self.map_selector.blockSignals(False)

    # ── Per-ROI MRF map spectra dialog ────────────────────────────────────────

    def _show_roi_spectra(self):
        """Open a dialog showing per-ROI mean±std for every available MRF map."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QMessageBox
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

        rois = getattr(self, '_last_rois', [])
        # Exclude phantom outline ROI from per-tube spectra
        rois = [r for r in rois if r.name != "Phantom_outline"]
        quant_maps = getattr(self, '_quant_maps', None)
        if not quant_maps:
            QMessageBox.information(self, "No Data", "Load and match MRF results first.")
            return

        # Collect map arrays — apply smart scale so ms maps aren't re-scaled
        _UNIT_FOR = {
            'fs': "mM", 'fs2': "mM",
            'ksw': "s⁻¹", 'ksw2': "s⁻¹", 'kss': "s⁻¹",
            't1w': "ms", 't2w': "ms", 't1s': "ms", 't2s': "ms",
            't1m': "ms", 't2m': "ms",
            'fm': "%", 'dmw': "ppm",
        }
        map_items: list[tuple] = []  # (display_label, unit, effective_scale, np.ndarray 2-D)
        for key, arr in quant_maps.items():
            if arr is None:
                continue
            a = np.array(arr, dtype=float)
            while a.ndim > 2:
                a = a[..., 0]
            lbl  = _MAP_DEFAULTS[key][0] if key in _MAP_DEFAULTS else key
            unit = _UNIT_FOR.get(key, "")
            sc   = _smart_scale(key, a)          # auto-detects ms vs s
            map_items.append((lbl, unit, sc, a))

        if not map_items:
            QMessageBox.information(self, "No Maps", "No quantitative maps available.")
            return

        # Build figure: one subplot per map, bars per ROI
        n_maps = len(map_items)
        nCols = min(3, n_maps)
        nRows = max(1, (n_maps + nCols - 1) // nCols)

        fig = Figure(figsize=(nCols * 5, nRows * 4), facecolor='white')
        fig.suptitle('Per-ROI MRF Map Values', fontsize=12, fontweight='bold')

        for mi, (lbl, unit, scale, map_arr) in enumerate(map_items):
            ax = fig.add_subplot(nRows, nCols, mi + 1)
            ax.set_facecolor('#f8f8f8')
            ax.set_title(lbl, fontsize=9, fontweight='bold')
            ax.set_ylabel(f"{lbl} ({unit})" if unit else lbl, fontsize=8)
            ax.tick_params(labelsize=7)

            if not rois:
                # Whole-image stats
                vals = map_arr[np.isfinite(map_arr)]
                if vals.size > 0:
                    ax.bar(['All pixels'], [vals.mean() * sc],
                           yerr=[vals.std() * sc], capsize=4, color='#4477aa')
                continue

            means, stds, names, colors = [], [], [], []
            for roi in rois:
                msk = roi.mask
                if msk.shape != map_arr.shape[:2]:
                    try:
                        from scipy.ndimage import zoom
                        zy = map_arr.shape[0] / max(msk.shape[0], 1)
                        zx = map_arr.shape[1] / max(msk.shape[1], 1)
                        msk = zoom(msk.astype(float), (zy, zx), order=1) > 0.5
                    except Exception:
                        continue
                vals = map_arr[msk]
                finite = vals[np.isfinite(vals)]
                if finite.size == 0:
                    continue
                means.append(finite.mean() * sc)
                stds.append(finite.std() * sc)
                names.append(roi.name)
                colors.append(getattr(roi, 'color', '#4477aa'))

            if means:
                x = np.arange(len(means))
                ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(names, rotation=30, ha='right', fontsize=7)

        fig.tight_layout()

        dlg = QDialog(self)
        dlg.setWindowTitle("ROI — MRF Map Values")
        dlg.resize(nCols * 420, nRows * 340 + 60)
        vl = QVBoxLayout(dlg)
        fc = FigureCanvas(fig)
        vl.addWidget(fc, stretch=1)
        btn_save = QPushButton("Save figure…")
        def _save():
            from my_gui.fig_export import save_figure, FIG_EXPORT_FILTER
            p, _ = QFileDialog.getSaveFileName(dlg, "Save figure", "roi_mrf_maps.png",
                                               FIG_EXPORT_FILTER)
            if p:
                save_figure(fig, p, dpi=300)
        btn_save.clicked.connect(_save)
        chk_bg = QCheckBox("Bg")
        chk_bg.setToolTip("Black background for the figure (for slides). "
                          "Only the surround and labels flip — the bars stay identical.")
        def _toggle_bg(on):
            apply_fig_dark_theme(fig, on)
            fc.draw()
        chk_bg.toggled.connect(_toggle_bg)
        bot_row = QHBoxLayout()
        bot_row.addWidget(btn_save)
        bot_row.addStretch()
        bot_row.addWidget(chk_bg)
        vl.addLayout(bot_row)
        apply_fig_dark_theme(fig, chk_bg.isChecked())
        dlg.show()

    # ── Per-ROI stats table ─────────────────────────────────────────────────────

    def _show_roi_table(self):
        """Show per-ROI mean ± std for all available MRF parametric maps."""
        from my_gui.roi_table_dialog import show_roi_table
        rois = getattr(self, '_last_rois', [])
        quant_maps = getattr(self, '_quant_maps', None)
        if not quant_maps:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "No Data", "Load and match MRF results first.")
            return

        map_items: list[tuple] = []
        for key, arr in quant_maps.items():
            if arr is None:
                continue
            a = np.array(arr, dtype=float)
            while a.ndim > 2:
                a = a[..., 0]
            lbl = _MAP_DEFAULTS[key][0] if key in _MAP_DEFAULTS else key
            sc  = _smart_scale(key, a)          # auto-detects ms vs s
            map_items.append((lbl, a * sc if sc != 1 else a))

        show_roi_table(self, rois, map_items, title="MRF Results — ROI Statistics")

    # ── Pixel-pick mode ───────────────────────────────────────────────────────

    def _open_simulate_menu(self):
        """Ask whether to simulate the drawn ROIs or pick a single pixel."""
        from PyQt6.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setWindowTitle("Simulate MRF fingerprint")
        box.setText("Compare the measured signal against the best-match\n"
                    "dictionary fingerprint for:")
        b_rois  = box.addButton("Drawn ROIs", QMessageBox.ButtonRole.AcceptRole)
        b_pixel = box.addButton("Pick a pixel", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is b_rois:
            self._simulate_rois()
        elif clicked is b_pixel:
            self._enter_pixel_pick_mode()

    def _simulate_rois(self):
        """Run the fingerprint comparison on the drawn ROIs (no pixel needed)."""
        from PyQt6.QtWidgets import QMessageBox
        rois = [r for r in getattr(self, '_rois', [])
                if getattr(r, 'name', '') != 'Phantom_outline'
                and getattr(r, 'mask', None) is not None]
        if not rois:
            QMessageBox.information(
                self, "No ROIs",
                "No ROIs found.  Draw ROIs in the ROI Manager tab first,\n"
                "then Simulate → Drawn ROIs.")
            return
        self._picked_pixel = None        # ROI-only comparison
        try:
            self._show_fingerprint_comparison()
        except Exception as exc:
            import traceback
            QMessageBox.critical(self, "Simulate — error",
                                 f"Could not open the fingerprint comparison:\n{exc}")
            print(traceback.format_exc())

    def _enter_pixel_pick_mode(self):
        """Arm single-pixel pick — the next image click opens the comparison."""
        # Pixel-pick and Window/Level both grab the mouse — keep exclusive.
        if self.btn_contrast.isChecked():
            self.btn_contrast.setChecked(False)
        if self._pp_cid is None:
            self._pp_cid = self.canvas.mpl_connect(
                'button_press_event', self._on_pixel_click)
        self.lbl_picked.setText("Click a pixel to simulate…")
        self.lbl_picked.setVisible(True)

    def _exit_pixel_pick_mode(self):
        """Disconnect the pixel-pick click handler."""
        if self._pp_cid is not None:
            self.canvas.mpl_disconnect(self._pp_cid)
            self._pp_cid = None

    # ── Window/Level (brightness–contrast) drag tool — OsiriX-style ──────────
    def _on_wl_tool_toggled(self, checked: bool):
        """Activate/deactivate the interactive window/level drag tool."""
        checked = bool(checked)
        if checked and self._pp_cid is not None:
            self._exit_pixel_pick_mode()            # mutually exclusive
        if checked == self._wl_active:
            return
        self._wl_active = checked
        if checked:
            self._wl_cids = [
                self.canvas.mpl_connect('button_press_event',   self._wl_on_press),
                self.canvas.mpl_connect('motion_notify_event',  self._wl_on_motion),
                self.canvas.mpl_connect('button_release_event', self._wl_on_release),
            ]
            try:
                self.canvas.setCursor(Qt.CursorShape.SizeAllCursor)
            except Exception:
                pass
        else:
            for cid in self._wl_cids:
                try:
                    self.canvas.mpl_disconnect(cid)
                except Exception:
                    pass
            self._wl_cids = []
            self._wl_drag = None
            try:
                self.canvas.unsetCursor()
            except Exception:
                pass

    def _wl_data_range(self):
        """Full finite range of the currently displayed image — the drag scale."""
        d = self._dc_img_data
        if d is None:
            return None
        fin = np.asarray(d)[np.isfinite(d)]
        if fin.size == 0:
            return None
        lo, hi = float(fin.min()), float(fin.max())
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def _wl_on_press(self, event):
        if not self._wl_active or event.button != 1:
            return
        if event.inaxes is None or self._im is None:
            return
        rng = self._wl_data_range()
        if rng is None:
            return
        lo, hi = rng
        vmin, vmax = self._im.get_clim()
        if vmin is None:
            vmin = lo
        if vmax is None:
            vmax = hi
        self._wl_drag = {
            'x': event.x, 'y': event.y,
            'w': float(vmax - vmin),
            'l': float((vmax + vmin) / 2.0),
            'scale': (hi - lo) / 250.0,   # ~250 px drag sweeps the full range
        }

    def _wl_on_motion(self, event):
        if self._wl_drag is None or event.x is None or event.y is None:
            return
        d = self._wl_drag
        width = d['w'] + (event.x - d['x']) * d['scale']
        level = d['l'] - (event.y - d['y']) * d['scale']
        min_w = abs(d['scale']) * 1e-2 + 1e-12   # keep a positive window
        if width < min_w:
            width = min_w
        self._wl_drag['vmin'] = level - width / 2.0
        self._wl_drag['vmax'] = level + width / 2.0
        try:
            self._im.set_clim(self._wl_drag['vmin'], self._wl_drag['vmax'])
        except Exception:
            return
        self.canvas.draw_idle()

    def _wl_on_release(self, event):
        if self._wl_drag is None:
            return
        vmin = self._wl_drag.get('vmin')
        vmax = self._wl_drag.get('vmax')
        self._wl_drag = None
        if vmin is not None and self._current_map_key is not None:
            # Remember per-map so the window survives later redraws (ROI toggle,
            # font tweaks, etc.) without touching colormap/custom settings.
            self._wl_override = (self._current_map_key, float(vmin), float(vmax))

    def _on_pixel_click(self, event):
        """Handle a canvas click in pixel-pick mode."""
        if event.inaxes is None or event.xdata is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))

        # Validate against the current image dimensions
        if self._dc_img_data is not None:
            H, W = self._dc_img_data.shape[:2]
            if not (0 <= row < H and 0 <= col < W):
                return

        self._picked_pixel = (row, col)

        # Draw/update a + crosshair on the image
        axes = self._fig.axes
        if axes:
            ax = axes[0]
            if self._pp_marker is not None:
                try:
                    for artist in self._pp_marker:
                        artist.remove()
                except Exception:
                    pass
            h_line, = ax.plot([col - 3, col + 3], [row, row],
                              color='#ffaa00', lw=1.0, zorder=10)
            v_line, = ax.plot([col, col], [row - 3, row + 3],
                              color='#ffaa00', lw=1.0, zorder=10)
            dot,    = ax.plot([col], [row], 'o',
                              color='#ffaa00', ms=3.0, zorder=11,
                              markeredgecolor='#ffffff', markeredgewidth=0.6)
            self._pp_marker = [h_line, v_line, dot]
            self.canvas.draw_idle()

        self.lbl_picked.setText(f"✕  pixel ({row}, {col})")
        self.lbl_picked.setVisible(True)

        # Open the fingerprint-comparison window for the just-picked pixel ONLY
        # (Simulate → Pick a pixel; drawn ROIs are excluded).
        try:
            self._show_fingerprint_comparison(pixel_only=True)
        except Exception as exc:
            import traceback
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self, "Simulate — error",
                f"Could not open the fingerprint comparison:\n{exc}")
            print(traceback.format_exc())

    def _clear_picked_pixel(self):
        """Remove the picked pixel and its marker."""
        self._picked_pixel = None
        if self._pp_marker is not None:
            try:
                for artist in self._pp_marker:
                    artist.remove()
            except Exception:
                pass
            self._pp_marker = None
            self.canvas.draw_idle()
        self.lbl_picked.setVisible(False)
        self._exit_pixel_pick_mode()

    # ── Data cursor ───────────────────────────────────────────────────────────

    def _toggle_datacursor(self, enabled: bool):
        if enabled:
            if self._dc_cid is None:
                self._dc_cid = self.canvas.mpl_connect(
                    "motion_notify_event", self._on_dc_hover
                )
            self._dc_annot = None  # will be created lazily in _on_dc_hover
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

    def _on_dc_hover(self, event):
        axes = self._fig.axes
        if not axes:
            return
        ax = axes[0]
        if event.inaxes is not ax or self._dc_img_data is None:
            if self._dc_annot is not None:
                self._dc_annot.set_visible(False)
                self.canvas.draw_idle()
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        xi, yi = int(round(x)), int(round(y))
        H, W = self._dc_img_data.shape[:2]
        if 0 <= yi < H and 0 <= xi < W:
            val = self._dc_img_data[yi, xi]
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

    # ── Fingerprint comparison dialog ─────────────────────────────────────────

    def _show_fingerprint_comparison(self, pixel_only: bool = False):
        """
        Open a dialog plotting measured vs best-match simulated MRF signal
        for each ROI (or for the whole image if no ROIs are defined).

        When `pixel_only` is True (Simulate → Pick a pixel), only the picked
        pixel is shown — the drawn ROIs are excluded.

        X-axis : measurement index (0 … N−1)
        Y-axis : normalised signal

        The best-match dictionary entry is found by re-running the dot-product
        for the ROI mean signal — fast (single matrix-vector multiply) and does
        not require dp_indexes to be stored in quant_maps.mat.
        """
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QScrollArea,
            QWidget, QMessageBox, QPushButton as _QPB,
            QLabel as _QL, QSizePolicy as _QSP,
        )
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        import scipy.io as sio

        # ── Pre-flight checks ──────────────────────────────────────────────
        if not self._dict_fn or not str(self._dict_fn).strip():
            QMessageBox.warning(
                self, "No Dictionary",
                "Generate or load a dictionary (dict.mat) first.\n"
                "Use the 'Dictionary' button or run matching."
            )
            return
        if self._acquired_data is None:
            QMessageBox.warning(
                self, "No Acquired Data",
                "Load acquired_data.mat first.\n"
                "Use the 'Acquired Data' button or run matching."
            )
            return

        # ── Load dictionary ────────────────────────────────────────────────
        try:
            if str(self._dict_fn).endswith(".npz"):
                synt_dict = dict(np.load(self._dict_fn, allow_pickle=True))
            else:
                synt_dict = sio.loadmat(self._dict_fn)
        except Exception as exc:
            QMessageBox.critical(self, "Error loading dictionary", str(exc))
            return

        # Mirrors the logic in cest_mrf/metrics/dot_product.py
        try:
            real_keys = [k for k in synt_dict if not k.startswith('_')]
            if len(real_keys) < 4:
                # Nested struct format (older MATLAB save)
                inner = synt_dict[real_keys[0]][0]
                synt_sig = np.array(inner['sig'][0], dtype=float).T   # (n_meas, n_dict)
                dict_fs  = inner['fs'][0].T.ravel()
                dict_ksw = inner['ksw'][0].T.ravel()
                dict_t1w = inner['t1w'][0].T.ravel()
                dict_t2w = inner['t2w'][0].T.ravel()
            else:
                # Flat format (standard)
                synt_sig = np.array(synt_dict['sig'], dtype=float).T  # (n_meas, n_dict)
                dict_fs  = np.array(synt_dict.get('fs_0',
                                    synt_dict.get('fs',  np.array([]))), dtype=float).ravel()
                dict_ksw = np.array(synt_dict.get('ksw_0',
                                    synt_dict.get('ksw', np.array([]))), dtype=float).ravel()
                dict_t1w = np.array(synt_dict.get('t1w', np.array([])), dtype=float).ravel()
                dict_t2w = np.array(synt_dict.get('t2w', np.array([])), dtype=float).ravel()
        except Exception as exc:
            QMessageBox.critical(
                self, "Error parsing dictionary",
                f"Could not extract 'sig' from dictionary:\n{exc}\n\n"
                "Make sure the file was generated by this pipeline."
            )
            return

        n_meas_dict, n_dict = synt_sig.shape

        # ── Acquired data ──────────────────────────────────────────────────
        # self._acquired_data has shape (n_meas, n_slices, Y, X)
        acq = self._acquired_data
        n_meas_acq = acq.shape[0]
        H, W        = acq.shape[2], acq.shape[3]

        if n_meas_dict != n_meas_acq:
            QMessageBox.warning(
                self, "Measurement count mismatch",
                f"Dictionary has {n_meas_dict} measurements but\n"
                f"acquired data has {n_meas_acq} measurements.\n\n"
                "They must match for fingerprint comparison."
            )
            return

        # Normalise dictionary columns (one column = one dict entry)
        norm_dict = synt_sig / (np.linalg.norm(synt_sig, axis=0, keepdims=True) + 1e-10)

        # ── Build list of (name, color, mask, is_pixel) tuples ───────────────
        rois = [r for r in getattr(self, '_rois', [])
                if getattr(r, 'name', '') != 'Phantom_outline'
                and getattr(r, 'mask', None) is not None]

        roi_specs = []   # (name, roi_color, mask, is_pixel_pick)

        if rois and not pixel_only:
            for r in rois:
                roi_specs.append((r.name, getattr(r, 'color', '#4a9eff'), r.mask, False))

        # Add single picked pixel if set
        if self._picked_pixel is not None:
            pr, pc = self._picked_pixel
            if 0 <= pr < H and 0 <= pc < W:
                px_mask = np.zeros((H, W), dtype=bool)
                px_mask[pr, pc] = True
                roi_specs.append((f"Pixel ({pr}, {pc})", '#ffaa00', px_mask, True))

        if not roi_specs:
            # Fallback: whole-image mean
            fb_mask = np.ones((H, W), dtype=bool)
            if self._dp_mask is not None:
                dp = self._dp_mask
                if dp.ndim == 3:
                    dp = dp[:, :, 0]
                if dp.shape == (H, W):
                    fb_mask = dp.astype(bool)
            roi_specs.append(('Whole image', '#4a9eff', fb_mask, False))

        # ── Compute best match per ROI / pixel ────────────────────────────
        results = []
        for name, color, raw_mask, is_pixel in roi_specs:
            msk = raw_mask
            if msk.shape != (H, W):
                try:
                    from scipy.ndimage import zoom as _zoom
                    msk = _zoom(msk.astype(float),
                                (H / max(msk.shape[0], 1),
                                 W / max(msk.shape[1], 1)), order=1) > 0.5
                except Exception:
                    continue
            if not msk.any():
                continue

            # Mean signal across all ROI voxels and all slices → (n_meas,)
            roi_vox  = acq[:, :, msk]               # (n_meas, n_slices, n_vox)
            mean_sig = roi_vox.mean(axis=(1, 2))     # (n_meas,)

            # Normalise measured fingerprint
            norm_meas = mean_sig / (np.linalg.norm(mean_sig) + 1e-10)

            # Dot product → best-matching dictionary entry
            scores  = norm_dict.T @ norm_meas        # (n_dict,)
            idx     = int(np.argmax(scores))
            r2_val  = float(scores[idx])
            sim_sig = norm_dict[:, idx]              # (n_meas,)

            # Look up best-match dictionary parameters
            def _dict_val(arr, i):
                if hasattr(arr, '__len__') and len(arr) > i:
                    return float(arr[i])
                return None

            fs_val  = _dict_val(dict_fs,  idx)
            ksw_val = _dict_val(dict_ksw, idx)
            t1w_val = _dict_val(dict_t1w, idx)
            t2w_val = _dict_val(dict_t2w, idx)

            # Convert to display units (fs: fraction→mM, T1/T2: s→ms)
            if fs_val  is not None: fs_val  = fs_val  * (110e3 / max(self.spin_protons.value(), 1))
            if t1w_val is not None: t1w_val = t1w_val * 1000.0
            if t2w_val is not None: t2w_val = t2w_val * 1000.0

            results.append({
                'name':     name,
                'color':    color,
                'is_pixel': is_pixel,
                'measured': norm_meas,
                'sim':      sim_sig,
                'r2':       r2_val,
                'fs':       fs_val,
                'ksw':      ksw_val,
                't1w':      t1w_val,
                't2w':      t2w_val,
            })

        if not results:
            QMessageBox.information(
                self, "No data",
                "No valid ROI voxels found.\n"
                "Draw ROIs in the ROI Manager tab, or load acquired data first."
            )
            return

        # ── Build dialog ───────────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle("MRF Fingerprint Comparison — Measured vs Simulated")
        n_rois = len(results)
        dlg.resize(1000, min(900, max(440, 280 * n_rois)))

        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.setSpacing(4)

        hdr = _QL(
            "<b>MRF Fingerprint Comparison</b>  —  "
            "normalised measured signal (ROI mean) vs best-match simulated signal"
        )
        hdr.setStyleSheet("font-size:12px; padding:4px;")
        dlg_layout.addWidget(hdr)

        # Scroll area so the figure is always accessible even if many ROIs
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        dlg_layout.addWidget(scroll, stretch=1)

        fig_h  = max(3.5, n_rois * 3.2)
        fig    = Figure(figsize=(10, fig_h), facecolor='white', constrained_layout=True)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(_QSP.Policy.Expanding, _QSP.Policy.Expanding)

        x_idx = np.arange(n_meas_dict)

        # Default font sizes (large — tuned for slides).  These are the spinbox
        # DEFAULTS, so the figure looks identical until a control is changed.
        _TITLE_FS, _TICK_FS, _LABEL_FS, _LEG_FS = 28, 28, 24, 20
        # Measured lines are theme-adaptive (white on black bg, black on white) —
        # recoloured by the "Bg" toggle so they stay visible and distinct from the
        # orange "Simulated" line.
        _theme_lines = []

        # ── Font-customization toolbar (family + per-role sizes) ────────────
        # This figure has NO suptitle, so there is deliberately no "Main"
        # control — only Title / Axes / Ticks / Legend.  A change triggers a
        # debounced full re-render (_render below), so sizes live-update.
        font_row = QHBoxLayout()
        font_row.addWidget(_QL("Font:"))
        combo_ff = QComboBox(); combo_ff.setFixedWidth(140)
        combo_ff.setToolTip("Font family for titles, axis labels and legends")
        for _ff in ("Default", "Arial", "Times New Roman",
                    "Helvetica", "DejaVu Sans", "DejaVu Serif"):
            combo_ff.addItem(_ff)
        font_row.addWidget(combo_ff)

        def _mkfs(label, default, tip):
            font_row.addWidget(_QL(label))
            sp = QSpinBox(); sp.setRange(5, 60); sp.setValue(default)
            sp.setFixedWidth(50); sp.setToolTip(tip)
            font_row.addWidget(sp)
            return sp

        spin_title  = _mkfs("Title:",  _TITLE_FS, "Subplot (ROI) title font size")
        spin_axes   = _mkfs("Axes:",   _LABEL_FS, "X / Y axis label font size")
        spin_ticks  = _mkfs("Ticks:",  _TICK_FS,  "Tick label font size")
        spin_legend = _mkfs("Legend:", _LEG_FS,   "Legend font size")
        font_row.addStretch()
        dlg_layout.insertLayout(1, font_row)

        # Shared font-properties helper — honours the chosen family + size
        # (+ bold for titles).  Used at every font site in _render below.
        from matplotlib.font_manager import FontProperties as _FP
        def _fp(size, bold=False):
            ff = combo_ff.currentText()
            kw = {'size': size}
            if ff and ff.lower() != "default":
                kw['family'] = ff
            if bold:
                kw['weight'] = 'bold'
            return _FP(**kw)

        # "Bg" dark-background toggle — created here so _render can read its
        # state; it is added to the bottom button row further below.
        chk_bg = QCheckBox("Bg")
        chk_bg.setToolTip("Black background for the figure (for slides). The "
                          "Measured line flips white↔black to stay visible.")

        def _recolor_theme(dark):
            c  = 'white' if dark else 'black'
            ec = 'black' if dark else 'white'
            for ln in _theme_lines:
                ln.set_color(c)
                ln.set_markerfacecolor(c)
                ln.set_markeredgecolor(ec)
            # Rebuild each legend so its "Measured" swatch shows the NEW colour —
            # a legend copies its handle colours at creation time, so recolouring
            # the line alone leaves the swatch black (invisible on a black bg).
            for _ax in fig.axes:
                if _ax.get_legend() is not None:
                    _lg = _ax.legend(prop=_fp(spin_legend.value()), framealpha=0.9,
                                     loc='upper right')
                    if _lg is not None:
                        _lg.set_draggable(True)

        # ── Full (re)draw of the figure from the current control values ─────
        # Refactored out of an inline one-shot loop so font/family changes can
        # re-run it.  Draws into the SAME canvas (fig.clf() + re-add subplots),
        # then re-applies the current Bg theme so a re-render never loses it.
        def _render():
            title_fs  = spin_title.value()
            ticks_fs  = spin_ticks.value()
            axes_fs   = spin_axes.value()
            legend_fs = spin_legend.value()

            fig.clf()
            _theme_lines.clear()

            for i, res in enumerate(results):
                ax = fig.add_subplot(n_rois, 1, i + 1)
                ax.set_facecolor('white')
                for spine in ax.spines.values():
                    spine.set_color('#333')
                ax.tick_params(colors='#333', labelsize=ticks_fs)
                ax.xaxis.label.set_color('#222')
                ax.yaxis.label.set_color('#222')

                # Measured — theme-adaptive default colour, circle markers + dashed line
                _mline, = ax.plot(x_idx, res['measured'],
                        'o--', color='black', lw=1.5, ms=3.5, alpha=0.95,
                        markeredgecolor='white', markeredgewidth=0.5,
                        label='Measured')
                _theme_lines.append(_mline)

                # Simulated — always bright orange, solid line, no markers
                ax.plot(x_idx, res['sim'],
                        '-', color='#ff9500', lw=2.2, alpha=0.95,
                        label='Simulated')

                # Title: ROI name only
                ax.set_title(
                    f"{res['name']}",
                    fontproperties=_fp(title_fs, bold=True), color='#222', pad=4
                )
                ax.set_xlabel("Image Acquisition Number", fontproperties=_fp(axes_fs))
                ax.set_ylabel("Normalised signal", fontproperties=_fp(axes_fs))
                _leg = ax.legend(prop=_fp(legend_fs), facecolor='white', edgecolor='#888',
                                 labelcolor='#222', framealpha=0.9, loc='upper right')
                # Make the legend draggable — user can move it anywhere on the plot
                if _leg is not None:
                    _leg.set_draggable(True)
                ax.axhline(0, color='#bbb', lw=0.8, ls='--')

                # Highlight measurements where |measured − simulated| > 0.05
                residual = np.abs(res['measured'] - res['sim'])
                for xi in x_idx[residual > 0.05]:
                    ax.axvspan(xi - 0.5, xi + 0.5, color='#ffaa00', alpha=0.08)

            # Re-apply the current Bg state after the rebuild so a font change
            # never resets the dark/light theme.
            _recolor_theme(chk_bg.isChecked())
            apply_fig_dark_theme(fig, chk_bg.isChecked())
            canvas.draw()

        scroll.setWidget(canvas)
        _render()

        # ── Debounced live re-render on any font control change ─────────────
        from PyQt6.QtCore import QTimer
        _font_timer = QTimer(dlg)
        _font_timer.setSingleShot(True)
        _font_timer.setInterval(180)
        _font_timer.timeout.connect(_render)
        def _sched_font(*_):
            _font_timer.start()
        for _sp in (spin_title, spin_axes, spin_ticks, spin_legend):
            _sp.valueChanged.connect(_sched_font)
        combo_ff.currentIndexChanged.connect(_sched_font)

        # ── Bottom buttons ─────────────────────────────────────────────────
        bot_row = QHBoxLayout()
        btn_export = _QPB("Export figure…")

        def _do_export():
            from my_gui.fig_export import save_figure, FIG_EXPORT_FILTER
            fn, _ = QFileDialog.getSaveFileName(
                dlg, "Export fingerprint comparison",
                "fingerprint_comparison",
                FIG_EXPORT_FILTER
            )
            if fn:
                save_figure(fig, fn, dpi=300,
                            facecolor=fig.get_facecolor())

        btn_export.clicked.connect(_do_export)
        bot_row.addWidget(btn_export)

        # chk_bg / _recolor_theme are defined above (before _render) so the
        # initial draw and font re-renders can honour the current Bg state.
        def _toggle_bg(on):
            _recolor_theme(on)
            apply_fig_dark_theme(fig, on)
            canvas.draw()
        chk_bg.toggled.connect(_toggle_bg)
        bot_row.addWidget(chk_bg)

        bot_row.addStretch()
        btn_close = _QPB("Close")
        btn_close.clicked.connect(dlg.accept)
        bot_row.addWidget(btn_close)
        dlg_layout.addLayout(bot_row)

        dlg.exec()
