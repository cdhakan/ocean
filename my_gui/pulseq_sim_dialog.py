"""
pulseq_sim_dialog.py
====================

"Pulseq Simulation" window for the CEST-MRF Sequence / Simulation tab.

Lets a user pick a ready-made Pulseq sequence from the *pulseq-cest-library*
(or browse to any ``.seq``), edit the scanner limits + a dictionary parameter
sweep in the **upper bar**, and in the **lower bar** (a) view the pulse-sequence
diagram and (b) run a Bloch–McConnell **dictionary simulation + dot-product
matching** demo — exactly the pipeline the acquisition uses, driven by an example
sequence so users can see how the fingerprint changes as they vary parameters.

The heavy simulation runs off the UI thread (``_SimWorker``).  It reuses the
verified backend:  ``write_yaml_dict`` → ``generate_mrf_cest_dictionary`` (from a
``.seq`` + synthesised YAML) → ``dot_prod_matching``.
"""
from __future__ import annotations

import glob
import os
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
try:
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as _NavToolbar
except Exception:                                    # pragma: no cover
    _NavToolbar = None

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QEvent
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox, QPushButton,
    QDoubleSpinBox, QSpinBox, QFileDialog, QTabWidget, QWidget, QScrollArea,
    QGroupBox, QMessageBox, QApplication, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame, QCheckBox, QProgressDialog,
)

# mplot3d registers the '3d' projection as a side effect of import
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import contextlib as _contextlib


@_contextlib.contextmanager
def _pypulseq_ctx(*_args, **_kwargs):
    """Yield the vendored Pulseq-1.3.1 build for reading/plotting a .seq.

    OCEAN ships a single, vendored pypulseq (1.3.1) — the exact format the C++
    BMCSimulator reads and the pulseq-cest-library uses — so authoring, reading,
    plotting and simulation all use one version. (Positional/keyword args are
    accepted and ignored for backwards compatibility with older call sites.)
    """
    from my_gui.worker import _vendored_pypulseq
    with _vendored_pypulseq():
        import pypulseq as pp
        yield pp


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_default_library() -> str:
    """Locate the pulseq-cest-library ``seq-library`` folder across environments.

    Search order: ``$OCEAN_PULSEQ_LIBRARY`` env var → a copy bundled next to the
    frozen app (``sys._MEIPASS``) or inside the repo → the developer's Downloads
    path (dev fallback).  Returns the first that exists, else the last candidate
    so the UI still shows a sensible (browsable) default.
    """
    candidates = [
        os.environ.get("OCEAN_PULSEQ_LIBRARY"),
        os.path.join(getattr(sys, "_MEIPASS", ""), "pulseq-cest-library", "seq-library"),
        os.path.join(_repo_root(), "pulseq-cest-library", "seq-library"),
        os.path.join(_repo_root(), "seq-library"),
        "/Users/cbd/Downloads/pulseq-cest-library-master/seq-library",
    ]
    candidates = [c for c in candidates if c]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[-1]


# Default location of the pulseq-cest-library (resolved per environment).
_DEFAULT_LIBRARY = _find_default_library()


def _ensure_paths():
    """Put repo root + open-py-cest-mrf on sys.path so ``cest_mrf`` imports."""
    for p in (_repo_root(), os.path.join(_repo_root(), "open-py-cest-mrf")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def discover_library_seqs(library: str = _DEFAULT_LIBRARY):
    """Return sorted ``[(display_name, seq_path), …]`` for every ``.seq`` found."""
    out: list[tuple[str, str]] = []
    if not os.path.isdir(library):
        return out
    for seq in sorted(glob.glob(os.path.join(library, "*", "*.seq"))):
        folder = os.path.basename(os.path.dirname(seq))
        fn = os.path.basename(seq)
        label = folder if fn == folder + ".seq" else f"{folder}  ·  {fn}"
        out.append((label, seq))
    return out


class _WheelForwarder(QObject):
    """Forward mouse-wheel events from a child (e.g. a matplotlib canvas that would
    otherwise swallow them) to a QScrollArea, so the wheel scrolls the window."""

    def __init__(self, scroll_area):
        super().__init__(scroll_area)
        self._sa = scroll_area

    def eventFilter(self, obj, event):  # noqa: N802
        if event.type() == QEvent.Type.Wheel:
            sb = self._sa.verticalScrollBar()
            sb.setValue(sb.value() - int(event.angleDelta().y()))
            return True                 # consume → the canvas won't also handle it
        return False


def _linspace_list(lo: float, hi: float, n: int) -> list[float]:
    """n≥2 → linspace; n≤1 → [lo].  Rounded to avoid float noise in the YAML."""
    if n <= 1 or hi <= lo:
        return [float(lo)]
    return [float(round(v, 8)) for v in np.linspace(lo, hi, int(n))]


# ─────────────────────────────────────────────────────────────────────────────
# Simulation worker (off the UI thread)
# ─────────────────────────────────────────────────────────────────────────────

class _SimWorker(QThread):
    finished = pyqtSignal(str)     # path to the generated dictionary .mat
    error    = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, seq_fn: str, cfg: dict, axes: str = "z", parent=None):
        super().__init__(parent)
        self._seq_fn = seq_fn
        self._cfg = cfg
        self._axes = axes            # 'z' → Mz fingerprint (library pseudo-ADC seqs); 'xy' → transverse

    def run(self):
        try:
            _ensure_paths()
            from cest_mrf.write_scenario import write_yaml_dict
            from cest_mrf.dictionary.generation import generate_mrf_cest_dictionary

            yaml_fn = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False).name
            dict_fn = tempfile.NamedTemporaryFile(suffix=".mat", delete=False).name
            cfg = dict(self._cfg)
            cfg["yaml_fn"] = yaml_fn
            cfg["seq_fn"] = self._seq_fn
            cfg["dict_fn"] = dict_fn
            self.progress.emit("Writing scenario YAML…")
            write_yaml_dict(cfg, yaml_fn)
            self.progress.emit("Simulating Bloch–McConnell dictionary… (this can take a moment)")
            generate_mrf_cest_dictionary(
                seq_fn=self._seq_fn, param_fn=yaml_fn, dict_fn=dict_fn,
                num_workers=1, axes=self._axes)
            self.finished.emit(dict_fn)
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# Dialog
# ─────────────────────────────────────────────────────────────────────────────

class PulseqSimDialog(QDialog):
    """Upper bar = parameters (example seq · scanner limits · dictionary sweep);
    lower bar = figures (sequence diagram · fingerprints · matching demo)."""

    def __init__(self, parent=None, library: str = _DEFAULT_LIBRARY):
        super().__init__(parent)
        self._library = library
        self._seq_path: str | None = None
        self._worker: _SimWorker | None = None
        self._wheel_fwds: dict = {}     # one _WheelForwarder per scroll area
        self._dark_fig = False          # "Bg" toggle — black background for all figures
        self.setWindowTitle("Pulseq Simulation  —  MRF dictionary generation & matching")
        # Universal (theme-adaptive) palette — dark text on light OS themes, light
        # text on dark OS themes, following the app/OS appearance.
        self._init_theme()
        self.setMinimumSize(940, 680)
        screen = QApplication.primaryScreen()
        if screen:
            ag = screen.availableGeometry()
            self.resize(min(1180, int(ag.width() * 0.82)),
                        min(900, int(ag.height() * 0.90)))
        self._build_ui()
        self._populate_library()
        self._update_count()

    # ── Theme (universal light/dark) ─────────────────────────────────────────

    def _init_theme(self):
        win = self.palette().color(QPalette.ColorRole.Window)
        self._dark = win.lightness() < 128
        if self._dark:
            self._c_text, self._c_mut = "#e6e8ee", "#9aa2b1"
            self._c_panel, self._c_border, self._c_accent = "#262a33", "#3a3f4b", "#5aa2ff"
            self._c_input, self._c_bg = "#1e222b", "#1a1d24"
            self._c_card_bg, self._c_card_border = "#242a35", "#5b6478"   # group-card fill + clear border
        else:
            self._c_text, self._c_mut = "#1a1c22", "#5c6470"
            self._c_panel, self._c_border, self._c_accent = "#f6f7fb", "#dfe3ea", "#1565c0"
            self._c_input, self._c_bg = "#ffffff", "#ffffff"
            self._c_card_bg, self._c_card_border = "#fbfcff", "#bcc3d0"
        # The figure AREA (tabs, plots, their labels/toolbars) is ALWAYS white —
        # every figure keeps a light background regardless of the OS theme.
        self._fig_bg, self._fig_fg, self._fig_grid = "#ffffff", "#222222", "#cfd4dc"
        self._fig_label = "#1565c0"      # figure section headings (on white)
        self._fig_note = "#555555"       # note text (on white)
        self._fig_tb_bg, self._fig_tb_fg, self._fig_tb_mut = "#f0f0f0", "#222222", "#555555"
        self.setStyleSheet(
            f"QDialog {{ background:{self._c_bg}; }}"
            # Labels: NO box/border, transparent, themed text (white on dark OS,
            # dark on light OS), larger for readability.
            f"QLabel {{ color:{self._c_text}; background:transparent; border:none; font-size:13px; }}"
            f"QDoubleSpinBox, QSpinBox, QComboBox {{ background:{self._c_input}; color:{self._c_text};"
            f" border:1px solid {self._c_border}; border-radius:3px; padding:1px 4px; font-size:13px; }}"
            f"QComboBox QAbstractItemView {{ background:{self._c_input}; color:{self._c_text};"
            f" selection-background-color:{self._c_accent}; }}"
            # Group cards — styled exactly like the T1/T2 scan cards: the title sits
            # INSIDE the box (subcontrol-origin: padding + padding-top) so all the
            # content is enclosed by the border.
            f"QGroupBox {{ color:{self._c_text}; background:{self._c_card_bg};"
            f" border:1.5px solid {self._c_card_border};"
            f" border-radius:8px; margin-top:8px; padding:34px 12px 12px 12px;"
            f" font-weight:bold; }}"
            f"QGroupBox::title {{ subcontrol-origin:padding; subcontrol-position:top left;"
            f" left:12px; top:7px; padding:0 6px; font-size:18px; color:{self._c_text}; }}"
            f"QCheckBox {{ color:{self._c_text}; background:transparent; font-size:13px; }}"
            f"QTabBar::tab {{ color:{self._c_text}; padding:4px 12px; }}"
            f"QTabBar::tab:selected {{ color:{self._c_accent}; font-weight:bold; }}"
        )

    def _style_fig(self, fig, axes=None):
        """Apply the current theme to a matplotlib figure + its axes."""
        fig.set_facecolor(self._fig_bg)
        axl = axes if axes is not None else fig.get_axes()
        for ax in axl:
            ax.set_facecolor(self._fig_bg)
            ax.tick_params(colors=self._fig_fg, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(self._fig_grid)
            ax.xaxis.label.set_color(self._fig_fg)
            ax.yaxis.label.set_color(self._fig_fg)
            if ax.get_title():
                ax.set_title(ax.get_title(), color=self._fig_fg, fontsize=9)
            try:
                ax.zaxis.label.set_color(self._fig_fg)                 # 3-D axes
                ax.tick_params(axis="z", colors=self._fig_fg)
            except Exception:
                pass
        # If the "Bg" toggle is on, flip this freshly-styled figure to black so
        # newly-created plots come up themed automatically.
        if getattr(self, "_dark_fig", False):
            from my_gui.fig_theme import apply_fig_dark_theme
            apply_fig_dark_theme(fig, True)

    def _on_dark_toggled(self, checked: bool):
        """Re-theme every existing figure across the 4 tabs to match the toggle."""
        self._dark_fig = bool(checked)
        from my_gui.fig_theme import apply_fig_dark_theme
        for tab in (self._tab_seq, self._tab_kspace, self._tab_fp, self._tab_match):
            lay = tab[2]
            for i in range(lay.count()):
                w = lay.itemAt(i).widget()
                if isinstance(w, FigureCanvas):
                    apply_fig_dark_theme(w.figure, self._dark_fig)
                    w.draw()

    # ── UI ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        title = QLabel("Pulseq Simulation")
        title.setStyleSheet(f"font-size:15px; font-weight:bold; color:{self._c_accent}; background:transparent;")
        root.addWidget(title)

        # ── UPPER BAR — parameters ────────────────────────────────────────────
        # No wrapping panel border/background — the group cards provide the
        # structure, and a panel-level border would cascade onto every child label
        # (the boxes-around-labels the user asked to remove).
        upper = QWidget()
        upper.setStyleSheet("background:transparent;")
        ub = QVBoxLayout(upper)
        ub.setContentsMargins(10, 8, 10, 10)
        ub.setSpacing(8)

        # sequence-picker row
        seqrow = QHBoxLayout()
        seqrow.addWidget(self._lbl("Pulseq Sequence (.seq):"))
        self.combo_seq = QComboBox()
        self.combo_seq.setMinimumWidth(420)
        self.combo_seq.currentIndexChanged.connect(self._on_seq_changed)
        seqrow.addWidget(self.combo_seq, stretch=1)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_seq)
        seqrow.addWidget(btn_browse)
        ub.addLayout(seqrow)
        # (the full .seq path line was removed at user request — the combo shows
        #  the sequence name; a hidden holder keeps _on_seq_changed simple.)
        self.lbl_seq_info = QLabel("", self); self.lbl_seq_info.setVisible(False)

        # scanner limits + sweep, side by side
        grids = QHBoxLayout(); grids.setSpacing(10)
        grids.addWidget(self._build_scanner_group(), stretch=1)
        grids.addWidget(self._build_sweep_group(), stretch=1)
        ub.addLayout(grids)

        # action buttons
        actions = QHBoxLayout()
        self.btn_view = QPushButton("View Sequence")
        self.btn_view.setStyleSheet(
            "QPushButton{background:#6a1b9a;color:white;border-radius:4px;padding:6px 16px;font-weight:bold;}"
            "QPushButton:hover{background:#8e24aa;}"
            "QPushButton:disabled{background:#bbb;color:#eee;}")
        self.btn_view.clicked.connect(self._view_sequence)
        self.btn_sim = QPushButton("Simulate Dictionary")
        self.btn_sim.setStyleSheet(
            "QPushButton{background:#1565c0;color:white;border-radius:4px;padding:6px 16px;font-weight:bold;}"
            "QPushButton:hover{background:#1976d2;}"
            "QPushButton:disabled{background:#bbb;color:#eee;}")
        self.btn_sim.clicked.connect(self._run_simulation)
        self.btn_export = QPushButton("Export .seq")
        self.btn_export.setStyleSheet(
            "QPushButton{background:#00695c;color:white;border-radius:4px;padding:6px 16px;font-weight:bold;}"
            "QPushButton:hover{background:#00897b;}"
            "QPushButton:disabled{background:#bbb;color:#eee;}")
        self.btn_export.setToolTip(
            "Export the currently selected sequence to a .seq file you choose,\n"
            "re-written with the current scanner-limit system settings\n"
            "(grad/slew/RF/ADC timing). Ready to load on a scanner.")
        self.btn_export.clicked.connect(self._export_seq)
        actions.addWidget(self.btn_view)
        actions.addWidget(self.btn_sim)
        actions.addWidget(self.btn_export)
        actions.addSpacing(8)
        actions.addWidget(self._lbl("Readout:"))
        self.combo_axes = QComboBox()
        self.combo_axes.addItem("Mz  (saturation)", "z")
        self.combo_axes.addItem("Mxy  (transverse)", "xy")
        self.combo_axes.setToolTip(
            "Library CEST-MRF sequences use a placeholder ADC, so the fingerprint is the\n"
            "longitudinal Mz after saturation. Use Mxy only for sequences with a real readout.")
        actions.addWidget(self.combo_axes)
        actions.addSpacing(12)
        self.chk_dark = QCheckBox("Bg")
        self.chk_dark.setToolTip(
            "Black background for all figures (for slides). Only the surround and "
            "labels flip — the plotted data / colours stay identical.")
        self.chk_dark.toggled.connect(self._on_dark_toggled)
        actions.addWidget(self.chk_dark)
        actions.addSpacing(12)
        self.lbl_count = QLabel("")
        self.lbl_count.setStyleSheet(f"font-size:11px; color:{self._c_text}; background:transparent;")
        actions.addWidget(self.lbl_count)
        actions.addStretch()
        ub.addLayout(actions)
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("font-size:11px; color:#c8871f; background:transparent;")
        ub.addWidget(self.lbl_status)

        root.addWidget(upper)

        # ── LOWER BAR — figures ───────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            f"QTabWidget::pane{{border:1px solid {self._c_border}; background:{self._fig_bg};}}")
        self._tab_seq    = self._make_scroll_tab()
        _kspace_widget   = self._build_kspace_tab()          # sets self._tab_kspace
        self._tab_fp     = self._make_scroll_tab()
        self._tab_match  = self._make_scroll_tab()
        self.tabs.addTab(self._tab_seq[0],   "Sequence Diagram")
        self.tabs.addTab(self._tab_fp[0],    "Simulation")
        # k-space and Matching Demo tabs removed for now (widgets kept in code,
        # simply not added to the tab bar so they can be re-enabled later).
        root.addWidget(self.tabs, stretch=1)

        # close row
        close_row = QHBoxLayout(); close_row.addStretch()
        btn_close = QPushButton("Close"); btn_close.clicked.connect(self.accept)
        close_row.addWidget(btn_close)
        root.addLayout(close_row)

    def _lbl(self, txt):
        q = QLabel(txt)
        q.setStyleSheet(f"background:transparent; color:{self._c_text}; border:none; font-size:15px;")
        return q

    def _spin(self, lo, hi, val, dec=3, step=0.1, w=90, suffix=""):
        s = QDoubleSpinBox()
        s.setRange(lo, hi); s.setDecimals(dec); s.setSingleStep(step); s.setValue(val)
        s.setFixedWidth(w)
        if suffix:
            s.setSuffix(suffix)
        return s

    def _ispin(self, lo, hi, val, w=64):
        s = QSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setFixedWidth(w)
        s.valueChanged.connect(self._update_count)
        return s

    def _build_scanner_group(self) -> QGroupBox:
        grp = QGroupBox("Scanner limits")
        g = QGridLayout(grp); g.setSpacing(6)
        r = 0
        g.addWidget(self._lbl("B₀ [T]:"), r, 0); self.sp_b0 = self._spin(0.5, 21.0, 3.0, 2, 0.1, 80); g.addWidget(self.sp_b0, r, 1)
        g.addWidget(self._lbl("γ [Hz/T]:"), r, 2); self.sp_gamma = self._spin(1e6, 5e8, 42576400.0, 0, 1e5, 120); g.addWidget(self.sp_gamma, r, 3)
        # γ is a fixed physical constant (¹H gyromagnetic ratio) — display only.
        from PyQt6.QtWidgets import QAbstractSpinBox as _QASB
        self.sp_gamma.setReadOnly(True)
        self.sp_gamma.setButtonSymbols(_QASB.ButtonSymbols.NoButtons)
        self.sp_gamma.setToolTip("Proton gyromagnetic ratio (fixed constant)")
        r += 1
        g.addWidget(self._lbl("Max grad [mT/m]:"), r, 0); self.sp_grad = self._spin(1, 300, 40.0, 0, 1, 80); g.addWidget(self.sp_grad, r, 1)
        g.addWidget(self._lbl("Max slew [T/m/s]:"), r, 2); self.sp_slew = self._spin(1, 1000, 130.0, 0, 5, 80); g.addWidget(self.sp_slew, r, 3)
        r += 1
        g.addWidget(self._lbl("RF ringdown [µs]:"), r, 0); self.sp_ring = self._spin(0, 1000, 30.0, 0, 5, 80)
        self.sp_ring.setToolTip("Coil ring-down delay after each RF pulse before the next event.")
        g.addWidget(self.sp_ring, r, 1)
        g.addWidget(self._lbl("RF dead [µs]:"), r, 2); self.sp_dead = self._spin(0, 5000, 100.0, 0, 10, 80)
        self.sp_dead.setToolTip("Dead time before each RF pulse (coil / gradient settle).")
        g.addWidget(self.sp_dead, r, 3)
        r += 1
        # ── ADC / raster timing limits (Pulseq system definitions) ──────────────
        # Defaults match a typical Siemens system: adcDeadTime 10 µs, rfRasterTime
        # 1 µs, gradRasterTime 10 µs, adcRasterTime 0.1 µs, blockDurationRaster 10 µs.
        g.addWidget(self._lbl("ADC dead [µs]:"), r, 0); self.sp_adc_dead = self._spin(0, 1000, 10.0, 2, 1, 80)
        self.sp_adc_dead.setToolTip("Dead time around each ADC event (adcDeadTime).")
        g.addWidget(self.sp_adc_dead, r, 1)
        g.addWidget(self._lbl("RF raster [µs]:"), r, 2); self.sp_rf_raster = self._spin(0.001, 100, 1.0, 3, 0.1, 80)
        self.sp_rf_raster.setToolTip("RF waveform raster time (rfRasterTime).")
        g.addWidget(self.sp_rf_raster, r, 3)
        r += 1
        g.addWidget(self._lbl("Grad raster [µs]:"), r, 0); self.sp_grad_raster = self._spin(0.001, 1000, 10.0, 3, 1, 80)
        self.sp_grad_raster.setToolTip("Gradient waveform raster time (gradRasterTime).")
        g.addWidget(self.sp_grad_raster, r, 1)
        g.addWidget(self._lbl("ADC raster [µs]:"), r, 2); self.sp_adc_raster = self._spin(0.001, 100, 0.1, 3, 0.05, 80)
        self.sp_adc_raster.setToolTip("ADC sampling raster time (adcRasterTime).")
        g.addWidget(self.sp_adc_raster, r, 3)
        r += 1
        g.addWidget(self._lbl("Block raster [µs]:"), r, 0); self.sp_block_raster = self._spin(0.001, 1000, 10.0, 3, 1, 80)
        self.sp_block_raster.setToolTip("Block-duration raster time (blockDurationRaster).")
        g.addWidget(self.sp_block_raster, r, 1)
        g.addWidget(self._lbl("B₁ scale (rel):"), r, 2)
        self.sp_rel_b1 = self._spin(0.05, 5.0, 1.0, 2, 0.05, 80)
        g.addWidget(self.sp_rel_b1, r, 3)
        return grp

    def _build_sweep_group(self) -> QGroupBox:
        grp = QGroupBox("Tissue Parameters")
        g = QGridLayout(grp); g.setSpacing(6)
        # header
        for c, h in enumerate(["", "min", "max", "steps"]):
            lab = QLabel(h); lab.setStyleSheet(f"font-size:11px; color:{self._c_mut}; background:transparent; border:none;")
            g.addWidget(lab, 0, c)
        r = 1
        # water T1w
        g.addWidget(self._lbl("Water T₁ [s]"), r, 0)
        self.t1w_lo = self._spin(0.1, 6, 2.8, 2, 0.1, 70); self.t1w_hi = self._spin(0.1, 6, 3.0, 2, 0.1, 70); self.t1w_n = self._ispin(1, 100, 1)
        g.addWidget(self.t1w_lo, r, 1); g.addWidget(self.t1w_hi, r, 2); g.addWidget(self.t1w_n, r, 3); r += 1
        # water T2w
        g.addWidget(self._lbl("Water T₂ [s]"), r, 0)
        self.t2w_lo = self._spin(0.01, 4, 0.7, 3, 0.05, 70); self.t2w_hi = self._spin(0.01, 4, 1.0, 3, 0.05, 70); self.t2w_n = self._ispin(1, 100, 1)
        g.addWidget(self.t2w_lo, r, 1); g.addWidget(self.t2w_hi, r, 2); g.addWidget(self.t2w_n, r, 3); r += 1
        # CEST fs — entered as concentration in mM (converted to a proton
        # fraction when the dictionary is built).
        g.addWidget(self._lbl("CEST fₛ [mM]"), r, 0)
        self.fs_lo = self._spin(0.1, 100000, 50.0, 1, 10, 96); self.fs_hi = self._spin(0.1, 100000, 300.0, 1, 10, 96); self.fs_n = self._ispin(1, 100, 6)
        g.addWidget(self.fs_lo, r, 1); g.addWidget(self.fs_hi, r, 2); g.addWidget(self.fs_n, r, 3); r += 1
        # CEST ksw
        g.addWidget(self._lbl("CEST kₛw [s⁻¹]"), r, 0)
        self.ksw_lo = self._spin(1, 20000, 100.0, 0, 50, 80); self.ksw_hi = self._spin(1, 20000, 1400.0, 0, 50, 80); self.ksw_n = self._ispin(1, 100, 8)
        g.addWidget(self.ksw_lo, r, 1); g.addWidget(self.ksw_hi, r, 2); g.addWidget(self.ksw_n, r, 3); r += 1
        # fixed CEST params
        fixed = QHBoxLayout()
        fixed.addWidget(self._lbl("Δω [ppm]:")); self.sp_dw = self._spin(-100, 100, 3.0, 2, 0.1, 70); fixed.addWidget(self.sp_dw)
        fixed.addWidget(self._lbl("T₁s [s]:")); self.sp_t1s = self._spin(0.05, 6, 2.8, 2, 0.1, 60); fixed.addWidget(self.sp_t1s)
        fixed.addWidget(self._lbl("T₂s [s]:")); self.sp_t2s = self._spin(0.001, 2, 0.04, 3, 0.005, 70); fixed.addWidget(self.sp_t2s)
        fixed.addStretch()
        g.addLayout(fixed, r, 0, 1, 4)
        return grp

    def _wheel_forwarder(self, scroll):
        fwd = self._wheel_fwds.get(id(scroll))
        if fwd is None:
            fwd = _WheelForwarder(scroll)
            self._wheel_fwds[id(scroll)] = fwd
        return fwd

    def _make_scroll_tab(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea{{border:none; background:{self._fig_bg};}}")
        inner = QWidget(); inner.setStyleSheet(f"background:{self._fig_bg};")
        lay = QVBoxLayout(inner); lay.setContentsMargins(6, 6, 6, 6); lay.setSpacing(6)
        scroll.setWidget(inner)
        # Mouse-wheel over the inner widget scrolls the area (figures forward too).
        inner.installEventFilter(self._wheel_forwarder(scroll))
        return (scroll, inner, lay)

    def _build_kspace_tab(self) -> QWidget:
        """k-space tab: a fixed control strip (trajectory · 3D · matrix · FOV · BW)
        above a scrollable figure area."""
        w = QWidget(); w.setStyleSheet(f"background:{self._c_bg};")
        v = QVBoxLayout(w); v.setContentsMargins(6, 6, 6, 6); v.setSpacing(6)

        strip = QHBoxLayout(); strip.setSpacing(5)
        strip.addWidget(self._lbl("Trajectory:"))
        self.combo_traj = QComboBox()
        for _t in ("EPI", "Cartesian", "Spiral", "Radial"):
            self.combo_traj.addItem(_t, _t)
        self.combo_traj.setToolTip(
            "Illustrative readout trajectory (used when the loaded sequence is CEST-prep\n"
            "with no imaging readout). A sequence that DOES contain a readout is plotted\n"
            "from its own calculate_kspace().")
        self.combo_traj.currentIndexChanged.connect(lambda *_: self._plot_kspace())
        strip.addWidget(self.combo_traj)
        self.chk_3d = QCheckBox("3D")
        self.chk_3d.setToolTip("Show the trajectory as a 3D (kx·ky·kz) readout instead of a 2D plane.")
        self.chk_3d.toggled.connect(self._on_3d_toggled)
        strip.addWidget(self.chk_3d)
        strip.addSpacing(6)
        # Matrix = a COUNT (unitless), split into the two encode directions.
        strip.addWidget(self._lbl("Matrix:"))
        self.sp_nx = QSpinBox(); self.sp_nx.setRange(4, 512); self.sp_nx.setValue(16); self.sp_nx.setFixedWidth(62)
        self.sp_nx.setToolTip("Frequency-encode (readout) samples per line — a count, not a length.")
        self.sp_nx.valueChanged.connect(lambda *_: self._plot_kspace())
        strip.addWidget(self.sp_nx)
        strip.addWidget(self._lbl("× phase:"))
        self.sp_ny = QSpinBox(); self.sp_ny.setRange(4, 512); self.sp_ny.setValue(16); self.sp_ny.setFixedWidth(62)
        self.sp_ny.setToolTip("Phase-encode lines — a count, not a length.")
        self.sp_ny.valueChanged.connect(lambda *_: self._plot_kspace())
        strip.addWidget(self.sp_ny)
        strip.addWidget(self._lbl("Nz:"))
        self.sp_nz = QSpinBox(); self.sp_nz.setRange(2, 128); self.sp_nz.setValue(6); self.sp_nz.setFixedWidth(56)
        self.sp_nz.setToolTip("Nz — number of partition (kz / slice-encode) steps; only used for a 3D readout.")
        self.sp_nz.valueChanged.connect(lambda *_: self._plot_kspace())
        strip.addWidget(self.sp_nz)
        strip.addWidget(self._lbl("FOV [mm]:"))
        self.sp_fov = QDoubleSpinBox(); self.sp_fov.setRange(10, 600); self.sp_fov.setDecimals(0)
        self.sp_fov.setValue(220); self.sp_fov.setFixedWidth(66)
        self.sp_fov.setToolTip("Field of view (millimetres) — sets kmax = matrix / (2·FOV) and Δk = 1/FOV.")
        self.sp_fov.valueChanged.connect(lambda *_: self._plot_kspace())
        strip.addWidget(self.sp_fov)
        strip.addWidget(self._lbl("BW [Hz/px]:"))
        self.sp_bw = QDoubleSpinBox(); self.sp_bw.setRange(10, 5000); self.sp_bw.setDecimals(0)
        self.sp_bw.setValue(250); self.sp_bw.setFixedWidth(72)
        self.sp_bw.setToolTip("Readout pixel bandwidth — sets the readout duration T = 1/BW and dwell time.")
        self.sp_bw.valueChanged.connect(lambda *_: self._plot_kspace())
        strip.addWidget(self.sp_bw)
        # Undersampling (acceleration) — skip lines/spokes to show accelerated sampling.
        self.chk_undersample = QCheckBox("Accel")
        self.chk_undersample.setToolTip(
            "Acquire only every R-th phase-encode line / radial spoke to show an\n"
            "accelerated (undersampled) k-space — the skipped lines are shown dashed/grey.")
        self.chk_undersample.toggled.connect(self._on_undersample_toggled)
        strip.addWidget(self.chk_undersample)
        strip.addWidget(self._lbl("R:"))
        self.sp_accel = QSpinBox(); self.sp_accel.setRange(2, 8); self.sp_accel.setValue(2)
        self.sp_accel.setFixedWidth(48); self.sp_accel.setEnabled(False)
        self.sp_accel.setToolTip("Undersampling / acceleration factor R — keep every R-th line or spoke.")
        self.sp_accel.valueChanged.connect(lambda *_: self._plot_kspace())
        strip.addWidget(self.sp_accel)
        strip.addStretch()
        v.addLayout(strip)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea{{border:none; background:{self._fig_bg};}}")
        inner = QWidget(); inner.setStyleSheet(f"background:{self._fig_bg};")
        lay = QVBoxLayout(inner); lay.setContentsMargins(6, 6, 6, 6); lay.setSpacing(6)
        scroll.setWidget(inner)
        inner.installEventFilter(self._wheel_forwarder(scroll))
        v.addWidget(scroll, stretch=1)
        self._tab_kspace = (scroll, inner, lay)
        return w

    # ── population / state ────────────────────────────────────────────────────

    _CUSTOM = "__custom__"     # sentinel data for the "Custom .seq file…" entry

    def _populate_library(self):
        self.combo_seq.blockSignals(True)
        self.combo_seq.clear()
        seqs = discover_library_seqs(self._library)
        for label, path in seqs:
            self.combo_seq.addItem(label, path)
        # Final entry: let the user browse to their own .seq file.
        self.combo_seq.addItem("＋  Custom .seq file…  (browse)", self._CUSTOM)
        # Default to the first MRF-CEST schedule.  This is the *MRF fingerprints*
        # simulator, so a real MRF schedule (pseudo-random offsets/B1 → the jagged
        # fingerprint) is the sensible default — a monotonic Z-spectrum sequence
        # would instead show a smooth Z-spectrum-like dip.  Fall back to index 0.
        _default_idx = 0
        for _i in range(self.combo_seq.count()):
            _d = self.combo_seq.itemData(_i)
            if _d and _d != self._CUSTOM and "mrf" in os.path.basename(str(_d)).lower():
                _default_idx = _i
                break
        self.combo_seq.setCurrentIndex(_default_idx)
        self.combo_seq.blockSignals(False)
        if self.combo_seq.currentData() != self._CUSTOM:
            self._on_seq_changed()
        else:
            self._seq_path = None
            self.btn_view.setEnabled(False); self.btn_sim.setEnabled(False)

    def _on_seq_changed(self, *_):
        data = self.combo_seq.currentData()
        if data == self._CUSTOM:
            # Open a browse dialog; insert the chosen file before this entry.
            if not self._browse_seq():
                # cancelled → fall back to the first library entry (if any)
                if self.combo_seq.count() > 1:
                    self.combo_seq.setCurrentIndex(0)
                else:
                    self._seq_path = None
                    self.btn_view.setEnabled(False); self.btn_sim.setEnabled(False)
            return
        self._seq_path = data
        ok = bool(data and os.path.isfile(data))
        self.btn_view.setEnabled(ok)
        self.btn_sim.setEnabled(ok)

    def _browse_seq(self) -> bool:
        """Browse for a user .seq file (or a folder containing .seq files).  Returns
        True if a sequence was chosen and selected."""
        start = self._library if os.path.isdir(self._library) else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select your Pulseq .seq file", start, "Pulseq sequence (*.seq);;All files (*)")
        if not path:
            # Offer folder selection as a fallback (pick the first .seq inside).
            folder = QFileDialog.getExistingDirectory(self, "…or select a folder containing a .seq file", start)
            if folder:
                found = sorted(glob.glob(os.path.join(folder, "*.seq")))
                if found:
                    path = found[0]
                else:
                    QMessageBox.information(self, "No .seq found", "That folder contains no .seq file.")
        if not path:
            return False
        # insert the browsed file just before the "Custom…" entry and select it
        insert_at = max(0, self.combo_seq.count() - 1)
        self.combo_seq.blockSignals(True)
        self.combo_seq.insertItem(insert_at, f"(your file)  ·  {os.path.basename(path)}", path)
        self.combo_seq.setCurrentIndex(insert_at)
        self.combo_seq.blockSignals(False)
        self._seq_path = path
        ok = os.path.isfile(path)
        self.btn_view.setEnabled(ok); self.btn_sim.setEnabled(ok)
        return True

    def _n_entries(self) -> int:
        return (max(1, self.t1w_n.value()) * max(1, self.t2w_n.value()) *
                max(1, self.fs_n.value()) * max(1, self.ksw_n.value()))

    def _update_count(self, *_):
        n = self._n_entries()
        self.lbl_count.setText(f"Dictionary size:  {n:,} entries")

    # ── config ────────────────────────────────────────────────────────────────

    def _build_cfg(self) -> dict:
        return {
            "water_pool": {
                "t1": _linspace_list(self.t1w_lo.value(), self.t1w_hi.value(), self.t1w_n.value()),
                "t2": _linspace_list(self.t2w_lo.value(), self.t2w_hi.value(), self.t2w_n.value()),
                "f": 1,
            },
            "cest_pool": {
                "CEST": {
                    "t1": [self.sp_t1s.value()],
                    "t2": [self.sp_t2s.value()],
                    "k":  _linspace_list(self.ksw_lo.value(), self.ksw_hi.value(), self.ksw_n.value()),
                    "dw": self.sp_dw.value(),
                    # fₛ entered in mM → proton fraction (÷ water-proton conc ≈ 111 000 mM)
                    "f":  [c / 111000.0 for c in _linspace_list(
                        self.fs_lo.value(), self.fs_hi.value(), self.fs_n.value())],
                },
            },
            "b0": self.sp_b0.value(),
            "gamma": 267.5153 if abs(self.sp_gamma.value() - 42576400.0) < 1 else self.sp_gamma.value() * 2 * np.pi / 1e6,
            "scale": 1,
            "reset_init_mag": 0,
            "b0_inhom": 0,
            "rel_b1": self.sp_rel_b1.value(),
            "verbose": 0,
            "max_pulse_samples": 100,
            "num_workers": 1,
        }

    # ── View Sequence ─────────────────────────────────────────────────────────

    def _clear_tab(self, tab):
        _scroll, _inner, lay = tab
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

    def _add_fig(self, tab, fig, height=340, label=None):
        _scroll, _inner, lay = tab
        if label:
            q = QLabel(label); q.setWordWrap(True)
            q.setStyleSheet(f"font-size:11px; font-weight:bold; color:{self._fig_label}; background:#ffffff;")
            lay.addWidget(q)
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(height)
        # Forward wheel over the canvas to the scroll area (else matplotlib eats it).
        canvas.installEventFilter(self._wheel_forwarder(_scroll))
        if _NavToolbar is not None:
            tb = _NavToolbar(canvas, self)
            tb.setStyleSheet(
                f"QToolBar{{background:{self._fig_tb_bg};border:1px solid #d5d5d5;}}"
                f"QToolButton{{color:{self._fig_tb_fg};}} QLabel{{color:{self._fig_tb_mut};font-size:10px;}}")
            lay.addWidget(tb)
        lay.addWidget(canvas)
        canvas.draw()

    def _view_sequence(self):
        if not (self._seq_path and os.path.isfile(self._seq_path)):
            return
        # NOTE: pypulseq's seq.plot() must run on the main (GUI) thread — creating
        # its figures off-thread segfaults with the live Qt backend.  Large CEST
        # sequences (thousands of blocks) can take up to a minute; we show a busy
        # cursor + status so it's clearly working, not hung.
        from PyQt6.QtGui import QGuiApplication
        self.lbl_status.setText("Reading and plotting sequence…  "
                                "(large sequences can take up to a minute)")
        self.lbl_status.setStyleSheet("font-size:11px; color:#888; background:transparent;")
        self.btn_view.setEnabled(False)
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            import matplotlib.pyplot as plt
            self._last_seq_k = None
            # Close any lingering pyplot figures so this plot's figures are the
            # only "new" ones — pypulseq 1.3.1's seq.plot() reuses fixed figure
            # numbers, so on a repeat click no NEW figure would be detected (→ 0
            # figures shown).  seq.plot uses pyplot; the app's other figures are
            # created via Figure() directly, so this is safe.
            plt.close("all")
            with _pypulseq_ctx() as pp:
                seq = pp.Sequence()
                seq.read(self._seq_path)
                figs_before = set(plt.get_fignums())
                _orig = plt.show; plt.show = lambda *a, **k: None
                try:
                    seq.plot(time_disp="ms")
                finally:
                    plt.show = _orig
                new = sorted(set(plt.get_fignums()) - figs_before)
                # k-space from the SAME already-read seq (avoids a second slow read)
                try:
                    kadc = np.asarray(seq.calculate_kspace()[0], dtype=float)
                    span = np.nanmax(kadc, axis=1) - np.nanmin(kadc, axis=1)
                    if np.nanmax(span) > 1.0:
                        self._last_seq_k = kadc
                except Exception:
                    self._last_seq_k = None
            self._clear_tab(self._tab_seq)
            labels = ["RF magnitude · RF/ADC phase · ADC events",
                      "Gradient waveforms (Gx, Gy, Gz)"]
            for i, fn in enumerate(new):
                fig = plt.figure(fn)
                self._style_fig(fig)
                fig.tight_layout(pad=1.1)
                self._add_fig(self._tab_seq, fig, height=max(320, len(fig.get_axes()) * 130),
                              label=(labels[i] if i < len(labels) else f"Figure {i+1}"))
            self._plot_kspace()                    # uses self._last_seq_k (no re-read)
            self.tabs.setCurrentWidget(self._tab_seq[0])
            self.lbl_status.setText(f"✓ Sequence plotted ({len(new)} figure(s)).")
            self.lbl_status.setStyleSheet("font-size:11px; color:#3aa655; background:transparent;")
        except Exception as exc:  # noqa: BLE001
            self.lbl_status.setText(f"Sequence plot error: {exc}")
            self.lbl_status.setStyleSheet("font-size:11px; color:#e06666; background:transparent;")
        finally:
            QGuiApplication.restoreOverrideCursor()
            self.btn_view.setEnabled(True)

    # ── k-space ───────────────────────────────────────────────────────────────

    def _on_3d_toggled(self, *_):
        # re-plot k-space only if it has already been built (after View Sequence)
        if self._tab_kspace[2].count() > 0:
            self._plot_kspace()

    @staticmethod
    def _kspace_illustration(kind: str, is3d: bool, kx_ro, ky_pe, kz_par, accel: int = 1):
        """Illustrative readout k-space trajectory.  Returns ``(acquired, skipped)``
        lists of polyline segments ``(kx, ky, kz|None)``.  With ``accel`` > 1 only
        every R-th phase-encode line / radial spoke is acquired (the rest go to
        ``skipped``); for a spiral the acquired trajectory uses a coarser pitch and
        the fully-sampled spiral is returned as the skipped reference."""
        kind = (kind or "EPI").lower()
        kx_ro = np.asarray(kx_ro, float); ky_pe = np.asarray(ky_pe, float)
        kzs = (np.asarray(kz_par, float) if is3d else [None])
        kmax = float(np.max(np.abs(kx_ro))) if kx_ro.size else 1.0
        n_pe = len(ky_pe)
        R = max(1, int(accel))

        def _z(arr, kz):
            return None if kz is None else np.full(len(arr), kz)

        acq, skip = [], []
        for kz in kzs:
            if kind == "spiral":
                base = max(4, n_pe // 3)
                turns = max(2, base // R)
                t = np.linspace(0, turns * 2 * np.pi, max(len(kx_ro), 8) * turns)
                r = kmax * t / t[-1]
                acq.append((r * np.cos(t), r * np.sin(t), _z(r, kz)))
                if R > 1:                                              # full spiral as reference
                    tf = np.linspace(0, base * 2 * np.pi, max(len(kx_ro), 8) * base)
                    rf = kmax * tf / tf[-1]
                    skip.append((rf * np.cos(tf), rf * np.sin(tf), _z(rf, kz)))
            elif kind == "radial":                                     # n_pe spokes
                rr = np.linspace(-kmax, kmax, len(kx_ro))
                for j, a in enumerate(np.linspace(0, np.pi, n_pe, endpoint=False)):
                    x, y = rr * np.cos(a), rr * np.sin(a)
                    (acq if j % R == 0 else skip).append((x, y, _z(x, kz)))
            else:                                                      # cartesian / epi
                for i, ky in enumerate(ky_pe):
                    xs = kx_ro[::-1] if (kind == "epi" and i % 2 == 1) else kx_ro
                    (acq if i % R == 0 else skip).append((xs, np.full(len(xs), ky), _z(xs, kz)))
        return acq, skip

    def _on_undersample_toggled(self, on: bool):
        self.sp_accel.setEnabled(on)
        self._plot_kspace()

    def _plot_kspace(self):
        """Show the k-space trajectory — the sequence's own if it has an imaging
        readout, otherwise an illustrative 2D/3D Cartesian-EPI readout."""
        if not (self._seq_path and os.path.isfile(self._seq_path)):
            return
        self._clear_tab(self._tab_kspace)
        is3d = self.chk_3d.isChecked()
        # Use the k-space trajectory computed by the plot worker (off-thread) —
        # recomputing here would re-freeze the UI for ~20 s.
        seq_k = getattr(self, "_last_seq_k", None)

        # Acquisition parameters that shape k-space (matrix is a COUNT; FOV is mm).
        Nx = int(self.sp_nx.value())        # frequency-encode (readout) samples
        Ny = int(self.sp_ny.value())        # phase-encode lines
        Nz = int(self.sp_nz.value())        # partitions (kz) — 3D only
        fov_m = max(1e-3, self.sp_fov.value() / 1000.0)      # mm → m
        bw = max(1.0, self.sp_bw.value())                    # Hz/px
        kmax = Nx / (2.0 * fov_m)                            # readout kmax [1/m]
        dk = 1.0 / fov_m                                     # 1/m
        t_read_ms = 1000.0 / bw                              # readout duration [ms]
        kx_ro = np.linspace(-Nx / (2.0 * fov_m), Nx / (2.0 * fov_m), Nx)
        ky_pe = np.linspace(-Ny / (2.0 * fov_m), Ny / (2.0 * fov_m), Ny)
        kz_par = np.linspace(-Nz / (2.0 * fov_m), Nz / (2.0 * fov_m), min(Nz, 7))

        fig = Figure(figsize=(6.6, 5.4), facecolor=self._fig_bg)
        ax = fig.add_subplot(111, projection="3d") if is3d else fig.add_subplot(111)
        acc, dot = self._c_accent, "#e06666"
        if seq_k is not None:
            kx, ky, kz = seq_k[0], seq_k[1], seq_k[2]
            if is3d:
                ax.plot(kx, ky, kz, color=acc, lw=0.7, alpha=0.7); ax.scatter(kx, ky, kz, s=6, color=dot)
                ax.set_zlabel("kz [1/m]")
            else:
                ax.plot(kx, ky, color=acc, lw=0.7, alpha=0.7); ax.scatter(kx, ky, s=8, color=dot)
            label = "k-space trajectory of the loaded sequence (from calculate_kspace)"
            note = (f"Acquisition: matrix {Nx}×{Ny} (readout × phase; counts, not mm), "
                    f"FOV {self.sp_fov.value():.0f} mm, BW {bw:.0f} Hz/px → readout {t_read_ms:.2f} ms.")
        else:
            kind = self.combo_traj.currentData() or "EPI"
            R = self.sp_accel.value() if self.chk_undersample.isChecked() else 1
            acq_segs, skip_segs = self._kspace_illustration(kind, is3d, kx_ro, ky_pe, kz_par, accel=R)
            # skipped (not-acquired) lines first — dashed grey
            for (x, y, z) in skip_segs:
                if is3d:
                    ax.plot(x, y, z, color="#b7b7b7", lw=0.5, ls="--", alpha=0.55)
                else:
                    ax.plot(x, y, color="#b7b7b7", lw=0.5, ls="--", alpha=0.55)
            # acquired lines + sample points
            xs_all, ys_all, zs_all = [], [], []
            for (x, y, z) in acq_segs:
                if is3d:
                    ax.plot(x, y, z, color=acc, lw=0.7, alpha=0.75); zs_all.append(z)
                else:
                    ax.plot(x, y, color=acc, lw=0.7, alpha=0.75)
                xs_all.append(x); ys_all.append(y)
            X = np.concatenate(xs_all); Y = np.concatenate(ys_all)
            if is3d:
                Z = np.concatenate(zs_all); ax.scatter(X, Y, Z, s=3, color=dot); ax.set_zlabel("kz [1/m]")
            else:
                ax.scatter(X, Y, s=6, color=dot)
            _kname = self.combo_traj.currentText()
            _us = f"  ·  undersampled R = {R}" if R > 1 else ""
            label = f"Illustrative {'3D ' if is3d else '2D '}{_kname} readout k-space{_us}"
            _mtx = f"{Nx}×{Ny}" + (f"×{Nz}" if is3d else "")
            _acq_note = (f"<br><b>Undersampling R = {R}</b>: {len(acq_segs)} lines/spokes acquired "
                         f"(coloured), {len(skip_segs)} skipped (dashed grey) → ~R-fold faster, "
                         f"aliasing unless reconstructed (parallel imaging / compressed sensing)."
                         if R > 1 else "")
            note = ("The loaded sequence is CEST preparation (placeholder ADC — no imaging "
                    "readout), so its own k-space is a single point. Shown is an illustrative "
                    f"{'3D stacked ' if is3d else '2D '}{_kname} readout trajectory.<br>"
                    f"<b>Matrix {_mtx}</b> (readout × phase" + ("× partitions" if is3d else "") +
                    " — a count, not mm/cm), "
                    f"<b>FOV {self.sp_fov.value():.0f} mm</b>, <b>BW {bw:.0f} Hz/px</b>  →  "
                    f"k<sub>max</sub> = {kmax:.1f} m⁻¹, Δk = {dk:.2f} m⁻¹, readout ≈ {t_read_ms:.2f} ms."
                    + _acq_note +
                    "<br>Change Trajectory / Matrix / FOV / BW / 3D / Undersample above to see the effect.")
        ax.set_xlabel("kx [1/m]"); ax.set_ylabel("ky [1/m]"); ax.set_title("k-space trajectory")
        self._style_fig(fig, [ax])
        fig.tight_layout()
        self._add_fig(self._tab_kspace, fig, height=460, label=label)
        if note:
            q = QLabel(note); q.setWordWrap(True)
            q.setStyleSheet(f"font-size:10px; color:{self._fig_note}; background:#ffffff; padding:2px 4px;")
            self._tab_kspace[2].addWidget(q)

    # ── Simulate + Match ──────────────────────────────────────────────────────

    def _make_export_opts(self, pp):
        """Build a pypulseq ``Opts`` (system limits) from the scanner-limit
        spinboxes, passing only the kwargs the installed pypulseq version
        accepts (they differ between 1.3.1 and 1.5)."""
        import inspect
        try:
            allowed = set(inspect.signature(pp.Opts.__init__).parameters)
        except Exception:
            allowed = set()

        kw = {}
        def add(k, v):
            if not allowed or k in allowed:
                kw[k] = v
        add('max_grad', self.sp_grad.value());  add('grad_unit', 'mT/m')
        add('max_slew', self.sp_slew.value());  add('slew_unit', 'T/m/s')
        add('rf_ringdown_time',      self.sp_ring.value() * 1e-6)
        add('rf_dead_time',          self.sp_dead.value() * 1e-6)
        add('adc_dead_time',         self.sp_adc_dead.value() * 1e-6)
        add('rf_raster_time',        self.sp_rf_raster.value() * 1e-6)
        add('grad_raster_time',      self.sp_grad_raster.value() * 1e-6)
        add('adc_raster_time',       self.sp_adc_raster.value() * 1e-6)
        add('block_duration_raster', self.sp_block_raster.value() * 1e-6)
        add('gamma', self.sp_gamma.value())
        try:
            return pp.Opts(**kw)
        except Exception:
            try:
                return pp.Opts()
            except Exception:
                return None

    def _export_seq(self):
        """Export the currently selected .seq to a file the user picks, re-written
        with the current scanner-limit system settings (grad/slew/RF/ADC timing)."""
        if not (self._seq_path and os.path.isfile(self._seq_path)):
            QMessageBox.information(self, "No sequence",
                                    "Select or load a .seq sequence first.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export sequence to .seq",
            os.path.basename(self._seq_path), "Pulseq sequence (*.seq)")
        if not out_path:
            return
        if not out_path.lower().endswith(".seq"):
            out_path += ".seq"
        try:
            with _pypulseq_ctx() as pp:
                opts = self._make_export_opts(pp)
                seq = pp.Sequence(system=opts) if opts is not None else pp.Sequence()
                seq.read(self._seq_path)
                seq.write(out_path)
            QMessageBox.information(self, "Export complete", f"Exported to:\n{out_path}")
        except Exception:
            # fall back to a verbatim copy if pypulseq can't re-write it
            try:
                import shutil
                shutil.copyfile(self._seq_path, out_path)
                QMessageBox.information(self, "Export complete", f"Exported to:\n{out_path}")
            except Exception as e:                       # noqa: BLE001
                QMessageBox.critical(self, "Export failed", str(e))

    def _run_simulation(self):
        if not (self._seq_path and os.path.isfile(self._seq_path)):
            return
        self.btn_sim.setEnabled(False); self.btn_view.setEnabled(False)
        self.lbl_status.setStyleSheet("font-size:11px; color:#c8871f; background:transparent;")
        self.lbl_status.setText("Starting simulation…")
        self._worker = _SimWorker(self._seq_path, self._build_cfg(),
                                  axes=self.combo_axes.currentData(), parent=self)
        self._worker.progress.connect(lambda m: self.lbl_status.setText(m))
        self._worker.finished.connect(self._on_sim_done)
        self._worker.error.connect(self._on_sim_error)
        self._worker.start()

    def _on_sim_error(self, msg: str):
        self.btn_sim.setEnabled(True); self._on_seq_changed()
        self.lbl_status.setText(f"Simulation error: {msg.splitlines()[0]}")
        self.lbl_status.setStyleSheet("font-size:11px; color:#e06666; background:transparent;")

    def _on_sim_done(self, dict_fn: str):
        self.btn_sim.setEnabled(True); self._on_seq_changed()
        try:
            import scipy.io as sio
            d = sio.loadmat(dict_fn)
            sig = np.asarray(d["sig"], dtype=float)          # (N, n_meas)
            if sig.ndim != 2 or sig.shape[0] < 2:
                raise ValueError("dictionary has too few entries to plot")
            N, n_meas = sig.shape
            fs = np.asarray(d["fs_0"]).ravel()
            ksw = np.asarray(d["ksw_0"]).ravel()
            self._plot_fingerprints(sig, fs, ksw)
            self.lbl_status.setText(f"✓ Simulated {N:,} fingerprints ({n_meas} images each).")
            self.lbl_status.setStyleSheet("font-size:11px; color:#3aa655; background:transparent;")
            self.tabs.setCurrentWidget(self._tab_fp[0])
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc()
            self.lbl_status.setText(f"Post-processing error: {exc}")
            self.lbl_status.setStyleSheet("font-size:11px; color:#e06666; background:transparent;")

    def _plot_fingerprints(self, sig, fs, ksw):
        self._clear_tab(self._tab_fp)
        N, n_meas = sig.shape
        # show up to ~12 representative fingerprints spanning the ksw range
        order = np.argsort(ksw)
        pick = order[np.linspace(0, N - 1, min(12, N)).astype(int)]
        fig = Figure(figsize=(7.6, 4.2), facecolor=self._fig_bg)
        ax = fig.add_subplot(111)
        x = np.arange(1, n_meas + 1)
        try:
            cmap = matplotlib.colormaps["viridis"]
        except Exception:
            cmap = matplotlib.cm.get_cmap("viridis")
        for j, idx in enumerate(pick):
            ax.plot(x, sig[idx], color=cmap(j / max(1, len(pick) - 1)), lw=1.4,
                    label=f"kₛw={ksw[idx]:.0f} s⁻¹, fₛ={fs[idx]*111000:.0f} mM")
        ax.set_xlabel("Schedule iteration (image #)"); ax.set_ylabel("Signal (a.u.)")
        ax.set_title("Simulated MRF fingerprints")
        leg = ax.legend(fontsize=10, ncol=2, loc="best", framealpha=0.9)
        if leg is not None:
            for t in leg.get_texts():
                t.set_color(self._fig_fg)
        ax.grid(alpha=0.25, color=self._fig_grid)
        self._style_fig(fig, [ax])
        fig.tight_layout()
        self._add_fig(self._tab_fp, fig, height=380,
                      label="A sample of the simulated dictionary — each curve is one tissue's fingerprint")

    def _matching_demo(self, dict_fn, sig, d):
        """Self-consistency demo: take K dictionary entries as ground truth, add
        noise, and match them back with the real dot-product pipeline.  Leads with
        match *quality* + fingerprint overlay (which always works); parameter
        recovery is shown too but honestly flagged when the schedule is degenerate."""
        _ensure_paths()
        try:
            from cest_mrf.metrics.dot_product_mt import dot_prod_matching
        except Exception:
            from cest_mrf.metrics.dot_product import dot_prod_matching
        N, n_meas = sig.shape
        rng = np.random.RandomState(0)
        side = int(np.floor(np.sqrt(min(64, N)))); K = side * side
        idx = rng.choice(N, size=K, replace=False)
        fs_all = np.asarray(d["fs_0"]).ravel(); ksw_all = np.asarray(d["ksw_0"]).ravel()
        true_fs, true_ksw = fs_all[idx], ksw_all[idx]
        clean = sig[idx]                                     # (K, n_meas)
        noise_pct = 0.5
        noise = (noise_pct / 100.0) * clean.max() * rng.randn(K, n_meas)
        acq2d = clean + noise                                # (K, n_meas)
        acquired = acq2d.T.reshape(n_meas, side, side)       # (n_iter, r, c)

        # Real pipeline for the recovered maps + match score.
        qm = dot_prod_matching(dict_fn=dict_fn, acquired_data=acquired, batch_size=side * side)
        dp = np.asarray(qm["dp"]).reshape(-1, order="F")
        rec_fs = np.asarray(qm["fs"]).reshape(-1, order="F")
        rec_ksw = np.asarray(qm["ksw"]).reshape(-1, order="F")
        # Matched fingerprint per voxel (same cosine argmax the matcher uses) for the overlay.
        nd = sig / (np.linalg.norm(sig, axis=1, keepdims=True) + 1e-12)
        na = acq2d / (np.linalg.norm(acq2d, axis=1, keepdims=True) + 1e-12)
        match_idx = (na @ nd.T).argmax(axis=1)

        # how degenerate is this schedule?  max cosine between distinct fingerprints
        cos = nd @ nd.T; np.fill_diagonal(cos, 0.0)
        max_cos = float(cos.max())
        degenerate = max_cos > 0.9995

        self._clear_tab(self._tab_match)
        x = np.arange(1, n_meas + 1)

        # ── Fig A — matching quality (always convincing) ──────────────────────
        figA = Figure(figsize=(7.8, 3.3), facecolor=self._fig_bg)
        a1 = figA.add_subplot(121); a2 = figA.add_subplot(122)
        for ax in (a1, a2):
            ax.grid(alpha=0.25, color=self._fig_grid)
        for v, col in zip(range(min(3, K)), ["#1565c0", "#2e7d32", "#c62828"]):
            a1.plot(x, acq2d[v], "o", ms=3, color=col, alpha=0.55)
            a1.plot(x, sig[match_idx[v]], "-", color=col, lw=1.5)
        a1.set_xlabel("image #"); a1.set_ylabel("signal (a.u.)")
        a1.set_title("acquired (dots) vs matched fingerprint (line)")
        a2.hist(dp, bins=20, color="#1565c0", alpha=0.85)
        a2.set_xlabel("dot-product match score (dp)"); a2.set_ylabel("voxels")
        a2.set_title(f"match quality — mean dp = {dp.mean():.4f}")
        self._style_fig(figA, [a1, a2])
        figA.tight_layout()
        self._add_fig(self._tab_match, figA, height=320,
                      label=f"Matching demo — {K} noisy fingerprints ({noise_pct:g}% noise) matched to the {N:,}-entry dictionary")

        # ── Fig B — parameter recovery ────────────────────────────────────────
        figB = Figure(figsize=(7.8, 3.3), facecolor=self._fig_bg)
        b1 = figB.add_subplot(121); b2 = figB.add_subplot(122)
        for ax in (b1, b2):
            ax.grid(alpha=0.25, color=self._fig_grid)
        _tfs_mM, _rfs_mM = true_fs * 111000.0, rec_fs * 111000.0
        b1.scatter(_tfs_mM, _rfs_mM, s=14, c="#1565c0", alpha=0.7)
        lim = [min(_tfs_mM.min(), _rfs_mM.min()), max(_tfs_mM.max(), _rfs_mM.max())]
        b1.plot(lim, lim, "k--", lw=0.8); b1.set_xlabel("true fₛ [mM]"); b1.set_ylabel("recovered fₛ [mM]"); b1.set_title("fₛ recovery")
        b2.scatter(true_ksw, rec_ksw, s=14, c="#c62828", alpha=0.7)
        lim2 = [min(true_ksw.min(), rec_ksw.min()), max(true_ksw.max(), rec_ksw.max())]
        b2.plot(lim2, lim2, "k--", lw=0.8); b2.set_xlabel("true kₛw [s⁻¹]"); b2.set_ylabel("recovered kₛw [s⁻¹]"); b2.set_title("kₛw recovery")
        self._style_fig(figB, [b1, b2])
        figB.tight_layout()
        self._add_fig(self._tab_match, figB, height=320,
                      label="Parameter recovery (true vs matched)")

        # ── honest summary ────────────────────────────────────────────────────
        if degenerate:
            note = (f"The matcher finds near-perfect fingerprint matches (mean dp = {dp.mean():.4f}), "
                    f"but this example schedule is <b>parameter-degenerate</b>: distinct (fₛ, kₛw) pairs "
                    f"produce almost identical fingerprints (max cosine = {max_cos:.5f}), so fₛ and kₛw "
                    f"scatter off the diagonal. This is the CEST-MRF ambiguity that <b>CRLB-optimised "
                    f"schedules</b> (see the CEST-MRF equations box) are designed to reduce — try a "
                    f"different example sequence to compare.")
        else:
            note = (f"The matcher recovers fₛ and kₛw close to the diagonal "
                    f"(mean dp = {dp.mean():.4f}, max in-dictionary cosine = {max_cos:.5f}).")
        lab = QLabel(note); lab.setWordWrap(True)
        lab.setStyleSheet(f"font-size:10px; color:{self._fig_note}; background:#ffffff; padding:2px 4px;")
        self._tab_match[2].addWidget(lab)
