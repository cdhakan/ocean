"""
synth_cest_tab.py  —  "Synthetic CEST MRI" sub-tab
===================================================
Lets a user define a synthetic pool system (water + any number of CEST/NOE pools
+ an optional MT pool) and scanner/saturation settings, then

  • plot the resulting CEST Z-spectrum for the pool system, and
  • build a synthetic phantom (controllable tiles or random shapes) whose regions
    carry varying pool parameters, simulate a per-pixel Z-stack, view parametric
    maps (Z @ offset, MTR_asym @ offset, region labels), click any pixel to see
    its Z-spectrum, and push the whole Z-stack into the "Quantitative Z Analysis"
    tab for the normal CEST pipeline.

The Bloch-McConnell physics lives in :mod:`my_gui.synth_cest`.  The map view
reuses :class:`ROICanvas`, so it inherits the colormap picker, custom clim,
window/level, the "Bg" black-background toggle and the "Log map" perceptual
scaling automatically.
"""
from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QGroupBox, QCheckBox, QSplitter,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QProgressBar, QLineEdit, QFileDialog, QMessageBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from my_gui import synth_cest as sc
from my_gui.roi_tools import ROICanvas
from my_gui.plot_custom_bar import PlotCustomBar
from my_gui.fig_theme import apply_fig_dark_theme


# ─────────────────────────────────────────────────────────────────────────────
# Background worker for the (slower) phantom simulation
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: the Bloch-McConnell simulation (BMCTool → np.linalg.eig / LAPACK) is run
# on the MAIN thread, NOT in a QThread.  Running LAPACK off the main thread
# bus-errors (SIGBUS) on macOS, so we keep the GUI responsive with processEvents
# in the per-region progress callback instead of threading.
class SynthCestTab(QWidget):
    """Synthetic phantom → CEST Z-spectrum generator."""

    # The 'Vary' dropdown is rebuilt dynamically from the current pools by
    # _refresh_vary_targets(): one entry-set per CEST pool (pool 1..N), MT (if
    # enabled) and water. Targets are stored as each item's userData.

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label_img = None          # current phantom labels
        self._zstack = None             # current phantom Z-stack (H,W,n_off)
        self._offsets = None            # current offsets (ppm)
        self._systems = None            # per-region PoolSystems (for param maps)
        self._param_maps = None         # per-pool 2-D parameter maps
        self._push_cb = None            # set by app.py → push Z-stack to Z-analysis
        self._dc_annot = None

        self._build_ui()
        self._load_defaults(sc.default_pool_system())

    # ── layout ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ============ LEFT: parameter editor (scrollable) ====================
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(390)
        left_scroll.setMaximumWidth(470)
        left = QWidget()
        L = QVBoxLayout(left)
        L.setContentsMargins(8, 8, 8, 8)

        # ── Water pool ───────────────────────────────────────────────────────
        gw = QGroupBox("Water pool")
        gwl = QGridLayout(gw)
        self.sp_wt1 = self._dspin(0.05, 10.0, 1.3, 2, 0.1, " s")
        self.sp_wt2 = self._dspin(0.001, 5.0, 0.05, 3, 0.01, " s")
        gwl.addWidget(QLabel("T1:"), 0, 0); gwl.addWidget(self.sp_wt1, 0, 1)
        gwl.addWidget(QLabel("T2:"), 0, 2); gwl.addWidget(self.sp_wt2, 0, 3)
        L.addWidget(gw)

        # ── CEST pools table ─────────────────────────────────────────────────
        gc = QGroupBox("CEST / NOE pools")
        gcl = QVBoxLayout(gc)
        self.tbl = QTableWidget(0, 6)
        self.tbl.setHorizontalHeaderLabels(
            ["Name", "f (rel.)", "k (Hz)", "Δω (ppm)", "T1 (s)", "T2 (s)"])
        self.tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl.setMinimumHeight(140)
        gcl.addWidget(self.tbl)
        brow = QHBoxLayout()
        b_add = QPushButton("＋ Add pool"); b_add.clicked.connect(self._add_pool_row)
        b_del = QPushButton("－ Remove selected"); b_del.clicked.connect(self._del_pool_row)
        brow.addWidget(b_add); brow.addWidget(b_del); brow.addStretch()
        gcl.addLayout(brow)
        L.addWidget(gc)

        # ── MT pool ──────────────────────────────────────────────────────────
        self.grp_mt = QGroupBox("MT pool")
        self.grp_mt.setCheckable(True)
        self.grp_mt.setChecked(True)
        self.grp_mt.toggled.connect(self._refresh_vary_targets)   # MT on/off → update Vary list
        gml = QGridLayout(self.grp_mt)
        self.sp_mt_f = self._dspin(0.0, 1.0, 0.05, 4, 0.01)
        self.sp_mt_k = self._dspin(0.0, 5000.0, 40.0, 1, 5.0, " Hz")
        self.sp_mt_dw = self._dspin(-30.0, 30.0, -2.5, 2, 0.5, " ppm")
        self.sp_mt_t2 = self._dspin(0.1, 1000.0, 9.1, 2, 0.5, " µs")
        self.cmb_mt_ls = QComboBox(); self.cmb_mt_ls.addItems(["SuperLorentzian", "Lorentzian"])
        gml.addWidget(QLabel("f:"), 0, 0); gml.addWidget(self.sp_mt_f, 0, 1)
        gml.addWidget(QLabel("k:"), 0, 2); gml.addWidget(self.sp_mt_k, 0, 3)
        gml.addWidget(QLabel("Δω:"), 1, 0); gml.addWidget(self.sp_mt_dw, 1, 1)
        gml.addWidget(QLabel("T2:"), 1, 2); gml.addWidget(self.sp_mt_t2, 1, 3)
        gml.addWidget(QLabel("Lineshape:"), 2, 0)
        gml.addWidget(self.cmb_mt_ls, 2, 1, 1, 3)
        L.addWidget(self.grp_mt)

        # ── Scanner / saturation ─────────────────────────────────────────────
        gs = QGroupBox("Scanner / saturation")
        gsl = QGridLayout(gs)
        self.sp_b0 = self._dspin(0.5, 14.0, 3.0, 2, 0.1, " T")
        self.sp_b1 = self._dspin(0.05, 10.0, 1.0, 2, 0.1, " µT")
        self.cmb_mode = QComboBox(); self.cmb_mode.addItems(["CW", "Pulsed"])
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.sp_tsat = self._dspin(0.05, 30.0, 2.0, 2, 0.1, " s")
        self.sp_tp = self._dspin(1.0, 1000.0, 100.0, 0, 5.0, " ms")
        self.sp_dc = self._dspin(1.0, 100.0, 50.0, 0, 5.0, " %")
        self.sp_np = self._ispin(1, 200, 20)
        self.sp_trec = self._dspin(0.0, 20.0, 3.0, 2, 0.1, " s")
        self.sp_range = self._dspin(1.0, 20.0, 6.0, 1, 0.5, " ppm")
        self.sp_noff = self._ispin(5, 401, 61)
        r = 0
        gsl.addWidget(QLabel("B0:"), r, 0); gsl.addWidget(self.sp_b0, r, 1)
        gsl.addWidget(QLabel("B1:"), r, 2); gsl.addWidget(self.sp_b1, r, 3); r += 1
        gsl.addWidget(QLabel("Saturation:"), r, 0); gsl.addWidget(self.cmb_mode, r, 1)
        self.lbl_tsat = QLabel("t<sub>sat</sub>:"); gsl.addWidget(self.lbl_tsat, r, 2)
        gsl.addWidget(self.sp_tsat, r, 3); r += 1
        self.lbl_tp = QLabel("t_p:"); gsl.addWidget(self.lbl_tp, r, 0); gsl.addWidget(self.sp_tp, r, 1)
        self.lbl_dc = QLabel("DC:"); gsl.addWidget(self.lbl_dc, r, 2); gsl.addWidget(self.sp_dc, r, 3); r += 1
        self.lbl_np = QLabel("# pulses:"); gsl.addWidget(self.lbl_np, r, 0); gsl.addWidget(self.sp_np, r, 1)
        gsl.addWidget(QLabel("T<sub>rec</sub>:"), r, 2); gsl.addWidget(self.sp_trec, r, 3); r += 1
        gsl.addWidget(QLabel("± range:"), r, 0); gsl.addWidget(self.sp_range, r, 1)
        gsl.addWidget(QLabel("# offsets:"), r, 2); gsl.addWidget(self.sp_noff, r, 3)
        L.addWidget(gs)

        # ── Phantom ──────────────────────────────────────────────────────────
        gp = QGroupBox("Synthetic phantom")
        gpl = QGridLayout(gp)
        self.cmb_ph_mode = QComboBox(); self.cmb_ph_mode.addItems(
            ["Controllable tiles", "Random shapes"])
        self.sp_ph_size = self._ispin(16, 256, 64)
        self.sp_ph_nreg = self._ispin(1, 64, 9)
        self.cmb_vary = QComboBox()   # populated by _refresh_vary_targets()
        self.sp_vary_lo = self._dspin(-1e6, 1e6, 0.0002, 5, 0.0001)
        self.sp_vary_hi = self._dspin(-1e6, 1e6, 0.002, 5, 0.0001)
        self.sp_seed = self._ispin(0, 99999, 1)
        self.sp_seed.setToolTip(
            "Random-number-generator seed that makes the synthetic phantom reproducible.")
        r = 0
        gpl.addWidget(QLabel("Phantom:"), r, 0); gpl.addWidget(self.cmb_ph_mode, r, 1, 1, 3); r += 1
        gpl.addWidget(QLabel("Size:"), r, 0); gpl.addWidget(self.sp_ph_size, r, 1)
        gpl.addWidget(QLabel("# regions:"), r, 2); gpl.addWidget(self.sp_ph_nreg, r, 3); r += 1
        gpl.addWidget(QLabel("Vary:"), r, 0); gpl.addWidget(self.cmb_vary, r, 1, 1, 3); r += 1
        gpl.addWidget(QLabel("from:"), r, 0); gpl.addWidget(self.sp_vary_lo, r, 1)
        gpl.addWidget(QLabel("to:"), r, 2); gpl.addWidget(self.sp_vary_hi, r, 3); r += 1
        gpl.addWidget(QLabel("Seed:"), r, 0); gpl.addWidget(self.sp_seed, r, 1)
        L.addWidget(gp)

        # ── Actions ──────────────────────────────────────────────────────────
        self.btn_plot = QPushButton("Plot Z-spectrum")
        self.btn_plot.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn_plot.clicked.connect(self._plot_single)
        self.btn_gen = QPushButton("Generate phantom")
        self.btn_gen.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn_gen.clicked.connect(self._generate_phantom)
        L.addWidget(self.btn_plot)
        L.addWidget(self.btn_gen)
        self.prog = QProgressBar(); self.prog.setVisible(False)
        L.addWidget(self.prog)
        self.lbl_status = QLabel(""); self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color:#4caf50; font-size:11px;")
        L.addWidget(self.lbl_status)
        L.addStretch()
        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        # ============ RIGHT: map + spectrum ==================================
        right = QWidget()
        R = QVBoxLayout(right)
        R.setContentsMargins(4, 4, 4, 4)

        # Display controls row
        disp = QHBoxLayout()
        disp.addWidget(QLabel("Display:"))
        self.cmb_display = QComboBox()
        self.cmb_display.addItems(["Region labels", "Z @ offset", "MTR_asym @ offset"])
        self.cmb_display.currentIndexChanged.connect(self._refresh_display)
        disp.addWidget(self.cmb_display)
        disp.addWidget(QLabel("offset:"))
        self.sp_show_off = self._dspin(-20.0, 20.0, 3.5, 2, 0.5, " ppm")
        self.sp_show_off.valueChanged.connect(self._refresh_display)
        disp.addWidget(self.sp_show_off)
        disp.addStretch()
        R.addLayout(disp)

        # ── Figure Customization (collapsible — mirrors the MRF Viewer) ───────
        self.grp_fig_custom = QGroupBox("Figure Customization")
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
        R.addWidget(self.grp_fig_custom)

        # Title + colour-bar / fonts controls (reused → gives Bg + Log map free)
        self.edit_map_title = QLineEdit()
        self.edit_map_title.setPlaceholderText("Map title (blank = default)")
        self._fcp_lay.addWidget(self.edit_map_title)
        self.plot_bar = PlotCustomBar(default_cmap="viridis", fonts_first=True)
        self.plot_bar.applied.connect(self._refresh_display)
        from my_gui.format_bar import add_title_format_bar, connect_title_debounced
        connect_title_debounced(self.edit_map_title, self._refresh_display)
        add_title_format_bar(self.edit_map_title, None,
                             target_row=self.plot_bar.font_row(),
                             default_getter=lambda: getattr(self.canvas, "_last_title", ""))
        self._fcp_lay.addWidget(self.plot_bar)

        # "Bg" + "Log map" toggles (same idiom as the other map tabs)
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
        self.chk_logmap = QCheckBox("Log map")
        self.chk_logmap.setToolTip(
            "Log-color scaling, redistribute the colormaps so equal color "
            "steps = equal % change in value.")
        self.chk_logmap.toggled.connect(self._refresh_display)
        _frow.insertWidget(_b_idx + 1, self.chk_logmap)

        # Map canvas + spectrum canvas in a vertical splitter
        vsplit = QSplitter(Qt.Orientation.Vertical)
        self.canvas = ROICanvas()
        self.canvas.mpl_connect("button_press_event", self._on_map_click)
        vsplit.addWidget(self.canvas)

        spec_w = QWidget(); spec_l = QVBoxLayout(spec_w)
        spec_l.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(7, 4.2))   # 2100×1260 px at 300 dpi on export
        self.spec_canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._init_spectrum_axes()
        spec_l.addWidget(self.spec_canvas)
        srow = QHBoxLayout()
        self.chk_asym = QCheckBox("Show MTR_asym")
        self.chk_asym.toggled.connect(self._redraw_last_spectrum)
        srow.addWidget(self.chk_asym)
        self.btn_param_maps = QPushButton("Parameter maps…")
        self.btn_param_maps.setToolTip(
            "Show the per-pool ground-truth parameter maps (f, k, Δω, R1, R2) of "
            "the generated phantom — the CEST-Generator figure.")
        self.btn_param_maps.setEnabled(False)
        self.btn_param_maps.clicked.connect(self._show_param_maps)
        srow.addWidget(self.btn_param_maps)
        srow.addStretch()
        self.btn_send = QPushButton("Send to Quantitative Z Analysis")
        self.btn_send.clicked.connect(self._send_to_zanalysis)
        self.btn_send.setEnabled(False)
        srow.addWidget(self.btn_send)
        self.btn_export = QPushButton("Export figure…")
        from PyQt6.QtWidgets import QMenu as _QMenu
        _exp_menu = _QMenu(self.btn_export)
        _exp_menu.addAction("Z-spectrum…", self._export_spectrum)
        _exp_menu.addAction("Phantom map…", self._export_phantom_map)
        self.btn_export.setMenu(_exp_menu)
        self.btn_export.setToolTip(
            "Export the Z-spectrum plot or the phantom map (PNG/JPEG/TIFF/PDF/SVG "
            "at 300 dpi, or the underlying data as .mat/.npz).")
        srow.addWidget(self.btn_export)
        spec_l.addLayout(srow)
        vsplit.addWidget(spec_w)
        vsplit.setSizes([460, 300])
        R.addWidget(vsplit, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 900])

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

        self._last_spectrum = None      # (offsets, Z, title) for redraws
        self._on_mode_changed()

    # ── small spinbox helpers ────────────────────────────────────────────────
    def _dspin(self, lo, hi, val, dec, step, suffix=""):
        s = QDoubleSpinBox()
        s.setRange(lo, hi); s.setDecimals(dec); s.setSingleStep(step)
        s.setValue(val)
        if suffix:
            s.setSuffix(suffix)
        s.setFixedWidth(96)
        return s

    def _ispin(self, lo, hi, val):
        s = QSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setFixedWidth(96)
        return s

    # ── defaults / table ─────────────────────────────────────────────────────
    def _load_defaults(self, ps: sc.PoolSystem):
        self.sp_wt1.setValue(ps.water.t1); self.sp_wt2.setValue(ps.water.t2)
        self.tbl.setRowCount(0)
        for c in ps.cest:
            self._add_pool_row(c)
        if ps.mt is not None:
            self.grp_mt.setChecked(True)
            self.sp_mt_f.setValue(ps.mt.f); self.sp_mt_k.setValue(ps.mt.k)
            self.sp_mt_dw.setValue(ps.mt.dw); self.sp_mt_t2.setValue(ps.mt.t2_us)
            self.cmb_mt_ls.setCurrentText(ps.mt.lineshape)
        self._refresh_vary_targets()

    def _add_pool_row(self, pool: sc.CESTPool = None):
        if not isinstance(pool, sc.CESTPool):
            pool = sc.CESTPool(name=f"pool{self.tbl.rowCount() + 1}")
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        for c, v in enumerate([pool.name, pool.f, pool.k, pool.dw, pool.t1, pool.t2]):
            self.tbl.setItem(r, c, QTableWidgetItem(str(v)))
        self._refresh_vary_targets()

    def _del_pool_row(self):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()}, reverse=True)
        if not rows:
            rows = [self.tbl.rowCount() - 1] if self.tbl.rowCount() else []
        for r in rows:
            if r >= 0:
                self.tbl.removeRow(r)
        self._refresh_vary_targets()

    def _refresh_vary_targets(self, *_):
        """Rebuild the 'Vary' dropdown from the current pools: one entry-set per
        CEST pool in the table (pool 1..N), MT (if enabled) and water, so pools 2
        and 3 can be swept just like pool 1. The previous selection is preserved
        where possible; each item carries its target string as userData."""
        cmb = getattr(self, "cmb_vary", None)
        if cmb is None:
            return
        prev = cmb.currentData()
        items = [("Randomize all pools", "randomize"), ("(no variation)", "none")]
        for i in range(self.tbl.rowCount()):
            items.append((f"CEST pool {i + 1} — conc. f",  f"cest{i}_f"))
            items.append((f"CEST pool {i + 1} — rate k",   f"cest{i}_k"))
            items.append((f"CEST pool {i + 1} — shift Δω", f"cest{i}_dw"))
        if self.grp_mt.isChecked():
            items.append(("MT pool — size f", "mt_f"))
        items += [("Water — T1", "water_t1"), ("Water — T2", "water_t2")]
        cmb.blockSignals(True)
        cmb.clear()
        for label, target in items:
            cmb.addItem(label, target)
        j = cmb.findData(prev) if prev is not None else -1
        cmb.setCurrentIndex(j if j >= 0 else 0)
        cmb.blockSignals(False)

    def _on_mode_changed(self):
        pulsed = self.cmb_mode.currentText() == "Pulsed"
        for w in (self.lbl_tp, self.sp_tp, self.lbl_dc, self.sp_dc, self.lbl_np, self.sp_np):
            w.setVisible(pulsed)
        for w in (self.lbl_tsat, self.sp_tsat):
            w.setVisible(not pulsed)

    # ── gather params from UI ────────────────────────────────────────────────
    def _gather_pool_system(self) -> sc.PoolSystem:
        water = sc.WaterPool(t1=self.sp_wt1.value(), t2=self.sp_wt2.value())
        cest = []
        for r in range(self.tbl.rowCount()):
            def _g(c, default=0.0):
                it = self.tbl.item(r, c)
                try:
                    return float(it.text())
                except (ValueError, AttributeError):
                    return default
            name_it = self.tbl.item(r, 0)
            cest.append(sc.CESTPool(
                name=name_it.text() if name_it else f"pool{r+1}",
                f=_g(1, 0.0009), k=_g(2, 30.0), dw=_g(3, 3.5),
                t1=_g(4, 1.0), t2=_g(5, 0.04)))
        mt = None
        if self.grp_mt.isChecked():
            mt = sc.MTPool(f=self.sp_mt_f.value(), k=self.sp_mt_k.value(),
                           dw=self.sp_mt_dw.value(), t2_us=self.sp_mt_t2.value(),
                           lineshape=self.cmb_mt_ls.currentText())
        return sc.PoolSystem(water=water, cest=cest, mt=mt)

    def _gather_scanner(self) -> sc.Scanner:
        return sc.Scanner(
            b0=self.sp_b0.value(), b1=self.sp_b1.value(),
            mode="pulsed" if self.cmb_mode.currentText() == "Pulsed" else "cw",
            tsat=self.sp_tsat.value(), tp=self.sp_tp.value() / 1000.0,
            dc=self.sp_dc.value() / 100.0, n_pulses=self.sp_np.value(),
            trec=self.sp_trec.value(), ppm_range=self.sp_range.value(),
            n_offsets=self.sp_noff.value())

    # ── single Z-spectrum ────────────────────────────────────────────────────
    def _plot_single(self):
        try:
            ps = self._gather_pool_system()
            scn = self._gather_scanner()
            off = sc.offset_list(scn)
            self.btn_plot.setEnabled(False); self.lbl_status.setText("Simulating…")
            self.repaint()
            Z = sc.simulate_zspectrum(scn, ps.water, ps.cest, ps.mt, off)
        except Exception as e:                       # noqa: BLE001
            self.btn_plot.setEnabled(True)
            QMessageBox.critical(self, "Simulation error", str(e))
            self.lbl_status.setText("")
            return
        self.btn_plot.setEnabled(True)
        mtlbl = "MT on" if ps.mt else "MT off"
        self._last_spectrum = (off, Z, f"Synthetic Z-spectrum  ({len(ps.cest)} CEST pools, {mtlbl})")
        self._redraw_last_spectrum()
        self.lbl_status.setText("Z-spectrum simulated.")

    def _init_spectrum_axes(self):
        self.ax.clear()
        self.ax.set_xlabel("Δω (ppm)"); self.ax.set_ylabel(r"Z = $M_z$ / $M_0$")
        self.ax.invert_xaxis()
        self.ax.grid(alpha=0.3)

    def _redraw_last_spectrum(self, *_):
        if self._last_spectrum is None:
            return
        off, Z, title = self._last_spectrum
        self.ax.clear()
        self.ax.plot(off, Z, "-o", ms=3, color="#1565c0", label="Z-spectrum")
        if self.chk_asym.isChecked():
            pos, asym = sc.mtr_asym(off, Z)
            self.ax.plot(pos, asym, "-s", ms=3, color="#c62828", label="MTR$_{asym}$")
            self.ax.axhline(0, color="0.6", lw=0.8)
        self.ax.set_xlabel("Δω (ppm)"); self.ax.set_ylabel(r"Z = $M_z$ / $M_0$")
        self.ax.set_title(title, fontsize=10)
        self.ax.invert_xaxis(); self.ax.grid(alpha=0.3); self.ax.legend(fontsize=8)
        apply_fig_dark_theme(self.fig, self.chk_dark_bg.isChecked())
        self.fig.tight_layout()
        self.spec_canvas.draw()

    # ── phantom ──────────────────────────────────────────────────────────────
    def _generate_phantom(self):
        try:
            ps = self._gather_pool_system()
            scn = self._gather_scanner()
            off = sc.offset_list(scn)
            mode = "random" if self.cmb_ph_mode.currentText().startswith("Random") else "tiles"
            size = self.sp_ph_size.value()
            nreg = self.sp_ph_nreg.value()
            label = sc.generate_phantom(size=size, mode=mode, n_regions=nreg,
                                        seed=self.sp_seed.value())
            n_actual = int(np.unique(label[label > 0]).size)
            target = self.cmb_vary.currentData() or "none"
            if target == "randomize":
                systems = sc.randomize_pool_systems(
                    ps, max(n_actual, 1), ppm_range=scn.ppm_range,
                    seed=self.sp_seed.value())
            else:
                systems = sc.vary_pool_systems(
                    ps, max(n_actual, 1), target,
                    self.sp_vary_lo.value(), self.sp_vary_hi.value())
        except Exception as e:                       # noqa: BLE001
            QMessageBox.critical(self, "Phantom error", str(e))
            return

        # Run on the MAIN thread — LAPACK (np.linalg.eig in BMCTool) bus-errors
        # off-thread on macOS.  processEvents keeps the UI alive between regions.
        from PyQt6.QtWidgets import QApplication
        self.btn_gen.setEnabled(False); self.btn_plot.setEnabled(False)
        self.prog.setVisible(True); self.prog.setRange(0, n_actual); self.prog.setValue(0)
        self.lbl_status.setText(f"Simulating {n_actual} regions × {len(off)} offsets…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        def _progress(done, total):
            self.prog.setValue(done)
            QApplication.processEvents()

        try:
            zstack = sc.simulate_phantom(label, systems, scn, off, progress=_progress)
        except Exception as e:                       # noqa: BLE001
            QApplication.restoreOverrideCursor()
            self.btn_gen.setEnabled(True); self.btn_plot.setEnabled(True)
            self.prog.setVisible(False)
            import traceback
            QMessageBox.critical(self, "Phantom simulation error",
                                 f"{e}\n\n{traceback.format_exc()}")
            self.lbl_status.setText("")
            return
        QApplication.restoreOverrideCursor()
        self._systems = systems
        self._on_phantom_done(label, zstack, off)

    def _on_phantom_done(self, label_img, zstack, offsets):
        self.btn_gen.setEnabled(True); self.btn_plot.setEnabled(True)
        self.prog.setVisible(False)
        self._label_img = label_img
        self._zstack = zstack
        self._offsets = offsets
        try:
            self._param_maps = sc.pool_param_maps(label_img, self._systems)
        except Exception:
            self._param_maps = None
        if hasattr(self, "btn_param_maps"):
            self.btn_param_maps.setEnabled(self._param_maps is not None)
        self.btn_send.setEnabled(True)
        self.lbl_status.setText(
            f"Phantom ready: {label_img.shape[0]}×{label_img.shape[1]} px, "
            f"{int(np.unique(label_img[label_img>0]).size)} regions, {len(offsets)} offsets. "
            "Click a pixel to see its Z-spectrum.")
        if self.cmb_display.currentText() == "Region labels":
            pass
        else:
            self.cmb_display.setCurrentText("MTR_asym @ offset")
        self._refresh_display()

    # ── map display ──────────────────────────────────────────────────────────
    def _refresh_display(self, *_):
        # propagate Bg + Log map to the ROICanvas (same idiom as other tabs)
        if hasattr(self, "chk_dark_bg"):
            self.canvas._dark_bg = self.chk_dark_bg.isChecked()
        if hasattr(self, "chk_logmap"):
            self.canvas._log_map = self.chk_logmap.isChecked()

        choice = self.cmb_display.currentText()
        cmap = self.plot_bar.get_cmap()
        vmin, vmax = self.plot_bar.get_clim()
        fs = self.plot_bar.get_font_sizes()
        ctitle = self.edit_map_title.text().strip()

        if choice == "Region labels":
            if self._label_img is None:
                return
            data = self._label_img.astype(float)
            data[data == 0] = np.nan
            self.canvas.show_map(data, ctitle or "Region labels",
                                 cmap=cmap if cmap != "viridis" else "tab20",
                                 vmin=vmin, vmax=vmax, **fs)
            return

        if self._zstack is None or self._offsets is None:
            return
        off = np.asarray(self._offsets)
        want = self.sp_show_off.value()
        if choice == "Z @ offset":
            idx = int(np.argmin(np.abs(off - want)))
            data = self._zstack[:, :, idx].astype(float)
            title = ctitle or f"Z @ {off[idx]:.2f} ppm"
        else:  # MTR_asym @ offset
            ip = int(np.argmin(np.abs(off - abs(want))))
            iN = int(np.argmin(np.abs(off + abs(want))))
            data = (self._zstack[:, :, iN] - self._zstack[:, :, ip]).astype(float)
            title = ctitle or f"MTR$_{{asym}}$ @ {abs(want):.2f} ppm"
        bg = (self._label_img == 0) if self._label_img is not None else (self._zstack.sum(2) == 0)
        data = data.copy(); data[bg] = np.nan
        self.canvas.show_map(data, title, cmap=cmap, vmin=vmin, vmax=vmax, **fs)

    def _on_map_click(self, event):
        if self._zstack is None or event.inaxes is None or event.xdata is None:
            return
        x = int(round(event.xdata)); y = int(round(event.ydata))
        H, W = self._zstack.shape[:2]
        if not (0 <= x < W and 0 <= y < H):
            return
        if self._label_img is not None and self._label_img[y, x] == 0:
            return
        Z = self._zstack[y, x, :]
        self._last_spectrum = (np.asarray(self._offsets), np.asarray(Z),
                               f"Pixel ({x}, {y})  Z-spectrum")
        self._redraw_last_spectrum()

    # ── send to Quantitative Z Analysis ──────────────────────────────────────
    def set_push_callback(self, cb):
        """app.py wires this so the phantom Z-stack can be sent to zspec_tab."""
        self._push_cb = cb

    def _send_to_zanalysis(self):
        if self._zstack is None:
            return
        if self._push_cb is None:
            QMessageBox.information(self, "Not connected",
                                    "Quantitative Z Analysis tab is not connected.")
            return
        # (Y, X, slices, n_off) + offsets (ppm)
        img_all = self._zstack[:, :, np.newaxis, :].astype(np.float32)
        try:
            self._push_cb(img_all, np.asarray(self._offsets, dtype=float))
            self.lbl_status.setText("Sent synthetic Z-stack → Quantitative Z Analysis.")
        except Exception as e:                        # noqa: BLE001
            QMessageBox.critical(self, "Send failed", str(e))

    # ── per-pool parameter maps (CEST-Generator figure) ───────────────────────
    def _show_param_maps(self):
        if not self._param_maps:
            return
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                     QCheckBox, QPushButton)
        from matplotlib.backends.backend_qt import NavigationToolbar2QT as _NTB

        pools = list(self._param_maps.keys())
        cols = ["f", "k", "dw", "R1", "R2"]
        col_titles = {"f": "fractional\nconcentration", "k": "exchange rate",
                      "dw": "chemical shift", "R1": "longitudinal\nrelaxation rate",
                      "R2": "transversal\nrelaxation rate"}
        units = {"f": "f", "k": "k [Hz]", "dw": "Δ [ppm]", "R1": "R1 [Hz]", "R2": "R2 [Hz]"}
        nrows, ncols = len(pools), len(cols)
        # Build with neutral colours (black text on a white ground); the "Bg"
        # checkbox flips the whole figure to a black background on demand via
        # apply_fig_dark_theme — so the maps render with and without Bg.
        fig = Figure(figsize=(2.5 * ncols, 2.3 * nrows))
        for ri, pool in enumerate(pools):
            pm = self._param_maps[pool]
            for ci, col in enumerate(cols):
                ax = fig.add_subplot(nrows, ncols, ri * ncols + ci + 1)
                ax.axis("off")
                if ri == 0:
                    ax.set_title(col_titles[col], fontsize=9)
                if ci == 0:
                    ax.text(-0.22, 0.5, pool, transform=ax.transAxes,
                            fontsize=11, ha="right", va="center")
                if col not in pm:
                    continue
                m = np.asarray(pm[col], float)
                if not np.any(np.isfinite(m)):
                    continue
                if col == "dw":
                    lim = float(np.nanmax(np.abs(m))) or 1.0
                    vmin, vmax = -lim, lim
                else:
                    vmin, vmax = 0.0, float(np.nanmax(m)) or 1.0
                im = ax.imshow(m, cmap="jet", vmin=vmin, vmax=vmax)
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=6)
                cb.set_label(units[col], fontsize=7)
        fig.tight_layout()

        dlg = QDialog(self)
        dlg.setWindowTitle("Phantom parameter maps  (ground truth)")
        dlg.resize(1180, min(240 * nrows + 110, 940))
        lay = QVBoxLayout(dlg)
        cv = FigureCanvas(fig)

        top = QHBoxLayout()
        top.addWidget(_NTB(cv, dlg))
        top.addStretch()
        chk_bg = QCheckBox("Bg")
        chk_bg.setToolTip("Black background for the parameter maps (for slides).")
        chk_bg.setChecked(True)
        def _retheme(on):
            apply_fig_dark_theme(fig, on)
            cv.draw_idle()
        chk_bg.toggled.connect(_retheme)
        top.addWidget(chk_bg)
        btn_exp = QPushButton("Export…")
        btn_exp.setToolTip("Export the parameter maps (PNG/JPEG/TIFF/PDF/SVG at "
                           "300 dpi, or data as .mat/.npz).")
        def _export_pm():
            from my_gui.fig_export import save_figure, FIG_EXPORT_FILTER
            p, _ = QFileDialog.getSaveFileName(
                dlg, "Export parameter maps", "phantom_param_maps", FIG_EXPORT_FILTER)
            if p:
                try:
                    save_figure(fig, p, dpi=300, facecolor=fig.get_facecolor())
                except Exception as e:            # noqa: BLE001
                    QMessageBox.critical(dlg, "Export failed", str(e))
        btn_exp.clicked.connect(_export_pm)
        top.addWidget(btn_exp)
        lay.addLayout(top)
        lay.addWidget(cv)

        apply_fig_dark_theme(fig, chk_bg.isChecked())   # initial: Bg on (as before)
        self._param_dlg = dlg          # keep a reference so it isn't GC'd
        dlg.show()

    # ── export ───────────────────────────────────────────────────────────────
    def _export_spectrum(self):
        from my_gui.fig_export import save_figure, FIG_EXPORT_FILTER
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Z-spectrum", "synthetic_zspectrum", FIG_EXPORT_FILTER)
        if path:
            try:
                save_figure(self.fig, path, dpi=300)
                self.lbl_status.setText(f"Saved {path}")
            except Exception as e:                    # noqa: BLE001
                QMessageBox.critical(self, "Export failed", str(e))

    def _export_phantom_map(self):
        cfig = getattr(self.canvas, "_fig", None)
        if cfig is None or self._label_img is None:
            QMessageBox.information(self, "No phantom",
                                    "Generate a phantom first, then export its map.")
            return
        from my_gui.fig_export import save_figure, FIG_EXPORT_FILTER
        path, _ = QFileDialog.getSaveFileName(
            self, "Export phantom map", "synthetic_phantom_map", FIG_EXPORT_FILTER)
        if path:
            try:
                save_figure(cfig, path, dpi=300, facecolor=cfig.get_facecolor())
                self.lbl_status.setText(f"Saved {path}")
            except Exception as e:                    # noqa: BLE001
                QMessageBox.critical(self, "Export failed", str(e))
