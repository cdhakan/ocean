from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QGroupBox, QLineEdit,
    QDoubleSpinBox, QSpinBox, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QComboBox, QScrollArea, QFrame,
    QPushButton, QFileDialog,
)
from PyQt6.QtCore import Qt


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_sweep_row(*pairs) -> QHBoxLayout:
    """
    Build a  [spinbox] label  [spinbox] label  [spinbox] label  row.
    Label is rendered to the RIGHT of each spinbox.
    """
    row = QHBoxLayout()
    row.setSpacing(4)
    for widget, label in pairs:
        row.addWidget(widget)
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        row.addWidget(lbl)
        row.addSpacing(10)
    row.addStretch()
    return row


class FixedOrSweepWidget(QWidget):
    """
    Reusable widget: a QCheckBox labelled 'Fixed' that toggles between
      • Fixed mode  →  single spinbox
      • Sweep mode  →  [spinbox] min  [spinbox] max  [spinbox] step

    Works with both QSpinBox (int) and QDoubleSpinBox (float).
    """

    def __init__(
        self,
        *,
        use_double: bool = False,
        fixed_value,
        fixed_range: tuple,
        fixed_suffix: str = "",
        fixed_decimals: int = 0,
        fixed_step=1,
        sweep_min_value,
        sweep_min_range: tuple,
        sweep_max_value,
        sweep_max_range: tuple,
        sweep_step_value,
        sweep_step_range: tuple,
        sweep_suffix: str = "",
        sweep_decimals: int = 0,
        sweep_step=1,
        starts_fixed: bool = True,
    ):
        super().__init__()

        SpinCls = QDoubleSpinBox if use_double else QSpinBox

        # ── Fixed spinbox ──────────────────────────────────────────────
        self.fixed_spin = SpinCls()
        self.fixed_spin.setRange(*fixed_range)
        self.fixed_spin.setValue(fixed_value)
        if fixed_suffix:
            self.fixed_spin.setSuffix(fixed_suffix)
        if use_double:
            self.fixed_spin.setDecimals(fixed_decimals)
            self.fixed_spin.setSingleStep(fixed_step)

        # ── Sweep spinboxes ────────────────────────────────────────────
        def _sw(lo, hi, val, step):
            sp = SpinCls()
            sp.setRange(lo, hi)
            sp.setValue(val)
            if sweep_suffix:
                sp.setSuffix(sweep_suffix)
            if use_double:
                sp.setDecimals(sweep_decimals)
                sp.setSingleStep(step)
            return sp

        self.sweep_min  = _sw(*sweep_min_range,  sweep_min_value,  sweep_step)
        self.sweep_max  = _sw(*sweep_max_range,  sweep_max_value,  sweep_step)
        self.sweep_step = _sw(*sweep_step_range, sweep_step_value, sweep_step)

        self._sweep_widget = QWidget()
        self._sweep_widget.setLayout(
            _make_sweep_row(
                (self.sweep_min,  "min"),
                (self.sweep_max,  "max"),
                (self.sweep_step, "step"),
            )
        )

        # ── Fixed checkbox ─────────────────────────────────────────────
        self.chk_fixed = QCheckBox("Fixed")
        self.chk_fixed.setChecked(starts_fixed)
        self.chk_fixed.setStyleSheet("font-size: 11px; color: #aaa;")

        # ── Layout ─────────────────────────────────────────────────────
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.fixed_spin)
        layout.addWidget(self._sweep_widget)
        layout.addWidget(self.chk_fixed)
        layout.addStretch()

        # Wire toggle
        self.chk_fixed.toggled.connect(self._on_toggle)
        self._on_toggle(starts_fixed)

    def _on_toggle(self, fixed: bool):
        self.fixed_spin.setVisible(fixed)
        self._sweep_widget.setVisible(not fixed)

    def get_value(self):
        """Return fixed float, or numpy array if sweep mode."""
        if self.chk_fixed.isChecked():
            return float(self.fixed_spin.value())
        return np.arange(
            self.sweep_min.value(),
            self.sweep_max.value() + self.sweep_step.value(),
            self.sweep_step.value(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main ConfigTab
# ─────────────────────────────────────────────────────────────────────────────

_SPIN_W = 82   # uniform width for all standalone spinboxes in ConfigTab


class ConfigTab(QWidget):
    def __init__(self):
        super().__init__()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(8)
        layout.addWidget(self._build_scanner_group())
        layout.addWidget(self._build_scanner_limits_group())
        layout.addWidget(self._build_water_group())
        layout.addWidget(self._build_cest_group())
        layout.addWidget(self._build_mt_group())
        layout.addWidget(self._build_sim_group())
        layout.addWidget(self._build_output_group())
        layout.addWidget(self._build_dict_size_group())
        layout.addStretch()

        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # ── Apply uniform spinbox width to all standalone spinboxes ──────
        self._apply_uniform_spinbox_width(inner)

    # ─────────────────────────────────────────────────────────────────────

    def _apply_uniform_spinbox_width(self, root: QWidget):
        """Set a fixed width on all QSpinBox / QDoubleSpinBox / QComboBox
        widgets that are *direct* form fields (not inside FixedOrSweepWidget)."""
        from PyQt6.QtWidgets import QAbstractSpinBox
        for w in root.findChildren(QAbstractSpinBox):
            # Skip spinboxes that live inside a FixedOrSweepWidget —
            # those already have their own compact layout.
            parent = w.parent()
            in_fosw = False
            while parent is not None:
                if isinstance(parent, FixedOrSweepWidget):
                    in_fosw = True
                    break
                parent = parent.parent()
            if not in_fosw:
                w.setFixedWidth(_SPIN_W)
        # Also size the lineshape combo uniformly
        for w in root.findChildren(QComboBox):
            parent = w.parent()
            in_fosw = False
            while parent is not None:
                if isinstance(parent, FixedOrSweepWidget):
                    in_fosw = True
                    break
                parent = parent.parent()
            if not in_fosw:
                w.setFixedWidth(_SPIN_W * 2)   # combos get double width

    # ─── Scanner ──────────────────────────────────────────────────────────

    def _build_scanner_group(self):
        grp = QGroupBox("Scanner")
        form = QFormLayout(grp)

        self.b0 = QDoubleSpinBox()
        self.b0.setRange(0.1, 20); self.b0.setValue(9.4); self.b0.setSuffix(" T")

        self.rel_b1 = QDoubleSpinBox()
        self.rel_b1.setRange(0.1, 2.0); self.rel_b1.setValue(1.0); self.rel_b1.setSingleStep(0.05)
        self.rel_b1.setEnabled(False)

        self.b0_inhom = QDoubleSpinBox()
        self.b0_inhom.setRange(-1, 1); self.b0_inhom.setValue(0)
        self.b0_inhom.setEnabled(False)

        # magnetization_scale → YAML key 'scale'
        self.mag_scale = QDoubleSpinBox()
        self.mag_scale.setRange(0.01, 10.0); self.mag_scale.setValue(1.0)
        self.mag_scale.setSingleStep(0.1); self.mag_scale.setDecimals(2)

        # magnetization_reset → YAML key 'reset_init_mag'
        self.reset_init_mag = QSpinBox()
        self.reset_init_mag.setRange(0, 1); self.reset_init_mag.setValue(0)
        self.reset_init_mag.setToolTip(
            "0 = keep magnetisation between pulses\n"
            "1 = reset to thermal equilibrium before each time point"
        )

        form.addRow("B0:", self.b0)
        form.addRow("Rel B1 (fixed):", self.rel_b1)
        form.addRow("B0 inhomogeneity (fixed):", self.b0_inhom)
        form.addRow("Magnetization scale:", self.mag_scale)
        form.addRow("Reset init mag:", self.reset_init_mag)
        return grp
    
    # ─── Water Pool ───────────────────────────────────────────────────────

    def _build_water_group(self):
        grp = QGroupBox("Water Pool")
        form = QFormLayout(grp)

        # T1 — fixed-or-sweep (ms)
        self.water_t1 = FixedOrSweepWidget(
            fixed_value=2500, fixed_range=(100, 15000), fixed_suffix=" ms",
            sweep_min_value=2000, sweep_min_range=(100, 15000),
            sweep_max_value=3300, sweep_max_range=(100, 15000),
            sweep_step_value=100, sweep_step_range=(1, 5000),
            sweep_suffix=" ms", starts_fixed=False,
        )
        self.water_t1.setToolTip("Water T1. Uncheck 'Fixed' to sweep min→max with step.")

        # T2 — fixed-or-sweep (ms)
        self.water_t2 = FixedOrSweepWidget(
            fixed_value=100, fixed_range=(1, 10000), fixed_suffix=" ms",
            sweep_min_value=50, sweep_min_range=(1, 10000),
            sweep_max_value=200, sweep_max_range=(1, 10000),
            sweep_step_value=10, sweep_step_range=(1, 5000),
            sweep_suffix=" ms", starts_fixed=False,
        )
        self.water_t2.setToolTip("Water T2. Uncheck 'Fixed' to sweep min→max with step.")

        # Proton volume fraction — always 1, not shown (written to YAML automatically)
        form.addRow("T1:", self.water_t1)
        form.addRow("T2:", self.water_t2)
        # f=1 is fixed by convention — no GUI row needed
        return grp

    # ─── CEST Pool ────────────────────────────────────────────────────────

    def _build_cest_group(self):
        grp = QGroupBox("CEST Pool")
        outer = QVBoxLayout(grp)

        # ── Optional toggle ────────────────────────────────────────────
        self.chk_cest = QCheckBox("Add CEST pool?")
        self.chk_cest.setChecked(True)
        self.chk_cest.setStyleSheet("font-weight: bold;")
        self.chk_cest.setToolTip(
            "Uncheck for MT-only or water-only dictionaries\n"
            "(matches the YAML example with no cest_pool key)"
        )
        outer.addWidget(self.chk_cest)

        # Container widget that shows/hides with checkbox
        self._cest_params_widget = QWidget()
        cest_outer = QVBoxLayout(self._cest_params_widget)
        cest_outer.setContentsMargins(0, 0, 0, 0)

        # ── Pool 1 ────────────────────────────────────────────────────
        form1 = QFormLayout()
        form1.setContentsMargins(0, 0, 0, 4)

        self.pool_name = QLineEdit("MyMolecule")
        self.dw        = QDoubleSpinBox()
        self.dw.setRange(-10, 10); self.dw.setValue(3.0); self.dw.setSuffix(" ppm")

        # T1 — fixed (2800 ms) or sweep
        self.cest_t1 = FixedOrSweepWidget(
            fixed_value=2800, fixed_range=(100, 5000), fixed_suffix=" ms",
            sweep_min_value=1000, sweep_min_range=(100, 5000),
            sweep_max_value=3000, sweep_max_range=(100, 5000),
            sweep_step_value=100, sweep_step_range=(10, 500),
            sweep_suffix=" ms", starts_fixed=True,
        )
        self.cest_t1.setToolTip("T1 of CEST pool. Default 2800 ms (fixed). Uncheck 'Fixed' for sweep.")

        # T2 — fixed (40 ms) or sweep
        self.cest_t2 = FixedOrSweepWidget(
            fixed_value=40, fixed_range=(1, 500), fixed_suffix=" ms",
            sweep_min_value=20, sweep_min_range=(1, 500),
            sweep_max_value=100, sweep_max_range=(1, 500),
            sweep_step_value=10, sweep_step_range=(1, 200),
            sweep_suffix=" ms", starts_fixed=True,
        )
        self.cest_t2.setToolTip("T2 of CEST pool. Default 40 ms (fixed). Uncheck 'Fixed' for sweep.")

        # Exchange rate k and concentration f (always swept)
        self.k_min  = QSpinBox(); self.k_min.setRange(10, 5000);  self.k_min.setValue(100);  self.k_min.setSuffix(" s⁻¹")
        self.k_max  = QSpinBox(); self.k_max.setRange(10, 20000);  self.k_max.setValue(3000); self.k_max.setSuffix(" s⁻¹")
        self.k_step = QSpinBox(); self.k_step.setRange(1, 5000);   self.k_step.setValue(100);  self.k_step.setSuffix(" s⁻¹")
        self.f_min  = QSpinBox(); self.f_min.setRange(1, 1000);   self.f_min.setValue(10)
        self.f_max  = QSpinBox(); self.f_max.setRange(1, 1000);   self.f_max.setValue(120)
        self.f_step = QSpinBox(); self.f_step.setRange(1, 100);   self.f_step.setValue(5)

        # CEST protons — integer 1-10, drives f = conc × n_protons / 110000
        self.cest_protons = QSpinBox()
        self.cest_protons.setRange(1, 10)
        self.cest_protons.setValue(3)
        self.cest_protons.setToolTip(
            "Number of exchangeable protons per molecule.\n"
            "Used in: f = Concentration (mM) × CEST_Protons / 110000\n"
            "e.g. Glutamate = 3 (NH₃ group), Creatine ≈ 3, Amine = 1"
        )

        form1.addRow("Pool name:", self.pool_name)
        form1.addRow("Chemical shift (dw):", self.dw)
        form1.addRow("T1:", self.cest_t1)
        form1.addRow("T2:", self.cest_t2)
        form1.addRow("CEST Protons (n):", self.cest_protons)
        form1.addRow("Exchange rate k (s⁻¹):", _make_sweep_row(
            (self.k_min, "min"), (self.k_max, "max"), (self.k_step, "step")))
        form1.addRow("Concentration f (mM × n/110000):", _make_sweep_row(
            (self.f_min, "min"), (self.f_max, "max"), (self.f_step, "step")))
        cest_outer.addLayout(form1)          # ← into container, not outer

        # ── Second CEST pool ──────────────────────────────────────────
        self.chk_cest2 = QCheckBox("Add second CEST pool?")
        self.chk_cest2.setChecked(False)
        self.chk_cest2.setStyleSheet("font-weight: bold;")
        cest_outer.addWidget(self.chk_cest2)  # ← into container

        self._cest2_widget = QWidget()
        form2 = QFormLayout(self._cest2_widget)
        form2.setContentsMargins(12, 4, 0, 0)  # indent to distinguish

        self.pool_name2 = QLineEdit("MyMolecule2")
        self.dw2 = QDoubleSpinBox()
        self.dw2.setRange(-10, 10); self.dw2.setValue(1.9); self.dw2.setSuffix(" ppm")
        self.dw2.setToolTip("Chemical shift for 2nd CEST pool (e.g. 1.9 ppm for creatine)")

        # T1 fixed-or-sweep for pool 2
        self.cest2_t1 = FixedOrSweepWidget(
            fixed_value=2800, fixed_range=(100, 5000), fixed_suffix=" ms",
            sweep_min_value=1000, sweep_min_range=(100, 5000),
            sweep_max_value=3000, sweep_max_range=(100, 5000),
            sweep_step_value=100, sweep_step_range=(10, 500),
            sweep_suffix=" ms", starts_fixed=True,
        )
        # T2 fixed-or-sweep for pool 2
        self.cest2_t2 = FixedOrSweepWidget(
            fixed_value=40, fixed_range=(1, 500), fixed_suffix=" ms",
            sweep_min_value=20, sweep_min_range=(1, 500),
            sweep_max_value=100, sweep_max_range=(1, 500),
            sweep_step_value=10, sweep_step_range=(1, 200),
            sweep_suffix=" ms", starts_fixed=True,
        )

        # ksw_0  — exchange rate for pool 2
        self.ksw0_min  = QSpinBox(); self.ksw0_min.setRange(10, 5000);  self.ksw0_min.setValue(50);   self.ksw0_min.setSuffix(" s⁻¹")
        self.ksw0_max  = QSpinBox(); self.ksw0_max.setRange(10, 5000);  self.ksw0_max.setValue(800);  self.ksw0_max.setSuffix(" s⁻¹")
        self.ksw0_step = QSpinBox(); self.ksw0_step.setRange(1, 500);   self.ksw0_step.setValue(50);  self.ksw0_step.setSuffix(" s⁻¹")

        # fs_0  — proton fraction for pool 2
        self.fs0_min  = QDoubleSpinBox(); self.fs0_min.setRange(0.0001, 1);  self.fs0_min.setValue(0.001);  self.fs0_min.setDecimals(4); self.fs0_min.setSingleStep(0.001)
        self.fs0_max  = QDoubleSpinBox(); self.fs0_max.setRange(0.0001, 1);  self.fs0_max.setValue(0.05);   self.fs0_max.setDecimals(4); self.fs0_max.setSingleStep(0.005)
        self.fs0_step = QDoubleSpinBox(); self.fs0_step.setRange(0.0001, 0.5); self.fs0_step.setValue(0.002); self.fs0_step.setDecimals(4); self.fs0_step.setSingleStep(0.001)

        self.cest2_protons = QSpinBox()
        self.cest2_protons.setRange(1, 10)
        self.cest2_protons.setValue(3)
        self.cest2_protons.setToolTip("Exchangeable protons for pool 2 (f = conc × n / 110000)")

        form2.addRow("Pool name:", self.pool_name2)
        form2.addRow("Chemical shift (dw):", self.dw2)
        form2.addRow("T1:", self.cest2_t1)
        form2.addRow("T2:", self.cest2_t2)
        form2.addRow("CEST Protons (n):", self.cest2_protons)
        form2.addRow("Exchange rate ksw_0 (s⁻¹):", _make_sweep_row(
            (self.ksw0_min, "min"), (self.ksw0_max, "max"), (self.ksw0_step, "step")))
        form2.addRow("Proton fraction fs_0:", _make_sweep_row(
            (self.fs0_min, "min"), (self.fs0_max, "max"), (self.fs0_step, "step")))

        cest_outer.addWidget(self._cest2_widget)   # ← into container
        self._cest2_widget.setVisible(False)
        self.chk_cest2.toggled.connect(self._cest2_widget.setVisible)

        # Wire the outer "Add CEST pool?" checkbox
        outer.addWidget(self._cest_params_widget)
        self._cest_params_widget.setVisible(True)   # shown by default
        self.chk_cest.toggled.connect(self._cest_params_widget.setVisible)

        return grp

    # ─── MT Pool ──────────────────────────────────────────────────────────

    def _build_mt_group(self):
        grp = QGroupBox("MT Pool")
        outer_layout = QVBoxLayout(grp)

        self.chk_mt = QCheckBox("Add MT Pool?")
        self.chk_mt.setChecked(False)
        self.chk_mt.setStyleSheet("font-weight: bold;")
        outer_layout.addWidget(self.chk_mt)

        self._mt_params_widget = QWidget()
        form = QFormLayout(self._mt_params_widget)
        form.setContentsMargins(0, 4, 0, 0)

        # T1 — fixed-or-sweep (in seconds)
        self.mt_t1 =FixedOrSweepWidget(
            fixed_value=100, fixed_range=(1, 5000), fixed_suffix=" ms",
            sweep_min_value=100, sweep_min_range=(1, 5000),
            sweep_max_value=5000, sweep_max_range=(1, 5000),
            sweep_step_value=1, sweep_step_range=(1, 200),
            sweep_suffix=" ms", starts_fixed=True,
        )
        self.mt_t1.setToolTip("MT pool T1 (ms). Default fixed at 100 ms.")

        # T2 — fixed-or-sweep (in µs, stored as µs, converted to s in get_config)
        self.mt_t2 = FixedOrSweepWidget(
            use_double=True,
            fixed_value=10.0,  fixed_range=(1.0, 500.0),  fixed_suffix=" µs",
            fixed_decimals=1,  fixed_step=1.0,
            sweep_min_value=5.0,  sweep_min_range=(1.0, 500.0),
            sweep_max_value=50.0, sweep_max_range=(1.0, 500.0),
            sweep_step_value=5.0, sweep_step_range=(0.1, 100.0),
            sweep_suffix=" µs", sweep_decimals=1, sweep_step=1.0,
            starts_fixed=True,
        )
        self.mt_t2.setToolTip("MT pool T2 (µs). Default fixed at 10 µs.")

        # Exchange rate k (s⁻¹) — fixed-or-sweep
        self.mt_k = FixedOrSweepWidget(
            use_double=True,
            fixed_value=50.0,  fixed_range=(1.0, 2000.0),  fixed_suffix=" s⁻¹",
            fixed_decimals=1,  fixed_step=10.0,
            sweep_min_value=20.0,  sweep_min_range=(1.0, 2000.0),
            sweep_max_value=300.0, sweep_max_range=(1.0, 2000.0),
            sweep_step_value=20.0, sweep_step_range=(1.0, 200.0),
            sweep_suffix=" s⁻¹", sweep_decimals=1, sweep_step=10.0,
            starts_fixed=False,
        )
        self.mt_k.setToolTip("MT exchange rate k (s⁻¹). Uncheck Fixed to sweep.")

        # Chemical shift dw (ppm) — fixed-or-sweep
        self.mt_dw = FixedOrSweepWidget(
            use_double=True,
            fixed_value=-2.5,  fixed_range=(-10.0, 10.0),  fixed_suffix=" ppm",
            fixed_decimals=2,  fixed_step=0.5,
            sweep_min_value=-2.5, sweep_min_range=(-10.0, 0.0),
            sweep_max_value=2.5,  sweep_max_range=(0.0, 10.0),
            sweep_step_value=0.5, sweep_step_range=(0.1, 5.0),
            sweep_suffix=" ppm", sweep_decimals=2, sweep_step=0.1,
            starts_fixed=False,
        )
        self.mt_dw.setToolTip("MT chemical shift dw (ppm). Uncheck Fixed to sweep.")

        # Proton fraction f — fixed-or-sweep
        self.mt_f = FixedOrSweepWidget(
            use_double=True,
            fixed_value=0.01,  fixed_range=(0.0001, 1.0),  fixed_suffix="",
            fixed_decimals=4,  fixed_step=0.001,
            sweep_min_value=0.001, sweep_min_range=(0.0001, 1.0),
            sweep_max_value=0.05,  sweep_max_range=(0.001, 1.0),
            sweep_step_value=0.001, sweep_step_range=(0.0001, 0.5),
            sweep_decimals=4, sweep_step=0.001,
            starts_fixed=False,
        )
        self.mt_f.setToolTip("MT proton fraction f. Uncheck Fixed to sweep.")

        self.mt_protons = QSpinBox()
        self.mt_protons.setRange(1, 20); self.mt_protons.setValue(1)

        self.mt_lineshape = QComboBox()
        self.mt_lineshape.addItems(["SuperLorentzian", "Lorentzian", "Gaussian"])
        self.mt_lineshape.setCurrentText("SuperLorentzian")

        form.addRow("T1:", self.mt_t1)
        form.addRow("T2:", self.mt_t2)
        form.addRow("Exchange rate k (s⁻¹):", self.mt_k)
        form.addRow("Chemical shift dw (ppm):", self.mt_dw)
        form.addRow("Proton fraction f:", self.mt_f)
        form.addRow("Protons:", self.mt_protons)
        form.addRow("Lineshape:", self.mt_lineshape)

        outer_layout.addWidget(self._mt_params_widget)
        self._mt_params_widget.setVisible(False)
        self.chk_mt.toggled.connect(self._mt_params_widget.setVisible)
        return grp

    # ─── Simulation Options ───────────────────────────────────────────────

    def _build_sim_group(self):
        grp = QGroupBox("Simulation Options")
        form = QFormLayout(grp)
        self.num_workers       = QSpinBox(); self.num_workers.setRange(1, 64);   self.num_workers.setValue(8)
        self.num_workers.setToolTip(
            "CPU processes simulating dictionary entries in parallel — "
            "higher is faster, up to the number of cores.")
        self.max_pulse_samples = QSpinBox(); self.max_pulse_samples.setRange(10, 500); self.max_pulse_samples.setValue(100)
        self.max_pulse_samples.setToolTip(
            "Time steps used to discretise each shaped RF pulse in the "
            "Bloch–McConnell simulation — higher is more accurate but slower.")
        form.addRow("Parallel workers:", self.num_workers)
        form.addRow("Max pulse samples:", self.max_pulse_samples)
        return grp

    def _build_scanner_limits_group(self):
        """Optional hardware limits — when enabled, the .seq is written by
        sequences_sl.write_sequence_clinical (system=lims, real spoilers) instead
        of the limit-free simulation writer. OFF by default → nothing changes."""
        from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QGridLayout,
                                     QLabel, QVBoxLayout, QHBoxLayout, QWidget)
        grp = QGroupBox("Scanner Limits")
        grp.setCheckable(True)
        grp.setChecked(False)
        grp.setToolTip(
            "OFF (default) and ON .seq file is written with these hardware limits.")
        self.grp_scanner_lims = grp

        outer = QVBoxLayout(grp)
        # Collapsible content — hidden until the group is enabled ("expand on click").
        content = QWidget()
        self._scanner_lims_content = content
        cl = QVBoxLayout(content); cl.setContentsMargins(0, 4, 0, 0); cl.setSpacing(6)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.combo_lim_preset = QComboBox()
        self.combo_lim_preset.addItems(
            ["Custom (user defined)", "Siemens (Prisma 3T)", "GE (Signa 3T)",
             "Philips (1.5T)", "United Imaging (3T)"])
        self.combo_lim_preset.setToolTip(
            "Custom (user defined) = enter the values yourself.  The vendor presets "
            "pre-fill hardware limits from the openMRF scanner system-definition files "
            "(one representative platform per vendor; values stay editable).")
        preset_row.addWidget(self.combo_lim_preset, 1)
        cl.addLayout(preset_row)

        def _dsb(lo, hi, val, dec, suffix):
            s = QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(dec)
            s.setValue(val); s.setSuffix(suffix); s.setMinimumWidth(110); return s
        self.lim_max_grad    = _dsb(1, 500, 30, 1, " mT/m")
        self.lim_max_slew    = _dsb(1, 1000, 100, 1, " T/m/s")
        self.lim_rf_dead     = _dsb(0, 1000, 100, 0, " µs")
        self.lim_rf_ring     = _dsb(0, 1000, 30, 0, " µs")
        self.lim_adc_dead    = _dsb(0, 1000, 10, 0, " µs")
        self.lim_rf_raster   = _dsb(0.1, 100, 1, 2, " µs")
        self.lim_grad_raster = _dsb(0.1, 100, 10, 2, " µs")

        # Fields side by side — 2 label+field pairs per row.
        grid = QGridLayout(); grid.setHorizontalSpacing(14); grid.setVerticalSpacing(6)
        fields = [
            ("Max grad:",    self.lim_max_grad),   ("Max slew:",    self.lim_max_slew),
            ("RF dead:",     self.lim_rf_dead),     ("RF ringdown:", self.lim_rf_ring),
            ("ADC dead:",    self.lim_adc_dead),    ("RF raster:",   self.lim_rf_raster),
            ("Grad raster:", self.lim_grad_raster),
        ]
        for i, (lbl, sp) in enumerate(fields):
            r, c = divmod(i, 2)
            grid.addWidget(QLabel(lbl), r, c * 2)
            grid.addWidget(sp,          r, c * 2 + 1)
        cl.addLayout(grid)

        outer.addWidget(content)
        content.setVisible(False)               # collapsed until enabled
        grp.toggled.connect(content.setVisible)

        # Hardware limits taken from the openMRF user_specifications/system_definitions
        # CSVs (one representative platform per vendor).  mT/m, T/m/s, and times in µs.
        # (max_grad, max_slew, rf_dead, rf_ring, adc_dead, rf_raster, grad_raster)
        _PRESETS = {
            "Siemens (Prisma 3T)":  (80,  200, 100, 100, 10,  1, 10),
            "GE (Signa 3T)":        (43,  180, 100, 100, 40,  2,  4),
            "Philips (1.5T)":       (40,  180, 100, 100, 10,  1, 10),
            "United Imaging (3T)":  (43,  180, 100, 100, 100, 2, 10),
        }

        def _apply_preset(name):
            if name in _PRESETS:
                for sp, v in zip((self.lim_max_grad, self.lim_max_slew, self.lim_rf_dead,
                                  self.lim_rf_ring, self.lim_adc_dead, self.lim_rf_raster,
                                  self.lim_grad_raster), _PRESETS[name]):
                    sp.setValue(v)
        self.combo_lim_preset.currentTextChanged.connect(_apply_preset)
        return grp

    # ─── Output File Paths ────────────────────────────────────────────────

    #: fixed output filenames written inside the chosen output folder
    _OUT_FILES = {
        "yaml_fn":      "scenario.yaml",
        "seq_fn":       "acq_protocol.seq",
        "dict_fn":      "dict.mat",
        "quantmaps_fn": "quant_maps.mat",
    }

    @staticmethod
    def _default_output_dir() -> str:
        """Fallback output folder when the field is left blank — resolved at
        runtime so it is correct per-user and in an exported/bundled app:
        ~/Documents/OCEAN_Output when bundled, <project>/OUTPUT_FILES in dev.
        Nothing is hardcoded to any one machine or user."""
        from my_gui.paths import get_output_dir
        return str(get_output_dir())

    def output_dir(self) -> str:
        """The chosen output folder (falls back to the per-user default)."""
        return (self.edit_out_dir.text().strip() or self._default_output_dir())

    def _build_output_group(self):
        """
        A single output FOLDER.  All four CEST-MRF outputs are written inside it:
          scenario.yaml · acq_protocol.seq · dict.mat · quant_maps.mat
        The default is per-user (~/MRF_Output) and any writable folder works — it is
        created automatically if missing, so this is portable across users/machines.
        """
        grp = QGroupBox("Choose the folder where you want to save output")
        vbox = QVBoxLayout(grp)

        row = QHBoxLayout()
        lbl = QLabel("Output folder:")
        row.addWidget(lbl)
        self.edit_out_dir = QLineEdit("")
        row.addWidget(self.edit_out_dir, stretch=1)
        btn = QPushButton("Browse…")
        btn.setFixedWidth(80)
        btn.clicked.connect(self._browse_output_dir)
        row.addWidget(btn)
        vbox.addLayout(row)
        return grp

    def _browse_output_dir(self):
        """Pick the output folder (a directory, not individual files)."""
        start = self.output_dir()
        d = QFileDialog.getExistingDirectory(self, "Choose output folder", start)
        if d:
            self.edit_out_dir.setText(d)

    # ─── Dictionary size calculator ──────────────────────────────────────────

    _DICT_LIMIT = 15_000_000   # 15 M combinations — hard limit

    def _build_dict_size_group(self) -> QGroupBox:
        """
        Live dictionary-size readout (ported from MATLAB DictSizeCalculator.m).
        Shows estimated number of parameter combinations for the current config.
        Turns red and shows a warning when > 15 M entries.
        """
        from PyQt6.QtWidgets import QMessageBox
        grp = QGroupBox("Dictionary Size Estimate")
        vbox = QVBoxLayout(grp)

        # Size display label
        self._lbl_dict_size = QLabel("—")
        self._lbl_dict_size.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_dict_size.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 4px;"
        )
        vbox.addWidget(self._lbl_dict_size)

        # Breakdown label — hidden (kept for tooltip on the size label)
        self._lbl_dict_breakdown = QLabel("")
        self._lbl_dict_breakdown.setVisible(False)

        # Warning bar (hidden by default)
        self._lbl_dict_warning = QLabel(
            "⚠  Dictionary exceeds 15 M entries — simulation will likely fail or run out of memory."
        )
        self._lbl_dict_warning.setStyleSheet(
            "background: #7a2020; color: #ffcccc; border-radius: 4px; "
            "padding: 4px 6px; font-weight: bold; font-size: 11px;"
        )
        self._lbl_dict_warning.setWordWrap(True)
        self._lbl_dict_warning.setVisible(False)
        vbox.addWidget(self._lbl_dict_warning)

        # Wire all relevant parameter widgets to recalculate on change
        self._wire_dict_size_signals()
        # Initial calculation
        self._update_dict_size()

        return grp

    def _wire_dict_size_signals(self):
        """Connect all config widgets that affect dictionary size to _update_dict_size."""

        def _connect_fosw(w: FixedOrSweepWidget):
            """Wire all spinboxes + the Fixed checkbox inside a FixedOrSweepWidget."""
            w.fixed_spin.valueChanged.connect(self._update_dict_size)
            w.sweep_min.valueChanged.connect(self._update_dict_size)
            w.sweep_max.valueChanged.connect(self._update_dict_size)
            w.sweep_step.valueChanged.connect(self._update_dict_size)
            w.chk_fixed.toggled.connect(self._update_dict_size)

        # CEST pool 1 — plain QSpinBox / QDoubleSpinBox
        for widget in [
            self.k_min, self.k_max, self.k_step,
            self.f_min, self.f_max, self.f_step,
        ]:
            widget.valueChanged.connect(self._update_dict_size)

        # CEST pool 2 — plain QSpinBox / QDoubleSpinBox
        for widget in [
            self.ksw0_min, self.ksw0_max, self.ksw0_step,
            self.fs0_min,  self.fs0_max,  self.fs0_step,
        ]:
            widget.valueChanged.connect(self._update_dict_size)

        # MT pool — FixedOrSweepWidgets
        for fosw in [self.mt_k, self.mt_f, self.mt_dw]:
            _connect_fosw(fosw)

        # Water T1 / T2 — FixedOrSweepWidgets
        for fosw in [self.water_t1, self.water_t2]:
            _connect_fosw(fosw)

        # Pool enable toggles
        self.chk_cest.toggled.connect(self._update_dict_size)
        self.chk_cest2.toggled.connect(self._update_dict_size)
        self.chk_mt.toggled.connect(self._update_dict_size)

    @staticmethod
    def _n_steps(start, stop, step) -> int:
        """Number of values in arange(start, stop+step, step), minimum 1."""
        if step <= 0:
            return 1
        n = int(round((stop - start) / step)) + 1
        return max(n, 1)

    @staticmethod
    def _fmt_entries(n: int) -> str:
        if n < 1_000:
            return str(n)
        if n < 1_000_000:
            return f"{n/1000:.2f} K"
        if n < 1_000_000_000:
            return f"{n/1_000_000:.2f} M"
        return f"{n/1_000_000_000:.2f} B"

    def _update_dict_size(self):
        """Recompute dictionary size from current widget values and update labels."""
        try:
            cfg = self.get_config()
        except Exception:
            return

        parts = []
        total = 1

        # ── Water T1 / T2 ─────────────────────────────────────────────────
        n_t1 = len(cfg['water_pool'].get('t1', [1]))
        n_t2 = len(cfg['water_pool'].get('t2', [1]))
        total *= n_t1 * n_t2
        parts.append(f"Water T1×T2: {n_t1}×{n_t2}")

        # ── CEST pool 1 ───────────────────────────────────────────────────
        if cfg.get('cest_pool'):
            pools = list(cfg['cest_pool'].values())
            p1 = pools[0]
            n_k   = len(p1.get('k', []))
            n_f   = len(p1.get('f', []))
            total *= max(n_k, 1) * max(n_f, 1)
            parts.append(f"CEST1 k×f: {n_k}×{n_f}")

            # ── CEST pool 2 ───────────────────────────────────────────────
            if len(pools) > 1:
                p2    = pools[1]
                n_k2  = len(p2.get('k', []))
                n_f2  = len(p2.get('f', []))
                total *= max(n_k2, 1) * max(n_f2, 1)
                parts.append(f"CEST2 k×f: {n_k2}×{n_f2}")

        # ── MT pool ───────────────────────────────────────────────────────
        if cfg.get('mt_pool'):
            mt = cfg['mt_pool']
            n_mtk  = len(mt.get('k',  [1]))
            n_mtf  = len(mt.get('f',  [1]))
            n_mtdw = len(mt.get('dw', [1]))
            total  *= max(n_mtk, 1) * max(n_mtf, 1) * max(n_mtdw, 1)
            parts.append(f"MT k×f×dw: {n_mtk}×{n_mtf}×{n_mtdw}")

        size_str = self._fmt_entries(total)
        over_limit = total > self._DICT_LIMIT

        self._lbl_dict_size.setText(f"Total entries: {size_str}")
        self._lbl_dict_size.setStyleSheet(
            f"font-size: 14px; font-weight: bold; padding: 4px; "
            f"color: {'#ff6666' if over_limit else '#88dd88'};"
        )
        # Store breakdown as tooltip — visible on hover, not cluttering the UI
        self._lbl_dict_size.setToolTip("  ×  ".join(parts))
        self._lbl_dict_warning.setVisible(over_limit)

    # ─── get_config ───────────────────────────────────────────────────────

    @staticmethod
    def _ms_to_s_list(val):
        """Convert a FixedOrSweepWidget value (ms) to a list (s)."""
        if isinstance(val, float):
            return [val / 1000.0]
        return (np.array(val) / 1000.0).tolist()

    def get_config(self) -> dict:
        # ── Output paths — all four files live inside the single output folder ──
        import os as _os
        _out = self.output_dir()
        try:
            _os.makedirs(_out, exist_ok=True)   # create it so any folder/user works
        except OSError:
            pass                                # non-fatal; surfaced later if unwritable
        yaml_fn      = _os.path.join(_out, self._OUT_FILES["yaml_fn"])
        seq_fn       = _os.path.join(_out, self._OUT_FILES["seq_fn"])
        dict_fn      = _os.path.join(_out, self._OUT_FILES["dict_fn"])
        quantmaps_fn = _os.path.join(_out, self._OUT_FILES["quantmaps_fn"])

        # ── Base config (always-present fields, mirrors YAML exactly) ─────
        cfg: dict = {
            # Output paths
            'yaml_fn':      yaml_fn,
            'seq_fn':       seq_fn,
            'dict_fn':      dict_fn,
            'quantmaps_fn': quantmaps_fn,
            # Scanner
            'b0':             self.b0.value(),
            'gamma':          267.5153,
            'b0_inhom':       self.b0_inhom.value(),
            'rel_b1':         self.rel_b1.value(),
            # Magnetization (MATLAB: magnetization_scale / magnetization_reset)
            'scale':          round(self.mag_scale.value(), 4),
            'reset_init_mag': self.reset_init_mag.value(),
            'verbose':        0,
            # Simulation
            'max_pulse_samples': self.max_pulse_samples.value(),
            'num_workers':       self.num_workers.value(),
            # Water pool — always present (ms → s conversion)
            'water_pool': {
                't1': self._ms_to_s_list(self.water_t1.get_value()),
                't2': self._ms_to_s_list(self.water_t2.get_value()),
                'f':  1,   # fixed by convention, not shown in GUI
            },
        }

        # ── CEST pool (optional) ──────────────────────────────────────────
        if self.chk_cest.isChecked():
            pool_name = self.pool_name.text() or "CEST"
            cest_t1   = self._ms_to_s_list(self.cest_t1.get_value())
            cest_t2   = self._ms_to_s_list(self.cest_t2.get_value())

            n_p = self.cest_protons.value()   # CEST protons (user-set)
            cfg['cest_pool'] = {
                pool_name: {
                    't1': cest_t1,
                    't2': cest_t2,
                    'k':  list(range(
                        self.k_min.value(),
                        self.k_max.value() + self.k_step.value(),
                        self.k_step.value())),
                    'dw': self.dw.value(),
                    'f':  (np.arange(
                        self.f_min.value(),
                        self.f_max.value() + self.f_step.value(),
                        self.f_step.value()) * n_p / 110000).tolist(),
                }
            }

            # Second CEST pool (fs_0 / ksw_0)
            if self.chk_cest2.isChecked():
                pool2  = self.pool_name2.text() or "CEST2"
                c2_t1  = self._ms_to_s_list(self.cest2_t1.get_value())
                c2_t2  = self._ms_to_s_list(self.cest2_t2.get_value())
                n_p2   = self.cest2_protons.value()
                cfg['cest_pool'][pool2] = {
                    't1': c2_t1,
                    't2': c2_t2,
                    'k':  list(range(
                        self.ksw0_min.value(),
                        self.ksw0_max.value() + self.ksw0_step.value(),
                        self.ksw0_step.value())),
                    'dw': self.dw2.value(),
                    'f':  (np.arange(
                        self.fs0_min.value(),
                        self.fs0_max.value() + self.fs0_step.value(),
                        self.fs0_step.value()) * n_p2 / 110000).tolist(),
                }

        # ── MT pool (optional) ────────────────────────────────────────────
        if self.chk_mt.isChecked():
            mt_t1_val = self.mt_t1.get_value()   # s
            mt_t2_val = self.mt_t2.get_value()   # µs → convert to s below
            mt_k_val  = self.mt_k.get_value()
            mt_dw_val = self.mt_dw.get_value()
            mt_f_val  = self.mt_f.get_value()

            def _to_list(v):
                return [float(v)] if isinstance(v, float) else np.array(v).tolist()

            cfg['mt_pool'] = {
                't1':        _to_list(mt_t1_val),
                't2':        _to_list(np.array(mt_t2_val) * 1e-6),  # µs → s
                'k':         _to_list(mt_k_val),
                'dw':        _to_list(mt_dw_val),
                'f':         _to_list(mt_f_val),
                'lineshape': self.mt_lineshape.currentText(),
                'protons':   self.mt_protons.value(),
            }

        # ── Optional scanner hardware limits (opt-in) ─────────────────────────
        if getattr(self, 'grp_scanner_lims', None) is not None \
                and self.grp_scanner_lims.isChecked():
            cfg['scanner_lims'] = {
                'max_grad_mT_m':    self.lim_max_grad.value(),
                'max_slew_T_m_s':   self.lim_max_slew.value(),
                'rf_dead_time_us':  self.lim_rf_dead.value(),
                'rf_ringdown_us':   self.lim_rf_ring.value(),
                'adc_dead_time_us': self.lim_adc_dead.value(),
                'rf_raster_us':     self.lim_rf_raster.value(),
                'grad_raster_us':   self.lim_grad_raster.value(),
            }
        else:
            cfg['scanner_lims'] = None

        return cfg
