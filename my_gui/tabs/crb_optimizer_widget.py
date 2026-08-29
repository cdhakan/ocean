"""
crb_optimizer_widget.py
=======================
CRB-based MRF schedule optimizer.

What is CRB?
------------
The Cramér-Rao Bound (CRB) — or Cramér-Rao Lower Bound (CRLB) — is a result
from statistical estimation theory. It gives the theoretical minimum variance
(uncertainty) any unbiased estimator can achieve when measuring a parameter
from noisy data.

In plain terms:
  • You acquire N MRF measurements.  Each has noise.
  • From those measurements you want to estimate tissue parameters
    (T1w, T2w, Ksw, fs, …).
  • No matter how smart your fitting algorithm is, estimation errors cannot
    be smaller than sqrt(CRB).
  • CRB = inverse of the Fisher Information Matrix (FIM).
    The FIM measures how sensitive the MRF signals are to each parameter
    (via the Jacobian / partial derivatives).
  • High sensitivity → large FIM → small CRB → accurate estimates.
  • Schedule optimisation: choose B1, offsets, Tsat, TR values that
    maximise FIM (minimise CRB) for the parameters you care about.

The normalised CRB (nCRB) used here is:
    nCRB(θ) = sqrt( CRB(θ) ) / θ   [%]
This is the predicted relative estimation error for parameter θ.

Workflow implemented here
-------------------------
tissue parameter ranges  →  dictionary config (YAML)
        ↓
optimizer loop (random search over B1, optionally offsets / Tsat / TR):
    candidate schedule  →  .seq file (pypulseq)
                        →  Bloch-McConnell dictionary
                        →  CRB via Fisher Information Matrix
                        →  scalar objective  (mean nCRB for selected params)
        ↓
best schedule  →  result_ready signal  →  loaded into Sequence tab table
"""

from __future__ import annotations
import os
import re
import tempfile
import traceback

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QTextEdit, QDoubleSpinBox,
    QSpinBox, QCheckBox, QGroupBox, QProgressBar,
    QScrollArea, QFrame, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal


# ─────────────────────────────────────────────────────────────────────────────
# Background worker
# ─────────────────────────────────────────────────────────────────────────────

class CRBOptimizerWorker(QThread):
    """
    Background QThread that runs random-search CRB schedule optimisation.

    Signals
    -------
    progress(iteration, total, best_crb_pct, message)
    finished(best_rows, best_crb_pct)
    error(message)
    """

    progress = pyqtSignal(int, int, float, str)
    finished = pyqtSignal(list, float)
    error    = pyqtSignal(str)

    def __init__(
        self,
        cfg:            dict,
        bounds:         dict,
        n_evals:        int,
        params_to_opt:  list[str],
        sigma:          float = 0.008,
    ):
        super().__init__()
        self._cfg           = cfg
        self._bounds        = bounds
        self._n_evals       = n_evals
        self._params_to_opt = params_to_opt
        self._sigma         = sigma
        self._stop_flag     = False

    def request_stop(self):
        self._stop_flag = True

    def run(self):
        try:
            self._optimize()
        except Exception:
            self.error.emit(traceback.format_exc())

    # ── core optimisation loop ────────────────────────────────────────────────

    def _optimize(self):
        # Lazy imports — keep GUI importable even if cest_mrf not installed
        try:
            from cest_mrf.write_scenario import write_yaml_dict
            from cest_mrf.dictionary.generation import generate_mrf_cest_dictionary
            from cest_mrf.metrics.crlb import crb_calc
        except ImportError as exc:
            self.error.emit(
                f"Cannot import cest_mrf modules:\n  {exc}\n\n"
                "Make sure open-py-cest-mrf is installed or in sys.path."
            )
            return

        bd = self._bounds
        niter    = bd['niter']
        b1_min   = bd['b1_min']
        b1_max   = bd['b1_max']
        off_vals = bd['offset_values'] or [3.0]
        tsat_min = bd['tsat_min'] / 1000     # ms → s
        tsat_max = bd['tsat_max'] / 1000
        tr_min   = bd['tr_min']   / 1000     # ms → s
        tr_max   = bd['tr_max']   / 1000

        best_crb  = float('inf')
        best_rows: list | None = None

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_fn = os.path.join(tmpdir, 'scenario.yaml')
            seq_fn  = os.path.join(tmpdir, 'schedule.seq')
            dict_fn = os.path.join(tmpdir, 'dict.mat')

            # Write YAML once — tissue parameter ranges are fixed
            cfg = dict(self._cfg)
            cfg.update(yaml_fn=yaml_fn, seq_fn=seq_fn, dict_fn=dict_fn)
            write_yaml_dict(cfg)

            for i in range(self._n_evals):
                if self._stop_flag:
                    self.progress.emit(i, self._n_evals, best_crb, "Stopped by user.")
                    break

                # ── sample random candidate schedule ──────────────────────────
                b1 = np.random.uniform(b1_min, b1_max, niter).tolist()

                offsets = (
                    np.random.choice(off_vals, niter).tolist()
                    if bd.get('vary_offsets') and len(off_vals) > 1
                    else [off_vals[0]] * niter
                )

                tp = (
                    np.random.uniform(tsat_min, tsat_max, niter).tolist()
                    if bd.get('vary_tsat')
                    else [float(tsat_min)] * niter
                )

                tr_raw = (
                    np.random.uniform(tr_min, tr_max, niter)
                    if bd.get('vary_tr')
                    else np.full(niter, tr_min)
                )

                # Trec = TR − Tsat − 3.1 ms (imaging + ADC overhead)
                OVERHEAD = 0.0031    # s
                trec = [max(float(tr - t - OVERHEAD), 0.0)
                        for tr, t in zip(tr_raw, tp)]

                seq_defs = {
                    'num_meas':      niter,
                    'n_pulses':      1,
                    'tp':            tp,
                    'td':            0.0,
                    'Trec':          trec,
                    'B1pa':          b1,
                    'excFA':         [90.0] * niter,
                    'SLFA':          [0.0]  * niter,
                    'SLflag':        [0]    * niter,
                    'DCsat':         [t / (t + 1e-9) for t in tp],
                    'offsets_ppm':   offsets,
                    'B0':            float(cfg.get('b0', 9.4)),
                    'seq_id_string': 'crb_opt',
                }

                try:
                    self._write_seq(seq_defs, seq_fn)

                    dictionary = generate_mrf_cest_dictionary(
                        seq_fn      = seq_fn,
                        param_fn    = yaml_fn,
                        dict_fn     = dict_fn,
                        num_workers = cfg.get('num_workers', 4),
                        axes        = 'xy',
                    )

                    # Reshape for crb_calc (needs 1-D param arrays)
                    signals = dictionary['sig']
                    d = {}
                    for k, v in dictionary.items():
                        if k == 'sig':
                            continue
                        arr = np.squeeze(np.asarray(v))
                        d[k] = np.expand_dims(arr, 0) if arr.ndim == 0 else arr

                    crb, dvars = crb_calc(
                        dictionary=d, signals=signals,
                        sigma=self._sigma, norm=True, verbose=False,
                    )
                    m_crb = np.mean(crb, axis=0)

                    # Objective = mean nCRB (%) over requested parameters
                    obj_vals = [
                        np.sqrt(max(float(m_crb[pi, pi]), 0.0)) * 100
                        for pi, p in enumerate(dvars)
                        if p in self._params_to_opt
                    ]
                    obj = float(np.mean(obj_vals)) if obj_vals else float('inf')

                    if obj < best_crb:
                        best_crb  = obj
                        best_rows = _rows_from_schedule(
                            b1, offsets,
                            [t * 1000 for t in tp],
                            [(float(tr) + t + OVERHEAD) * 1000
                             for tr, t in zip(trec, tp)],
                        )

                    crb_detail = "  ".join(
                        f"{p}={np.sqrt(max(float(m_crb[pi,pi]),0))*100:.1f}%"
                        for pi, p in enumerate(dvars)
                        if p in self._params_to_opt
                    )
                    msg = (
                        f"[{i+1}/{self._n_evals}]  {crb_detail}"
                        f"  →  best={best_crb:.2f}%"
                    )

                except Exception as exc:
                    msg = f"[{i+1}/{self._n_evals}]  failed: {exc}"

                self.progress.emit(i + 1, self._n_evals, best_crb, msg)

        if best_rows:
            self.finished.emit(best_rows, best_crb)
        else:
            self.error.emit("Optimizer produced no valid schedule. Check log.")

    # ── sequence writer ───────────────────────────────────────────────────────

    @staticmethod
    def _write_seq(seq_defs: dict, seq_fn: str):
        """
        Try writing the sequence using the preclinical SL writer first,
        then fall back to the supplementary preclinical writer.
        """
        # Primary: per-measurement SL writer (sequences_sl.py)
        try:
            from sequences_sl import write_sequence_sl
            write_sequence_sl(seq_defs=seq_defs, seq_fn=seq_fn)
            return
        except ImportError:
            pass
        except Exception as exc:
            raise RuntimeError(
                f"sequences_sl.write_sequence_sl failed: {exc}"
            )

        # Fallback: supplementary preclinical writer (scalar tp / Trec)
        try:
            from supplementary.published_pulse_sequences.cest_mrf.sequences import (
                write_sequence_preclinical,
            )
            sd = dict(seq_defs)
            sd['tp']   = float(seq_defs['tp'][0])
            sd['Trec'] = float(seq_defs['Trec'][0])
            write_sequence_preclinical(seq_defs=sd, seq_fn=seq_fn)
            return
        except (ImportError, Exception) as exc:
            raise RuntimeError(
                f"No sequence writer available. "
                f"Make sure sequences_sl.py is on sys.path. "
                f"Last error: {exc}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rows_from_schedule(
    b1: list, offsets: list, tsat_ms: list, tr_ms: list
) -> list[list]:
    """Convert per-measurement arrays to sequence-table row format."""
    return [
        [round(tr_ms[i], 1), round(b1[i], 4), round(offsets[i], 4),
         90.0, round(tsat_ms[i], 1), 0, 0.0]
        for i in range(len(b1))
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CRB Optimizer widget  (tab inside ScheduleGeneratorDialog)
# ─────────────────────────────────────────────────────────────────────────────

class CRBOptimizerWidget(QWidget):
    """
    Full UI for the CRB-based schedule optimizer.

    Emits ``result_ready(rows, crb_pct)`` when a best schedule is found.
    Emits ``load_and_close(rows, saved_filepath)`` when the user asks to save the
    optimised schedule AND load it straight into the Sequence table.
    The parent dialog reads ``.best_rows`` and ``.best_crb`` to load the result.
    """

    result_ready   = pyqtSignal(list, float)
    load_and_close = pyqtSignal(list, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker    = None
        self._best_rows: list | None = None
        self._best_crb:  float | None = None
        self._setup_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(6, 6, 6, 6)

        # Scrollable settings area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        inner = QWidget()
        vl = QVBoxLayout(inner)
        vl.setSpacing(8)
        vl.setContentsMargins(2, 2, 2, 2)

        _GRP_SS = (
            "QGroupBox { font-weight: bold; color: #5dade2; "
            "border: 1px solid #444; border-radius: 5px; margin-top: 6px; padding-top: 22px; }"
            "QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; "
            "left: 10px; top: 4px; padding: 0 4px; font-size: 14px; }"
        )

        def _hdr(text: str) -> QLabel:
            l = QLabel(text)
            l.setStyleSheet("font-size: 10px; color: #777;")
            return l

        # ──────────────────────────────────────────────────────────────────────
        # Section 1 — Tissue parameter ranges
        # ──────────────────────────────────────────────────────────────────────
        grp1 = QGroupBox("Tissue Parameter Ranges  (dictionary sweep)")
        grp1.setStyleSheet(_GRP_SS)
        g1 = QGridLayout(grp1)
        g1.setSpacing(5)

        # Column header tooltips
        _col_tips = [
            "",
            "",
            "Minimum value of this tissue parameter in the dictionary",
            "Maximum value of this tissue parameter in the dictionary",
            (
                "# Dictionary points\n\n"
                "How many evenly-spaced values to sample between Min and Max "
                "when building the dictionary.\n\n"
                "Example: T1w Min=1200, Max=2800, 4 points\n"
                "→ dictionary uses T1w = [1200, 1733, 2267, 2800] ms\n\n"
                "More points = more accurate CRB evaluation,\n"
                "but the dictionary grows and each evaluation takes longer.\n"
                "4–6 points per parameter is a good starting value."
            ),
            (
                "Sweep (vary this parameter)\n\n"
                "✓ Checked  →  this parameter varies across the dictionary\n"
                "   (sampled at the # Dict points between Min and Max).\n"
                "   The CRB is computed over this full range — the schedule\n"
                "   must be informative across all these values.\n\n"
                "✗ Unchecked  →  this parameter is fixed at the Min value.\n"
                "   It is treated as already known and does not contribute\n"
                "   to the dictionary size or CRB calculation."
            ),
        ]
        _col_labels = ["Parameter", "Unit", "Min", "Max", "# Dict points", "Sweep (vary)"]
        for ci, (h, tip) in enumerate(zip(_col_labels, _col_tips)):
            lbl = _hdr(h)
            if tip:
                lbl.setToolTip(tip)
            g1.addWidget(lbl, 0, ci)

        # (name, unit, lo, hi, dec, n_def, sweep_default, row_tooltip)
        _TISSUE = [
            ("T1w",  "ms",    1200,  2800,  0, 4, True,
             "Water longitudinal relaxation time.\n"
             "Sweep checked → dictionary varies T1w between Min and Max.\n"
             "Typical brain tissue range: 1200–2800 ms at 9.4 T."),
            ("T2w",  "ms",      30,  1600,  0, 4, True,
             "Water transverse relaxation time.\n"
             "Sweep checked → dictionary varies T2w between Min and Max.\n"
             "Typical tissue range: 30–1600 ms."),
            ("Ksw",  "Hz",     900, 12000,  0, 4, True,
             "Solute-to-water proton exchange rate (= k_sw in s⁻¹).\n"
             "Sweep checked → dictionary varies Ksw between Min and Max.\n"
             "Typical amine range: 900–12000 Hz at physiological pH."),
            ("Kssw", "Hz",       1,   100,  1, 3, False,
             "Semi-solid (MT) pool exchange rate.\n"
             "Sweep checked → dictionary includes MT pool variation.\n"
             "Uncheck if your experiment does not include an MT pool."),
            ("fs",   "mM",       1,    40,  1, 4, True,
             "Solute (CEST pool) concentration in mM.\n"
             "Converted to proton fraction using:\n"
             "  f = fs[mM] × (# exchangeable H) / 110 000\n"
             "Typical amine concentration: 1–40 mM."),
            ("fss",  "mM",       1, 15000,  0, 3, False,
             "Semi-solid (MT) pool concentration in mM.\n"
             "Converted to proton fraction: f = fss / 110 000.\n"
             "Uncheck if no MT pool is modelled."),
        ]
        self._tissue: dict[str, tuple] = {}
        for row_i, (nm, unit, lo, hi, dec, n_def, en, tip) in enumerate(_TISSUE, start=1):
            lbl = QLabel(nm); lbl.setToolTip(tip)
            g1.addWidget(lbl,          row_i, 0)
            g1.addWidget(QLabel(unit), row_i, 1)

            sp_min = QDoubleSpinBox()
            sp_min.setRange(0, 1e7); sp_min.setDecimals(dec); sp_min.setValue(lo)
            sp_min.setToolTip(f"Minimum {nm} value in the dictionary")

            sp_max = QDoubleSpinBox()
            sp_max.setRange(0, 1e7); sp_max.setDecimals(dec); sp_max.setValue(hi)
            sp_max.setToolTip(f"Maximum {nm} value in the dictionary")

            sp_n = QSpinBox()
            sp_n.setRange(1, 10000); sp_n.setValue(n_def)
            sp_n.setToolTip(
                f"Number of evenly-spaced {nm} values between Min and Max.\n"
                "Recommended: 3–6 per swept parameter.\n"
                "Total dict entries = product of all swept parameters' points:\n"
                "  4 params × 4 pts = 256  ✓ fast & accurate\n"
                "  4 params × 6 pts = 1296  ✓ good accuracy\n"
                "  4 params × 10 pts = 10000 ⚠ slow\n"
                "  4 params × 20 pts = 160000 ✗ very slow, numerically unstable\n\n"
                "NOTE: High nCRLB (>1000%) is usually caused by too-low\n"
                "concentration (fs min), not by too few dict points."
            )

            chk = QCheckBox()
            chk.setChecked(en)
            chk.setToolTip(
                f"Sweep (vary) {nm} in the dictionary\n\n"
                "✓ Checked  → dictionary sweeps this parameter Min→Max\n"
                "             (uses # Dict points evenly-spaced values)\n\n"
                "✗ Unchecked → fixed at Min value; not estimated;\n"
                "              does not affect dictionary size or CRB"
            )

            g1.addWidget(sp_min, row_i, 2)
            g1.addWidget(sp_max, row_i, 3)
            g1.addWidget(sp_n,   row_i, 4)
            g1.addWidget(chk,    row_i, 5)
            self._tissue[nm] = (chk, sp_min, sp_max, sp_n)

        # ── Live dictionary-size counter ──────────────────────────────────
        self._lbl_dict_size = QLabel("")
        self._lbl_dict_size.setWordWrap(True)
        self._lbl_dict_size.setStyleSheet("font-size: 10px; padding: 4px 6px; color: #aaa;")
        # Place well below the B0 / CEST / nH / MT / pool-guide rows to avoid overlap
        row_sz = len(_TISSUE) + 6
        g1.addWidget(self._lbl_dict_size, row_sz, 0, 1, 6)

        def _update_dict_size():
            n = 1
            _mt = getattr(self, '_mt_enable', None)
            _mt_on = _mt.isChecked() if _mt is not None else False
            for nm_, (chk_, _, _, sp_n_) in self._tissue.items():
                if nm_ in ("Kssw", "fss") and not _mt_on:
                    continue          # MT pool off → these params are ignored
                if chk_.isChecked():
                    n *= sp_n_.value()
            self._lbl_dict_size.setStyleSheet(
                "font-size: 10px; padding: 4px 6px; color: #aaa;"
            )
            self._lbl_dict_size.setText(
                f"Dictionary size: {n:,} entries   "
                "— 3–5 points per swept parameter is recommended."
            )

        self._update_dict_size_fn = _update_dict_size
        for nm_, (chk_, sp_min_, sp_max_, sp_n_) in self._tissue.items():
            chk_.toggled.connect(_update_dict_size)
            sp_n_.valueChanged.connect(_update_dict_size)
        _update_dict_size()   # initial state

        # B0 / CEST dw / exchangeable H
        row_b = len(_TISSUE) + 1
        g1.addWidget(QLabel("B0"),             row_b,   0); g1.addWidget(QLabel("T"), row_b, 1)
        self._b0 = QDoubleSpinBox(); self._b0.setRange(0.1, 20); self._b0.setValue(9.4); self._b0.setDecimals(2)
        g1.addWidget(self._b0, row_b, 2, 1, 2)

        g1.addWidget(QLabel("CEST shift"),     row_b+1, 0); g1.addWidget(QLabel("ppm"), row_b+1, 1)
        self._dw = QDoubleSpinBox(); self._dw.setRange(-20, 20); self._dw.setValue(3.5); self._dw.setDecimals(1)
        self._dw.setToolTip("Chemical shift of CEST pool (3.5 ppm amide, 2 ppm amine, 1 ppm OH)")
        g1.addWidget(self._dw, row_b+1, 2)

        g1.addWidget(QLabel("Exchangeable H"), row_b+2, 0); g1.addWidget(QLabel("fs pool"), row_b+2, 1)
        self._nh = QSpinBox(); self._nh.setRange(1, 10); self._nh.setValue(3)
        self._nh.setToolTip(
            "Exchangeable protons per solute molecule:\n"
            "  1 → Trp indole NH (7.3 ppm),  OH hydroxyl (1 ppm)\n"
            "  2 → backbone amide NH (3.5 ppm),  guanidinium (2 ppm)\n"
            "  3 → amine –NH₃⁺ (2–3 ppm),  lysine side-chain\n"
            "  Affects: f_fraction = conc[mM] × n_H / 110 000"
        )
        g1.addWidget(self._nh, row_b+2, 2)

        # ── Optional MT (semi-solid) pool toggle ──────────────────────────
        self._mt_enable = QCheckBox("Add MT (semi-solid) pool  —  uses the Kssw + fss rows")
        self._mt_enable.setChecked(False)
        self._mt_enable.setToolTip(
            "Off → 2-pool model (water + CEST); the Kssw and fss rows are ignored.\n"
            "On  → adds a semi-solid MT pool (SuperLorentzian lineshape) built from\n"
            "      the Kssw (exchange rate, Hz) and fss (concentration, mM) rows.\n"
            "Tick the Sweep box on Kssw / fss to also estimate them in the CRB."
        )
        g1.addWidget(self._mt_enable, row_b+3, 0, 1, 6)

        def _sync_mt_rows():
            on = self._mt_enable.isChecked()
            for nm_ in ("Kssw", "fss"):
                for w in self._tissue[nm_]:
                    w.setEnabled(on)
            self._update_dict_size_fn()
        self._mt_enable.toggled.connect(_sync_mt_rows)
        _sync_mt_rows()   # apply initial enabled/disabled state

        # Pool quick-reference — shown as a tooltip on the Tissue Ranges button
        self._pool_guide_html = (
            "<b>Pool quick-reference</b> (CEST shift  |  n_H  |  ksw range  |  detectable fs min):<br>"
            "&nbsp;Amide &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.5 ppm &nbsp;| 2 | 10–100 Hz &nbsp;&nbsp;| ≥ 20 mM<br>"
            "&nbsp;Amine &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2–3 ppm &nbsp;| 3 | 500–5 000 Hz | ≥ 5 mM<br>"
            "&nbsp;Hydroxyl &nbsp;1.0 ppm &nbsp;| 1 | 1 000–3 000 Hz | ≥ 10 mM<br>"
            "&nbsp;Trp (7.3) &nbsp;7.3 ppm &nbsp;| <b>1</b> | 100–5 000 Hz &nbsp;| ≥ <b>1 mM</b><br>"
            "&nbsp;NOE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;−3.5 ppm | 1 | 10–50 Hz &nbsp;&nbsp;&nbsp;| model only<br>"
            "<i>High nCRLB (>500%) = concentration below detection threshold for that sequence.</i>"
        )

        # Tissue Parameter Ranges live in a pop-up dialog opened from a button
        self._grp_tissue = grp1
        _btn_row = QHBoxLayout()
        self._btn_tissue = QPushButton("Tissue Parameter Ranges…")
        self._btn_tissue.clicked.connect(self._open_tissue_dialog)
        _btn_row.addWidget(self._btn_tissue)
        _btn_row.addStretch()
        self._btn_optset = QPushButton("Optimization Settings…")
        self._btn_optset.setToolTip("Evaluations, noise σ, parallel workers, and which "
                                    "parameters to optimise for.")
        self._btn_optset.clicked.connect(self._open_optset_dialog)
        _btn_row.addWidget(self._btn_optset)
        vl.addLayout(_btn_row)

        # ──────────────────────────────────────────────────────────────────────
        # Section 2 — Schedule search bounds
        # ──────────────────────────────────────────────────────────────────────
        grp2 = QGroupBox("Schedule Search Bounds  (what the optimiser randomises)")
        grp2.setStyleSheet(_GRP_SS)
        g2 = QGridLayout(grp2)
        g2.setSpacing(5)


        _sched_col_tips = [
            "",
            "",
            "Minimum value the optimizer will try for this parameter",
            "Maximum value the optimizer will try for this parameter",
            (
                "Randomise per measurement\n\n"
                "✓ Checked  → each measurement in the candidate schedule gets\n"
                "   a randomly drawn value between Min and Max.\n"
                "   The optimiser searches over these random patterns.\n\n"
                "✗ Unchecked → all measurements use the Min value (fixed).\n"
                "   That parameter is constant across the schedule."
            ),
        ]
        for ci, (h, tip) in enumerate(zip(
            ["Parameter", "Unit", "Value / Min", "Max (if randomised)", "Randomise per measurement"],
            _sched_col_tips,
        )):
            lbl = _hdr(h)
            if tip:
                lbl.setToolTip(tip)
            g2.addWidget(lbl, 1, ci)

        def _sched_row(label, unit, lo, hi, dec, vary, row_i, tip=""):
            lbl = QLabel(label)
            if tip:
                lbl.setToolTip(tip)
            g2.addWidget(lbl,          row_i, 0)
            g2.addWidget(QLabel(unit), row_i, 1)
            sp_lo = QDoubleSpinBox()
            sp_lo.setRange(-1e5, 1e5); sp_lo.setDecimals(dec); sp_lo.setValue(lo)
            sp_hi = QDoubleSpinBox()
            sp_hi.setRange(-1e5, 1e5); sp_hi.setDecimals(dec); sp_hi.setValue(hi)
            g2.addWidget(sp_lo, row_i, 2)
            g2.addWidget(sp_hi, row_i, 3)
            cb = QCheckBox()
            cb.setChecked(vary)
            cb.setToolTip(
                f"Randomise {label} per measurement\n\n"
                "✓  Each measurement gets a random value in [Min, Max]\n"
                "✗  All measurements use the Min value (constant)"
            )
            g2.addWidget(cb, row_i, 4)
            # Fixed (unchecked) → show a single value (Min) only.
            # Randomise (checked) → show Min + Max range.
            def _sync_hi(checked, _hi=sp_hi):
                _hi.setVisible(bool(checked))
            cb.toggled.connect(_sync_hi)
            _sync_hi(cb.isChecked())
            return sp_lo, sp_hi, cb

        self._b1_lo, self._b1_hi, self._b1_vary = _sched_row(
            "B1 amplitude", "µT", 0.5, 4.0, 2, True, 2,
            "B1 is always randomised per measurement — it is the primary variable "
            "the optimiser tunes to minimise CRB.\nMin/Max define the search range.",
        )
        self._tsat_lo, self._tsat_hi, self._tsat_vary = _sched_row(
            "Tsat", "ms", 500, 3000, 0, True, 3,
            "Saturation pulse duration.\n"
            "Randomise checked → each measurement gets a random Tsat in [Min, Max].\n"
            "Unchecked → all measurements use Min (constant Tsat schedule).",
        )
        self._tr_lo, self._tr_hi, self._tr_vary = _sched_row(
            "TR", "ms", 3000, 7000, 0, False, 4,
            "Repetition time.\n"
            "Unchecked → all measurements use the Min value (constant TR).\n"
            "Checked → random TR per measurement in [Min, Max].",
        )

        # Offsets row
        off_lbl = QLabel("Offset(s)")
        off_lbl.setToolTip(
            "Saturation frequency offset(s) in ppm.\n\n"
            "Enter one value for a fixed offset (e.g. 3.0).\n"
            "Enter multiple comma-separated values to allow the optimizer to\n"
            "randomly assign one offset per measurement (if 'Randomise' is checked).\n\n"
            "Example:  3.0, 3.5, 2.0, 100\n"
            "  → each measurement randomly picks from these four ppm values."
        )
        g2.addWidget(off_lbl, 5, 0)
        g2.addWidget(QLabel("ppm"), 5, 1)
        self._off_edit = QTextEdit()
        self._off_edit.setFixedHeight(34)
        self._off_edit.setPlainText("3.0")
        self._off_edit.setToolTip(off_lbl.toolTip())
        g2.addWidget(self._off_edit, 5, 2, 1, 2)
        self._off_vary = QCheckBox()
        self._off_vary.setToolTip(
            "Randomise offset per measurement\n\n"
            "✓  Each measurement is assigned a random offset from the list above\n"
            "✗  All measurements use the first offset in the list (fixed)"
        )
        g2.addWidget(self._off_vary, 5, 4)

        # niter
        niter_lbl = QLabel("Measurements")
        niter_lbl.setToolTip("Total number of measurements in the generated schedule (niter)")
        g2.addWidget(niter_lbl, 6, 0)
        self._niter = QSpinBox()
        self._niter.setRange(4, 500)
        self._niter.setValue(40)
        self._niter.setToolTip("Total number of measurements (time points) in the schedule")
        g2.addWidget(self._niter, 6, 2)

        vl.addWidget(grp2)

        # ──────────────────────────────────────────────────────────────────────
        # Section 3 — Optimisation settings
        # ──────────────────────────────────────────────────────────────────────
        grp3 = QGroupBox("Optimisation Settings")
        grp3.setStyleSheet(_GRP_SS)
        g3 = QGridLayout(grp3)
        g3.setSpacing(5)

        g3.addWidget(QLabel("Evaluations"),      0, 0)
        self._n_evals = QSpinBox(); self._n_evals.setRange(5, 2000); self._n_evals.setValue(50)
        self._n_evals.setToolTip("Number of random schedule candidates to evaluate.\n50–100 is a good starting point.")
        g3.addWidget(self._n_evals, 0, 1)

        g3.addWidget(QLabel("Noise  σ"),         1, 0)
        self._sigma = QDoubleSpinBox()
        self._sigma.setRange(0.0001, 1.0); self._sigma.setDecimals(4); self._sigma.setValue(0.008)
        self._sigma.setToolTip("Assumed noise standard deviation (dimensionless).\n"
                               "Typical value: 0.008  (0.8% of M0).")
        g3.addWidget(self._sigma, 1, 1)

        g3.addWidget(QLabel("Parallel workers"),  2, 0)
        self._workers = QSpinBox(); self._workers.setRange(1, 64); self._workers.setValue(4)
        g3.addWidget(self._workers, 2, 1)

        g3.addWidget(QLabel("Optimise for:"),    3, 0)
        pb_widget = QWidget()
        pb_hl = QHBoxLayout(pb_widget); pb_hl.setContentsMargins(0, 0, 0, 0)
        self._param_cbs: dict[str, QCheckBox] = {}
        for p_name, checked in [("fs_0", True), ("ksw_0", True),
                                 ("T1w_0", False), ("T2w_0", False)]:
            cb = QCheckBox(p_name); cb.setChecked(checked)
            pb_hl.addWidget(cb)
            self._param_cbs[p_name] = cb
        pb_hl.addStretch()
        g3.addWidget(pb_widget, 3, 1, 1, 2)

        # Optimisation Settings live in a pop-up dialog opened from a button
        self._grp_optset = grp3
        scroll.setWidget(inner)
        root.addWidget(scroll, stretch=2)

        # ── Progress ──────────────────────────────────────────────────────────
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #444;"); root.addWidget(sep)

        self._prog_bar = QProgressBar()
        self._prog_bar.setRange(0, 100); self._prog_bar.setValue(0)
        root.addWidget(self._prog_bar)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(80)
        self._log.setMaximumHeight(130)
        self._log.setStyleSheet(
            "QTextEdit { font-family: Menlo, Consolas, 'DejaVu Sans Mono', 'Courier New'; font-size: 10px; "
            "background: #111; color: #7ec97e; border: 1px solid #444; border-radius: 3px; }"
        )
        self._log.setPlaceholderText("Optimiser log will appear here…")
        root.addWidget(self._log)

        self._status = QLabel("Ready — configure parameters above and click  Run Optimizer")
        self._status.setStyleSheet("color: #888; font-size: 11px;")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        # ── Run / Stop buttons ────────────────────────────────────────────────
        btn_hl = QHBoxLayout()
        self._btn_run  = QPushButton("Run Optimizer")
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)

        self._btn_run.setStyleSheet(
            "QPushButton { background:#1e8449; color:white; border-radius:4px; padding:5px 20px; }"
            "QPushButton:hover { background:#27ae60; }"
        )
        self._btn_stop.setStyleSheet(
            "QPushButton { background:#922b21; color:white; border-radius:4px; padding:5px 20px; }"
            "QPushButton:hover { background:#c0392b; }"
            "QPushButton:disabled { background:#444; color:#666; }"
        )
        self._btn_save = QPushButton("Save Results (.txt)")
        self._btn_save.setEnabled(False)
        self._btn_save.setToolTip(
            "Save the optimised schedule and settings to a .txt file.\n"
            "Available after the optimizer finishes."
        )
        self._btn_save.setStyleSheet(
            "QPushButton { background:#1a4a6a; color:white; border-radius:4px; padding:5px 16px; }"
            "QPushButton:hover { background:#2a6a9a; }"
            "QPushButton:disabled { background:#333; color:#666; }"
        )
        self._btn_save_load = QPushButton("Save && Load in schedule table")
        self._btn_save_load.setEnabled(False)
        self._btn_save_load.setToolTip(
            "Save the optimised schedule and load it straight into the Sequence\n"
            "table (closes this window).  Available after the optimizer finishes."
        )
        self._btn_save_load.setStyleSheet(
            "QPushButton { background:#1e8449; color:white; border-radius:4px; padding:5px 16px; }"
            "QPushButton:hover { background:#27ae60; }"
            "QPushButton:disabled { background:#333; color:#666; }"
        )
        self._btn_run.clicked.connect(self._start)
        self._btn_stop.clicked.connect(self._stop)
        self._btn_save.clicked.connect(self._save_txt)
        self._btn_save_load.clicked.connect(self._save_and_load)

        btn_hl.addWidget(self._btn_run)
        btn_hl.addWidget(self._btn_stop)
        btn_hl.addWidget(self._btn_save)
        btn_hl.addWidget(self._btn_save_load)
        btn_hl.addStretch()
        root.addLayout(btn_hl)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _start(self):
        params = [p for p, cb in self._param_cbs.items() if cb.isChecked()]
        if not params:
            self._log.append("Select at least one parameter to optimise for.")
            return

        # Pre-flight: dictionary size sanity check
        dict_size = 1
        for chk_, _, _, sp_n_ in self._tissue.values():
            if chk_.isChecked():
                dict_size *= sp_n_.value()
        if dict_size > 5000:
            from PyQt6.QtWidgets import QMessageBox
            ans = QMessageBox.question(
                self, "Dictionary too large",
                f"The current settings produce a dictionary with {dict_size:,} entries.\n\n"
                "This will make each CRB evaluation extremely slow and numerically unstable "
                "(results like 3 000%+ nCRLB).\n\n"
                "Recommended: reduce '# Dict points' to 3–5 per swept parameter.\n"
                f"  4 swept params × 4 pts = 4⁴ = 256 entries  ✓\n"
                f"  4 swept params × 5 pts = 5⁴ = 625 entries  ✓\n\n"
                "Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return

        self._log.clear()
        self._log.append("Building config and starting optimizer…")
        self._prog_bar.setValue(0)
        self._status.setText("Running…")
        self._status.setStyleSheet("color: #f0a500; font-size: 11px;")
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._best_rows = None

        self._worker = CRBOptimizerWorker(
            cfg           = self._build_cfg(),
            bounds        = self._build_bounds(),
            n_evals       = self._n_evals.value(),
            params_to_opt = params,
            sigma         = self._sigma.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._log.append("Stop requested — finishing current evaluation…")
            self._btn_stop.setEnabled(False)

    def _on_progress(self, cur: int, total: int, best: float, msg: str):
        self._prog_bar.setValue(int(cur / total * 100))
        self._log.append(msg)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )

    def _on_finished(self, rows: list, crb_pct: float):
        self._best_rows = rows
        self._best_crb  = crb_pct
        self._prog_bar.setValue(100)
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_save.setEnabled(True)
        self._btn_save_load.setEnabled(True)
        n = len(rows)
        self._status.setText(
            f"Done — best mean nCRB = {crb_pct:.2f}%  ({n} measurements)   "
            "→ click  'Save & Load in schedule table'  to use this schedule"
        )
        self._status.setStyleSheet(
            "color: #7ec97e; font-size: 11px; font-weight: bold;"
        )
        self._log.append(
            f"\n✓  Optimisation complete.  Best mean nCRB = {crb_pct:.2f}%\n"
            "Click  'Save & Load in schedule table'  below to use it."
        )
        self.result_ready.emit(rows, crb_pct)

    def _on_error(self, msg: str):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._status.setText("Error — see log for details")
        self._status.setStyleSheet("color: #e57373; font-size: 11px;")
        self._log.append(f"\nERROR:\n{msg}")

    # ── Save results ──────────────────────────────────────────────────────────

    def _save_txt(self):
        """Save the best optimised schedule as a loadable Bruker/pulseq .txt.

        Produces the compact schedule format (num_meas → tab-separated rows →
        column legend) that loads directly into the Sequence table — not the
        verbose on-screen report (that stays in the log).
        """
        if not self._best_rows:
            QMessageBox.warning(self, "No results", "Run the optimizer first.")
            return None

        path, _ = QFileDialog.getSaveFileName(
            self, "Save optimised schedule", "crb_optimized_schedule.txt",
            "Text files (*.txt);;All files (*)"
        )
        if not path:
            return None
        if not path.lower().endswith(".txt"):
            path += ".txt"

        try:
            from my_gui.tabs.sequence_tab import format_schedule_txt
            txt = format_schedule_txt(self._best_rows)
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
            self._log.append(
                f"✓  Schedule saved → {path}\n"
                f"   {len(self._best_rows)} measurements — loadable .txt format."
            )
            return path
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return None

    def _save_and_load(self):
        """Save the optimised schedule .txt, then load it into the Sequence
        table and close the generator dialog."""
        path = self._save_txt()
        if path:
            self.load_and_close.emit(self._best_rows, path)

    # ── Config builders ───────────────────────────────────────────────────────

    def _open_dialog(self, attr_name: str, groupbox, title: str):
        """Lazily wrap a section's QGroupBox in a reusable pop-up dialog and show it."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout
        dlg = getattr(self, attr_name, None)
        if dlg is None:
            dlg = QDialog(self)
            dlg.setWindowTitle(title)
            lay = QVBoxLayout(dlg)
            lay.setContentsMargins(8, 8, 8, 8)
            lay.addWidget(groupbox)        # re-parents the groupbox into the dialog
            setattr(self, attr_name, dlg)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _open_tissue_dialog(self):
        self._open_dialog("_tissue_dialog", self._grp_tissue, "Tissue Parameter Ranges")

    def _open_optset_dialog(self):
        self._open_dialog("_optset_dialog", self._grp_optset, "Optimization Settings")

    def _make_range(self, name: str) -> list[float]:
        chk, sp_min, sp_max, sp_n = self._tissue[name]
        if not chk.isChecked():
            return [float(sp_min.value())]
        return np.linspace(sp_min.value(), sp_max.value(), sp_n.value()).tolist()

    def _build_cfg(self) -> dict:
        b0  = float(self._b0.value())
        n_H = int(self._nh.value())
        dw  = float(self._dw.value())

        t1w = [v / 1000 for v in self._make_range("T1w")]   # ms → s
        t2w = [v / 1000 for v in self._make_range("T2w")]
        ksw = self._make_range("Ksw")                          # Hz = s⁻¹
        fs  = [v * n_H / 110_000 for v in self._make_range("fs")]  # mM → fraction

        cfg: dict = {
            'water_pool': {'t1': t1w, 't2': t2w, 'f': 1},
            'cest_pool':  {
                'CEST': {
                    't1': [max(t1w)],
                    't2': [40 / 1000],     # typical 40 ms
                    'k':  ksw,
                    'dw': dw,
                    'f':  fs,
                }
            },
            'scale':          1,
            'reset_init_mag': 0,
            'b0':             b0,
            'gamma':          267.5153,
            'b0_inhom':       0,
            'rel_b1':         1,
            'verbose':        0,
            'max_pulse_samples': 100,
            'num_workers':    self._workers.value(),
        }

        # MT pool — include only if the user explicitly enables it
        if self._mt_enable.isChecked():
            cfg['mt_pool'] = {
                't1': [1.0],
                't2': [10e-6],
                'k':  self._make_range("Kssw"),
                'dw': 0.0,
                'f':  [v / 110_000 for v in self._make_range("fss")],
                'lineshape': 'SuperLorentzian',
            }

        return cfg

    def _build_bounds(self) -> dict:
        off_text = self._off_edit.toPlainText().strip()
        off_vals = []
        for tok in re.split(r'[,\s]+', off_text):
            try:
                off_vals.append(float(tok))
            except ValueError:
                pass
        if not off_vals:
            off_vals = [3.0]

        return {
            'niter':         self._niter.value(),
            'b1_min':        self._b1_lo.value(),
            'b1_max':        self._b1_hi.value(),
            'tsat_min':      self._tsat_lo.value(),
            'tsat_max':      self._tsat_hi.value(),
            'tr_min':        self._tr_lo.value(),
            'tr_max':        self._tr_hi.value(),
            'offset_values': off_vals,
            'vary_offsets':  self._off_vary.isChecked(),
            'vary_tsat':     self._tsat_vary.isChecked(),
            'vary_tr':       self._tr_vary.isChecked(),
        }

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def best_rows(self) -> list | None:
        return self._best_rows

    @property
    def best_crb(self) -> float | None:
        return self._best_crb
