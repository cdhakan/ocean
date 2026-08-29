"""
sequence_tab.py
Pulse-sequence schedule editor + 4-panel MATLAB-style visualisation.

Layout:
  Left  — toolbar / filename label / table editor
  Right — 4-panel schedule figure (B1, Offset, Tsat, TR/Td)
          + export button

The visualisation replicates the MATLAB MRF schedule figure exactly:
  Panel 1 : B₁ (µT)
  Panel 2 : Ω  (ppm)
  Panel 3 : Tsat (ms)
  Panel 4 : Td / TR (ms)
  Shared x-axis, blue lines, dark background, title = loaded filename.
"""

from __future__ import annotations
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTableWidget, QTableWidgetItem, QPushButton,
    QLabel, QHeaderView, QFileDialog, QMessageBox,
    QSizePolicy, QFrame, QDialog, QTextEdit, QLineEdit,
    QSpinBox, QDoubleSpinBox, QGridLayout, QScrollArea,
    QTabWidget, QApplication, QProgressBar, QCheckBox,
    QComboBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QEvent

from my_gui.tabs.crb_optimizer_widget import CRBOptimizerWidget
from my_gui.fig_theme import apply_fig_dark_theme

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ─────────────────────────────────────────────────────────────────────────────
# Column definitions
# ─────────────────────────────────────────────────────────────────────────────

COLUMNS = [
    "TR (ms)", "Ampl (µT)", "Offset (ppm)", "Excitation FA (deg)",
    "Sat/Lock time (ms)", "SL=1 / Sat=0", "SL prep FA (deg)",
]

_COL_TR    = 0
_COL_B1    = 1
_COL_OFF   = 2
_COL_FA    = 3
_COL_TSAT  = 4
_COL_ISLSL = 5
_COL_SLFA  = 6


# ─────────────────────────────────────────────────────────────────────────────
# .txt parser  (mirrors MATLAB logic exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_txt(filepath: str) -> list[list]:
    """
    Parse a Bruker MRF schedule .txt file.

    Rules (matching the MATLAB textscan / sscanf logic):
      - Blank lines and lines starting with '#' are skipped
      - Lines with a single integer token are treated as a row-count
        header and skipped
      - Rows with >= 6 numeric tokens are accepted; if < 7 tokens the
        7th value (SL prep FA) defaults to 0  ← matches MATLAB
        'if numel(nums) < 7, nums(7) = 0; end'
      - Returns list of 7-element float lists
    """
    rows: list[list] = []
    with open(filepath, "r") as fh:
        lines = fh.readlines()

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # Single-token integer header line → skip
        if len(parts) == 1:
            try:
                int(parts[0])
                continue
            except ValueError:
                pass
        # Parse numeric tokens (sscanf behaviour: stop at first non-numeric)
        nums: list[float] = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                break
        if len(nums) < 6:
            continue                    # not enough values — skip
        if len(nums) < 7:
            nums.append(0.0)            # pad SL prep FA to 0
        rows.append(nums[:7])

    if not rows:
        raise ValueError("No valid data rows found in file.")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Schedule .txt writer  (shared by manual generator + CRB optimizer)
# ─────────────────────────────────────────────────────────────────────────────

_SCHEDULE_TXT_FOOTER = (
    "TR[ms] |\tAmpl[uT] |\tOffset[ppm] |\t"
    "Excit FA[deg] |\tSat/lock time[ms] |\t"
    "SL[1] or sat[0] |\tSL prep FA[deg]"
)


def format_schedule_txt(rows: list[list]) -> str:
    """
    Format schedule rows as the loadable Bruker/pulseq .txt:

        <num_meas>
        TR  B1  Offset  ExcFA  Tsat  SL/sat  SLprepFA      (tab-separated)
        ...
        TR[ms] | Ampl[uT] | …                              (column legend)

    Each row is [TR(ms), B1(µT), Offset(ppm), ExcFA(°), Tsat(ms), SL(0/1),
    SLprepFA(°)].  Columns are tab-separated; TR/Tsat are integers, B1 is
    right-aligned to 1 decimal, offset to 2 decimals.
    """
    lines = [str(len(rows))]
    for r in rows:
        tr, b1, off, fa, tsat, issl, slfa = (list(r) + [0.0] * 7)[:7]
        lines.append(
            f"{tr:.0f}\t{b1:4.1f}\t{off:.2f}\t{fa:.1f}\t"
            f"{tsat:.0f}\t{int(round(issl))}\t{slfa:.1f}"
        )
    lines.append(_SCHEDULE_TXT_FOOTER)
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Value parser for the schedule generator
# ─────────────────────────────────────────────────────────────────────────────

def _parse_values(text: str, niter: int) -> list[float]:
    """
    Parse user-entered text as either:
      - A single scalar  → replicated niter times
      - An array of exactly niter values

    Accepted separators: comma, semicolon, whitespace, newlines.
    Raises ValueError with a human-readable message on any error.
    """
    tokens = [t for t in re.split(r"[,;\s]+", text.strip()) if t]
    if not tokens:
        raise ValueError("No values entered.")
    if len(tokens) == 1:
        try:
            return [float(tokens[0])] * niter
        except ValueError:
            raise ValueError(f"Cannot parse '{tokens[0]}' as a number.")
    elif len(tokens) == niter:
        result: list[float] = []
        for t in tokens:
            try:
                result.append(float(t))
            except ValueError:
                raise ValueError(f"Cannot parse '{t}' as a number.")
        return result
    else:
        raise ValueError(
            f"Expected 1 value (uniform) or {niter} values (array), "
            f"got {len(tokens)}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Schedule Generator Dialog
# ─────────────────────────────────────────────────────────────────────────────

class ScheduleGeneratorDialog(QDialog):
    """
    Two-tab dialog for creating an MRF schedule:

    Tab 1 — Manual Parameters
        Enter TR, B1, offsets, Tsat, etc. as scalar or per-measurement arrays.
        Generates a MATLAB-compatible .txt file and optionally loads it into
        the Sequence tab table.

    Tab 2 — CRB Optimizer
        Specify tissue parameter ranges (T1w, T2w, Ksw, fs, …) and schedule
        search bounds (B1 min/max, Tsat range, …).  Runs a random-search
        optimiser that minimises the Cramér-Rao Lower Bound (CRB) — the
        theoretical minimum estimation error — over the parameter space.
        Returns the best schedule found.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CEST-MRF Schedule  —  Generator & CRB Optimiser")
        self.setMinimumWidth(720)
        self.setMinimumHeight(500)
        # Size to 90 % of available screen height so it never overflows
        screen = QApplication.primaryScreen() if QApplication.primaryScreen() else None
        if screen:
            ag = screen.availableGeometry()
            self.resize(min(820, int(ag.width() * 0.65)),
                        min(720, int(ag.height() * 0.88)))

        # Shared result — set by whichever tab produces a schedule
        self.generated_rows: list[list] | None = None
        self.saved_filepath:  str | None = None

        self._setup_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        main = QVBoxLayout(self)
        main.setSpacing(6)
        main.setContentsMargins(12, 12, 12, 12)

        # Title
        title = QLabel("CEST-MRF Schedule  —  Generator & CRB Optimiser")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #5dade2; padding: 4px;"
        )
        main.addWidget(title)

        # ── Tab widget ────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            "QTabBar::tab { padding: 6px 20px; min-width: 160px; }"
            "QTabBar::tab:selected { background: #1565c0; color: white; font-weight: bold; }"
        )

        # Tab 1: Manual
        self._manual_widget = _ManualScheduleWidget()
        self._manual_widget.schedule_ready.connect(self._on_manual_ready)
        self._manual_widget.load_and_close.connect(self._on_load_and_close)
        self._tabs.addTab(self._manual_widget, "Manual Parameters")

        # Tab 2: CRB Optimizer
        self._crb_widget = CRBOptimizerWidget()
        self._crb_widget.result_ready.connect(self._on_crb_ready)
        self._crb_widget.load_and_close.connect(self._on_load_and_close)
        self._tabs.addTab(self._crb_widget, "CRB Optimizer")

        main.addWidget(self._tabs, stretch=1)

        # ── Shared result status ──────────────────────────────────────────────
        self._result_lbl = QLabel("No schedule ready yet.")
        self._result_lbl.setStyleSheet("color: #888; font-size: 11px; padding: 2px 0;")
        self._result_lbl.setWordWrap(True)
        main.addWidget(self._result_lbl)

        # ── Bottom buttons ────────────────────────────────────────────────────
        # Loading into the table is handled per-tab by the "Save & Load in
        # schedule table" button, which fires load_and_close → self.accept().
        btn_hl = QHBoxLayout()
        btn_hl.setSpacing(8)

        btn_close = QPushButton("Close")
        btn_close.setStyleSheet(
            "QPushButton { border-radius: 4px; padding: 6px 14px; }"
        )
        btn_close.clicked.connect(self.reject)

        btn_hl.addStretch()
        btn_hl.addWidget(btn_close)
        main.addLayout(btn_hl)

    # ─────────────────────────────────────────────────────────────────────────
    # Slots from inner widgets
    # ─────────────────────────────────────────────────────────────────────────

    def _on_manual_ready(self, rows: list, filepath: str):
        """Called when the Manual tab saves a schedule (Save button)."""
        self.generated_rows = rows
        self.saved_filepath  = filepath
        n = len(rows)
        self._result_lbl.setText(
            f"Manual schedule ready — {n} measurements  |  {filepath or 'unsaved'}"
        )
        self._result_lbl.setStyleSheet("color: #f0a500; font-size: 11px;")

    def _on_crb_ready(self, rows: list, crb_pct: float):
        """Called when the CRB Optimizer finds a best schedule."""
        self.generated_rows = rows
        self.saved_filepath  = None
        n = len(rows)
        self._result_lbl.setText(
            f"CRB-optimised schedule ready — {n} measurements  |  "
            f"best mean nCRB = {crb_pct:.2f}%"
        )
        self._result_lbl.setStyleSheet(
            "color: #7ec97e; font-size: 11px; font-weight: bold;"
        )

    def _on_load_and_close(self, rows: list, filepath: str):
        """A tab asked to load its schedule into the table and close the dialog."""
        if not rows:
            return
        self.generated_rows = rows
        self.saved_filepath  = filepath or None
        self.accept()


# ─────────────────────────────────────────────────────────────────────────────
# Manual schedule widget  (Tab 1 content)
# ─────────────────────────────────────────────────────────────────────────────

class _ManualScheduleWidget(QWidget):
    """
    Form-based schedule generator.
    Emits schedule_ready(rows, saved_filepath) when the user saves a schedule.
    Emits load_and_close(rows, saved_filepath) when the user asks to save AND
    load the schedule straight into the Sequence table (closing the dialog).
    """

    schedule_ready = pyqtSignal(list, str)
    load_and_close = pyqtSignal(list, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        main = QVBoxLayout(self)
        main.setSpacing(8)
        main.setContentsMargins(8, 8, 8, 8)

        hint = QLabel(
            "Enter a single number for a uniform value, or paste a column vector / "
            "comma-separated list for a per-measurement array.  "
            "Arrays must contain exactly niter values."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #aaa; font-size: 11px; padding: 0 0 4px 0;")
        main.addWidget(hint)

        # ── Filename + niter ──────────────────────────────────────────────────
        top = QHBoxLayout()
        top.addWidget(QLabel("Filename root:"))
        self.edit_fname = QLineEdit("mrf_schedule")
        self.edit_fname.setPlaceholderText("e.g.  unsupervised_MT")
        top.addWidget(self.edit_fname, stretch=2)
        top.addSpacing(16)
        top.addWidget(QLabel("Iterations (niter):"))
        self.spin_niter = QSpinBox()
        self.spin_niter.setRange(1, 2000)
        self.spin_niter.setValue(40)
        self.spin_niter.setFixedWidth(80)
        top.addWidget(self.spin_niter)
        main.addLayout(top)

        # ── Parameter grid ────────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(6)
        grid.setContentsMargins(2, 2, 2, 2)

        for ci, text in enumerate([
            "Parameter", "Unit",
            "Values  —  scalar (uniform)  or  array  (one per line / comma-separated)",
        ]):
            h = QLabel(text)
            h.setStyleSheet(
                "font-weight: bold; color: #ccc; "
                "border-bottom: 1px solid #555; padding-bottom: 3px;"
            )
            grid.addWidget(h, 0, ci)

        _PARAMS = [
            ("tr",     "TR",            "ms",
             "5000\n\n— or paste array:\n3900\n4900\n5800\n…",
             "Repetition time (ms).  Single value = uniform; array = per-measurement."),
            ("b1",     "B1 amplitude",  "µT",
             "1.5\n\n— or paste array:\n1.0\n1.3\n1.7\n…",
             "Saturation/spin-lock amplitude (µT)."),
            ("offset", "Offsets",       "ppm",
             "3.0\n\n— or paste array:\n8\n8\n9\n9\n…",
             "Saturation frequency offset (ppm).  Use ~100 ppm for M0 reference."),
            ("excfa",  "Excitation FA", "deg",
             "90.0",
             "Excitation flip angle before readout (°).  Typically 90°."),
            ("tsat",   "Tsat",          "ms",
             "1000\n\n— or paste array:\n400\n900\n1300\n…",
             "Saturation/spin-lock pulse duration (ms)."),
            ("issl",   "Sat=0 / SL=1",  "",
             "0\n\n(0 = CW saturation,  1 = spin-lock)\n— or paste array: 0\n0\n1\n1\n…",
             "Pulse type per measurement.  0 = saturation,  1 = spin-lock."),
            ("slfa",   "SL prep FA",    "deg",
             "0.0\n\n(0 = auto-calculate 90° for SL rows)",
             "Pre/post spin-lock preparation flip angle.  0 = auto."),
        ]

        self._param_edits: dict[str, QTextEdit] = {}
        for row_i, (attr, label, unit, placeholder, tooltip) in enumerate(_PARAMS, start=1):
            lbl = QLabel(label); lbl.setToolTip(tooltip); lbl.setStyleSheet("padding: 2px 4px;")
            grid.addWidget(lbl, row_i, 0)

            u = QLabel(unit); u.setStyleSheet("color: #888; font-size: 11px;")
            u.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(u, row_i, 1)

            te = QTextEdit()
            te.setPlaceholderText(placeholder)
            te.setToolTip(tooltip)
            te.setFixedHeight(60)
            te.setStyleSheet(
                "QTextEdit { font-family: Menlo, Consolas, 'DejaVu Sans Mono', 'Courier New'; font-size: 11px; "
                "background: #1e1e1e; color: #e0e0e0; border: 1px solid #444; "
                "border-radius: 3px; padding: 3px 6px; }"
                "QTextEdit:focus { border-color: #1565c0; }"
            )
            grid.addWidget(te, row_i, 2)
            self._param_edits[attr] = te

        grid.setColumnStretch(2, 1)
        grid.setColumnMinimumWidth(0, 120)
        grid.setColumnMinimumWidth(1, 38)
        scroll.setWidget(container)
        main.addWidget(scroll, stretch=1)

        # ── Preview ───────────────────────────────────────────────────────────
        prev_lbl = QLabel("File preview (first rows + annotation):")
        prev_lbl.setStyleSheet("color: #888; font-size: 11px; margin-top: 2px;")
        main.addWidget(prev_lbl)

        self.preview_box = QTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setMinimumHeight(80)
        self.preview_box.setMaximumHeight(140)
        self.preview_box.setPlaceholderText(
            "Click  Preview  to validate and see how the .txt file will look…"
        )
        self.preview_box.setStyleSheet(
            "QTextEdit { font-family: Menlo, Consolas, 'DejaVu Sans Mono', 'Courier New'; font-size: 10px; "
            "background: #111; color: #7ec97e; border: 1px solid #444; border-radius: 3px; }"
        )
        main.addWidget(self.preview_box)

        # ── Action buttons ────────────────────────────────────────────────────
        # Step 1: Generate (validate + preview).  Once a valid schedule exists,
        # the two save buttons enable: Save (write .txt) and Save & Load (write
        # .txt + load into the Sequence table, closing the dialog).
        btn_hl = QHBoxLayout()
        btn_hl.setSpacing(8)
        btn_gen             = QPushButton("Generate")
        self._btn_save      = QPushButton("Save")
        self._btn_save_load = QPushButton("Save && Load in schedule table")
        self._btn_save.setEnabled(False)
        self._btn_save_load.setEnabled(False)

        btn_gen.setStyleSheet(
            "QPushButton { background:#6a1b9a; color:white; border-radius:4px; padding:5px 18px; font-weight:bold; }"
            "QPushButton:hover { background:#8e24aa; }"
        )
        self._btn_save.setStyleSheet(
            "QPushButton { background:#1a5276; color:white; border-radius:4px; padding:5px 14px; }"
            "QPushButton:hover { background:#2471a3; }"
            "QPushButton:disabled { background:#333; color:#666; }"
        )
        self._btn_save_load.setStyleSheet(
            "QPushButton { background:#1e8449; color:white; border-radius:4px; padding:5px 14px; }"
            "QPushButton:hover { background:#27ae60; }"
            "QPushButton:disabled { background:#333; color:#666; }"
        )

        btn_gen.clicked.connect(self._do_generate)
        self._btn_save.clicked.connect(lambda: self._do_save(load=False))
        self._btn_save_load.clicked.connect(lambda: self._do_save(load=True))

        btn_hl.addWidget(btn_gen)
        btn_hl.addStretch()
        btn_hl.addWidget(self._btn_save)
        btn_hl.addWidget(self._btn_save_load)
        main.addLayout(btn_hl)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _read_param(self, attr: str, niter: int) -> list[float]:
        text = self._param_edits[attr].toPlainText().strip()
        if not text:
            raise ValueError(f"'{attr}' field is empty.")
        return _parse_values(text, niter)

    def _build_rows(self) -> tuple[list[list], int]:
        niter  = self.spin_niter.value()
        errors: list[str] = []
        results: dict[str, list[float]] = {}
        labels = {
            "tr": "TR", "b1": "B1 amplitude", "offset": "Offsets",
            "excfa": "Excitation FA", "tsat": "Tsat",
            "issl": "Sat/SL flag", "slfa": "SL prep FA",
        }
        for attr in ("tr", "b1", "offset", "excfa", "tsat", "issl", "slfa"):
            try:
                results[attr] = self._read_param(attr, niter)
            except ValueError as e:
                errors.append(f"• {labels[attr]}: {e}")
        if errors:
            raise ValueError("\n".join(errors))
        rows: list[list] = []
        for i in range(niter):
            rows.append([
                results["tr"][i], results["b1"][i], results["offset"][i],
                results["excfa"][i], results["tsat"][i],
                int(round(results["issl"][i])), results["slfa"][i],
            ])
        return rows, niter

    def _format_txt(self, rows: list[list]) -> str:
        return format_schedule_txt(rows)

    def _render_preview(self, rows: list[list], niter: int):
        """Show the generated schedule in the preview box (green = OK)."""
        self.preview_box.setStyleSheet(
            "QTextEdit { font-family: Menlo, Consolas, 'DejaVu Sans Mono', 'Courier New'; font-size: 10px; "
            "background: #111; color: #7ec97e; border: 1px solid #444; border-radius: 3px; }"
        )
        all_lines = self._format_txt(rows).strip().split("\n")
        n_show = min(8, niter)
        preview = [f"── {niter} measurements ──"] + all_lines[:1 + n_show]
        if niter > n_show:
            preview.append(f"  … ({niter - n_show} more rows) …")
        preview += all_lines[-1:]
        self.preview_box.setPlainText("\n".join(preview))

    def _do_generate(self):
        """Step 1 — validate the fields, build the schedule, and preview it.
        On success the Save / Save & Load buttons are enabled."""
        try:
            rows, niter = self._build_rows()
        except ValueError as e:
            self.preview_box.setStyleSheet(
                "QTextEdit { font-family: Menlo, Consolas, 'DejaVu Sans Mono', 'Courier New'; font-size: 10px; "
                "background: #1a0000; color: #e57373; border: 1px solid #555; border-radius: 3px; }"
            )
            self.preview_box.setPlainText(f"Validation errors:\n{e}")
            self._btn_save.setEnabled(False)
            self._btn_save_load.setEnabled(False)
            return
        self._render_preview(rows, niter)
        self._btn_save.setEnabled(True)
        self._btn_save_load.setEnabled(True)

    def _do_save(self, *, load: bool):
        """Step 2 — write the schedule .txt.  When load=True, also emit
        load_and_close so the dialog loads it into the table and closes."""
        try:
            rows, niter = self._build_rows()
        except ValueError as e:
            QMessageBox.warning(self, "Input error", str(e))
            return
        fname_root   = self.edit_fname.text().strip() or "mrf_schedule"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save MRF schedule", fname_root + ".txt",
            "Text files (*.txt);;All files (*)"
        )
        if not save_path:
            return
        try:
            Path(save_path).write_text(self._format_txt(rows), encoding="utf-8")
        except OSError as e:
            QMessageBox.critical(self, "Save error", str(e))
            return

        if load:
            self.load_and_close.emit(rows, save_path)
        else:
            QMessageBox.information(
                self, "Saved",
                f"Schedule saved:\n{save_path}\n\n{niter} measurements."
            )
            self.schedule_ready.emit(rows, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# PyPulseq sequence-generation worker  (runs write_sequence_sl off the UI thread)
# ─────────────────────────────────────────────────────────────────────────────

class _SeqGenWorker(QThread):
    """
    Calls write_sequence_sl in a background thread so the UI stays responsive
    while the .seq file is being assembled.

    Signals
    -------
    finished(object)  — the PyPulseq Sequence object on success
    error(str)        — full traceback string on failure
    """

    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, seq_defs: dict, seq_fn: str, use_vendored: bool = True,
                 parent=None):
        super().__init__(parent)
        self._seq_defs      = seq_defs
        self._seq_fn        = seq_fn
        self._use_vendored  = use_vendored   # True → write with vendored 1.3.1

    def run(self):
        import contextlib
        import traceback
        # Make sure both the project root and the CEST library are importable.
        _root = Path(__file__).resolve().parent.parent
        _cest = _root / "open-py-cest-mrf"
        for p in [str(_root), str(_cest)]:
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            from sequences_sl import write_sequence_sl  # noqa: PLC0415
            # Route to the requested pypulseq: 1.3.1 (vendored, simulator format)
            # or 1.5 (whatever is installed).  write_sequence_sl resolves
            # `import pypulseq` at call time, so the context takes effect.
            if self._use_vendored:
                from my_gui.worker import _vendored_pypulseq
                _ctx = _vendored_pypulseq()
            else:
                _ctx = contextlib.nullcontext()
            with _ctx:
                seq = write_sequence_sl(seq_defs=self._seq_defs, seq_fn=self._seq_fn)
            self.finished.emit(seq)
        except Exception:
            self.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Pulse-sequence viewer dialog
# ─────────────────────────────────────────────────────────────────────────────

class PulseSeqViewerDialog(QDialog):
    """
    Modal dialog that shows the PyPulseq timing diagram produced by
    write_sequence_sl for the current MRF schedule.

    seq.plot() normally calls plt.show(); we suppress that call and instead
    capture every matplotlib Figure it creates, then embed each one inside a
    FigureCanvasQTAgg in a vertically-scrollable area.

    Two figures are produced:
      Figure 1 — RF magnitude, RF phase, ADC events
      Figure 2 — Gradient channels Gx / Gy / Gz
                  (near-zero for a CW-CEST sequence — informational only)
    """

    def __init__(self, seq, parent=None, title: str = "Current Schedule",
                 pp_version: str = None):
        super().__init__(parent)
        self._pp_version = pp_version
        self.setWindowTitle(f"Pulse Sequence Viewer  —  {title}")
        self.setMinimumWidth(820)
        self.setMinimumHeight(480)
        screen = QApplication.primaryScreen()
        if screen:
            ag = screen.availableGeometry()
            self.resize(min(980, int(ag.width()  * 0.72)),
                        min(840, int(ag.height() * 0.88)))

        self._figs: list = []
        self._canvases: list = []
        self._setup_ui()
        self._render(seq)

    # ── UI skeleton ───────────────────────────────────────────────────────────

    def _setup_ui(self):
        # White background for the whole viewer (figures + chrome), as requested.
        self.setStyleSheet("QDialog { background: #ffffff; }")
        vbox = QVBoxLayout(self)
        vbox.setSpacing(6)
        vbox.setContentsMargins(10, 10, 10, 10)

        # Header
        hdr = QLabel("PyPulseq Sequence Diagram")
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #1565c0; "
            "background: transparent; padding: 4px;"
        )
        vbox.addWidget(hdr)

        # Status / progress
        self._status = QLabel("Rendering diagram…")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("font-size: 11px; color: #b26a00; background: transparent; padding: 2px;")
        vbox.addWidget(self._status)

        # Scrollable figure area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(
            "QScrollArea { border: none; background: #ffffff; }")
        self._inner = QWidget()
        self._inner.setStyleSheet("background: #ffffff;")
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.setSpacing(6)
        self._inner_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll.setWidget(self._inner)
        vbox.addWidget(self._scroll, stretch=1)

        # Bottom button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._btn_save = QPushButton("Save PNG…")
        self._btn_save.setEnabled(False)
        self._btn_save.setStyleSheet(
            "QPushButton { background:#1a5276; color:white; "
            "border-radius:4px; padding:5px 14px; }"
            "QPushButton:hover { background:#2471a3; }"
            "QPushButton:disabled { background:#333; color:#666; }"
        )
        self._btn_save.clicked.connect(self._save_png)

        btn_close = QPushButton("Close")
        btn_close.setStyleSheet(
            "QPushButton { border-radius:4px; padding:5px 14px; }"
        )
        btn_close.clicked.connect(self.accept)

        self.chk_dark = QCheckBox("Bg")
        self.chk_dark.setToolTip(
            "Black background for the figures (for slides). Only the white "
            "surround and labels flip - the plots stay identical."
        )
        self.chk_dark.toggled.connect(self._apply_dark_bg)

        btn_row.addWidget(self._btn_save)
        btn_row.addWidget(self.chk_dark)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        vbox.addLayout(btn_row)

    def _apply_dark_bg(self):
        """(Re)theme every rendered figure to match the 'Bg' checkbox state."""
        for fig, cv in zip(self._figs, self._canvases):
            apply_fig_dark_theme(fig, self.chk_dark.isChecked())
            cv.draw()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self, seq):
        """
        Call seq.plot(), intercept the figures it creates, apply dark styling,
        add a NavigationToolbar per figure (zoom · pan · hover coordinates),
        handle near-zero gradient channels, and embed each canvas in the scroll area.
        """
        import matplotlib.pyplot as plt

        # Navigation toolbar — zoom, pan, home, and live x/y coordinate readout.
        _NavToolbar = None
        try:
            from matplotlib.backends.backend_qtagg import \
                NavigationToolbar2QT as _NavToolbar          # matplotlib ≥ 3.6
        except ImportError:
            try:
                from matplotlib.backends.backend_qt import \
                    NavigationToolbar2QT as _NavToolbar      # fallback
            except ImportError:
                pass  # no toolbar — still fully functional, just no zoom/pan

        _TOOLBAR_SS = (
            "QToolBar { background:#f0f0f0; border:1px solid #d5d5d5; spacing:4px; }"
            "QToolButton { color:#222222; background:transparent; border:none;"
            "              padding:2px 6px; border-radius:3px; }"
            "QToolButton:hover { background:#dde3ee; }"
            "QLabel { color:#444444; font-size:10px; }"
        )

        # pypulseq reuses fixed figure numbers (1, 2), so a previous "View Pulse
        # Sequence" leaves them registered and seq.plot() then creates NO new
        # figures — nothing is captured and the viewer looks empty. Close all
        # pyplot figures first so this render's figures (1, 2) are always new.
        plt.close("all")
        figs_before = set(plt.get_fignums())

        # Silence plt.show() — Agg already ignores it, but be explicit.
        _orig_show = plt.show
        plt.show = lambda *a, **k: None
        try:
            seq.plot(time_disp="ms")
        except Exception as exc:
            self._status.setText(f"Plot error: {exc}")
            self._status.setStyleSheet("font-size:11px; color:#ef5350;")
            return
        finally:
            plt.show = _orig_show

        # Determine which figures seq.plot() created.
        new_nums = sorted(set(plt.get_fignums()) - figs_before)

        if not new_nums:
            self._status.setText(
                "seq.plot() created no figures.  "
                "Check that the schedule has at least one measurement."
            )
            self._status.setStyleSheet("font-size:11px; color:#ef5350;")
            return

        panel_labels = [
            "Figure 1  —  RF Pulses  (amplitude, phase)  &  ADC readout events",
            "Figure 2  —  Gradient waveforms  (Gx, Gy, Gz)",
        ]

        for i, fn in enumerate(new_nums):
            fig = plt.figure(fn)
            n_axes = len(fig.get_axes())

            # ── White styling ──────────────────────────────────────────────────
            fig.set_facecolor("white")
            for ax in fig.get_axes():
                ax.set_facecolor("white")
                ax.tick_params(colors="#222222", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333333")
                ax.xaxis.label.set_color("#222222")
                ax.yaxis.label.set_color("#222222")
                if ax.get_title():
                    ax.set_title(ax.get_title(), color="#111111", fontsize=9)

                # ── Gradient figure: make near-zero channels readable ──────────
                if i == 1:
                    # Measure max absolute value across all plotted lines
                    max_abs = 0.0
                    for ln in ax.get_lines():
                        yd = np.asarray(ln.get_ydata(), dtype=float)
                        if yd.size > 0:
                            max_abs = max(max_abs, float(np.nanmax(np.abs(yd))))

                    if max_abs < 1e-3:
                        # Near-zero channel — force a visible symmetric y-range
                        # and add an informational annotation
                        ax.set_ylim(-0.5, 0.5)
                        ax.axhline(0, color="#999999", linewidth=0.6,
                                   linestyle="--", zorder=0)
                        ax.text(
                            0.02, 0.80,
                            "≈ 0  (near-zero — CW-CEST uses no slice gradients)",
                            transform=ax.transAxes,
                            color="#666666", fontsize=8, style="italic",
                            va="center",
                        )
                    else:
                        # Real gradient data — ensure the y-range isn't collapsed
                        ylo, yhi = ax.get_ylim()
                        if abs(yhi - ylo) < 1e-9:
                            ax.set_ylim(ylo - 0.5, yhi + 0.5)

            try:
                fig.tight_layout(pad=1.2)
            except Exception:
                pass

            self._figs.append(fig)

            # ── Section label ──────────────────────────────────────────────────
            lbl_text = panel_labels[i] if i < len(panel_labels) else f"Figure {i + 1}"
            lbl = QLabel(lbl_text)
            lbl.setStyleSheet(
                "font-size:11px; font-weight:bold; color:#64b5f6; padding:4px 2px 0 2px;"
            )
            self._inner_layout.addWidget(lbl)

            # ── Canvas ────────────────────────────────────────────────────────
            canvas = FigureCanvas(fig)
            # Scale canvas height to number of sub-panels so nothing is squished
            canvas.setMinimumHeight(max(320, n_axes * 140))
            canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            # The canvas swallows wheel events; forward them so the mouse wheel
            # scrolls the diagram (previously scrolling did nothing over a figure).
            canvas.installEventFilter(self)

            # ── Navigation toolbar (zoom · pan · home · hover coords) ─────────
            if _NavToolbar is not None:
                toolbar = _NavToolbar(canvas, self)
                toolbar.setStyleSheet(_TOOLBAR_SS)
                self._inner_layout.addWidget(toolbar)

            self._canvases.append(canvas)
            self._inner_layout.addWidget(canvas)
            canvas.draw()

        self._status.setText("")               # no "ready" banner — figures speak
        self._status.setVisible(False)
        self._btn_save.setEnabled(True)

        # Honour the 'Bg' checkbox for the freshly-rendered figures.
        self._apply_dark_bg()

    def eventFilter(self, obj, event):
        """Forward mouse-wheel events from the matplotlib canvases (which would
        otherwise swallow them) to the scroll area, so the wheel scrolls the
        diagram view."""
        if event.type() == QEvent.Type.Wheel and getattr(self, "_scroll", None):
            sb = self._scroll.verticalScrollBar()
            sb.setValue(sb.value() - int(event.angleDelta().y()))
            return True
        return super().eventFilter(obj, event)

    # ── Export ────────────────────────────────────────────────────────────────

    def _save_png(self):
        from my_gui.fig_export import save_figure, FIG_EXPORT_FILTER
        if not self._figs:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save sequence diagram", "pulse_sequence",
            FIG_EXPORT_FILTER
        )
        if not path:
            return
        stem = Path(path).stem
        ext  = Path(path).suffix or ".png"
        saved = []
        for i, fig in enumerate(self._figs):
            suffix = f"_{i + 1}" if len(self._figs) > 1 else ""
            fn = str(Path(path).parent / f"{stem}{suffix}{ext}")
            save_figure(fig, fn, dpi=300, facecolor=fig.get_facecolor())
            saved.append(fn)
        QMessageBox.information(
            self, "Saved",
            f"Saved {len(saved)} file(s):\n" + "\n".join(saved)
        )


# ─────────────────────────────────────────────────────────────────────────────
# SequenceTab
# ─────────────────────────────────────────────────────────────────────────────

class SequenceTab(QWidget):
    def __init__(self):
        super().__init__()
        self._loaded_fname: str = ""
        self._dict_tab_ref = None        # injected by CestMrfTab.set_dict_tab()
        self._sim_dialog = None          # lazily created QDialog

        splitter = QSplitter(Qt.Orientation.Vertical)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

        # ── LEFT: toolbar + filename label + table ────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(4)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        btn_load     = QPushButton("Load .txt")
        btn_generate = QPushButton("Generate MRF schedule…")
        btn_add      = QPushButton("+ Row")
        btn_remove   = QPushButton("− Row")
        btn_clear    = QPushButton("Clear table")
        self.lbl_count = QLabel()

        # ── View Pulse Sequence button + B0 field ──────────────────────────
        self._btn_view_seq = QPushButton("View Pulse Sequence")
        self._btn_view_seq.setEnabled(False)          # enabled once rows exist
        self._btn_view_seq.setToolTip(
            "Generate the PyPulseq .seq file from the current schedule and display\n"
            "the timing diagram (RF amplitude, phase, ADC events, gradient channels)."
        )
        self._btn_view_seq.setStyleSheet(
            "QPushButton{background:#6a1b9a;color:white;border-radius:3px;"
            "padding:2px 10px;font-weight:bold;}"
            "QPushButton:hover{background:#8e24aa;}"
            "QPushButton:disabled{background:#333;color:#666;}"
        )

        # ── Pulseq Simulation — open the example-sequence simulation window ──
        self._btn_pulseq_sim = QPushButton("Pulseq Simulation")
        self._btn_pulseq_sim.setToolTip(
            "Open the Pulseq Simulation window: load an example sequence from the\n"
            "pulseq-cest-library, edit scanner limits + a tissue-parameter sweep,\n"
            "view the sequence diagram, and run a Bloch–McConnell dictionary\n"
            "simulation + dot-product matching to see how the fingerprint changes."
        )
        self._btn_pulseq_sim.setStyleSheet(
            "QPushButton{background:#1565c0;color:white;border-radius:3px;"
            "padding:2px 10px;font-weight:bold;}"
            "QPushButton:hover{background:#1976d2;}"
        )

        # B0 is now requested in a small dialog when the user clicks View Pulse
        # Sequence (see _view_pulse_sequence); the last entry is remembered here.
        self._b0_value = 9.4

        btn_generate.setToolTip(
            "Open the MRF schedule generator to define TR, B1, offsets, Tsat, etc.\n"
            "and export them as a .txt file."
        )
        btn_generate.setStyleSheet(
            "QPushButton{background:#1e6b3a;color:white;border-radius:3px;padding:2px 10px;font-weight:bold;}"
            "QPushButton:hover{background:#27ae60;}"
        )

        btn_load.clicked.connect(self._load_txt)
        btn_generate.clicked.connect(self._open_generator)
        self._btn_view_seq.clicked.connect(self._view_pulse_sequence)
        self._btn_pulseq_sim.clicked.connect(self._open_pulseq_simulation)
        btn_add.clicked.connect(self._add_row)
        btn_remove.clicked.connect(self._remove_row)
        btn_clear.clicked.connect(self._clear_table)

        toolbar.addWidget(btn_load)
        toolbar.addWidget(btn_generate)
        # Thin separator before the pulseq viewer group
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #444;")
        toolbar.addWidget(sep)
        # View Pulse Sequence always writes the 1.3.1 simulator format (no version
        # choice here); the 1.3.1 / 1.5 selector lives in the Pulseq Simulation window.
        toolbar.addWidget(self._btn_view_seq)
        toolbar.addWidget(self._btn_pulseq_sim)
        toolbar.addSpacing(8)
        # ── Thin separator before the simulation button ───────────────────
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.VLine)
        sep3.setStyleSheet("color: #444;")
        toolbar.addWidget(sep3)

        # ── Simulation button (shown once dict_tab is injected) ───────────
        # Placed inline in the toolbar next to Clear Table.
        # No fixed height — size is driven by font + padding so text never clips.
        self._btn_sim = QPushButton("Generate CEST-MRF Dictionary\nSimulation && Matching")
        self._btn_sim.setToolTip(
            "Open the Dictionary Simulation & Matching dialog.\n"
            "Load acquired data, generate dictionary and run voxelwise matching."
        )
        # Allow the button to grow to its size hint (text + padding) in both axes
        self._btn_sim.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self._btn_sim.setMinimumHeight(54)
        self._btn_sim.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0d6efd, stop:0.45 #6610f2, stop:1 #d63384);
                color: white;
                font-size: 12px;
                font-weight: bold;
                border: none;
                border-radius: 7px;
                padding: 8px 18px;
                letter-spacing: 0.3px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3d8bfd, stop:0.45 #8540f5, stop:1 #e25c92);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0a58ca, stop:0.45 #520dc2, stop:1 #ab296a);
            }
        """)
        self._btn_sim.clicked.connect(self._open_sim_dialog)
        self._btn_sim.hide()
        toolbar.addWidget(self._btn_sim)

        toolbar.addStretch()
        toolbar.addWidget(self.lbl_count)
        left_layout.addLayout(toolbar)

        # Table-editing buttons — kept directly above the table and separate from
        # the toolbar's pulseq / schedule buttons, so it's clear they edit the
        # sequence table rows.
        _tbl_btn_row = QHBoxLayout()
        _tbl_btn_row.addWidget(btn_add)
        _tbl_btn_row.addWidget(btn_remove)
        _tbl_btn_row.addWidget(btn_clear)
        _tbl_btn_row.addStretch()
        left_layout.addLayout(_tbl_btn_row)

        # Status label — starts empty (the verbose "No schedule loaded…" placeholder
        # and the column legend were removed at user request; the table's own column
        # headers already label the columns).  Populated with "Loaded: …" on load.
        self.lbl_loaded = QLabel("")
        self.lbl_loaded.setStyleSheet(
            "font-size: 11px; color: #888; padding: 2px 0px;"
        )
        self.lbl_loaded.setWordWrap(True)
        left_layout.addWidget(self.lbl_loaded)

        # Table
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setAlternatingRowColors(True)
        self.table.itemChanged.connect(self._on_table_changed)
        left_layout.addWidget(self.table, stretch=1)

        splitter.addWidget(left)

        # ── RIGHT: 4-panel schedule figure ───────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(4)
        right_layout.setContentsMargins(4, 4, 4, 4)

        # Plot title (filename)
        self.lbl_plot_title = QLabel("CEST-MRF Schedule")
        self.lbl_plot_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_plot_title.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #ddd;"
        )
        right_layout.addWidget(self.lbl_plot_title)

        # Matplotlib figure — vertically stacked panels sharing the x-axis
        self._fig = Figure(facecolor="white")
        self.canvas = FigureCanvas(self._fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # Navigation toolbar — lets the user zoom/pan and, via the "Edit axis,
        # curves and parameters" button, change each panel's y-axis limits/ticks.
        try:
            from matplotlib.backends.backend_qtagg import \
                NavigationToolbar2QT as _SchedToolbar
        except Exception:
            try:
                from matplotlib.backends.backend_qt5agg import \
                    NavigationToolbar2QT as _SchedToolbar
            except Exception:
                _SchedToolbar = None
        if _SchedToolbar is not None:
            self._sched_toolbar = _SchedToolbar(self.canvas, self)
            self._sched_toolbar.setToolTip(
                "Zoom / pan, and use the axis-parameters button to change "
                "y-axis limits and ticks for any panel.")
            right_layout.addWidget(self._sched_toolbar)

        right_layout.addWidget(self.canvas, stretch=1)

        # Y-axis tick density control + Export button
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(QLabel("Y-axis ticks:"))
        self._yticks_spin = QSpinBox()
        self._yticks_spin.setRange(2, 12)
        self._yticks_spin.setValue(6)
        self._yticks_spin.setFixedWidth(60)
        self._yticks_spin.valueChanged.connect(self._refresh_plot)
        bottom_row.addWidget(self._yticks_spin)

        # Per-panel y-axis range / tick overrides
        self._btn_yaxis = QPushButton("Y-axis…")
        self._btn_yaxis.clicked.connect(self._open_yaxis_dialog)
        bottom_row.addWidget(self._btn_yaxis)

        self.chk_dark_bg = QCheckBox("Bg")
        self.chk_dark_bg.setToolTip(
            "Black background for the figure (for slides). Only the white "
            "surround and labels flip - the schedule plot stays identical."
        )
        self.chk_dark_bg.toggled.connect(self._refresh_plot)
        bottom_row.addWidget(self.chk_dark_bg)

        bottom_row.addStretch()

        btn_export = QPushButton("Export schedule figure…")
        btn_export.clicked.connect(self._export_figure)
        bottom_row.addWidget(btn_export)
        right_layout.addLayout(bottom_row)

        # Per-panel y-axis config:  key → {auto, min, max, nticks}
        self._yaxis_cfg: dict = {}
        # Populated each redraw:  [(key, human_label), …] for the dialog
        self._current_panels: list = []
        # key → (data_min, data_max) from the last draw (pre-fills the dialog)
        self._panel_autorange: dict = {}

        splitter.addWidget(right)
        splitter.setSizes([360, 460])

        # Start with an empty table (no default schedule)
        self._update_count()
        self._refresh_plot()

    # ─────────────────────────────────────────────────────────────────────
    # Table helpers
    # ─────────────────────────────────────────────────────────────────────

    def _set_row(self, row_idx: int, values: list):
        for col, val in enumerate(values):
            item = QTableWidgetItem(str(val))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, col, item)

    def _add_row(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self._set_row(r, [7000, 3.0, 3.0, 60, 3000, 0, 0])
        self._update_count()
        self._refresh_plot()

    def _remove_row(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)
        self._update_count()
        self._refresh_plot()

    def _clear_table(self):
        """Clear all rows from the schedule table."""
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.table.blockSignals(False)
        self._loaded_fname = ""
        self.lbl_loaded.setText("Table cleared.")
        self.lbl_loaded.setStyleSheet("font-size: 11px; color: #888; padding: 2px 0px;")
        self.lbl_plot_title.setText("CEST-MRF Schedule")
        self._update_count()
        self._refresh_plot()

    def _update_count(self):
        n = self.table.rowCount()
        self.lbl_count.setText(f"{n} measurement{'s' if n != 1 else ''}")
        # Enable "View Pulse Sequence" only when there's something to visualise.
        self._btn_view_seq.setEnabled(n > 0)

    def _on_table_changed(self):
        self._update_count()
        self._refresh_plot()

    # ─────────────────────────────────────────────────────────────────────
    # Load from .txt
    # ─────────────────────────────────────────────────────────────────────

    def _load_txt(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Load schedule from .txt", "",
            "Text files (*.txt);;All files (*)"
        )
        if not fn:
            return
        try:
            rows = _parse_txt(fn)
        except Exception as e:
            QMessageBox.critical(self, "Parse error", str(e))
            return

        self._load_rows(rows)

        self._loaded_fname = fn
        self.lbl_loaded.setText(f"Loaded: {fn}")
        self.lbl_loaded.setStyleSheet(
            "font-size: 11px; color: #7ec97e; padding: 2px 0px;"
        )
        title = Path(fn).stem.replace("_", " ")
        self.lbl_plot_title.setText(title)
        self._refresh_plot()

    def _load_from_bruker(self):
        """
        Read seq_defs from a Bruker pdata/1 directory (method + acqp files)
        and populate the schedule table.
        """
        scan_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Bruker pdata/1 directory (containing 2dseq / method+acqp two levels up)",
            "",
        )
        if not scan_dir:
            return

        pv360_guess = False
        pv_version_str = "PV6 / PV7"
        try:
            import os, re as _re
            subject_fp = os.path.normpath(
                os.path.join(scan_dir, "..", "..", "..", "subject")
            )
            if os.path.isfile(subject_fp):
                with open(subject_fp, "r", errors="replace") as fh:
                    _content = fh.read()
                if "ParaVision 360" in _content or "PV360" in _content:
                    pv360_guess = True
                    pv_version_str = "PV360"
                else:
                    m = _re.search(r"ParaVision[\s_]?(\d+)", _content, _re.IGNORECASE)
                    pv_version_str = f"PV{m.group(1)}" if m else "PV6 / PV7"
        except Exception:
            pass

        try:
            from my_gui.bruker_reader import read_seq_defs_from_bruker
            seq_defs, info = read_seq_defs_from_bruker(scan_dir, pv360=pv360_guess)
        except Exception as e:
            QMessageBox.critical(self, "Bruker read error", str(e))
            return

        import numpy as np
        def _to_list(v):
            return list(v) if hasattr(v, '__len__') and not isinstance(v, str) else [v]

        n        = int(seq_defs.get("num_meas", 1))
        tp_arr   = _to_list(seq_defs.get("tp",   [0.0]))
        trec_arr = _to_list(seq_defs.get("Trec", [0.0]))
        b1_arr   = _to_list(seq_defs.get("B1pa", [0.0]))
        excfa    = _to_list(seq_defs.get("excFA",[60.0]))
        slfa     = _to_list(seq_defs.get("SLFA", [60.0]))
        sl_flag  = _to_list(seq_defs.get("SLflag",[False]))
        offsets  = _to_list(seq_defs.get("offsets_ppm",[0.0]))

        def _get(arr, i, default=0.0):
            try:
                return float(arr[i])
            except (IndexError, TypeError, ValueError):
                return default

        rows = []
        for i in range(n):
            tr_ms   = (_get(trec_arr, i) + _get(tp_arr, i)) * 1000
            b1      = _get(b1_arr,   i)
            off_ppm = _get(offsets,  i)
            fa      = _get(excfa,    i, 60.0)
            tsat_ms = _get(tp_arr,   i) * 1000
            is_sl   = int(bool(_get(sl_flag, i)))
            sl_fa   = _get(slfa,     i, fa)
            rows.append([round(tr_ms, 1), round(b1, 4), round(off_ppm, 4),
                         round(fa, 1), round(tsat_ms, 1), is_sl, round(sl_fa, 1)])

        self._load_rows(rows)

        b0     = info.get("B0", "?")
        sched  = info.get("schedule", "")
        pv_str = pv_version_str
        label  = f"Loaded from Bruker [{pv_str}]: {Path(scan_dir).name}"
        if sched:
            label += f"  (schedule: {sched})"
        self._loaded_fname = str(scan_dir)
        self.lbl_loaded.setText(label)
        self.lbl_loaded.setStyleSheet(
            "font-size: 11px; color: #5dade2; padding: 2px 0px;"
        )
        self.lbl_plot_title.setText(
            f"CEST-MRF Schedule — Bruker {pv_str}  B0={b0} T  n={n}"
        )
        self._update_count()
        self._refresh_plot()

    def _open_generator(self):
        """Open the two-tab ScheduleGeneratorDialog and load the result."""
        dlg = ScheduleGeneratorDialog(parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.generated_rows:
            self._load_rows(dlg.generated_rows)
            self._loaded_fname = dlg.saved_filepath or ""
            src = self._loaded_fname or "(CRB-optimised, unsaved)"
            self.lbl_loaded.setText(f"Loaded from generator: {src}")
            self.lbl_loaded.setStyleSheet(
                "font-size: 11px; color: #f0a500; padding: 2px 0px;"
            )
            stem = Path(self._loaded_fname).stem if self._loaded_fname else "Generated Schedule"
            self.lbl_plot_title.setText(stem.replace("_", " "))
            self._update_count()
            self._refresh_plot()

    # ─────────────────────────────────────────────────────────────────────
    # Shared row loader
    # ─────────────────────────────────────────────────────────────────────

    def _load_rows(self, rows: list[list]):
        """Populate table from a list of 7-element rows (blocks signals)."""
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for row in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self._set_row(r, row)
        self.table.blockSignals(False)
        self._update_count()

    # ─────────────────────────────────────────────────────────────────────
    # 4-panel MATLAB-style schedule visualisation
    # ─────────────────────────────────────────────────────────────────────

    def _get_data_array(self) -> np.ndarray | None:
        """Read all rows from table → (n, 7) float array. Returns None if empty."""
        n = self.table.rowCount()
        if n == 0:
            return None
        arr = np.zeros((n, 7))
        for r in range(n):
            for c in range(7):
                item = self.table.item(r, c)
                try:
                    arr[r, c] = float(item.text()) if item else 0.0
                except ValueError:
                    arr[r, c] = 0.0
        return arr

    def _refresh_plot(self):
        """
        Redraw the schedule figure on a white background:
          B₁ (µT) · Ω (ppm) · Tsat (ms) · Td (ms)
        If the schedule contains spin-lock (CESL/wrCESL) measurements, the
        panels are shaded green (CEST) / blue (spin-lock) per iteration and an
        extra "SL FA (°)" panel is added. Panel count adapts automatically.
        """
        data = self._get_data_array()
        self._fig.clf()

        if data is None:
            # Empty state — draw a placeholder message
            ax = self._fig.add_subplot(111)
            ax.set_facecolor("white")
            self._fig.set_facecolor("white")
            ax.text(
                0.5, 0.5,
                "No schedule loaded\n\n"
                "Load a .txt file,\nor use  Generate MRF schedule…",
                ha="center", va="center",
                fontsize=11, color="#888",
                transform=ax.transAxes,
            )
            ax.axis("off")
            apply_fig_dark_theme(self._fig, self.chk_dark_bg.isChecked())
            self.canvas.draw()
            return

        n   = data.shape[0]
        idx = np.arange(1, n + 1)

        # ── Spin-lock detection (CESL / wrCESL) ─────────────────────────────
        # _COL_ISLSL = per-measurement spin-lock flag; _COL_SLFA = SL flip angle.
        if data.shape[1] > _COL_ISLSL:
            is_sl = data[:, _COL_ISLSL] > 0.5
        else:
            is_sl = np.zeros(n, dtype=bool)
        has_sl = bool(np.any(is_sl))

        # Base panels (per-panel line colour, matching the reference figure).
        # Excitation flip angle (_COL_FA) sits between Ω and Tsat.
        signals = [
            data[:, _COL_B1],
            data[:, _COL_OFF],
            data[:, _COL_FA],
            data[:, _COL_TSAT],
            data[:, _COL_TR],
        ]
        labels = [
            r"$B_1$ (µT)",
            r"$\Omega$ (ppm)",
            r"Excitation FA (°)",
            r"$T_{sat}$ (ms)",
            r"$T_R$ (ms)",
        ]
        # Stable keys + human-readable names for the per-panel y-axis dialog
        keys       = ["b1", "off", "excfa", "tsat", "tr"]
        human      = ["B1 (µT)", "Ω (ppm)", "Excitation FA (°)",
                      "Tsat (ms)", "TR (ms)"]
        colors = [
            (0.90, 0.49, 0.13),   # B1        — orange
            (0.84, 0.19, 0.15),   # Ω         — red
            (0.55, 0.35, 0.15),   # Exc. FA   — brown
            (0.58, 0.20, 0.58),   # Tsat      — purple
            (0.13, 0.55, 0.55),   # TR        — teal
        ]
        # Extra panel showing the Spin-Lock on/off flag — only when present
        flag_panel = -1
        if has_sl:
            signals.append(is_sl.astype(float))
            labels.append("Spin Lock")
            keys.append("spinlock")
            human.append("Spin Lock")
            colors.append((0.40, 0.30, 0.70))   # indigo
            flag_panel = len(signals) - 1

        n_panels = len(signals)
        # Expose current panels (key, label) so the Y-axis dialog can list them
        self._current_panels = list(zip(keys, human))

        # Background colours per acquisition type (reference figure)
        CEST_BG = (0.62, 0.84, 0.59)   # green
        SL_BG   = (0.62, 0.76, 0.92)   # blue

        # Adaptive geometry — generous title/legend margins so nothing overlaps
        left      = 0.14
        wid       = 0.80
        gap       = 0.034
        top_pad   = 0.085                       # room for the (long) title
        bot_start = 0.150 if has_sl else 0.075  # room for x-label + CEST/SL legend
        hp        = (1.0 - bot_start - top_pad - (n_panels - 1) * gap) / n_panels
        tc        = (0.15, 0.15, 0.15)
        lw        = 1.6
        fs        = 10

        # Contiguous runs of the same acquisition type → one axvspan each
        def _runs(flags):
            out, start = [], 0
            for i in range(1, len(flags)):
                if bool(flags[i]) != bool(flags[i - 1]):
                    out.append((start, i - 1, bool(flags[start]))); start = i
            out.append((start, len(flags) - 1, bool(flags[start])))
            return out
        runs = _runs(is_sl) if has_sl else []

        for p in range(n_panels):
            bot = bot_start + (n_panels - 1 - p) * (hp + gap)
            ax  = self._fig.add_axes([left, bot, wid, hp], facecolor="white")
            ax.set_facecolor("white")

            # Green (CEST) / blue (spin-lock) background bands
            for s, e, fl in runs:
                ax.axvspan(s + 0.5, e + 1.5,
                           color=(SL_BG if fl else CEST_BG),
                           alpha=0.55, linewidth=0, zorder=0)

            ax.tick_params(colors=tc, direction="in", labelsize=fs - 1)
            for spine in ax.spines.values():
                spine.set_edgecolor(tc)
                spine.set_linewidth(0.8)
            ax.xaxis.label.set_color(tc)
            ax.yaxis.label.set_color(tc)

            sig = signals[p]
            ax.plot(idx, sig, color=colors[p], linewidth=lw,
                    marker='o', markersize=3.5, markerfacecolor=colors[p],
                    markeredgecolor=colors[p], zorder=3)

            self._panel_autorange[keys[p]] = (float(np.nanmin(sig)),
                                              float(np.nanmax(sig)))
            from matplotlib.ticker import MaxNLocator
            cfg = self._yaxis_cfg.get(keys[p])
            if p == flag_panel:
                # Binary on/off indicator
                ax.set_ylim(-0.3, 1.3)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["off", "on"])
            elif cfg and not cfg.get("auto", True):
                # ── User-defined y-axis range / tick count for THIS panel ──
                # Place ticks exactly at min…max (evenly spaced) so the chosen
                # endpoints — e.g. 7.3 — are shown, not "nice" rounded values.
                lo_c, hi_c = float(cfg["min"]), float(cfg["max"])
                nt = max(2, int(cfg.get("nticks", 6)))
                ax.set_ylim(lo_c, hi_c)
                ticks = np.round(np.linspace(lo_c, hi_c, nt), 6)
                from matplotlib.ticker import FixedLocator
                ax.yaxis.set_major_locator(FixedLocator(ticks))
                try:
                    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                except Exception:
                    pass
            else:
                lo, hi = float(np.nanmin(sig)), float(np.nanmax(sig))
                yspan  = hi - lo
                if yspan <= 1e-9:
                    # Constant-value panel (e.g. FA=60, TR=7000): show a real
                    # range from 0 instead of a degenerate ±offset window.
                    if abs(hi) < 1e-9:
                        lo2, hi2 = -1.0, 1.0
                    elif hi > 0:
                        lo2, hi2 = 0.0, hi * 1.25
                    else:
                        lo2, hi2 = hi * 1.25, 0.0
                    ax.set_ylim(lo2, hi2)
                else:
                    ypad = yspan * 0.12
                    ax.set_ylim(lo - ypad, hi + ypad)
                nbins = self._yticks_spin.value() if hasattr(self, "_yticks_spin") else 6
                ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune=None))
                try:
                    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                except Exception:
                    pass
            ax.set_xlim(0.5, n + 0.5)

            ax.set_ylabel(
                labels[p], fontsize=fs, color=tc,
                rotation=0, labelpad=50,
                ha="right", va="center",
            )

            if p < n_panels - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time point index", fontsize=fs, color=tc)

        title_text = self.lbl_plot_title.text()
        self._fig.text(
            0.5, 0.985, title_text,
            ha="center", va="top", fontsize=fs + 1,
            fontweight="bold", color=tc,
        )

        # CEST / spin-lock colour legend — a clear strip below the x-label
        if has_sl:
            self._fig.text(0.34, 0.018, "■ CEST", ha="center", va="bottom",
                           fontsize=fs, fontweight="bold", color=(0.30, 0.62, 0.30))
            self._fig.text(0.66, 0.018, "■ spin-lock (CESL)", ha="center",
                           va="bottom", fontsize=fs, fontweight="bold",
                           color=(0.35, 0.55, 0.80))

        self._fig.set_facecolor("white")
        apply_fig_dark_theme(self._fig, self.chk_dark_bg.isChecked())
        self.canvas.draw()

    # ─────────────────────────────────────────────────────────────────────
    # Per-panel y-axis editor
    # ─────────────────────────────────────────────────────────────────────

    def _open_yaxis_dialog(self):
        """Per-panel y-axis range / tick editor (independent for each panel)."""
        from PyQt6.QtWidgets import (
            QDialog, QGridLayout, QLabel as _QL, QCheckBox as _QCB,
            QDoubleSpinBox as _QDSP, QSpinBox as _QSB, QDialogButtonBox,
        )
        panels = [(k, lbl) for k, lbl in self._current_panels if k != "spinlock"]
        if not panels:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "No schedule",
                                   "Load or generate a schedule first.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Custom Y-axis (per panel)")
        grid = QGridLayout(dlg)
        for c, txt in enumerate(["Panel", "Auto", "Min", "Max", "# ticks"]):
            h = _QL(f"<b>{txt}</b>"); grid.addWidget(h, 0, c)

        widgets = {}
        for r, (key, label) in enumerate(panels, start=1):
            cfg = self._yaxis_cfg.get(key, {})
            lo, hi = self._panel_autorange.get(key, (0.0, 1.0))
            # sensible default span for a constant panel
            if abs(hi - lo) < 1e-9:
                hi = hi * 1.25 if hi > 0 else (1.0 if hi == 0 else 0.0)
                lo = lo if lo < 0 else 0.0

            grid.addWidget(_QL(label), r, 0)
            chk = _QCB(); chk.setChecked(cfg.get("auto", True))
            grid.addWidget(chk, r, 1)
            sp_min = _QDSP(); sp_min.setRange(-1e6, 1e6); sp_min.setDecimals(2)
            sp_min.setValue(float(cfg.get("min", lo)))
            grid.addWidget(sp_min, r, 2)
            sp_max = _QDSP(); sp_max.setRange(-1e6, 1e6); sp_max.setDecimals(2)
            sp_max.setValue(float(cfg.get("max", hi)))
            grid.addWidget(sp_max, r, 3)
            sp_n = _QSB(); sp_n.setRange(2, 20)
            sp_n.setValue(int(cfg.get("nticks", self._yticks_spin.value())))
            grid.addWidget(sp_n, r, 4)

            # Min/Max/#ticks editable only when Auto is off
            def _sync(state, _w=(sp_min, sp_max, sp_n), _c=chk):
                on = not _c.isChecked()
                for w in _w:
                    w.setEnabled(on)
            chk.toggled.connect(lambda _=False, f=_sync, c=chk: f(c.isChecked()))
            _sync(None)
            widgets[key] = (chk, sp_min, sp_max, sp_n)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        grid.addWidget(bb, len(panels) + 1, 0, 1, 5)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            for key, (chk, sp_min, sp_max, sp_n) in widgets.items():
                if chk.isChecked():
                    self._yaxis_cfg[key] = {"auto": True}
                else:
                    lo, hi = sp_min.value(), sp_max.value()
                    if hi <= lo:
                        hi = lo + 1.0
                    self._yaxis_cfg[key] = {
                        "auto": False, "min": lo, "max": hi,
                        "nticks": sp_n.value(),
                    }
            self._refresh_plot()

    # ─────────────────────────────────────────────────────────────────────
    # Export figure
    # ─────────────────────────────────────────────────────────────────────

    def _export_figure(self):
        from my_gui.fig_export import save_figure, FIG_EXPORT_FILTER
        default = Path(self._loaded_fname).stem if self._loaded_fname else "MRF_schedule"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export schedule figure", default,
            FIG_EXPORT_FILTER
        )
        if path:
            save_figure(self._fig, path, dpi=300,
                        facecolor=self._fig.get_facecolor())

    # ─────────────────────────────────────────────────────────────────────
    # View Pulse Sequence  (PyPulseq timing diagram)
    # ─────────────────────────────────────────────────────────────────────

    def _open_pulseq_simulation(self):
        """Open the Pulseq Simulation window (example-sequence dictionary sim + match)."""
        try:
            from my_gui.pulseq_sim_dialog import PulseqSimDialog
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Pulseq Simulation",
                                 f"Could not open the Pulseq Simulation window:\n{exc}")
            return
        dlg = PulseqSimDialog(parent=self)
        dlg.exec()

    def _view_pulse_sequence(self):
        """
        Build seq_defs from the current table, call write_sequence_sl in a
        background thread, then open PulseSeqViewerDialog with the result.
        """
        if self.table.rowCount() == 0:
            QMessageBox.warning(self, "No schedule",
                                "Load or generate a schedule first.")
            return

        # Ask for the field strength B₀ used to convert ppm offsets → Hz. The RF
        # frequency offsets scale with B₀ (offset_Hz = offset_ppm · γ · B₀), so the
        # plotted sequence (offset frequencies / phase) changes with the field.
        from PyQt6.QtWidgets import QInputDialog
        b0, ok = QInputDialog.getDouble(
            self, "Field strength B₀",
            "Enter the magnetic field strength B₀ (Tesla) used to convert the\n"
            "ppm offsets to Hz:",
            self._b0_value, 0.55, 14.0, 2)
        if not ok:
            return
        self._b0_value = b0
        try:
            seq_defs = self.get_seq_defs(b0)
        except Exception as exc:
            QMessageBox.critical(self, "seq_defs error",
                                 f"Could not build sequence definition:\n{exc}")
            return

        # Write to a temp file — write_sequence_sl requires a file path.
        tmp = tempfile.NamedTemporaryFile(suffix=".seq", delete=False)
        tmp.close()
        tmp_fn = tmp.name

        # Always the vendored 1.3.1 writer (simulator format).
        self._pp_ver_used = "1.3.1"
        _use_vendored = True

        # Disable button and show spinner text while the worker runs.
        self._btn_view_seq.setEnabled(False)
        self._btn_view_seq.setText("Generating…")

        self._seq_worker = _SeqGenWorker(seq_defs, tmp_fn,
                                         use_vendored=_use_vendored, parent=self)
        self._seq_worker.finished.connect(
            lambda seq: self._on_seq_ready(seq, tmp_fn))
        self._seq_worker.error.connect(self._on_seq_error)
        self._seq_worker.start()

    def _on_seq_ready(self, seq, tmp_fn: str):
        """Receive the finished Sequence object and open the viewer dialog."""
        import os

        ver = getattr(self, '_pp_ver_used', '1.3.1')
        # Re-read the written .seq with the vendored pypulseq (OCEAN ships one
        # version) to get a clean object off the worker thread; fall back to the
        # worker's own object if the re-read fails.
        plot_seq = seq
        try:
            import pypulseq as _pp
            _s = _pp.Sequence()
            _s.read(tmp_fn)
            plot_seq = _s
        except Exception:
            plot_seq = seq
        try:
            os.unlink(tmp_fn)
        except OSError:
            pass

        # Restore button.
        self._btn_view_seq.setEnabled(self.table.rowCount() > 0)
        self._btn_view_seq.setText("View Pulse Sequence")

        base = Path(self._loaded_fname).stem if self._loaded_fname else "Current Schedule"
        title = f"{base}   [pypulseq {ver}]"
        dlg = PulseSeqViewerDialog(plot_seq, parent=self, title=title,
                                   pp_version=ver)
        dlg.exec()

    def _on_seq_error(self, msg: str):
        """Show write_sequence_sl errors to the user."""
        self._btn_view_seq.setEnabled(self.table.rowCount() > 0)
        self._btn_view_seq.setText("View Pulse Sequence")
        QMessageBox.critical(
            self, "Pulse Sequence Generation Error",
            f"write_sequence_sl failed:\n\n{msg}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Public API (used by app.py)
    # ─────────────────────────────────────────────────────────────────────

    def get_schedule(self) -> list[dict]:
        keys = ["tr_ms", "ampl_ut", "offset_ppm", "fa_deg",
                "sat_time_ms", "is_sl", "sl_fa_deg"]
        schedule = []
        for r in range(self.table.rowCount()):
            row = {}
            for c, key in enumerate(keys):
                try:
                    row[key] = float(self.table.item(r, c).text())
                except (ValueError, AttributeError):
                    row[key] = 0.0
            schedule.append(row)
        return schedule

    def get_seq_defs(self, b0: float) -> dict:
        """
        Build seq_defs with per-measurement arrays for write_sequence_sl.

        Timing model (must match sequences_sl.write_sequence_sl):
          Each TR consists of:
            (if idx>0)  recovery delay = Trec[idx-1] - te
            saturation block            = tsat
            imaging pulse               = t_img (2.1 ms)
            imaging delay               = te    (20 ms)
            pseudo-ADC                  = t_adc (1 ms)
          Total = (Trec[idx-1] - te) + tsat + t_img + te + t_adc
                = Trec[idx-1] + tsat + t_img + t_adc
                = Trec[idx-1] + tsat + 3.1 ms  == TR
          ⟹  Trec[i] = TR[i] - tsat[i] - 3.1 ms
        """
        schedule = self.get_schedule()
        n        = len(schedule)

        n_pulses = 1
        td_s     = 1e-5
        pw_90    = 0.1e-3
        t_img    = 2.1e-3
        t_adc    = 1e-3

        tp_arr    = [r["sat_time_ms"] / 1000.0 for r in schedule]
        tr_arr    = [r["tr_ms"]       / 1000.0 for r in schedule]
        excfa_arr = [r["fa_deg"]                for r in schedule]
        slflag    = [int(r["is_sl"])             for r in schedule]

        slfa_arr = []
        for r, is_sl in zip(schedule, slflag):
            slfa = r["sl_fa_deg"]
            if is_sl and slfa == 0.0:
                slfa = 90.0
            slfa_arr.append(slfa)

        trec_arr = []
        for i in range(n):
            tp    = tp_arr[i]
            tr    = tr_arr[i]
            is_sl = slflag[i]
            tsat  = n_pulses * tp + max(n_pulses - 1, 0) * td_s
            if is_sl:
                tsat += 2.0 * pw_90
            trec = tr - tsat - t_img - t_adc
            trec_arr.append(max(trec, 0.0))

        dcsat = [
            n_pulses * tp / (n_pulses * tp + td_s)
            if (n_pulses * tp + td_s) > 0 else 0.0
            for tp in tp_arr
        ]

        return {
            "num_meas":      n,
            "n_pulses":      n_pulses,
            "tp":            tp_arr,
            "td":            td_s,
            "Trec":          trec_arr,
            "B1pa":          [r["ampl_ut"]    for r in schedule],
            "excFA":         excfa_arr,
            "SLFA":          slfa_arr,
            "SLflag":        slflag,
            "DCsat":         dcsat,
            "offsets_ppm":   [r["offset_ppm"] for r in schedule],
            "B0":            b0,
            "seq_id_string": "seq",
            "_loaded_fname": self._loaded_fname,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Simulation dialog integration
    # ─────────────────────────────────────────────────────────────────────

    def set_dict_tab(self, dict_tab) -> None:
        """
        Inject the DictTab widget and reveal the simulation button below the toolbar.

        Called by CestMrfTab after both seq_tab and dict_tab are created.
        The DictTab lives inside a resizable, non-modal dialog so the user
        can keep the sequence table visible alongside the simulation controls.
        """
        self._dict_tab_ref = dict_tab
        self._btn_sim.show()

    def _open_sim_dialog(self) -> None:
        """Open (or raise) the Dictionary Simulation & Matching dialog."""
        if self._dict_tab_ref is None:
            return

        if self._sim_dialog is None:
            from PyQt6.QtWidgets import QDialog, QVBoxLayout as _VL, QSizeGrip
            from PyQt6.QtCore import Qt as _Qt

            dlg = QDialog(self.window())           # top-level, non-modal
            dlg.setWindowTitle("🔬  Dictionary Simulation & Matching")
            dlg.setMinimumSize(860, 640)
            dlg.resize(1000, 740)
            dlg.setWindowFlags(
                _Qt.WindowType.Window |
                _Qt.WindowType.WindowCloseButtonHint |
                _Qt.WindowType.WindowMinMaxButtonsHint
            )
            # Prevent the widget from being destroyed when the dialog closes
            dlg.setAttribute(_Qt.WidgetAttribute.WA_DeleteOnClose, False)

            vl = _VL(dlg)
            vl.setContentsMargins(6, 6, 6, 6)
            vl.addWidget(self._dict_tab_ref, stretch=1)

            self._sim_dialog = dlg

        self._sim_dialog.show()
        self._sim_dialog.raise_()
        self._sim_dialog.activateWindow()
