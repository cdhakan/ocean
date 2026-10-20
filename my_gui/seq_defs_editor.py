"""
seq_defs_editor.py
==================

"Seq Parameter Definitions" dialog for the Pulseq Simulation window: shows the
sequence definitions of the selected .seq the way the pulseq-cest-library
generator scripts define them (b0, b1pa, n_pulses, tp, td, trec, trec_m0,
m0_offset, offsets_ppm, spoiling, seq_id_string + the derived dcsat /
num_meas / tsat / freq), lets the user edit every value, and regenerates the
.seq with the new physics (see ``seq_rebuild``) — ready to view, simulate and
export.
"""
from __future__ import annotations

import ast
import os
from types import SimpleNamespace

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QGuiApplication
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QFileDialog, QApplication,
    QAbstractItemView,
)

from my_gui import seq_rebuild as sr


# ─────────────────────────────────────────────────────────────────────────────
# value parsing — numbers, "1, 2, 3", or small numpy expressions
# ─────────────────────────────────────────────────────────────────────────────

_FUNCS = {
    "linspace": np.linspace, "arange": np.arange, "ones": np.ones, "zeros": np.zeros,
    "append": np.append, "concatenate": np.concatenate, "repeat": np.repeat,
    "array": np.array, "abs": np.abs, "round": np.round, "pi": np.pi, "nan": float("nan"),
}
_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Constant, ast.List,
            ast.Tuple, ast.Name, ast.Attribute, ast.Load, ast.Add, ast.Sub, ast.Mult,
            ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Mod, ast.FloorDiv, ast.keyword)


def parse_value(text: str):
    """'3' → 3.0 · '1, 2, 3' → [1, 2, 3] · 'append(-300, linspace(-4, 4, 33))' →
    list · anything else (seq_id_string) → str."""
    s = text.strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        pass
    parts = [p for p in s.replace(";", ",").replace("\n", ",").split(",") if p.strip()]
    if len(parts) > 1 or (len(parts) == 1 and " " in parts[0].strip()):
        try:
            toks = []
            for p in parts:
                toks += p.split()
            return [float(t) for t in toks]
        except ValueError:
            pass
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return s
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            return s
        if isinstance(node, ast.Name) and node.id not in _FUNCS and node.id != "np":
            return s
        if isinstance(node, ast.Attribute) and not (isinstance(node.value, ast.Name) and node.value.id == "np"
                                                    and node.attr in _FUNCS):
            return s
        if isinstance(node, ast.Call):
            f = node.func
            ok = (isinstance(f, ast.Name) and f.id in _FUNCS) or (
                isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "np")
            if not ok:
                return s
    env = dict(_FUNCS)
    env["np"] = SimpleNamespace(**_FUNCS)
    try:
        val = eval(compile(tree, "<defs>", "eval"), {"__builtins__": {}}, env)  # noqa: S307 (AST whitelisted)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"cannot evaluate '{s}': {exc}")
    if isinstance(val, (list, tuple, np.ndarray)):
        return [float(x) for x in np.asarray(val, dtype=float).ravel()]
    return float(val)


def fmt_value(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple, np.ndarray)):
        return ", ".join(f"{float(x):.6g}" for x in v)
    return f"{float(v):.6g}"


# ─────────────────────────────────────────────────────────────────────────────
# Dialog
# ─────────────────────────────────────────────────────────────────────────────

class SeqDefinitionsDialog(QDialog):
    """Edit the sequence definitions of ``seq_path`` and regenerate the .seq.

    ``applied(path, defs)`` is emitted after a successful regeneration / export
    so the caller can select the new file for viewing and simulation."""

    applied = pyqtSignal(str, dict)

    _ROWS = list(sr.EDITABLE) + list(sr.DERIVED)

    def __init__(self, seq_path: str, parent=None, out_dir: str | None = None,
                 opts_factory=None, b0_hint: float | None = None):
        super().__init__(parent)
        self._seq_path = seq_path
        self._out_dir = out_dir or os.path.dirname(seq_path)
        self._opts_factory = opts_factory          # callable(pp) → pp.Opts (scanner limits)
        self._b0_hint = b0_hint
        self._model: sr.SeqModel | None = None
        self._orig: dict = {}
        self._src: dict = {}
        self._updating = False
        self.setWindowTitle(f"Seq Parameter Definitions  —  {os.path.basename(seq_path)}")
        self.resize(820, 620)
        self._build_ui()
        self._load_model()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        title = QLabel("Sequence parameter definitions")
        title.setStyleSheet("font-size:15px; font-weight:bold;")
        root.addWidget(title)
        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("font-size:11px; color:#6b7280;")
        root.addWidget(self.lbl_info)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["parameter", "value", "unit", "description  (in file as)"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.cellChanged.connect(self._on_cell_changed)
        root.addWidget(self.table, stretch=1)

        hint = QLabel("Value: a number, a comma-separated list (one per measurement), or an expression such as "
                      "append(-300, linspace(-4, 4, 33)) · the length of offsets_ppm sets the number of measurements")
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size:11px; color:#6b7280;")
        root.addWidget(hint)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-size:11px; color:#c8871f;")
        root.addWidget(self.lbl_status)

        row = QHBoxLayout()
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset)
        self.btn_apply = QPushButton("Apply  →  regenerate .seq")
        self.btn_apply.setStyleSheet(
            "QPushButton{background:#1565c0;color:white;border-radius:4px;padding:6px 16px;font-weight:bold;}"
            "QPushButton:hover{background:#1976d2;}")
        self.btn_apply.clicked.connect(lambda: self._apply(export=False))
        self.btn_export = QPushButton("Export .seq…")
        self.btn_export.setStyleSheet(
            "QPushButton{background:#00695c;color:white;border-radius:4px;padding:6px 16px;font-weight:bold;}"
            "QPushButton:hover{background:#00897b;}")
        self.btn_export.clicked.connect(lambda: self._apply(export=True))
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        row.addWidget(self.btn_reset)
        row.addStretch()
        row.addWidget(self.btn_apply)
        row.addWidget(self.btn_export)
        row.addWidget(btn_close)
        root.addLayout(row)

    # ── model ───────────────────────────────────────────────────────────────
    def _load_model(self):
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            self._model = sr.parse_seq(self._seq_path)
            self._orig, self._src = self._model.current_defs(self._b0_hint)
        except Exception as exc:  # noqa: BLE001
            QGuiApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Cannot read sequence", f"{exc}")
            self.btn_apply.setEnabled(False)
            self.btn_export.setEnabled(False)
            return
        finally:
            QGuiApplication.restoreOverrideCursor()
        notes = ("  ·  " + "; ".join(self._model.notes)) if self._model.notes else ""
        self.lbl_info.setText(f"{self._seq_path}\n{self._model.summary()}{notes}")
        self._fill_table(self._orig)

    def _fill_table(self, defs: dict):
        self._updating = True
        self.table.setRowCount(0)
        extra = [k for k in self._model.definitions.keys()
                 if k not in {kk for keys in sr.CANON_KEYS.values() for kk in keys}]
        rows = self._ROWS + (["b1rms"] if "b1rms" in self._model.file_defs() else []) + extra
        mono = QFont("Menlo" if os.uname().sysname == "Darwin" else "Monospace")
        mono.setPointSize(11)
        for name in rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            editable = name in sr.EDITABLE
            if name in sr.DERIVED or name == "b1rms":
                val = sr.derived(defs).get(name, defs.get(name)) if name != "b1rms" else self._model.file_defs().get("b1rms")
                unit, desc = sr.DESCRIPTIONS.get(name, ("", ""))
                src = self._src.get(name, "derived")
            elif editable:
                val = defs.get(name)
                unit, desc = sr.DESCRIPTIONS.get(name, ("", ""))
                src = self._src.get(name, "")
            else:                                       # extra definition from the file
                val = sr._def_value(self._model.definitions[name])
                unit, desc, src = "", "other definition (kept as is)", name
            items = [QTableWidgetItem(name), QTableWidgetItem(fmt_value(val)),
                     QTableWidgetItem(unit), QTableWidgetItem(f"{desc}" + (f"   ({src})" if src else ""))]
            items[1].setFont(mono)
            for c, it in enumerate(items):
                if not (editable and c == 1):
                    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if not editable:
                    it.setForeground(QColor("#8a8f98"))
                self.table.setItem(r, c, it)
            if editable:
                items[0].setFont(QFont(items[0].font().family(), -1, QFont.Weight.Bold))
        self._updating = False

    def _row_of(self, name: str) -> int:
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).text() == name:
                return r
        return -1

    def _read_defs(self) -> dict:
        defs = {}
        for name in sr.EDITABLE:
            r = self._row_of(name)
            if r < 0:
                continue
            txt = self.table.item(r, 1).text()
            if name == "seq_id_string":
                defs[name] = txt.strip() or self._orig.get(name, "seq")
                continue
            v = parse_value(txt)
            if v is None:
                if name in ("trec_m0", "m0_offset"):
                    continue                       # optional
                raise ValueError(f"{name}: value missing")
            if isinstance(v, str):
                raise ValueError(f"{name}: '{v}' is not a number / list / expression")
            defs[name] = v
        return defs

    def _on_cell_changed(self, row: int, col: int):
        if self._updating or col != 1:
            return
        try:
            defs = self._read_defs()
        except ValueError as exc:
            self.lbl_status.setText(str(exc))
            return
        self.lbl_status.setText("")
        d = sr.derived(defs)
        self._updating = True
        for name in sr.DERIVED:
            r = self._row_of(name)
            if r >= 0 and name in d:
                self.table.item(r, 1).setText(fmt_value(d[name]))
        self._updating = False

    def _reset(self):
        self._fill_table(self._orig)
        self.lbl_status.setText("")

    # ── regenerate ──────────────────────────────────────────────────────────
    def _apply(self, export: bool):
        try:
            defs = self._read_defs()
        except ValueError as exc:
            QMessageBox.warning(self, "Check the values", str(exc))
            return
        seq_id = str(defs.get("seq_id_string") or "seq")
        safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in seq_id) or "seq"
        if export:
            out_path, _ = QFileDialog.getSaveFileName(
                self, "Export regenerated sequence", os.path.join(self._out_dir, safe + ".seq"),
                "Pulseq sequence (*.seq)")
            if not out_path:
                return
            if not out_path.lower().endswith(".seq"):
                out_path += ".seq"
        else:
            os.makedirs(self._out_dir, exist_ok=True)
            out_path = os.path.join(self._out_dir, safe + ".seq")
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            opts = None
            if self._opts_factory is not None:
                try:
                    with sr.vendored_pypulseq() as pp:
                        opts = self._opts_factory(pp)
                except Exception:
                    opts = None
            info = sr.rebuild(self._model, defs, out_path, opts=opts)
        except Exception as exc:  # noqa: BLE001
            QGuiApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Regeneration failed", f"{exc}")
            return
        finally:
            QGuiApplication.restoreOverrideCursor()
        warn = ("\n" + "\n".join(info["warnings"])) if info["warnings"] else ""
        self.lbl_status.setText(
            f"✓ {'Exported' if export else 'Regenerated'}: {info['num_meas']} measurements, "
            f"{info['n_blocks']} blocks, {info['duration_s']:.2f} s  →  {out_path}{warn}")
        self.lbl_status.setStyleSheet(f"font-size:11px; color:{'#c8871f' if warn else '#3aa655'};")
        self.applied.emit(out_path, dict(defs))
