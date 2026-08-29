"""
pool_dialog.py
Shared pool-selection dialog and catalog for Z-Spec, QUESP, and 1/Z tabs.
"""
from __future__ import annotations
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox,
    QLabel, QDialogButtonBox, QGridLayout, QFrame, QScrollArea, QWidget
)
from PyQt6.QtCore import Qt

# ─────────────────────────────────────────────────────────────────────────────
# Pool catalog: (key, display_name, ppm_center, default_on)
# 'water' is always included and not shown as a checkbox.
# ─────────────────────────────────────────────────────────────────────────────
POOL_CATALOG: list[tuple[str, str, float, bool]] = [
    # key             display                       ppm    default
    ("amide",         "amide (3.5 ppm)",            3.5,   True),
    ("amine",         "amine (3.0 ppm)",            3.0,   True),
    ("OH",            "OH (0.8 ppm)",               0.8,   True),
    ("MT",            "MT (broad, ~−2 ppm)",        -2.0,  True),
    ("NOE",           "NOE / rNOE (−3.5 ppm)",      -3.5,  True),
    ("Trp",           "Trp (5.4 ppm)",              5.4,   True),
    ("7.3ppm",        "7.3 ppm",                    7.3,   False),
    ("9.8ppm",        "9.8 ppm",                    9.8,   False),
    ("glucose",       "Glucose (1.2 ppm)",          1.2,   False),
    ("creatine",      "Creatine (1.9 ppm)",         1.9,   False),
    # One checkbox fits BOTH iopamidol amide pools (4.2 & 5.5 ppm); the fitter
    # expands the 'iopamidol' key into iopamidol43 + iopamidol55 internally.
    ("iopamidol",     "Iopamidol (4.2 & 5.5 ppm)",  4.2,   False),
    ("3omg",          "3-OMG (1.2 ppm)",            1.2,   False),
    ("GAG",           "GAG (~1.0 ppm)",             1.0,   False),
]

def default_pools() -> list[str]:
    """Return keys of pools that are on by default."""
    return [key for key, *_, default in POOL_CATALOG if default]


class PoolSelectionWidget(QWidget):
    """Reusable pool checkbox grid — 3 columns."""

    def __init__(self, initial_selection: list[str] | None = None, parent=None):
        super().__init__(parent)
        if initial_selection is None:
            initial_selection = default_pools()
        self._checks: dict[str, QCheckBox] = {}
        self._build(initial_selection)

    def _build(self, initial: list[str]):
        grp = QGroupBox("Pools to Fit  (water always included)")
        grid = QGridLayout(grp)
        grid.setSpacing(4)
        for i, (key, label, ppm, _) in enumerate(POOL_CATALOG):
            chk = QCheckBox(label)
            chk.setChecked(key in initial)
            self._checks[key] = chk
            row, col = divmod(i, 3)
            grid.addWidget(chk, row, col)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(grp)

    def selected_pools(self) -> list[str]:
        return [k for k, chk in self._checks.items() if chk.isChecked()]

    def set_selection(self, keys: list[str]):
        for k, chk in self._checks.items():
            chk.setChecked(k in keys)
