"""
scan_dir_tab.py
Scan Directory tab — browse a Bruker study root or GE / Siemens data folder
and assign scan paths to each modality (T1, T2, B1, WASSR, CEST, MRF).

Placed first in the tab bar so the user can set up all paths before anything
else.  Other tabs (ZSpec, T1T2, Dict) can read assigned paths via the
get_scan_paths() method wired through app.py.
"""
from __future__ import annotations

import os
import re

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QLineEdit, QComboBox,
    QTextEdit, QCheckBox, QSplitter, QFrame,
    QScrollArea, QSizePolicy, QFileDialog, QGridLayout,
    QStackedWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

# Re-use the Bruker helpers that live in dict_tab
from my_gui.tabs.dict_tab import build_scan_list, _detect_pv360, _detect_pv_version


# Shared style for every section header group-box: the title sits *inside* the
# rounded box (top-left) and is rendered larger so it clearly reads as a header.
_GROUP_BOX_STYLE = """
    QGroupBox { border: 1px solid #444; border-radius: 6px;
                margin-top: 8px; padding-top: 28px; font-weight: bold; }
    QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left;
                       left: 10px; top: 5px; padding: 0 6px;
                       font-size: 17px; color: #e6e6e6; }
"""


def _strip_expno(s: str) -> str:
    """Remove trailing Bruker experiment-number suffixes like (E1), (E2) …"""
    return re.sub(r'\s*\(E\d+\)\s*$', '', s).strip()


# ── Modality definitions ───────────────────────────────────────────────────────
#  (key, display_name, colour, description, typical_name_hint)
# "__b1vfa__" is a special placeholder \u2192 rendered as a single combined box
# holding both B1 double-angle sub-scans (\u03b1\u2081 + \u03b1\u2082); see _B1VFAScanCard.
_MODALITIES = [
    ("t1",       "T1 Scan",               "#e67e22",  "#f0a860", "", ""),
    ("t2",       "T2 Scan",               "#27ae60",  "#5dbb8a", "", ""),
    ("__b1vfa__","B1 VFA Scan",           "#8e44ad",  "#b47dd4", "", ""),
    ("wasabi",   "WASABI Scan",           "#6c5ce7",  "#8577e8", "", ""),
    ("wassr",    "WASSR / B0 Scan",       "#2980b9",  "#5ba8db", "", ""),
    ("cest",     "CEST Scan",             "#16a085",  "#4ec9b0", "", ""),
    ("mrf",      "MR Fingerprinting Scan","#c0392b",  "#e06060", "", ""),
    ("quesp",    "QUESP Scan",            "#d35400",  "#e8895a", "", ""),
]


# ── DICOM tags surfaced per series (mirrors explore_dicom_folder.m + extras) ──
#  (dicom_keyword, short_label).  Numeric tags are formatted with %g.
_GS_TAGS = [
    ("SeriesNumber",          "Series#"),
    ("SeriesDescription",     "Description"),
    ("ProtocolName",          "Protocol"),
    ("SequenceName",          "Sequence"),
    ("MRAcquisitionType",     "AcqType"),
    ("ManufacturerModelName", "Model"),
    ("ImagingFrequency",      "Freq (MHz)"),
    ("MagneticFieldStrength", "Field (T)"),
    ("FlipAngle",             "FA (deg)"),
    ("SliceThickness",        "Slice (mm)"),
    ("EchoTime",              "TE (ms)"),
    ("RepetitionTime",        "TR (ms)"),
    ("SAR",                   "SAR"),
]
_GS_NUMERIC = {"ImagingFrequency", "MagneticFieldStrength", "FlipAngle",
               "SliceThickness", "EchoTime", "RepetitionTime", "SAR"}


class _ScanCard(QWidget):
    """
    One card per modality — large readable title + coloured rounded frame.

    Layout (matching the mockup):
        ┌─────────────────────────────┐
        │  T1 Scan          (large)   │  ← title label, coloured
        ├─────────────────────────────┤  ← thin colour accent line
        │  Scan: [dropdown    ▾     ] │
        │  [Browse folder…] [Clear]   │
        │  ✔ /path/to/scan            │
        └─────────────────────────────┘  ← rounded coloured border
    """

    path_changed = pyqtSignal(str, str)   # (modality_key, abs_path)

    def __init__(self, key: str, name: str,
                 colour: str, colour_light: str,
                 description: str, hint: str,
                 parent=None, embedded: bool = False):
        super().__init__(parent)
        self._key   = key
        self._path  = ""
        self._colour       = colour
        self._colour_light = colour_light
        self._embedded  = embedded       # True → compact sub-entry (no own frame)
        self._file_mode = False          # True → pick a single file (.MRD) not a folder
        self._file_filter = "All files (*)"

        # ── Outer frame — coloured border + rounded corners ──────────────────
        # (skipped in embedded mode, where a parent container draws the box)
        frame = None
        if not embedded:
            frame = QFrame(self)
            frame.setObjectName("scanFrame")
            frame.setStyleSheet(
                f"QFrame#scanFrame {{"
                f"  border: 2px solid {colour};"
                f"  border-radius: 10px;"
                f"  background: #1c1c1c;"
                f"}}"
            )

        # ── Title / sub-label ──────────────────────────────────────────────────
        self.title_lbl = QLabel(name)
        if embedded:
            # Smaller coloured sub-heading (e.g. "α₁") inside a shared box
            self.title_lbl.setStyleSheet(
                f"QLabel {{ color: {colour_light}; font-size: 14px; font-weight: bold;"
                f"  background: transparent; border: none; padding: 0px; }}"
            )
        else:
            self.title_lbl.setStyleSheet(
                f"QLabel {{"
                f"  color: {colour_light};"
                f"  font-size: 16px;"
                f"  font-weight: bold;"
                f"  background: transparent;"
                f"  border: none;"
                f"  padding: 0px;"
                f"}}"
            )

        # Thin horizontal accent line under the title (full card only)
        accent = QFrame()
        accent.setFrameShape(QFrame.Shape.HLine)
        accent.setFixedHeight(2)
        accent.setStyleSheet(
            f"QFrame {{ background: {colour}; border: none; max-height: 2px; }}"
        )

        # ── Controls ──────────────────────────────────────────────────────────
        # Scan picker
        pick_row = QHBoxLayout()
        scan_lbl = QLabel("Scan:")
        scan_lbl.setStyleSheet("font-size: 12px; color: #ccc; border: none;")
        pick_row.addWidget(scan_lbl)
        self.combo = QComboBox()
        # Embedded sub-scans (α₁ / α₂) sit side-by-side inside one card, so they
        # get a narrower minimum to fit two per box.
        self.combo.setMinimumWidth(110 if embedded else 210)
        self.combo.setMinimumHeight(28)
        self.combo.setStyleSheet(
            "QComboBox { background: #2a2a2a; color: #ddd; border: 1px solid #555;"
            "  border-radius: 4px; padding: 3px 6px; font-size: 12px; }"
            "QComboBox::drop-down { border: none; }"
            # Suppress the native (black) popup frame; keep a thin neutral edge
            "QComboBox QFrame { border: none; }"
            "QComboBox QAbstractScrollArea { border: none; }"
            "QComboBox QAbstractItemView {"
            "  background: #2a2a2a; color: #eee;"
            "  border: 1px solid #666; border-radius: 4px; outline: none;"
            # White highlight follows the cursor so the active row is obvious
            "  selection-background-color: #ffffff; selection-color: #000000; }"
            "QComboBox QAbstractItemView::item {"
            "  padding: 5px 10px; border: none; min-height: 22px; }"
            "QComboBox QAbstractItemView::item:selected,"
            "QComboBox QAbstractItemView::item:hover {"
            "  background: #ffffff; color: #000000; border: none; }"
        )
        self.combo.addItem("—  not assigned  —", "")
        self.combo.currentIndexChanged.connect(self._on_combo)
        pick_row.addWidget(self.combo, stretch=1)

        # Browse / Clear buttons  (compact labels when embedded side-by-side)
        browse_row = QHBoxLayout()
        if embedded:
            browse_row.setSpacing(4)
        self.btn_browse = QPushButton("Browse…" if embedded else "Browse folder…")
        self.btn_browse.setFixedHeight(28)
        self.btn_browse.setStyleSheet(
            f"QPushButton {{ background: {colour}; color: white; border: none;"
            f"  border-radius: 5px; padding: 2px {'6' if embedded else '12'}px;"
            f"  font-size: 12px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: {colour_light}; color: #111; }}"
        )
        self.btn_browse.clicked.connect(self._on_browse)
        browse_row.addWidget(self.btn_browse, stretch=1)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setFixedHeight(28)
        if embedded:
            # Size to the label (never fix it too narrow) so "Clear" is not clipped
            self.btn_clear.setMinimumWidth(58)
        else:
            self.btn_clear.setFixedWidth(58)
        self.btn_clear.setStyleSheet(
            "QPushButton { background: #3a3a3a; color: #aaa; border: 1px solid #555;"
            "  border-radius: 5px; font-size: 11px; padding: 2px 8px; }"
            "QPushButton:hover { background: #555; color: white; }"
        )
        self.btn_clear.clicked.connect(self._clear)
        browse_row.addWidget(self.btn_clear)
        browse_row.addStretch()

        # Status label + tiny path-view button row
        self.lbl_path = QLabel("No path assigned.")
        self.lbl_path.setStyleSheet(
            "QLabel { font-size: 11px; color: #666; background: transparent; border: none; }"
        )

        self._btn_view_path = QPushButton("📁")
        self._btn_view_path.setFixedSize(22, 22)
        self._btn_view_path.setVisible(False)
        self._btn_view_path.setToolTip("Click to view full path")
        self._btn_view_path.setStyleSheet(
            "QPushButton { background: #2a2a2a; color: #aaa; border: 1px solid #444;"
            "  border-radius: 4px; font-size: 11px; padding: 0px; }"
            "QPushButton:hover { background: #444; color: white; }"
        )

        def _on_view_path(checked=False, _w=self):
            from PyQt6.QtWidgets import (
                QDialog, QVBoxLayout, QLabel as _QL2,
                QPushButton as _QPB2, QHBoxLayout as _QHL2,
            )
            _dlg = QDialog(_w)
            _dlg.setWindowTitle("Scan Path")
            _dlg.setMinimumWidth(520)
            _vl = QVBoxLayout(_dlg)
            _lbl = _QL2(_w._path)
            _lbl.setWordWrap(True)
            _lbl.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            _lbl.setStyleSheet(
                "font-size: 12px; color: #4ec9b0; padding: 10px;"
            )
            _vl.addWidget(_lbl)
            _br = _QHL2()
            _br.addStretch()
            _ok = _QPB2("Close")
            _ok.clicked.connect(_dlg.accept)
            _br.addWidget(_ok)
            _vl.addLayout(_br)
            _dlg.exec()

        self._btn_view_path.clicked.connect(_on_view_path)

        _path_row = QHBoxLayout()
        _path_row.setContentsMargins(0, 0, 0, 0)
        _path_row.addWidget(self.lbl_path, stretch=1)
        _path_row.addWidget(self._btn_view_path)

        if embedded:
            # ── Compact sub-entry — no frame/accent; parent box draws the border
            outer = QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(6)
            outer.addWidget(self.title_lbl)
            outer.addLayout(pick_row)
            outer.addLayout(browse_row)
            outer.addLayout(_path_row)
        else:
            # ── Frame inner layout ──────────────────────────────────────────────
            # A trailing stretch absorbs any extra card height (from the card's
            # minimum height) so rows keep their natural spacing and never overlap.
            frame_lay = QVBoxLayout(frame)
            frame_lay.setContentsMargins(14, 10, 14, 12)
            frame_lay.setSpacing(9)
            frame_lay.addWidget(self.title_lbl)
            frame_lay.addWidget(accent)
            frame_lay.addLayout(pick_row)
            frame_lay.addLayout(browse_row)
            frame_lay.addLayout(_path_row)
            frame_lay.addStretch(1)

            # ── Outer widget layout ─────────────────────────────────────────────
            outer = QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.addWidget(frame)

    # ── public ────────────────────────────────────────────────────────────────

    def populate(self, entries: list[str], scan_nums: list[str], study_dir: str):
        """Fill the dropdown from the study browser scan list."""
        self._study_dir = study_dir
        self._scan_nums  = scan_nums
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("—  not assigned  —", "")
        for entry, num in zip(entries, scan_nums):
            # "2 ----T1map_RARE" → "2  T1map_RARE"  (remove ---- separator)
            clean = re.sub(r'\s*-{2,}\s*', '  ', entry).strip()
            self.combo.addItem(clean, num)
        self.combo.blockSignals(False)

    def get_path(self) -> str:
        return self._path

    # ── internals ─────────────────────────────────────────────────────────────

    def _set_path(self, path: str):
        self._path = path
        if path:
            self.lbl_path.setText(" Assigned")
            self.lbl_path.setStyleSheet(
                "QLabel { font-size: 11px; color: #4ec9b0; "
                "background: transparent; border: none; }"
            )
            self._btn_view_path.setVisible(True)
        else:
            self.lbl_path.setText("No path assigned.")
            self.lbl_path.setStyleSheet(
                "QLabel { font-size: 11px; color: #666; "
                "background: transparent; border: none; }"
            )
            self._btn_view_path.setVisible(False)
        self.path_changed.emit(self._key, path)

    def set_file_mode(self, on: bool, file_filter: str = "All files (*)"):
        """Switch the card between folder-assignment and single-file (.MRD) mode."""
        self._file_mode = on
        self._file_filter = file_filter
        if self._embedded:      # keep the compact label for side-by-side sub-scans
            self.btn_browse.setText("File…" if on else "Browse…")
        else:
            self.btn_browse.setText("Browse file…" if on else "Browse folder…")

    def populate_files(self, files: list[str], study_dir: str):
        """Fill the dropdown with files (abs paths stored in itemData)."""
        self._study_dir = study_dir
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("—  not assigned  —", "")
        for f in files:
            label = os.path.relpath(f, study_dir) if study_dir else os.path.basename(f)
            self.combo.addItem(label, f)      # itemData = absolute file path
        self.combo.setCurrentIndex(0)
        self.combo.blockSignals(False)

    def populate_gs_series(self, entries: list):
        """Fill the dropdown from GE / Siemens detected series.

        entries : list of (display_label, abs_series_dir). The single-series
        folder is stored as itemData so assigning it resolves straight to a
        folder the DICOM readers can consume — exactly like a Bruker scan.
        """
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("—  not assigned  —", "")
        for label, series_dir in entries:
            self.combo.addItem(label, series_dir)   # itemData = abs series dir
        self.combo.setCurrentIndex(0)
        self.combo.blockSignals(False)

    def _on_combo(self, idx: int):
        data = self.combo.itemData(idx)
        if not data:
            self._set_path("")
            return
        # Absolute path in itemData → GE/Siemens series folder or MR Solutions file
        if os.path.isabs(str(data)):
            self._set_path(data if (os.path.isdir(data) or os.path.isfile(data)) else "")
            return
        # File mode → itemData is already an absolute file path
        if self._file_mode:
            self._set_path(data if os.path.isfile(data) else "")
            return
        # Resolve absolute path: study_dir/num/pdata/1
        base = getattr(self, "_study_dir", "")
        if base:
            candidate = os.path.join(base, data, "pdata", "1")
            if os.path.isdir(candidate):
                self._set_path(candidate)
                return
            # Fall back to raw scan directory
            candidate2 = os.path.join(base, data)
            if os.path.isdir(candidate2):
                self._set_path(candidate2)
                return
        self._set_path("")

    def _on_browse(self):
        if self._file_mode:
            f, _ = QFileDialog.getOpenFileName(
                self, f"Select {self.title_lbl.text().strip()} file", "",
                self._file_filter
            )
            if f:
                self.combo.blockSignals(True)
                self.combo.setCurrentIndex(0)
                self.combo.blockSignals(False)
                self._set_path(f)
            return
        d = QFileDialog.getExistingDirectory(
            self, f"Select {self.title_lbl.text().strip()} directory", ""
        )
        if d:
            self.combo.blockSignals(True)
            self.combo.setCurrentIndex(0)   # reset to "not assigned"
            self.combo.blockSignals(False)
            self._set_path(d)

    def _clear(self):
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(0)
        self.combo.blockSignals(False)
        self._set_path("")


class _B1VFAScanCard(QWidget):
    """One box titled *B1 VFA Scan* holding both double-angle sub-scans
    (α₁ left, α₂ right) side-by-side, so the card stays the same height as the
    single-scan cards and does not stretch its grid row. Each sub-scan is a
    compact (embedded)
    ``_ScanCard`` — exposed as ``card_fa1`` / ``card_fa2`` so the tab registers
    them under the usual ``b1_fa1`` / ``b1_fa2`` keys and every existing
    populate / file-mode / path-lookup path keeps working unchanged."""

    _COLOUR       = "#8e44ad"
    _COLOUR_LIGHT = "#b47dd4"

    def __init__(self, parent=None):
        super().__init__(parent)

        frame = QFrame(self)
        frame.setObjectName("scanFrame")
        frame.setStyleSheet(
            f"QFrame#scanFrame {{ border: 2px solid {self._COLOUR};"
            f"  border-radius: 10px; background: #1c1c1c; }}"
        )

        title = QLabel("B1 VFA Scan")
        title.setStyleSheet(
            f"QLabel {{ color: {self._COLOUR_LIGHT}; font-size: 16px; font-weight: bold;"
            f"  background: transparent; border: none; padding: 0px; }}"
        )
        accent = QFrame()
        accent.setFrameShape(QFrame.Shape.HLine)
        accent.setFixedHeight(2)
        accent.setStyleSheet(
            f"QFrame {{ background: {self._COLOUR}; border: none; max-height: 2px; }}"
        )

        # Two compact sub-scans (keep the original per-flip-angle colours)
        self.card_fa1 = _ScanCard("b1_fa1", "α₁", "#8e44ad", "#b47dd4",
                                  "", "", embedded=True)
        self.card_fa2 = _ScanCard("b1_fa2", "α₂", "#6c3483", "#9b59b6",
                                  "", "", embedded=True)

        # Vertical divider between the two side-by-side sub-scans
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setFixedWidth(1)
        divider.setStyleSheet("QFrame { background: #333; border: none; max-width: 1px; }")

        # α₁ on the left, α₂ on the right — keeps this card the same height as
        # the single-scan cards so the grid row is not stretched.
        subs = QHBoxLayout()
        subs.setContentsMargins(0, 0, 0, 0)
        subs.setSpacing(10)
        subs.addWidget(self.card_fa1, stretch=1)
        subs.addWidget(divider)
        subs.addWidget(self.card_fa2, stretch=1)

        frame_lay = QVBoxLayout(frame)
        frame_lay.setContentsMargins(14, 10, 14, 12)
        frame_lay.setSpacing(9)
        frame_lay.addWidget(title)
        frame_lay.addWidget(accent)
        frame_lay.addLayout(subs)
        frame_lay.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)


# ── Main tab ──────────────────────────────────────────────────────────────────

class ScanDirTab(QWidget):
    """
    Scan directory tab.

    Supports two scanner families:

    Bruker
        Browse a study root → scan list → assign each scan number to a
        modality via dropdown or folder browse.

    GE / Siemens
        Browse individual DICOM or NIfTI folders for each modality.
        Format is selected once (DICOM / NIfTI) and applies to all scans.
    """

    # Emitted whenever any modality path changes: (key, abs_path)
    scan_assigned = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._study_dir: str = ""
        self._cards: dict[str, _ScanCard] = {}

        # Scrollable content — cards keep their natural size and the user can
        # scroll on short windows instead of the rows compressing/overlapping.
        _scroll = QScrollArea(self)
        _scroll.setWidgetResizable(True)
        _scroll.setFrameShape(QFrame.Shape.NoFrame)
        _content = QWidget()
        _scroll.setWidget(_content)
        _self_lay = QVBoxLayout(self)
        _self_lay.setContentsMargins(0, 0, 0, 0)
        _self_lay.addWidget(_scroll)

        outer = QVBoxLayout(_content)
        outer.setSpacing(10)
        outer.setContentsMargins(10, 10, 10, 10)

        # ── Scanner / vendor selector ─────────────────────────────────────
        scanner_grp = QGroupBox("Scanner")
        scanner_grp.setStyleSheet(_GROUP_BOX_STYLE)
        scanner_lay = QHBoxLayout(scanner_grp)
        scanner_lay.addWidget(QLabel("Platform:"))
        self.combo_scanner = QComboBox()
        self.combo_scanner.addItems(["Bruker", "GE / Siemens", "MR Solutions"])
        self.combo_scanner.setFixedWidth(160)
        self.combo_scanner.setToolTip(
            "Bruker: browse a study root directory.\n"
            "GE / Siemens: browse individual DICOM or NIfTI folders.\n"
            "MR Solutions: browse a folder of .MRD files and assign one per scan."
        )
        self.combo_scanner.currentIndexChanged.connect(self._on_scanner_changed)
        scanner_lay.addWidget(self.combo_scanner)

        scanner_lay.addStretch()
        outer.addWidget(scanner_grp)

        # ── Stacked widget: Bruker browser | GE/Siemens info ─────────────
        self._dir_stack = QStackedWidget()

        # Page 0 — Bruker study root browser
        bruker_page = QWidget()
        bruker_lay  = QVBoxLayout(bruker_page)
        bruker_lay.setContentsMargins(0, 0, 0, 0)

        root_grp = QGroupBox("Bruker Study Directory")
        root_grp.setStyleSheet(_GROUP_BOX_STYLE)
        root_lay = QVBoxLayout(root_grp)

        dir_row = QHBoxLayout()
        self.edit_study = QLineEdit()
        self.edit_study.setReadOnly(True)
        self.edit_study.setStyleSheet(
            "QLineEdit { background: #2a2a2a; color: #ccc; "
            "border: 1px solid #555; border-radius: 4px; padding: 4px 8px; }"
        )
        dir_row.addWidget(self.edit_study, stretch=1)

        self.btn_browse_root = QPushButton("Browse…")
        self.btn_browse_root.setFixedHeight(32)
        self.btn_browse_root.setStyleSheet(
            "QPushButton { background: #0d6efd; color: white; border: none; "
            "border-radius: 5px; padding: 4px 14px; font-weight: bold; }"
            "QPushButton:hover { background: #3d8bfd; }"
        )
        self.btn_browse_root.clicked.connect(self._browse_root)
        dir_row.addWidget(self.btn_browse_root)
        root_lay.addLayout(dir_row)

        # Bruker version selector + info
        meta_row = QHBoxLayout()
        meta_row.addWidget(QLabel("Bruker version:"))
        self.combo_pv = QComboBox()
        self.combo_pv.addItems(["PV360", "PV6 / PV7"])
        self.combo_pv.setToolTip(
            "Bruker ParaVision version — auto-detected from subject file."
        )
        self.combo_pv.setFixedWidth(110)
        meta_row.addWidget(self.combo_pv)
        self.lbl_study_meta = QLabel("")
        self.lbl_study_meta.setStyleSheet("font-size: 11px; color: #888;")
        meta_row.addWidget(self.lbl_study_meta, stretch=1)

        self.btn_refresh = QPushButton("↻  Refresh")
        self.btn_refresh.setFixedHeight(26)
        self.btn_refresh.setStyleSheet(
            "QPushButton { background: #333; color: #aaa; border: none; "
            "border-radius: 4px; padding: 2px 10px; font-size: 11px; }"
            "QPushButton:hover { background: #555; color: white; }"
        )
        self.btn_refresh.clicked.connect(self._refresh_scans)
        meta_row.addWidget(self.btn_refresh)
        root_lay.addLayout(meta_row)
        bruker_lay.addWidget(root_grp)

        # Scan list preview (Bruker only)
        list_grp = QGroupBox("Detected Scan List")
        list_grp.setStyleSheet(_GROUP_BOX_STYLE)
        list_lay = QVBoxLayout(list_grp)

        from PyQt6.QtWidgets import QListWidget, QListWidgetItem
        self.lst_scan_list = QListWidget()
        self.lst_scan_list.setFont(QFont("Arial", 13))
        self.lst_scan_list.setFixedHeight(130)
        self.lst_scan_list.setStyleSheet("""
            QListWidget {
                background: #1a1a1a;
                color: #ddd;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 4px;
                font-family: Arial;
                font-size: 13px;
                outline: none;
            }
            QListWidget::item {
                padding: 5px 8px;
                border-radius: 3px;
            }
            QListWidget::item:selected {
                background: #1565c0;
                color: white;
            }
            QListWidget::item:hover:!selected {
                background: #2a2a2a;
            }
        """)
        self.lst_scan_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.lst_scan_list.setToolTip("Click a scan entry to see its detected path below.")
        self.lst_scan_list.currentRowChanged.connect(self._on_scanlist_row_changed)
        self.txt_scan_list = self.lst_scan_list  # backward-compat alias

        self.lbl_scan_detail = QLabel("")
        self.lbl_scan_detail.setFont(QFont("Arial", 11))
        self.lbl_scan_detail.setWordWrap(True)
        self.lbl_scan_detail.setStyleSheet(
            "QLabel { color: #4ec9b0; font-size: 11px; padding: 2px 4px; "
            "background: transparent; border: none; }"
        )

        list_lay.addWidget(self.lst_scan_list)
        list_lay.addWidget(self.lbl_scan_detail)
        bruker_lay.addWidget(list_grp)

        self._dir_stack.addWidget(bruker_page)   # index 0

        # Page 1 — GE / Siemens study directory browser
        gs_page = QWidget()
        gs_lay  = QVBoxLayout(gs_page)
        gs_lay.setContentsMargins(0, 0, 0, 0)
        gs_lay.setSpacing(6)

        gs_grp = QGroupBox("GE / Siemens Study Directory")
        gs_grp.setStyleSheet(_GROUP_BOX_STYLE)
        gs_root_lay = QVBoxLayout(gs_grp)

        # Directory path row
        gs_dir_row = QHBoxLayout()
        self.edit_gs_study = QLineEdit()
        self.edit_gs_study.setReadOnly(True)
        self.edit_gs_study.setPlaceholderText("Select the folder containing DICOM / NIfTI files…")
        self.edit_gs_study.setStyleSheet(
            "QLineEdit { background: #2a2a2a; color: #ccc; "
            "border: 1px solid #555; border-radius: 4px; padding: 4px 8px; }"
        )
        gs_dir_row.addWidget(self.edit_gs_study, stretch=1)
        self.btn_browse_gs = QPushButton("Browse…")
        self.btn_browse_gs.setFixedHeight(32)
        self.btn_browse_gs.setStyleSheet(
            "QPushButton { background: #0d6efd; color: white; border: none; "
            "border-radius: 5px; padding: 4px 14px; font-weight: bold; }"
            "QPushButton:hover { background: #3d8bfd; }"
        )
        self.btn_browse_gs.clicked.connect(self._browse_gs_study)
        gs_dir_row.addWidget(self.btn_browse_gs)
        gs_root_lay.addLayout(gs_dir_row)

        # Format + Detect row
        gs_fmt_row = QHBoxLayout()
        gs_fmt_row.addWidget(QLabel("Format:"))
        self.combo_gs_format = QComboBox()
        self.combo_gs_format.addItems(["DICOM (.dcm)", "NIfTI (.nii)"])
        self.combo_gs_format.setFixedWidth(140)
        self.combo_gs_format.setToolTip(
            "DICOM: folder contains .dcm files (one per TR/TE/offset).\n"
            "NIfTI: T1 filenames include TR value (e.g. Tr_100); "
            "T2 echo files named *_e1.nii, *_e2.nii, …"
        )
        gs_fmt_row.addWidget(self.combo_gs_format)
        self.btn_gs_detect = QPushButton("↻  Detect")
        self.btn_gs_detect.setFixedHeight(26)
        self.btn_gs_detect.setStyleSheet(
            "QPushButton { background: #333; color: #aaa; border: none; "
            "border-radius: 4px; padding: 2px 10px; font-size: 11px; }"
            "QPushButton:hover { background: #555; color: white; }"
        )
        self.btn_gs_detect.clicked.connect(self._refresh_gs_info)
        gs_fmt_row.addWidget(self.btn_gs_detect)
        gs_fmt_row.addStretch()
        gs_root_lay.addLayout(gs_fmt_row)

        # Auto-detected scanner info
        self.lbl_gs_info = QLabel("Browse a folder to auto-detect scanner info.")
        self.lbl_gs_info.setStyleSheet("font-size: 11px; color: #888;")
        self.lbl_gs_info.setWordWrap(True)
        gs_root_lay.addWidget(self.lbl_gs_info)

        gs_lay.addWidget(gs_grp)

        # GE/Siemens detected scan list
        gs_list_grp = QGroupBox("Detected Scan Series")
        gs_list_grp.setStyleSheet(_GROUP_BOX_STYLE)
        gs_list_lay = QVBoxLayout(gs_list_grp)
        from PyQt6.QtWidgets import QListWidget, QListWidgetItem as _LWI
        self.lst_gs_scans = QListWidget()
        self.lst_gs_scans.setFont(QFont("Arial", 12))
        self.lst_gs_scans.setFixedHeight(130)
        self.lst_gs_scans.setStyleSheet("""
            QListWidget { background: #1a1a1a; color: #ddd; border: 1px solid #333;
                          border-radius: 4px; padding: 4px; font-family: Arial; font-size: 12px; }
            QListWidget::item { padding: 3px 6px; border-radius: 3px; }
            QListWidget::item:selected { background: #1565c0; color: white; }
            QListWidget::item:hover:!selected { background: #2a2a2a; }
        """)
        self.lst_gs_scans.setToolTip("Click a series to see its path and DICOM details.")
        self.lst_gs_scans.currentRowChanged.connect(self._on_gs_scanlist_row_changed)
        gs_list_lay.addWidget(self.lst_gs_scans)
        self.lbl_gs_scan_detail = QLabel("")
        self.lbl_gs_scan_detail.setStyleSheet("font-size: 10px; color: #888;")
        self.lbl_gs_scan_detail.setWordWrap(True)
        gs_list_lay.addWidget(self.lbl_gs_scan_detail)
        gs_lay.addWidget(gs_list_grp)

        # Internal state for GE/Siemens
        self._gs_study_dir: str = ""
        self._gs_scan_dirs: list[str] = []
        # Detected series: list of dict(dir, num, desc, n, tags) — one per series
        self._gs_series: list[dict] = []

        self._dir_stack.addWidget(gs_page)       # index 1

        # Page 2 — MR Solutions (.MRD) folder browser
        mrd_page = QWidget()
        mrd_lay  = QVBoxLayout(mrd_page)
        mrd_lay.setContentsMargins(0, 0, 0, 0)
        mrd_lay.setSpacing(6)

        mrd_grp = QGroupBox("MR Solutions Data Folder")
        mrd_grp.setStyleSheet(_GROUP_BOX_STYLE)
        mrd_root_lay = QVBoxLayout(mrd_grp)

        mrd_dir_row = QHBoxLayout()
        self.edit_mrd_study = QLineEdit()
        self.edit_mrd_study.setReadOnly(True)
        self.edit_mrd_study.setPlaceholderText("Select the folder that holds the .MRD files…")
        self.edit_mrd_study.setStyleSheet(
            "QLineEdit { background: #2a2a2a; color: #ccc; "
            "border: 1px solid #555; border-radius: 4px; padding: 4px 8px; }"
        )
        mrd_dir_row.addWidget(self.edit_mrd_study, stretch=1)
        self.btn_browse_mrd = QPushButton("Browse…")
        self.btn_browse_mrd.setFixedHeight(32)
        self.btn_browse_mrd.setStyleSheet(
            "QPushButton { background: #0d6efd; color: white; border: none; "
            "border-radius: 5px; padding: 4px 14px; font-weight: bold; }"
            "QPushButton:hover { background: #3d8bfd; }"
        )
        self.btn_browse_mrd.clicked.connect(self._browse_mrd_study)
        mrd_dir_row.addWidget(self.btn_browse_mrd)
        mrd_root_lay.addLayout(mrd_dir_row)

        self.lbl_mrd_info = QLabel("")   # populated after a folder is browsed
        self.lbl_mrd_info.setStyleSheet("font-size: 11px; color: #888;")
        self.lbl_mrd_info.setWordWrap(True)
        mrd_root_lay.addWidget(self.lbl_mrd_info)
        mrd_lay.addWidget(mrd_grp)

        self._mrd_study_dir: str = ""
        self._mrd_files: list[str] = []

        self._dir_stack.addWidget(mrd_page)      # index 2

        outer.addWidget(self._dir_stack)

        # ── Modality cards ────────────────────────────────────────────────
        cards_grp = QGroupBox("Assign respective scans")
        # Match the scan-card titles ("T1 Scan", …) exactly: 16px bold.
        cards_grp.setStyleSheet(_GROUP_BOX_STYLE + """
            QGroupBox::title { font-size: 16px; font-weight: bold; }
        """)
        grid = QGridLayout(cards_grp)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(18)
        grid.setContentsMargins(12, 18, 12, 12)

        for i, (key, name, col, col_l, desc, hint) in enumerate(_MODALITIES):
            if key == "__b1vfa__":
                # One box holding both B1 double-angle sub-scans (α₁ + α₂)
                widget = _B1VFAScanCard()
                for subcard, subkey in ((widget.card_fa1, "b1_fa1"),
                                        (widget.card_fa2, "b1_fa2")):
                    subcard.path_changed.connect(self.scan_assigned)
                    subcard.path_changed.connect(self._on_path_changed)
                    self._cards[subkey] = subcard
                # α₁ / α₂ sit side-by-side, so this box needs barely more height
                # than a single-scan card — keeping it small stops the whole grid
                # row (T1 / T2) from being stretched.
                widget.setMinimumHeight(170)
            else:
                widget = _ScanCard(key, name, col, col_l, desc, hint)
                widget.path_changed.connect(self.scan_assigned)
                widget.path_changed.connect(self._on_path_changed)
                self._cards[key] = widget
                # Firm minimum height so cards never compress into overlapping rows.
                widget.setMinimumHeight(150)
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            row, col_idx = divmod(i, 3)
            grid.addWidget(widget, row, col_idx)

        outer.addWidget(cards_grp)

        # ── Summary bar ───────────────────────────────────────────────────
        self.lbl_summary = QLabel("No scans assigned yet.")
        self.lbl_summary.setStyleSheet(
            "background: #1e2a1e; color: #4ec9b0; border-radius: 6px; "
            "padding: 6px 12px; font-size: 12px;"
        )
        self.lbl_summary.setWordWrap(True)
        outer.addWidget(self.lbl_summary)

        outer.addStretch()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_scan_paths(self) -> dict[str, str]:
        """Return {modality_key: abs_path} for all assigned modalities."""
        return {k: c.get_path() for k, c in self._cards.items() if c.get_path()}

    def get_study_dir(self) -> str:
        return self._study_dir

    def is_pv360(self) -> bool:
        return self.combo_pv.currentText() == "PV360"

    def get_vendor(self) -> str:
        """Return 'bruker', 'ge_siemens', or 'mr_solutions'."""
        return {0: "bruker", 1: "ge_siemens", 2: "mr_solutions"}.get(
            self.combo_scanner.currentIndex(), "bruker"
        )

    def get_mrd_larmor_mhz(self) -> float:
        """Larmor frequency (MHz) for MR Solutions ppm conversion."""
        return 199.7502

    def get_gs_format(self) -> str:
        """Return 'dicom' or 'nifti' (only meaningful when vendor is GE/Siemens)."""
        return "dicom" if self.combo_gs_format.currentIndex() == 0 else "nifti"

    # ── Internals ─────────────────────────────────────────────────────────────

    def _on_scanner_changed(self, idx: int):
        """Switch between Bruker / GE-Siemens / MR Solutions UI panels."""
        self._dir_stack.setCurrentIndex(idx)
        # MR Solutions → cards pick single .MRD files; otherwise folders
        mrd_mode = (idx == 2)
        for card in self._cards.values():
            card.set_file_mode(mrd_mode, "MR Solutions raw (*.MRD *.mrd);;All files (*)")
        if mrd_mode:
            self._refresh_mrd_files()
        elif idx == 1 and self._gs_series:
            # GE/Siemens — restore the detected-series dropdowns on the cards
            self._populate_gs_cards()
        elif idx == 0 and getattr(self, "_scan_entries", None):
            # Bruker — restore the scan-number dropdowns
            for card in self._cards.values():
                card.populate(self._scan_entries, self._scan_nums_list, self._study_dir)

    def _browse_mrd_study(self):
        """Browse a folder of .MRD files and populate the modality cards."""
        d = QFileDialog.getExistingDirectory(
            self, "Select MR Solutions data folder (containing .MRD files)", ""
        )
        if not d:
            return
        self._mrd_study_dir = d
        self.edit_mrd_study.setText(d)
        self._refresh_mrd_files()

    def _refresh_mrd_files(self):
        """Scan the MR Solutions folder (recursively) for .MRD files."""
        import glob as _glob
        path = self._mrd_study_dir
        if not path:
            return
        files = sorted(set(
            _glob.glob(os.path.join(path, "*.MRD")) +
            _glob.glob(os.path.join(path, "*.mrd")) +
            _glob.glob(os.path.join(path, "**", "*.MRD"), recursive=True) +
            _glob.glob(os.path.join(path, "**", "*.mrd"), recursive=True)
        ))
        self._mrd_files = files
        for card in self._cards.values():
            card.populate_files(files, path)
        if files:
            self.lbl_mrd_info.setText(
                f"{len(files)} .MRD file(s) found — assign CEST and WASSR / B0 "
                f"in the cards below."
            )
            self.lbl_mrd_info.setStyleSheet("font-size: 11px; color: #4ec9b0;")
        else:
            self.lbl_mrd_info.setText("No .MRD files found in the selected folder.")
            self.lbl_mrd_info.setStyleSheet("font-size: 11px; color: orange;")

    def _browse_gs_study(self):
        """Browse a GE / Siemens data folder and auto-detect scanner info."""
        d = QFileDialog.getExistingDirectory(
            self, "Select GE / Siemens data folder (containing DICOM or NIfTI files)", ""
        )
        if not d:
            return
        self._gs_study_dir = d
        self.edit_gs_study.setText(d)
        self._refresh_gs_info()

    def _refresh_gs_info(self):
        """
        Auto-detect scanner info and build the DICOM series list for the chosen
        GE / Siemens folder.  Works for BOTH layouts:

          * a flat folder holding every scan's DICOM files together (grouped by
            SeriesInstanceUID / SeriesNumber, like explore_dicom_folder.m), and
          * an already-sorted study with one sub-folder per series.

        Each detected series is materialised into its own single-series folder
        of .dcm links, then offered in every modality card's dropdown so the
        user can assign it to T1 / T2 / CEST / MRF … exactly like Bruker.
        """
        path = self._gs_study_dir
        if not path:
            self.lbl_gs_info.setText("No folder selected.")
            return

        import glob as _glob
        fmt = self.get_gs_format()

        if fmt == "nifti":
            # NIfTI mode — just list .nii files
            nii_files = (
                _glob.glob(os.path.join(path, "*.nii")) +
                _glob.glob(os.path.join(path, "*.nii.gz")) +
                _glob.glob(os.path.join(path, "**/*.nii"), recursive=True)
            )
            if nii_files:
                self.lbl_gs_info.setText(
                    f"NIfTI mode  —  {len(nii_files)} .nii file(s) found in folder."
                )
                self.lbl_gs_info.setStyleSheet("font-size: 11px; color: #4ec9b0;")
            else:
                self.lbl_gs_info.setText("No .nii files found in the selected folder.")
                self.lbl_gs_info.setStyleSheet("font-size: 11px; color: orange;")
            self.lst_gs_scans.clear()
            self._gs_series = []
            return

        # DICOM mode
        try:
            import pydicom
        except ImportError:
            self.lbl_gs_info.setText(
                "pydicom not installed — cannot read DICOM.\n"
                "Install with:  pip install pydicom"
            )
            self.lbl_gs_info.setStyleSheet("font-size: 11px; color: orange;")
            return

        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt as _Qt
        QApplication.setOverrideCursor(_Qt.CursorShape.WaitCursor)
        self.lbl_gs_info.setText("Scanning DICOM headers…")
        self.lbl_gs_info.setStyleSheet("font-size: 11px; color: #888;")
        QApplication.processEvents()
        try:
            ordered = self._scan_gs_dicom_series(path, pydicom)
        finally:
            QApplication.restoreOverrideCursor()

        if not ordered:
            self.lbl_gs_info.setText(
                "No DICOM series found. Check the folder or switch to NIfTI format."
            )
            self.lbl_gs_info.setStyleSheet("font-size: 11px; color: orange;")
            self.lst_gs_scans.clear()
            self._gs_series = []
            return

        # Vendor / scanner banner from the first series' header
        ds0 = ordered[0][1]['first']
        manufacturer = str(getattr(ds0, 'Manufacturer', '')).strip()
        model        = str(getattr(ds0, 'ManufacturerModelName', '')).strip()
        field_T      = getattr(ds0, 'MagneticFieldStrength', None)
        freq_MHz     = getattr(ds0, 'ImagingFrequency', None)
        mfg_up = manufacturer.upper()
        if 'GE' in mfg_up or 'GEMS' in mfg_up:
            vendor_label = 'GE Medical Systems'
        elif 'SIEMENS' in mfg_up:
            vendor_label = 'Siemens'
        else:
            vendor_label = manufacturer or 'Unknown vendor'
        parts = [f"Platform: {vendor_label}"]
        if model:
            parts.append(f"Model: {model}")
        if field_T is not None:
            try:
                parts.append(f"Field: {float(field_T):.2g} T")
            except (TypeError, ValueError):
                pass
        if freq_MHz is not None:
            try:
                parts.append(f"Freq: {float(freq_MHz):.2f} MHz")
            except (TypeError, ValueError):
                pass
        parts.append(f"{len(ordered)} series")
        self.lbl_gs_info.setText("   |   ".join(parts))
        self.lbl_gs_info.setStyleSheet("font-size: 11px; color: #4ec9b0;")

        self._build_gs_series(path, ordered)

    def _scan_gs_dicom_series(self, root: str, pydicom_mod) -> list:
        """Read every DICOM header under *root* (flat or sub-foldered) and group
        by SeriesInstanceUID.  Returns a list of (key, info) ordered by
        SeriesNumber, where info = dict(num, files, first, dirs)."""
        import glob as _glob
        from collections import OrderedDict

        files: list[str] = []
        for ext in ("*.dcm", "*.DCM", "*.ima", "*.IMA"):
            files += _glob.glob(os.path.join(root, ext))
            files += _glob.glob(os.path.join(root, "**", ext), recursive=True)
        files = sorted(set(files))
        if not files:
            # Extension-less DICOMs — walk and probe every plausible file
            skip_ext = {'.txt', '.mat', '.json', '.png', '.jpg', '.jpeg', '.eps',
                        '.pdf', '.nii', '.gz', '.py', '.csv', '.xml', '.zip',
                        '.log', '.md', '.html'}
            for dp, _dn, fns in os.walk(root):
                for fn in fns:
                    if fn.startswith('.'):
                        continue
                    if os.path.splitext(fn)[1].lower() in skip_ext:
                        continue
                    files.append(os.path.join(dp, fn))
            files = sorted(set(files))

        MAX_FILES = 30000
        if len(files) > MAX_FILES:
            files = files[:MAX_FILES]

        series: "OrderedDict[str, dict]" = OrderedDict()
        for f in files:
            try:
                ds = pydicom_mod.dcmread(f, stop_before_pixels=True, force=True)
            except Exception:
                continue
            uid = getattr(ds, 'SeriesInstanceUID', None)
            num = getattr(ds, 'SeriesNumber', None)
            if uid is None and num is None:
                continue                       # not a real image DICOM
            key = str(uid) if uid is not None else f"num_{num}"
            info = series.get(key)
            if info is None:
                info = dict(num=num, files=[], first=ds, dirs=set())
                series[key] = info
            info['files'].append(f)
            info['dirs'].add(os.path.dirname(f))

        def _skey(item):
            n = item[1]['num']
            try:
                return (0, int(n))
            except (TypeError, ValueError):
                return (1, str(n))
        return sorted(series.items(), key=_skey)

    def _build_gs_series(self, root: str, ordered: list):
        """Materialise each detected series into a clean folder, fill the preview
        list, and populate every modality card's dropdown."""
        from PyQt6.QtWidgets import QListWidgetItem

        self.lst_gs_scans.clear()
        self._gs_series = []

        for key, info in ordered:
            ds  = info['first']
            num = info['num']
            n   = len(info['files'])
            desc = (str(getattr(ds, 'SeriesDescription', '')).strip()
                    or str(getattr(ds, 'ProtocolName', '')).strip()
                    or str(getattr(ds, 'SequenceName', '')).strip()
                    or 'series')
            num_s = str(num) if num is not None else '?'
            series_dir = self._materialize_series_dir(root, key, num_s, desc, info['files'])
            tags = self._read_gs_tags(ds)

            label = f"  {num_s:>4}   ·   {desc}   ·   {n} img"
            item  = QListWidgetItem(label)
            item.setFont(QFont("Arial", 12))
            self.lst_gs_scans.addItem(item)
            self._gs_series.append(
                dict(dir=series_dir, num=num_s, desc=desc, n=n, tags=tags)
            )

        if not self._gs_series:
            item = QListWidgetItem("  (no DICOM series found)")
            item.setFont(QFont("Arial", 11))
            item.setForeground(QColor("#666"))
            self.lst_gs_scans.addItem(item)

        self._populate_gs_cards()

    def _populate_gs_cards(self):
        """Push the detected GE/Siemens series into every modality-card dropdown."""
        entries = [(f"{s['num']}  ·  {s['desc']}  ({s['n']})", s['dir'])
                   for s in self._gs_series]
        for card in self._cards.values():
            card.populate_gs_series(entries)

    @staticmethod
    def _read_gs_tags(ds) -> list:
        """Pull the display tags (from explore_dicom_folder.m + extras) off a
        DICOM header as a list of (label, value) pairs, numeric ones formatted."""
        out = []
        for tag, label in _GS_TAGS:
            val = getattr(ds, tag, None)
            if val is None or str(val).strip() == '':
                continue
            if tag in _GS_NUMERIC:
                try:
                    val = f"{float(val):g}"
                except (TypeError, ValueError):
                    val = str(val)
            out.append((label, str(val)))
        return out

    def _materialize_series_dir(self, root: str, key: str, num: str,
                                desc: str, files: list) -> str:
        """Create a clean single-series folder of .dcm links (fallback copies) so
        the folder-based DICOM readers see only this series' files.  Handles both
        flat folders (all scans together) and already-sub-foldered studies, and
        guarantees a .dcm extension even for extension-less source DICOMs."""
        import hashlib
        import tempfile
        import shutil

        tag  = hashlib.md5(f"{root}|{key}".encode()).hexdigest()[:8]
        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', str(desc))[:40].strip('_') or 'series'
        d = os.path.join(tempfile.gettempdir(), 'mrf_gui_gs_series',
                         f"{num}_{safe}_{tag}")
        os.makedirs(d, exist_ok=True)
        # Clear any stale contents from a previous detection
        for old in os.listdir(d):
            try:
                os.remove(os.path.join(d, old))
            except OSError:
                pass
        for i, f in enumerate(sorted(files)):
            src  = os.path.abspath(f)
            link = os.path.join(d, f"{i:04d}.dcm")
            try:
                os.symlink(src, link)
            except (OSError, NotImplementedError, AttributeError):
                try:
                    shutil.copy2(src, link)
                except OSError:
                    pass
        return d

    def _on_gs_scanlist_row_changed(self, row: int):
        """Show the full DICOM tag set for the selected series."""
        if row < 0 or row >= len(self._gs_series):
            self.lbl_gs_scan_detail.setText("")
            return
        s = self._gs_series[row]
        bits = [f"{lbl}: {val}" for lbl, val in s['tags']]
        detail = f"Path: {s['dir']}   |   {s['n']} files"
        if bits:
            detail += "   |   " + "   |   ".join(bits)
        self.lbl_gs_scan_detail.setText(detail)

    def _browse_root(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Bruker study root directory", ""
        )
        if not d:
            return
        self._study_dir = d
        self.edit_study.setText(d)
        self._refresh_scans()

    def _refresh_scans(self):
        if not self._study_dir:
            return

        ver = _detect_pv_version(self._study_dir)
        self.combo_pv.setCurrentText("PV360" if ver == "PV360" else "PV6 / PV7")
        pv_str = f"Bruker {ver}" if ver != "unknown" else "Bruker (version unknown)"
        self.lbl_study_meta.setText(f"Auto-detected: {pv_str}")

        entries, scan_nums = build_scan_list(self._study_dir)

        # Strip Bruker experiment-number suffixes (E1), (E2) … from all entries
        entries = [_strip_expno(e) for e in entries]

        # Populate scan list widget — clean up "----" separator
        from PyQt6.QtWidgets import QListWidgetItem, QMessageBox
        from PyQt6.QtGui import QColor
        self.lst_scan_list.clear()
        self.lbl_scan_detail.setText("")
        self._scan_entries = entries    # store for detail lookup
        self._scan_nums_list = scan_nums
        if entries:
            for entry in entries:
                # Convert "11 ----fpSL_EPI_an30nSL" → "11  ·  fpSL_EPI_an30nSL"
                clean = entry.replace("----", " ").strip()
                parts = clean.split(None, 1)  # split on first whitespace
                if len(parts) == 2:
                    num, name = parts[0], parts[1].strip()
                    display = f"  {num:>4}   ·   {name}"
                else:
                    display = f"  {clean}"
                item = QListWidgetItem(display)
                item.setFont(QFont("Arial", 13))
                self.lst_scan_list.addItem(item)
        else:
            item = QListWidgetItem("  (no scans found)")
            item.setFont(QFont("Arial", 12))
            item.setForeground(QColor("#666"))
            self.lst_scan_list.addItem(item)
            QMessageBox.warning(
                self,
                "No scans found",
                "No Bruker scan sub-folders were found in the selected directory.\n\n"
                "Please select the study root folder that contains numbered scan "
                "sub-directories (e.g. 1, 2, 3 … each with a pdata/1/ sub-folder).\n\n"
                f"Selected path:\n{self._study_dir}"
            )

        # Populate all cards (already-cleaned entries propagate to dropdowns too)
        for card in self._cards.values():
            card.populate(entries, scan_nums, self._study_dir)

    def _on_scanlist_row_changed(self, row: int):
        """Show path details when user clicks a scan entry."""
        if row < 0:
            self.lbl_scan_detail.setText("")
            return
        entries  = getattr(self, '_scan_entries', [])
        nums     = getattr(self, '_scan_nums_list', [])
        if row >= len(nums):
            self.lbl_scan_detail.setText("")
            return
        num  = nums[row]
        name = entries[row].replace("----", " ").strip() if row < len(entries) else num
        # Check what pdata/1/ contains
        pdata = os.path.join(self._study_dir, num, "pdata", "1")
        if os.path.isdir(pdata):
            has_mat = os.path.isfile(os.path.join(pdata, "acquired_data.mat"))
            has_2dseq = os.path.isfile(os.path.join(pdata, "2dseq"))
            flags = []
            if has_mat:
                flags.append("✔ acquired_data.mat")
            if has_2dseq:
                flags.append("✔ 2dseq binary")
            detail = f"Path: {pdata}"
            if flags:
                detail += "   |   " + "   ".join(flags)
            self.lbl_scan_detail.setText(detail)
        else:
            self.lbl_scan_detail.setText(f"Scan #{num}: {name}  (pdata/1/ not found)")

    # Short display names for the summary bar
    _SUMMARY_SHORT: dict = {
        't1': 'T1', 't2': 'T2', 'b1_fa1': 'B1₁', 'b1_fa2': 'B1₂',
        'wassr': 'WASSR', 'cest': 'CEST', 'mrf': 'MRF', 'quesp': 'QUESP',
    }

    @staticmethod
    def _path_to_scan_id(p: str) -> str:
        """Extract the scan/experiment number from a Bruker path .../scan_num/pdata/1."""
        parts = p.replace('\\', '/').rstrip('/').split('/')
        if len(parts) >= 3 and parts[-2] == 'pdata':
            return parts[-3]   # .../scan_num/pdata/1 → scan_num
        return parts[-1] if parts else p

    def _on_path_changed(self, key: str, path: str):
        """Update the summary bar; auto-convert MRF if acquired_data.mat missing."""
        assigned = {k: c.get_path() for k, c in self._cards.items() if c.get_path()}
        if assigned:
            parts = [
                f"<b>{self._SUMMARY_SHORT.get(k, k.upper())}</b> - {self._path_to_scan_id(v)}"
                for k, v in assigned.items()
            ]
            self.lbl_summary.setText("   |   ".join(parts))
        else:
            self.lbl_summary.setText("No scans assigned yet.")

        # ── MRF: auto-check / auto-convert acquired_data.mat ──────────────
        if key == "mrf" and path:
            mat_path = os.path.join(path, "acquired_data.mat")
            if os.path.isfile(mat_path):
                self.lbl_summary.setStyleSheet(
                    "background: #1e2a1e; color: #4ec9b0; border-radius: 6px; "
                    "padding: 6px 12px; font-size: 12px;"
                )
                self.lbl_summary.setText(
                    self.lbl_summary.text() +
                    "   ✔ MRF: acquired_data.mat already exists"
                )
            else:
                # Auto-convert 2dseq → acquired_data.mat in background thread
                self._auto_convert_mrf(path)

    def _auto_convert_mrf(self, path: str):
        """Auto-convert Bruker 2dseq → acquired_data.mat in a background thread."""
        from PyQt6.QtCore import QThread, pyqtSignal as _sig, QObject

        class _Worker(QObject):
            done  = _sig(str)   # saved path
            error = _sig(str)

            def __init__(self, path, pv360):
                super().__init__()
                self._path  = path
                self._pv360 = pv360

            def run(self):
                try:
                    from my_gui.bruker_reader import read_2dseq_mrf, save_acquired_data
                    acq, info, seq = read_2dseq_mrf(self._path, pv360=self._pv360)
                    saved = save_acquired_data(
                        os.path.join(self._path, "acquired_data"),
                        acq, info, seq, fmt="mat"
                    )
                    self.done.emit(saved)
                except Exception as exc:
                    self.error.emit(str(exc))

        # Indicate conversion in progress
        cur = self.lbl_summary.text()
        self.lbl_summary.setText(cur + "   ⏳ Converting MRF 2dseq → .mat…")
        self.lbl_summary.setStyleSheet(
            "background: #1e2a1e; color: #f0c040; border-radius: 6px; "
            "padding: 6px 12px; font-size: 12px;"
        )

        pv360 = self.combo_pv.currentText() == "PV360"
        thread = QThread(self)
        worker = _Worker(path, pv360)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def _on_done(saved):
            thread.quit()
            cur2 = self.lbl_summary.text().replace("⏳ Converting MRF 2dseq → .mat…", "")
            self.lbl_summary.setText(cur2.rstrip() + "   ✔ MRF: acquired_data.mat saved")
            self.lbl_summary.setStyleSheet(
                "background: #1e2a1e; color: #4ec9b0; border-radius: 6px; "
                "padding: 6px 12px; font-size: 12px;"
            )

        def _on_error(msg):
            thread.quit()
            cur2 = self.lbl_summary.text().replace("⏳ Converting MRF 2dseq → .mat…", "")
            self.lbl_summary.setText(cur2.rstrip() + f"   ✖ MRF convert error: {msg[:60]}")
            self.lbl_summary.setStyleSheet(
                "background: #2a1e1e; color: #e74c3c; border-radius: 6px; "
                "padding: 6px 12px; font-size: 12px;"
            )

        worker.done.connect(_on_done)
        worker.error.connect(_on_error)
        self._mrf_thread  = thread   # keep reference to avoid GC
        self._mrf_worker  = worker
        thread.start()

