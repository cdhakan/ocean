"""
dict_tab.py
Dictionary tab — four sections:

  0. Study Browser      : select Bruker study directory → parse scan list
                          → choose which scan is MRF / QUESP / T1map / T2map / WASSR / zSpec
  1. Raw Bruker Data    : browse pdata/1 directly → convert 2dseq → acquired_data.mat
  2. Dictionary         : generate dictionary + run matching
  3. Previous Results   : load an existing quant_maps.mat directly

Scan-list parsing ports the MATLAB logic from:
  File_directory_navigation/genProtocolList.m     (PV360 ScanProgram.scanProgram)
  File_directory_navigation/displayScanList.m     (*.txt protocol lists)
  File_directory_navigation/loadDirectories.m     (numeric sub-directory scan)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QProgressBar, QLabel, QLineEdit,
    QGroupBox, QComboBox, QCheckBox, QFileDialog,
    QSizePolicy, QGridLayout, QFrame,
)
from PyQt6.QtCore import Qt


# ── Scan-list helpers (ported from MATLAB) ───────────────────────────────────

_SCAN_TYPES = ["MRF"]   # QUESP / T1map / T2map / WASSR / zSpec handled in their own tabs
_SELECT_PLACEHOLDER = "Select:"


def _find_numeric_scan_dirs(study_dir: str) -> list[str]:
    """
    Return sorted list of numeric sub-directory names inside *study_dir*
    (mirrors loadDirectories.m). Non-numeric dirs are skipped.
    """
    try:
        entries = os.listdir(study_dir)
    except OSError:
        return []
    nums = []
    for e in entries:
        if os.path.isdir(os.path.join(study_dir, e)):
            try:
                nums.append(int(e))
            except ValueError:
                pass
    return [str(n) for n in sorted(nums)]


def _detect_pv_version(study_dir: str) -> str:
    """
    Read the 'subject' file from the study root and detect the Bruker ParaVision
    version. Returns 'PV360', 'PV7', 'PV6', 'PV5', or 'unknown'.
    Mirrors StudyLoad.m PVverstr detection logic.
    """
    import re as _re
    for fname in ("subject", "Subject"):
        fp = os.path.join(study_dir, fname)
        if os.path.isfile(fp):
            try:
                with open(fp, "r", errors="replace") as fh:
                    content = fh.read()
                if "ParaVision 360" in content or "PV360" in content:
                    return "PV360"
                m = _re.search(r"ParaVision[\s_]?(\d+)", content, _re.IGNORECASE)
                if m:
                    major = int(m.group(1))
                    if major >= 7:
                        return "PV7"
                    elif major == 6:
                        return "PV6"
                    elif major == 5:
                        return "PV5"
            except OSError:
                pass
    return "unknown"


def _detect_pv360(study_dir: str) -> bool:
    """
    Backward-compatible wrapper — returns True only for ParaVision 360.
    Use _detect_pv_version() for explicit version detection.
    """
    return _detect_pv_version(study_dir) == "PV360"


def _parse_scan_program(study_dir: str) -> list[str]:
    """
    Parse ScanProgram.scanProgram (PV360) to get scan entries.
    Mirrors genProtocolList.m:
        scanEntries = scanNo + ' ----' + scanNames
    Returns list of strings like  '11 ----fpSL_EPI_an30nSL'.
    """
    sp_path = os.path.join(study_dir, "ScanProgram.scanProgram")
    if not os.path.isfile(sp_path):
        return []

    entries: list[str] = []
    try:
        with open(sp_path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return []

    sie_flag = False
    current_name = ""
    for line in lines:
        stripped = line.strip()
        if "ScanInstructionEntity>" in stripped:
            sie_flag = True
            current_name = ""
        elif "<displayName>" in stripped and sie_flag:
            m = re.search(r"<displayName>(.*?)</displayName>", stripped)
            if m:
                current_name = m.group(1).strip()
        elif "<expno>" in stripped and sie_flag:
            m = re.search(r"<expno>(.*?)</expno>", stripped)
            if m:
                expno = m.group(1).strip()
                entries.append(f"{expno} ----{current_name}")
                sie_flag = False

    return entries


def _find_txt_protocol(study_dir: str) -> list[str]:
    """
    Look for *.txt files whose names contain 'list', 'protocol', or 'scan'
    (mirrors displayScanList.m) and return their lines.
    """
    try:
        txts = [
            f for f in os.listdir(study_dir)
            if f.lower().endswith(".txt")
            and any(k in f.lower() for k in ("list", "protocol", "scan"))
        ]
    except OSError:
        return []

    if not txts:
        return []

    lines: list[str] = []
    try:
        with open(os.path.join(study_dir, txts[0]), "r", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)
    except OSError:
        pass
    return lines


def build_scan_list(study_dir: str) -> tuple[list[str], list[str]]:
    """
    Return (scan_entries, scan_numbers) for *study_dir*.

    scan_entries : human-readable strings like '11 ----fpSL_EPI_an30nSL'
    scan_numbers : bare numeric strings like '11', '6', ... (sorted)

    Logic mirrors MATLAB:
      1. Try ScanProgram.scanProgram (PV360) — expno IS the directory name
      2. Try *.txt protocol list
      3. Fall back to raw numeric directory listing
    """
    entries = _parse_scan_program(study_dir)
    disk_nums = _find_numeric_scan_dirs(study_dir)

    if entries:
        # In PV360, the expno from ScanProgram IS the directory name.
        # Extract the leading number from each entry (format: "16 ----name")
        # and use it as the actual directory to look up.
        derived_nums: list[str] = []
        for e in entries:
            m = re.match(r'^(\d+)', e.strip())
            derived_nums.append(m.group(1) if m else "")

        # Validate: check which expno directories actually exist on disk
        valid_pairs = [
            (e, n) for e, n in zip(entries, derived_nums)
            if n and os.path.isdir(os.path.join(study_dir, n))
        ]

        if valid_pairs:
            return [e for e, _ in valid_pairs], [n for _, n in valid_pairs]

        # All derived dirs are missing — fall through to disk listing below
        entries = []  # discard scan-program entries, use disk listing

    if not entries:
        entries = _find_txt_protocol(study_dir)

    if not entries:
        # Fallback: use numeric directory names as entries
        entries = [f"{n}" for n in disk_nums]

    return entries, disk_nums


# ── Dict tab widget ───────────────────────────────────────────────────────────

class DictTab(QWidget):
    def __init__(self, on_generate_clicked):
        super().__init__()
        self._on_generate = on_generate_clicked
        self._study_dir: str = ""
        self._bruker_dir: str = ""
        self._acquired_data_path: str = ""
        self._quant_maps_path: str = ""
        self._scan_dirs: list[str] = []  # numeric scan dirs in study

        # Callback set by app.py so that "Load quant_maps" can push to ResultsTab
        self.on_quant_maps_loaded = None   # callable(quant_maps: dict) | None
        # Callback set by app.py so acquired data is forwarded to MRF Viewer immediately
        self.on_acquired_data_loaded = None  # callable(path: str) | None

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Section 0: MRF Scan status ────────────────────────────────────
        layout.addWidget(self._build_mrf_status())

        # ── Section 1: Raw Bruker Data — instantiate widgets but keep hidden
        # (widgets are used internally by set_mrf_scan_path / _set_acquired_data;
        #  conversion UI is in the Scan Directory tab instead)
        self._raw_section_widget = self._build_raw_section()

        # _build_study_browser() is not added to the layout (moved to ScanDirTab)
        # but _set_acquired_data references lbl_study_load_status — create a hidden one
        if not hasattr(self, 'lbl_study_load_status'):
            from PyQt6.QtWidgets import QLabel as _QL
            self.lbl_study_load_status = _QL("")  # hidden fallback, not shown in UI

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # ── Section 2: Dictionary Generation ─────────────────────────────
        layout.addWidget(self._build_gen_section(), stretch=1)

    # ─────────────────────────────────────────────────────────────────────
    # Section builders
    # ─────────────────────────────────────────────────────────────────────

    def _build_study_browser(self) -> QGroupBox:
        grp = QGroupBox("Bruker Study Browser  (select study directory → assign scans)")
        v = QVBoxLayout(grp)
        v.setSpacing(6)

        # ── Study directory row ───────────────────────────────────────────
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Study directory:"))
        self.edit_study_dir = QLineEdit()
        self.edit_study_dir.setPlaceholderText(
            "…/20260227_130353_Phantom_1/   (root containing scan sub-folders)"
        )
        self.edit_study_dir.setReadOnly(True)
        dir_row.addWidget(self.edit_study_dir, stretch=1)
        self.btn_study_browse = QPushButton("Browse…")
        self.btn_study_browse.clicked.connect(self._browse_study)
        dir_row.addWidget(self.btn_study_browse)
        v.addLayout(dir_row)

        # ── Bruker version selector ───────────────────────────────────────
        pv_row = QHBoxLayout()
        pv_row.addWidget(QLabel("Bruker version:"))
        self.combo_study_pv = QComboBox()
        self.combo_study_pv.addItems(["PV360", "PV6 / PV7"])
        self.combo_study_pv.setToolTip(
            "Bruker ParaVision version — auto-detected from subject file."
        )
        self.combo_study_pv.setFixedWidth(110)
        pv_row.addWidget(self.combo_study_pv)
        self.lbl_study_info = QLabel("")
        self.lbl_study_info.setStyleSheet("font-size: 11px; color: gray;")
        pv_row.addWidget(self.lbl_study_info, stretch=1)
        v.addLayout(pv_row)

        # ── Scan list display ─────────────────────────────────────────────
        self.txt_scan_list = QTextEdit()
        self.txt_scan_list.setReadOnly(True)
        self.txt_scan_list.setFixedHeight(120)
        from my_gui.theme import mono_font
        self.txt_scan_list.setFont(mono_font(10))
        self.txt_scan_list.setPlaceholderText(
            "Scan list will appear here after loading a study directory…\n"
            "e.g.\n"
            "  3 ----1_Localizer\n"
            "  6 ----T1map_RARE\n"
            " 11 ----fpSL_EPI_an30nSL\n"
            " 10 ----fpSL_EPI_Glu_MRF_OCohen_2"
        )
        v.addWidget(self.txt_scan_list)

        # ── MRF scan selector ─────────────────────────────────────────────
        self._scan_combos: dict[str, QComboBox] = {}

        mrf_row = QHBoxLayout()
        mrf_lbl = QLabel("MRF scan #:")
        mrf_lbl.setStyleSheet("font-weight: bold;")
        mrf_row.addWidget(mrf_lbl)
        mrf_combo = QComboBox()
        mrf_combo.addItem(_SELECT_PLACEHOLDER)
        mrf_combo.setMinimumWidth(100)
        mrf_combo.setToolTip(
            "Select the scan number for the MRF acquisition.\n"
            "Other scan types (WASSR, zSpec, T1map, etc.) are\n"
            "configured in their respective tabs."
        )
        self._scan_combos["MRF"] = mrf_combo
        mrf_row.addWidget(mrf_combo)
        mrf_row.addStretch()
        v.addLayout(mrf_row)

        # ── Load button ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_load_mrf_scan = QPushButton("Load Selected MRF Scan")
        self.btn_load_mrf_scan.setFixedHeight(34)
        self.btn_load_mrf_scan.setToolTip(
            "Reads pdata/1/acquired_data.mat (or converts 2dseq) from\n"
            "the selected MRF scan number, ready for dictionary generation."
        )
        self.btn_load_mrf_scan.clicked.connect(self._load_mrf_from_study)
        btn_row.addWidget(self.btn_load_mrf_scan)
        btn_row.addStretch()
        v.addLayout(btn_row)

        self.lbl_study_load_status = QLabel("")
        self.lbl_study_load_status.setStyleSheet("font-size: 11px; color: gray;")
        self.lbl_study_load_status.setWordWrap(True)
        v.addWidget(self.lbl_study_load_status)

        return grp

    def _build_raw_section(self) -> QGroupBox:
        grp = QGroupBox("Raw Bruker Data  (2dseq → acquired_data.mat)")
        v = QVBoxLayout(grp)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("2dseq folder (pdata/1/):"))
        self.edit_bruker_dir = QLineEdit()
        self.edit_bruker_dir.setPlaceholderText("…/pdata/1/  (directory containing 2dseq)")
        self.edit_bruker_dir.setReadOnly(True)
        dir_row.addWidget(self.edit_bruker_dir, stretch=1)
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse_bruker)
        dir_row.addWidget(self.btn_browse)
        v.addLayout(dir_row)

        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("Bruker version:"))
        self.combo_pv = QComboBox()
        self.combo_pv.addItems(["PV360", "PV6 / PV7"])
        self.combo_pv.setToolTip(
            "PV360 = ParaVision 360  |  PV6 / PV7 = ParaVision 6 or 7"
        )
        self.combo_pv.setFixedWidth(110)
        opt_row.addWidget(self.combo_pv)
        opt_row.addWidget(QLabel("Save format:"))
        self.combo_fmt = QComboBox()
        self.combo_fmt.addItems(["MATLAB (.mat)", "NumPy (.npz)"])
        opt_row.addWidget(self.combo_fmt)
        opt_row.addStretch()
        v.addLayout(opt_row)

        conv_row = QHBoxLayout()
        self.btn_convert = QPushButton("Load & Convert 2dseq")
        self.btn_convert.setFixedHeight(34)
        self.btn_convert.clicked.connect(self._convert_bruker)
        conv_row.addWidget(self.btn_convert)

        self.btn_load_acqdata = QPushButton("Load existing acquired_data.mat")
        self.btn_load_acqdata.setFixedHeight(34)
        self.btn_load_acqdata.setToolTip(
            "Load a previously generated acquired_data.mat directly\n"
            "(skips Bruker binary conversion)"
        )
        self.btn_load_acqdata.clicked.connect(self._load_existing_acqdata)
        conv_row.addWidget(self.btn_load_acqdata)
        v.addLayout(conv_row)

        self.lbl_convert_status = QLabel("No data loaded.")
        self.lbl_convert_status.setStyleSheet("color: gray;")
        v.addWidget(self.lbl_convert_status)

        self.lbl_raw_info = QLabel("")
        self.lbl_raw_info.setWordWrap(True)
        self.lbl_raw_info.setStyleSheet("font-size: 11px; color: #444;")
        v.addWidget(self.lbl_raw_info)

        return grp

    def _build_gen_section(self) -> QGroupBox:
        grp = QGroupBox("Dictionary Generation && Matching")
        v = QVBoxLayout(grp)

        self.lbl_status = QLabel("Ready.")
        v.addWidget(self.lbl_status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        v.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        from my_gui.theme import mono_font
        self.log.setFont(mono_font(10))
        self.log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        v.addWidget(self.log)

        btn_row = QHBoxLayout()
        self.btn_gen = QPushButton("  Generate CEST-MRF Dictionary Simulation && Matching ")
        self.btn_gen.setFixedHeight(46)
        self.btn_gen.setFixedWidth(580)
        self.btn_gen.setStyleSheet("""
            QPushButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0   #0d6efd,
                    stop:0.4 #6610f2,
                    stop:1   #d63384
                );
                color: white;
                font-size: 14px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
                padding: 4px 16px;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0   #3d8bfd,
                    stop:0.4 #8540f5,
                    stop:1   #e25c92
                );
            }
            QPushButton:pressed {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0   #0a58ca,
                    stop:0.4 #520dc2,
                    stop:1   #ab296a
                );
                padding-top: 6px;
            }
            QPushButton:disabled {
                background: #444;
                color: #888;
            }
        """)
        self.btn_gen.clicked.connect(self._on_generate)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_gen)

        self.btn_cancel_gen = QPushButton("Cancel")
        self.btn_cancel_gen.setFixedHeight(46)
        self.btn_cancel_gen.setEnabled(False)
        self.btn_cancel_gen.setStyleSheet(
            "QPushButton { background: #c0392b; color: white; font-weight: bold; "
            "font-size: 13px; border: none; border-radius: 8px; padding: 4px 14px; }"
            "QPushButton:hover { background: #e74c3c; }"
            "QPushButton:disabled { background: #555; color: #999; }"
        )
        # Cancel callback set by app.py via set_cancel_callback()
        self._cancel_callback = None
        self.btn_cancel_gen.clicked.connect(self._on_cancel_gen)
        btn_row.addWidget(self.btn_cancel_gen)
        v.addLayout(btn_row)

        return grp

    def _build_prev_section(self) -> QGroupBox:
        grp = QGroupBox("Load Previous Results")
        v = QVBoxLayout(grp)

        row = QHBoxLayout()
        self.btn_load_qm = QPushButton("Load quant_maps.mat")
        self.btn_load_qm.setFixedHeight(34)
        self.btn_load_qm.setToolTip(
            "Load a previously computed quant_maps.mat file.\n"
            "Maps will be shown directly in the Results tab."
        )
        self.btn_load_qm.clicked.connect(self._load_quant_maps)
        row.addWidget(self.btn_load_qm)

        self.lbl_qm_path = QLabel("No quant_maps file loaded.")
        self.lbl_qm_path.setStyleSheet("font-size: 11px; color: gray;")
        self.lbl_qm_path.setWordWrap(True)
        row.addWidget(self.lbl_qm_path, stretch=1)
        v.addLayout(row)

        return grp

    def _build_mrf_status(self) -> QGroupBox:
        grp = QGroupBox("MRF Scan")
        v = QVBoxLayout(grp)
        self.lbl_mrf_scan_path = QLabel("No MRF scan loaded.")
        self.lbl_mrf_scan_path.setWordWrap(True)
        self.lbl_mrf_scan_path.setStyleSheet("color: gray; font-size: 11px;")
        v.addWidget(self.lbl_mrf_scan_path)
        return grp

    def set_mrf_scan_path(self, path: str):
        """Called by app.py when a MRF scan is assigned in scan_dir_tab."""
        if path:
            self.lbl_mrf_scan_path.setText(f"✔  {path}")
            self.lbl_mrf_scan_path.setStyleSheet("color: #4ec9b0; font-size: 11px;")
            # Try to auto-load acquired_data.mat if present
            import os
            acqmat = os.path.join(path, "acquired_data.mat")
            if os.path.isfile(acqmat):
                self._set_acquired_data(acqmat, source="scan directory")
            else:
                self._bruker_dir = path
                self.edit_bruker_dir.setText(path)
                self.lbl_convert_status.setText(
                    "MRF scan path set — click 'Load & Convert 2dseq'."
                )
        else:
            self.lbl_mrf_scan_path.setText("No MRF scan loaded.")
            self.lbl_mrf_scan_path.setStyleSheet("color: gray; font-size: 11px;")

    # ─────────────────────────────────────────────────────────────────────
    # Study browser logic
    # ─────────────────────────────────────────────────────────────────────

    def _browse_study(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Bruker study root directory (containing scan sub-folders)", ""
        )
        if not d:
            return
        self._study_dir = d
        self.edit_study_dir.setText(d)
        self._refresh_scan_list()

    def _refresh_scan_list(self):
        """Parse scan list from self._study_dir and populate UI."""
        if not self._study_dir:
            return

        # Auto-detect ParaVision version
        ver = _detect_pv_version(self._study_dir)
        self.combo_study_pv.setCurrentText("PV360" if ver == "PV360" else "PV6 / PV7")
        ver_label = ver if ver != "unknown" else "version unknown"
        self.lbl_study_info.setText(f"Detected: Bruker {ver_label}")

        entries, scan_nums = build_scan_list(self._study_dir)
        self._scan_dirs = scan_nums

        # Display scan list
        if entries:
            self.txt_scan_list.setPlainText(
                f"Information on scan list ({Path(self._study_dir).name}):\n"
                + "\n".join(f"  {e}" for e in entries)
            )
        else:
            self.txt_scan_list.setPlainText(
                "No scan information found. Check the study directory."
            )

        # Repopulate all dropdowns
        for combo in self._scan_combos.values():
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(_SELECT_PLACEHOLDER)
            for sn in scan_nums:
                # Find matching entry label for tooltip
                label = next(
                    (e for e in entries if e.split(" ")[0] == sn), sn
                )
                combo.addItem(sn, label)
            combo.blockSignals(False)

        self.lbl_study_load_status.setText(
            f"Loaded study: {len(scan_nums)} scan directories found."
        )
        self.lbl_study_load_status.setStyleSheet("font-size: 11px; color: green;")

    def _get_scan_path(self, dtype: str) -> str | None:
        """Return the pdata/1 path for the selected scan type, or None."""
        combo = self._scan_combos.get(dtype)
        if combo is None:
            return None
        val = combo.currentText()
        if val == _SELECT_PLACEHOLDER or not val.strip():
            return None
        return os.path.join(self._study_dir, val.strip(), "pdata", "1")

    def _load_mrf_from_study(self):
        """Load acquired_data from the MRF scan selected in the study browser."""
        scan_path = self._get_scan_path("MRF")
        if not scan_path:
            self.lbl_study_load_status.setText(
                "Please select an MRF scan number first."
            )
            self.lbl_study_load_status.setStyleSheet("font-size: 11px; color: red;")
            return

        # Prefer a pre-existing acquired_data.mat in pdata/1
        acqmat = os.path.join(scan_path, "acquired_data.mat")
        if os.path.isfile(acqmat):
            self._set_acquired_data(acqmat, source="study browser (existing .mat)")
            return

        # Otherwise try to convert the 2dseq binary
        if not os.path.isdir(scan_path):
            self.lbl_study_load_status.setText(
                f"Scan path not found:\n{scan_path}"
            )
            self.lbl_study_load_status.setStyleSheet("font-size: 11px; color: red;")
            return

        try:
            from my_gui.bruker_reader import read_2dseq_mrf, save_acquired_data
            pv360 = self.combo_study_pv.currentText() == "PV360"
            acquired_data, info, seq_defs = read_2dseq_mrf(scan_path, pv360=pv360)
            out_base = os.path.join(scan_path, "acquired_data")
            saved = save_acquired_data(out_base, acquired_data, info, seq_defs, fmt="mat")
            self._set_acquired_data(saved, source="study browser (converted 2dseq)")
            sz = info.get("size", [])
            n_meas = seq_defs.get("num_meas", "?")
            self.lbl_raw_info.setText(
                f"Matrix: {sz[0]}×{sz[1]}  Slices: {sz[2]}  Meas: {n_meas}\n{saved}"
            )
        except Exception as exc:
            self.lbl_study_load_status.setText(f"Error loading MRF scan: {exc}")
            self.lbl_study_load_status.setStyleSheet("font-size: 11px; color: red;")
            self.log.append(f"[Study MRF ERROR] {exc}")

    def _set_acquired_data(self, path: str, source: str = ""):
        """Store acquired_data path and update UI labels."""
        self._acquired_data_path = path
        # Also mirror into the Raw Bruker section labels
        self.edit_bruker_dir.setText(str(Path(path).parent))
        self._bruker_dir = str(Path(path).parent)
        self.lbl_convert_status.setText(f"Loaded: {Path(path).name}")
        self.lbl_convert_status.setStyleSheet("color: green;")
        self.lbl_study_load_status.setText(
            f"MRF data ready ({source}):\n{path}"
        )
        self.lbl_study_load_status.setStyleSheet("font-size: 11px; color: green;")
        self.log.append(f"Acquired data loaded → {Path(path).name}")

        # Forward path to MRF Viewer so it shows the raw images immediately
        if callable(self.on_acquired_data_loaded):
            try:
                self.on_acquired_data_loaded(path)
            except Exception:
                pass

        # Try to show shape info
        try:
            import scipy.io as sio
            d = sio.loadmat(path)
            ad = d.get("acquired_data")
            if ad is not None:
                self.lbl_raw_info.setText(
                    f"acquired_data shape: {ad.shape}\n{path}"
                )
        except Exception:
            self.lbl_raw_info.setText(path)

    # Stubs for future scan-type accessors (WASSR/T1map/T2map live in their own tabs)
    def get_wassr_scan_path(self) -> str | None:
        return None

    def get_t1map_scan_path(self) -> str | None:
        return None

    def get_t2map_scan_path(self) -> str | None:
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Raw Bruker section helpers
    # ─────────────────────────────────────────────────────────────────────

    def _browse_bruker(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Bruker pdata/1/ directory (containing 2dseq)", ""
        )
        if d:
            self._bruker_dir = d
            self.edit_bruker_dir.setText(d)
            self.lbl_convert_status.setText("Directory selected — click 'Load & Convert 2dseq'.")
            self.lbl_convert_status.setStyleSheet("color: #555;")
            self.lbl_raw_info.setText("")

    def _fmt_code(self) -> str:
        return "npz" if "npz" in self.combo_fmt.currentText().lower() else "mat"

    def _convert_bruker(self):
        if not self._bruker_dir:
            self.lbl_convert_status.setText("Please browse to a 2dseq directory first.")
            self.lbl_convert_status.setStyleSheet("color: red;")
            return

        self.btn_convert.setEnabled(False)
        self.lbl_convert_status.setText("Loading…")
        self.lbl_convert_status.setStyleSheet("color: #555;")

        try:
            from my_gui.bruker_reader import read_2dseq_mrf, save_acquired_data

            acquired_data, info, seq_defs = read_2dseq_mrf(
                self._bruker_dir, pv360=self.combo_pv.currentText() == "PV360"
            )
            out_base = str(Path(self._bruker_dir) / "acquired_data")
            saved_path = save_acquired_data(
                out_base, acquired_data, info, seq_defs, fmt=self._fmt_code()
            )

            sz     = info.get("size", [])
            b0     = info.get("B0", "?")
            n_meas = seq_defs.get("num_meas", "?")

            self._acquired_data_path = saved_path
            self.lbl_convert_status.setText(f"Saved: {Path(saved_path).name}")
            self.lbl_convert_status.setStyleSheet("color: green;")
            self.lbl_raw_info.setText(
                f"Matrix: {sz[0]}×{sz[1]}  |  Slices: {sz[2]}  |  "
                f"Measurements: {n_meas}  |  B0: {b0} T\n{saved_path}"
            )
            self.log.append(f"[Bruker] Converted → {saved_path}")

        except Exception as exc:
            self.lbl_convert_status.setText(f"Error: {exc}")
            self.lbl_convert_status.setStyleSheet("color: red;")
            self.log.append(f"[Bruker ERROR] {exc}")
        finally:
            self.btn_convert.setEnabled(True)

    def _load_existing_acqdata(self):
        """Browse to an already-generated acquired_data.mat file."""
        fn, _ = QFileDialog.getOpenFileName(
            self, "Select acquired_data file", "",
            "Data files (*.mat *.npz);;All files (*)"
        )
        if not fn:
            return
        self._set_acquired_data(fn, source="manual file selection")

    def get_acquired_data_path(self) -> str:
        return self._acquired_data_path

    # ─────────────────────────────────────────────────────────────────────
    # Dictionary generation callbacks (called from app.py / DictWorker)
    # ─────────────────────────────────────────────────────────────────────

    def set_running(self, running: bool):
        self.btn_gen.setEnabled(not running)
        self.btn_cancel_gen.setEnabled(running)
        self.progress.setVisible(running)
        self.lbl_status.setText("Generating…" if running else "Done.")

    def set_cancel_callback(self, cb):
        """Called by app.py to wire the Cancel button to the active DictWorker."""
        self._cancel_callback = cb

    def _on_cancel_gen(self):
        if callable(self._cancel_callback):
            self._cancel_callback()
        self.btn_cancel_gen.setEnabled(False)
        self.lbl_status.setText("Cancelling…")

    def append_log(self, msg: str):
        self.log.append(msg)

    def on_error(self, msg: str):
        self.set_running(False)
        self.lbl_status.setText("Error!")
        self.append_log(f"ERROR: {msg}")

    # ─────────────────────────────────────────────────────────────────────
    # Load existing quant_maps.mat
    # ─────────────────────────────────────────────────────────────────────

    def _load_quant_maps(self):
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
                k: v for k, v in raw.items()
                if not k.startswith("_") and isinstance(v, np.ndarray)
            }

            self._quant_maps_path = fn
            keys = ", ".join(quant_maps.keys())
            self.lbl_qm_path.setText(f"{Path(fn).name}  [{keys}]")
            self.lbl_qm_path.setStyleSheet("font-size: 11px; color: green;")
            self.log.append(f"[quant_maps] Loaded → {fn}  keys: {keys}")

            if callable(self.on_quant_maps_loaded):
                self.on_quant_maps_loaded(quant_maps)

        except Exception as exc:
            self.lbl_qm_path.setText(f"Error: {exc}")
            self.lbl_qm_path.setStyleSheet("font-size: 11px; color: red;")
            self.log.append(f"[quant_maps ERROR] {exc}")
