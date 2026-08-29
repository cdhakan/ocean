"""
zspec_tab.py
Z-Spectroscopy tab — Python equivalent of the MATLAB z-spec pipeline:
  - Load Bruker CEST 2dseq data
  - Optional B0 correction from WASSR scan
  - SNR thresholding
  - Voxelwise multi-peak fitting (Lorentzian / Pseudo-Voigt)
  - MTR asymmetry map
  - Interactive visualization

Pools matched to MATLAB: water, NOE, MT, amide, OH
"""

from __future__ import annotations

import os
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QPushButton, QLabel, QLineEdit, QGroupBox,
    QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox,
    QTextEdit, QProgressBar, QFileDialog, QSizePolicy,
    QSlider, QFrame, QApplication, QMessageBox, QDialog,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

import matplotlib
matplotlib.use("Agg")

from PyQt6.QtWidgets import QScrollArea
from my_gui.roi_tools import ROICanvas, ROIPanel, roi_union_mask
from my_gui.roi_manager import apply_analysis_mask
from my_gui.plot_custom_bar import WindowLevelToolButton
from my_gui.plot_custom_bar import PlotCustomBar
from my_gui.fig_export import save_figure, FIG_EXPORT_FILTER


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers for multi-vendor loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_dicom_4d(folder: str, ppm_sidecar: str, log_fn) -> tuple:
    """
    Read a DICOM folder and return (img_4d, ppm_array).
    img_4d shape: (rows, cols, n_slices, n_offsets)  float32, raw pixel values.
    ppm_array: 1-D float64, length n_offsets.
    """
    import pydicom
    from pathlib import Path
    from collections import defaultdict
    import re

    def _to_gray_frames(pix):
        """Yield 2-D grayscale float frames from a DICOM pixel_array of any shape.

        Handles: 2-D grayscale, RGB/RGBA (→ luminance), multiframe (N,H,W),
        and multiframe-colour (N,H,W,3).
        """
        a = np.asarray(pix)
        if a.ndim == 2:
            yield a.astype(np.float32)
        elif a.ndim == 3:
            if a.shape[-1] in (3, 4):                       # single RGB(A) image
                yield (a[..., :3].astype(np.float32) @ [0.299, 0.587, 0.114])
            else:                                           # multiframe grayscale (N,H,W)
                for k in range(a.shape[0]):
                    yield a[k].astype(np.float32)
        elif a.ndim == 4 and a.shape[-1] in (3, 4):         # multiframe colour (N,H,W,C)
            for k in range(a.shape[0]):
                yield (a[k, ..., :3].astype(np.float32) @ [0.299, 0.587, 0.114])
        else:
            yield np.asarray(a).reshape(a.shape[-2], a.shape[-1]).astype(np.float32)

    def _si(ds, attr, default=0):
        try: return int(getattr(ds, attr, default))
        except: return default

    def _sf(ds, attr, default=0.0):
        try: return float(getattr(ds, attr, default))
        except: return default

    folder_path = Path(folder)
    # Collect (ds, gray2d) for every frame in every readable DICOM file
    items = []                       # list of (ds, gray2d)
    n_files = 0
    n_rgb = 0
    for fpath in sorted(folder_path.rglob("*")):
        if not fpath.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(fpath), stop_before_pixels=False, force=True)
            if not hasattr(ds, 'Rows'):
                continue
            pix = ds.pixel_array
        except Exception:
            continue
        n_files += 1
        if getattr(ds, 'SamplesPerPixel', 1) and int(getattr(ds, 'SamplesPerPixel', 1)) >= 3:
            n_rgb += 1
        rs = float(getattr(ds, 'RescaleSlope', 1.0))
        ri = float(getattr(ds, 'RescaleIntercept', 0.0))
        for g in _to_gray_frames(pix):
            items.append((ds, g * rs + ri))

    if not items:
        raise ValueError("No readable DICOM images found.")

    log_fn(f"  Found {n_files} DICOM file(s) → {len(items)} image frame(s)"
           + (f"  ({n_rgb} colour/RGB, converted to grayscale)." if n_rgb else "."))

    # ── Siemens MOSAIC? de-tile each frame into slices ────────────────────
    # Each mosaic DICOM file packs every slice of the acquisition into one tiled
    # 2-D frame.  When present, each file is one dynamic (offset) and its mosaic
    # is split into the slice axis → (rows, cols, n_slices, n_offsets).
    from my_gui.dicom_mosaic import is_mosaic, mosaic_to_volume
    if items and is_mosaic(items[0][0]):
        ordered = sorted(items, key=lambda t: _si(t[0], 'InstanceNumber', 0))
        vols = []
        for ds, g in ordered:
            v = mosaic_to_volume(g, ds)
            if v.ndim == 2:
                v = v[:, :, None]
            vols.append(v)
        tr, tc, nsl = vols[0].shape
        rows, cols, n_slices = int(tr), int(tc), int(nsl)
        n_offsets = len(vols)
        img_all = np.zeros((rows, cols, n_slices, n_offsets), dtype=np.float32)
        _dropped = 0
        for oi, v in enumerate(vols):
            if v.shape == (rows, cols, n_slices):
                img_all[:, :, :, oi] = v
            else:
                _dropped += 1
        offset_ds_list = [ds for ds, _ in ordered]
        log_fn(f"  Siemens MOSAIC detected → de-tiled to {n_slices} slice(s) "
               f"× {n_offsets} offset(s) of {rows}×{cols}.")
        if _dropped:
            log_fn(f"  Warning: {_dropped} frame(s) had a different tile geometry "
                   f"than the first and were left blank (mixed-content folder?).")
    else:
        rows = int(items[0][1].shape[0])
        cols = int(items[0][1].shape[1])

        # Group by SeriesNumber (each series = one saturation offset is common GE/Siemens)
        series_map = defaultdict(list)
        for ds, g in items:
            series_map[_si(ds, 'SeriesNumber', 1)].append((ds, g))
        sorted_series = sorted(series_map.keys())
        n_series = len(sorted_series)

        def _frame_ok(g):
            return g.shape == (rows, cols)

        if n_series > 1:
            # Multi-series CEST: offsets = series, slices = frames within a series
            n_offsets = n_series
            n_slices  = max(len(series_map[s]) for s in sorted_series)
            img_all   = np.zeros((rows, cols, n_slices, n_offsets), dtype=np.float32)
            for oi, sn in enumerate(sorted_series):
                sl_sorted = sorted(series_map[sn],
                                   key=lambda t: _sf(t[0], 'SliceLocation', _si(t[0], 'InstanceNumber')))
                for si2, (ds, g) in enumerate(sl_sorted[:n_slices]):
                    if _frame_ok(g):
                        img_all[:, :, si2, oi] = g
            offset_ds_list = [series_map[sn][0][0] for sn in sorted_series]
        else:
            # Single series → expose EVERY frame on the scrollable offset axis so the
            # user can browse all images (color maps, coefficients, DL, etc.).
            all_items = sorted(series_map[sorted_series[0]],
                               key=lambda t: _si(t[0], 'InstanceNumber', 0))
            all_items = [(ds, g) for ds, g in all_items if _frame_ok(g)]
            n_slices  = 1
            n_offsets = max(1, len(all_items))
            img_all   = np.zeros((rows, cols, n_slices, n_offsets), dtype=np.float32)
            for oi, (ds, g) in enumerate(all_items):
                img_all[:, :, 0, oi] = g
            offset_ds_list = [ds for ds, _ in all_items]
            log_fn(f"  Single-series data → {n_offsets} frame(s) on the scroll axis.")

    # ── ppm offsets ──────────────────────────────────────────────────────
    ppm_all = None

    # 1. User sidecar
    if ppm_sidecar:
        try:
            ppm_all = np.loadtxt(ppm_sidecar)
            log_fn(f"  PPM from sidecar ({len(ppm_all)} values).")
        except Exception as e:
            log_fn(f"  Warning: sidecar read failed: {e}")

    # 2. Auto-detect from DICOM tags
    if ppm_all is None:
        vals = []
        for ds0 in offset_ds_list:
            v   = None
            # Siemens: ImageComments may contain "3.5 ppm" or "#offset=3.5"
            try:
                ic = str(getattr(ds0, 'ImageComments', ''))
                m  = re.search(r'([-+]?\d+\.?\d*)\s*ppm', ic, re.IGNORECASE)
                if m:
                    v = float(m.group(1))
            except Exception:
                pass
            # Siemens private tag 0x0019,0x109c
            if v is None:
                try:
                    raw = ds0[0x0019, 0x109c].value
                    v   = float(raw) if not isinstance(raw, bytes) else float(raw.decode().strip())
                except Exception:
                    pass
            # GE private tag 0x0043,0x1038
            if v is None:
                try:
                    raw = ds0[0x0043, 0x1038].value
                    v   = float(raw) if not isinstance(raw, bytes) else float(raw.decode().strip())
                except Exception:
                    pass
            vals.append(v)

        if all(v is not None for v in vals):
            ppm_all = np.array(vals, dtype=float)
            log_fn(f"  PPM auto-detected from DICOM tags: {ppm_all}")
        else:
            log_fn("  Warning: ppm offsets not found in DICOM tags. Using sequential indices.")
            ppm_all = np.arange(n_offsets, dtype=float)

    if len(ppm_all) != n_offsets:
        log_fn(f"  Warning: ppm array length {len(ppm_all)} != {n_offsets} offsets. Trimming/padding.")
        ppm_all = np.resize(ppm_all, n_offsets)

    return img_all, ppm_all


def _load_ppm_sidecar(ppm_path: str, data_path: str, n_offsets: int, log_fn) -> np.ndarray:
    """
    Try to load ppm offsets from:
      1. explicit ppm_path (.txt)
      2. BIDS JSON sidecar next to data_path
      3. Sequential fallback
    """
    import json
    from pathlib import Path

    # 1. User-specified file
    if ppm_path:
        try:
            ppm = np.loadtxt(ppm_path)
            log_fn(f"  PPM from user file ({len(ppm)} values).")
            return np.resize(ppm, n_offsets) if len(ppm) != n_offsets else ppm
        except Exception as e:
            log_fn(f"  Warning: ppm file read failed: {e}")

    # 2. BIDS JSON sidecar
    try:
        base = Path(data_path)
        for suffix in ('.json',):
            candidate = base.with_suffix('').with_suffix(suffix)
            if not candidate.exists():
                candidate = Path(str(data_path).replace('.nii.gz', '.json').replace('.nii', '.json'))
            if candidate.exists():
                with open(candidate) as f:
                    bids = json.load(f)
                for key in ['SaturationFrequency', 'CESTOffsets', 'offset_ppm', 'ppm', 'offsets']:
                    if key in bids:
                        ppm = np.array(bids[key], dtype=float)
                        log_fn(f"  PPM from BIDS JSON key '{key}' ({len(ppm)} values).")
                        return np.resize(ppm, n_offsets) if len(ppm) != n_offsets else ppm
    except Exception:
        pass

    log_fn("  Warning: no ppm offsets found. Using sequential indices.")
    return np.arange(n_offsets, dtype=float)


def _finish_cest_load(tab, img_all: np.ndarray, ppm_all: np.ndarray, source: str,
                      m0_override: "np.ndarray | None" = None):
    """
    Post-load normalisation and state assignment shared by DICOM + NIfTI CEST loaders.
    `tab` is the ZSpecTab instance.

    If `m0_override` (Y, X, n_sl) is given it is used as the M0/S0 reference for
    normalisation (e.g. a dedicated GE S0 frame) instead of the max-|ppm| frame.
    """
    n_offsets = img_all.shape[3]
    if m0_override is not None:
        m0_idx = -1
        M0_img = np.asarray(m0_override, dtype=img_all.dtype)
    else:
        m0_idx = int(np.argmax(np.abs(ppm_all)))
        M0_img = img_all[:, :, :, m0_idx]

    # Store RAW images in _z_img_full (consistent with the Bruker loader).
    # Consumers (ROI spectra, analysis) normalise by _get_m0() themselves —
    # storing pre-normalised data here caused a double-normalisation → flat ~0 Z.
    tab._z_img_full        = np.asarray(img_all, dtype=np.float32)
    tab._M0_img            = M0_img
    tab._ppm               = ppm_all
    tab._z_img_all         = img_all.copy()
    tab._ppm_all           = ppm_all.copy()
    tab._cest_m0_auto_idx  = m0_idx

    sz = img_all.shape
    tab.lbl_cest_info.setText(
        f"Loaded ({source}): {sz[0]}x{sz[1]} px  |  {sz[2]} slice(s)  |  {sz[3]} offsets"
    )
    tab.lbl_cest_info.setStyleSheet("font-size: 11px; color: green;")
    tab._log(f"{source} CEST loaded. Shape: {img_all.shape}  ppm range: "
             f"[{ppm_all.min():.2f}, {ppm_all.max():.2f}]")

    if tab.combo_m0_source.currentText() == "CEST":
        tab._populate_m0_frame_combo()
    tab.combo_display.setCurrentText("M0 image (unsaturated)")
    tab._refresh_display()
    tab._update_m0_label()
    tab.btn_preview_denoise.setEnabled(tab._denoise_method_key() != 'None')
    tab._roi_spectra_dlg_key = ()   # force rebuild of ROI Spectra dialog on next open

    # Motion correction: drop the previous raw cache; auto-apply if the box is on.
    tab._z_img_all_raw = None
    if getattr(tab, 'chk_moco', None) is not None and tab.chk_moco.isChecked():
        tab._apply_moco()


def _finish_wassr_load(tab, img_all: np.ndarray, ppm_all: np.ndarray, source: str,
                       m0_override: "np.ndarray | None" = None):
    """
    Post-load normalisation and B0 map computation shared by DICOM + NIfTI WASSR loaders.

    If `m0_override` (Y, X, n_sl) is given it is used as the WASSR M0/S0 reference.
    """
    if m0_override is not None:
        m0_idx = -1
        M0_img = np.asarray(m0_override, dtype=img_all.dtype)
    else:
        m0_idx  = int(np.argmax(np.abs(ppm_all)))
        M0_img  = img_all[:, :, :, m0_idx]

    M0_safe = M0_img[:, :, :, np.newaxis].copy()
    M0_safe[M0_safe < 1.0] = 1.0
    z_norm  = np.clip(img_all / M0_safe, 0.0, 2.0).astype(np.float32)

    tab._wassr_M0_img       = M0_img
    tab._wassr_img_full     = z_norm
    tab._wassr_img_all      = img_all.copy()
    tab._wassr_ppm_all      = ppm_all.copy()
    tab._wassr_ppm          = ppm_all
    tab._wassr_m0_auto_idx  = m0_idx

    # B0 map by SNR-masked per-voxel Lorentzian fit (matches WASSR_load_proc.m)
    from my_gui.zspec_processing import compute_b0_map_wassr
    _ftol, _max_nfev = {
        0: (1e-3, 400),    # Fast
        1: (1e-5, 800),    # Balanced
        2: (1e-7, 2000),   # Precise
    }.get(tab.combo_fit_quality.currentIndex(), (1e-3, 400))
    tab._log("Fitting WASSR B0 map (Lorentzian)…")
    tab._b0_map_ppm = compute_b0_map_wassr(
        z_norm, ppm_all,
        m0_img=M0_img,
        snr_thresh=tab.spin_snr.value(),
        larmor_mhz=tab.spin_larmor_mhz.value(),
        n_workers=tab.spin_workers.value(),
        ftol=_ftol,
        max_nfev=_max_nfev,
    )

    if tab.combo_m0_source.currentText() == "WASSR":
        tab._populate_m0_frame_combo()

    tab.lbl_wassr_info.setText(f"B0 map loaded ({source}).")
    tab.lbl_wassr_info.setStyleSheet("font-size: 11px; color: green;")
    tab._log(f"WASSR {source} loaded. Shape: {img_all.shape}")
    tab._update_m0_label()


# ─────────────────────────────────────────────────────────────────────────────
# Z-Spectroscopy Tab
# ─────────────────────────────────────────────────────────────────────────────

class ZSpecTab(QWidget):
    """
    Full z-spectroscopy processing tab.

    Left panel  — data loading + processing settings
    Right panel — visualization canvas + map selector
    """

    # Pools to fit — matches MATLAB zSpec_load_proc.m default exactly:
    #   water, NOE, MT, and the 3.0 ppm pool.
    # In MATLAB the 3.0 ppm pool is stored as pf.amide (its struct field name),
    # but its bounds (zspecSetPVPeakBounds x.amide) are centred at 3.0 ppm and
    # the plotting script labels it "OH".  In Python we call it "amine" and its
    # bounds are [2.5, 3.5] ppm — identical to MATLAB.
    # Adding extra pools (amide@3.5, OH@0.8) on top of these 4 causes water to
    # absorb all signal (too many overlapping free parameters → ill-conditioned).
    POOL_NAMES = ["water", "NOE", "MT", "amine"]

    def __init__(self):
        super().__init__()
        self._cest_dir: str = ""
        self._wassr_dir: str = ""
        self._worker = None
        self._total_vox: int = 0
        self._results: dict = {}
        self._z_img_full: np.ndarray | None = None    # (Y, X, slices, offsets) analysis subset
        self._roi_bg_img = None                        # user-picked grayscale bkg for "ROIs + Bkg"
        self._scan_paths_getter = None                 # set by app.py via set_scan_paths_getter
        self._M0_img: np.ndarray | None = None         # M0 auto-extracted from CEST load
        self._wassr_M0_img: np.ndarray | None = None   # M0 auto-extracted from WASSR load
        self._wassr_img_full: np.ndarray | None = None # (Y, X, slices, offsets) WASSR analysis
        self._wassr_ppm: np.ndarray | None = None      # WASSR offset ppm (analysis subset)
        self._ppm: np.ndarray | None = None
        # Full datasets (all frames, original order) — used for manual M0 selection
        self._z_img_all: np.ndarray | None = None      # (Y, X, slices, n_all) complete CEST
        self._ppm_all: np.ndarray | None = None        # ppm for every CEST frame
        self._cest_m0_auto_idx: int = 0                # auto-detected M0 frame index (CEST)
        self._wassr_img_all: np.ndarray | None = None  # (Y, X, slices, n_all) complete WASSR
        self._wassr_ppm_all: np.ndarray | None = None  # ppm for every WASSR frame
        self._wassr_m0_auto_idx: int = 0               # auto-detected M0 frame index (WASSR)
        self._last_rois: list = []
        self._dc_annot = None
        self._dc_cid: int | None = None
        # Global pool selection — used by DROF / MPLF fits in ROI Spectra dialog
        self._global_pools: list = ["water", "amide", "NOE", "MT", "amine", "OH"]
        # 2-D scalar maps generated from on-demand ROI spectral fits
        # Structure: {method_label: {pool_name: 2D np.ndarray}}
        self._roi_spectra_maps: dict = {}
        # Track which source was last used for each method, so ROI Stats Table
        # shows the most-recently-computed version.
        # Values: 'voxelwise' or 'roi_spectra'
        self._roi_stats_last_source: dict = {}
        # Persistent ROI Spectra dialog — kept alive (hidden) between opens so
        # all fitted plots survive close/reopen without needing to re-run.
        self._roi_spectra_dlg = None
        self._roi_spectra_dlg_key: tuple = ()   # (roi_names, ppm_len) validity key

        # GE/Siemens data paths (unified: file OR folder, DICOM OR NIfTI)
        self._cest_data_path: str = ""
        self._wassr_data_path: str = ""
        self._cest_ppm_path: str = ""          # CEST offsets .txt
        self._wassr_ppm_path: str = ""         # WASSR offsets .txt
        # Legacy aliases (kept so any stray reference doesn't crash)
        self._cest_dicom_dir = self._cest_nifti_path = ""
        self._wassr_dicom_dir = self._wassr_nifti_path = ""

        # MR Solutions (.MRD) state variables
        self._cest_mrd_path: str = ""
        self._wassr_mrd_path: str = ""


        # Main horizontal splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer = QHBoxLayout(self)
        outer.addWidget(splitter)

        # ── Left panel (scrollable) ───────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(300)
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(6)

        left_scroll = QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMaximumWidth(430)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # ── CEST data group ───────────────────────────────────────────────
        grp_cest = QGroupBox("CEST Data")
        g1 = QVBoxLayout(grp_cest)

        # Vendor selector row
        vendor_cest_row = QHBoxLayout()
        vendor_cest_row.addWidget(QLabel("Platform:"))
        self.combo_vendor_cest = QComboBox()
        self.combo_vendor_cest.addItems(
            ["Bruker", "GE/Siemens", "MR Solutions"]
        )
        vendor_cest_row.addWidget(self.combo_vendor_cest, stretch=1)
        g1.addLayout(vendor_cest_row)

        # ── Bruker sub-widget ──────────────────────────────────────────────
        self._cest_bruker_w = QWidget()
        bruker_cest_lay = QVBoxLayout(self._cest_bruker_w)
        bruker_cest_lay.setContentsMargins(0, 0, 0, 0)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Folder:"))
        self.edit_cest_dir = QLineEdit()
        self.edit_cest_dir.setPlaceholderText("Select the scan folder…")
        self.edit_cest_dir.setReadOnly(True)
        r1.addWidget(self.edit_cest_dir, stretch=1)
        self.btn_cest_browse = QPushButton("Browse…")
        self.btn_cest_browse.clicked.connect(self._browse_cest)
        r1.addWidget(self.btn_cest_browse)
        bruker_cest_lay.addLayout(r1)

        # Bruker version kept in the backend (auto/PV360) but not shown.
        self.combo_pv_cest = QComboBox()
        self.combo_pv_cest.addItems(["PV360", "PV6 / PV7"])
        self.combo_pv_cest.setVisible(False)
        g1.addWidget(self._cest_bruker_w)

        # ── GE/Siemens — data (file/folder) browse only ───────────────────
        # The acquisition type, offsets file and WASSR+CEST option live in the
        # "CEST parameters" dialog so the main panel stays just Platform + Folder.
        self._cest_ge_w = QWidget()
        ge_cest_lay = QVBoxLayout(self._cest_ge_w)
        ge_cest_lay.setContentsMargins(0, 0, 0, 0)
        ge_cest_lay.setSpacing(3)
        gd = QHBoxLayout()
        gd.addWidget(QLabel("Folder:"))
        self.edit_cest_data = QLineEdit(); self.edit_cest_data.setReadOnly(True)
        self.edit_cest_data.setPlaceholderText("DICOM/NIfTI — file or folder")
        gd.addWidget(self.edit_cest_data, stretch=1)
        _bcf = QPushButton("File…"); _bcf.clicked.connect(lambda: self._browse_cest_data(False))
        gd.addWidget(_bcf)
        _bcd = QPushButton("Folder…"); _bcd.clicked.connect(lambda: self._browse_cest_data(True))
        gd.addWidget(_bcd)
        ge_cest_lay.addLayout(gd)
        self._cest_ge_w.setVisible(False)
        g1.addWidget(self._cest_ge_w)

        # ── MR Solutions (.MRD) — data browse only ─────────────────────────
        self._cest_mrd_w = QWidget()
        mrd_cest_lay = QVBoxLayout(self._cest_mrd_w)
        mrd_cest_lay.setContentsMargins(0, 0, 0, 0)
        mrd_cest_lay.setSpacing(3)
        mc1 = QHBoxLayout()
        mc1.addWidget(QLabel("Folder:"))
        self.edit_cest_mrd_path = QLineEdit()
        self.edit_cest_mrd_path.setReadOnly(True)
        self.edit_cest_mrd_path.setPlaceholderText("…/<scan>_000_0.MRD  (CEST, SMIS raw)")
        mc1.addWidget(self.edit_cest_mrd_path, stretch=1)
        btn_cest_mrd = QPushButton("Browse…")
        btn_cest_mrd.clicked.connect(self._browse_cest_mrd)
        mc1.addWidget(btn_cest_mrd)
        mrd_cest_lay.addLayout(mc1)
        self._cest_mrd_w.setVisible(False)
        g1.addWidget(self._cest_mrd_w)

        # ── CEST parameters (kept in a dialog; opened by the button below) ──
        self._cest_params_w = QWidget()
        _cp = QVBoxLayout(self._cest_params_w); _cp.setSpacing(6)

        # GE / Siemens options
        self._cest_ge_params_w = QWidget()
        _gep = QVBoxLayout(self._cest_ge_params_w)
        _gep.setContentsMargins(0, 0, 0, 0); _gep.setSpacing(4)
        _geh = QLabel("GE / Siemens options"); _geh.setStyleSheet("font-weight:bold; color:#4ec9b0;")
        _gep.addWidget(_geh)
        ga = QHBoxLayout()
        ga.addWidget(QLabel("Acquisition:"))
        self.combo_cest_acq = QComboBox()
        self.combo_cest_acq.addItems(["Custom  (Siemens / any)", "SSFSE", "CUBE (3D) — GE"])
        self.combo_cest_acq.setToolTip(
            "CEST acquisition layout:\n"
            "  • Custom — one image per saturation offset (Siemens or any vendor);\n"
            "    the loaded offsets .txt (Hz) is used directly, no idling/S0 prefix.\n"
            "  • SSFSE / CUBE (GE) — GE combined layout; idling / S0 / WASSR / CEST\n"
            "    ordering and slices-per-offset are handled automatically.")
        ga.addWidget(self.combo_cest_acq, stretch=1)
        _gep.addLayout(ga)
        gc3 = QHBoxLayout()
        gc3.addWidget(QLabel("Offsets (.txt):"))
        self.edit_cest_ppm_path = QLineEdit()
        self.edit_cest_ppm_path.setReadOnly(True)
        self.edit_cest_ppm_path.setPlaceholderText("CEST offsets — one per line (Hz)")
        gc3.addWidget(self.edit_cest_ppm_path, stretch=1)
        btn_cest_ppm = QPushButton("Browse…")
        btn_cest_ppm.clicked.connect(self._browse_cest_ppm)
        gc3.addWidget(btn_cest_ppm)
        _gep.addLayout(gc3)
        self.chk_cest_combined = QCheckBox(
            "This offset .txt file contains WASSR + CEST together")
        self.chk_cest_combined.setToolTip(
            "Check if the loaded data and offsets .txt include BOTH the WASSR/B0\n"
            "block and the CEST block (idling + S0 + idling + WASSR + CEST in   \n"
            "one file).Loading CEST then also splits off and loads the WASSR/B0 \n"
            "map — no separate WASSR load needed. Leave unchecked to load CEST  \n"
            "and WASSR separately with their own offsets files.")
        _gep.addWidget(self.chk_cest_combined)
        # Backend-only: Larmor MHz + Hz flag (kept; read from DICOM where possible)
        self.spin_gecest_mhz = QDoubleSpinBox(); self.spin_gecest_mhz.setDecimals(4)
        self.spin_gecest_mhz.setRange(1.0, 1000.0); self.spin_gecest_mhz.setValue(127.7415)
        self.spin_gecest_mhz.setVisible(False)
        self.chk_gecest_hz = QCheckBox("Offsets in Hz"); self.chk_gecest_hz.setChecked(True)
        self.chk_gecest_hz.setVisible(False)
        _gep.addWidget(self.spin_gecest_mhz); _gep.addWidget(self.chk_gecest_hz)
        _cp.addWidget(self._cest_ge_params_w)

        # MR Solutions options
        self._cest_mrd_params_w = QWidget()
        _mrp = QVBoxLayout(self._cest_mrd_params_w)
        _mrp.setContentsMargins(0, 0, 0, 0); _mrp.setSpacing(4)
        _mrh = QLabel("MR Solutions options"); _mrh.setStyleSheet("font-weight:bold; color:#4ec9b0;")
        _mrp.addWidget(_mrh)
        mc2 = QHBoxLayout()
        mc2.addWidget(QLabel("Larmor (MHz):"))
        self.spin_cest_mrd_mhz = QDoubleSpinBox()
        self.spin_cest_mrd_mhz.setDecimals(4)
        self.spin_cest_mrd_mhz.setRange(1.0, 1000.0)
        self.spin_cest_mrd_mhz.setValue(199.7502)
        self.spin_cest_mrd_mhz.setToolTip(
            "Proton Larmor frequency used to convert saturation offsets (Hz) → ppm.\n"
            "MR Solutions .MRD files do not store this; default 199.7502 MHz (4.7 T).")
        self.spin_cest_mrd_mhz.setFixedWidth(110)
        mc2.addWidget(self.spin_cest_mrd_mhz)
        mc2.addStretch()
        _mrp.addLayout(mc2)
        _cp.addWidget(self._cest_mrd_params_w)

        # Bruker note (Bruker reads everything from the method file)
        self._cest_bruker_note = QLabel(
            "Bruker: offsets, Larmor and saturation power are read automatically "
            "from the method file — no parameters needed here.")
        self._cest_bruker_note.setWordWrap(True)
        self._cest_bruker_note.setStyleSheet("font-size:11px; color:#888;")
        _cp.addWidget(self._cest_bruker_note)
        _cp.addStretch()

        # Connect Platform combo to show/hide sub-widgets + params
        self.combo_vendor_cest.currentIndexChanged.connect(self._on_cest_vendor_changed)

        # ── Buttons: CEST parameters (dialog) + Load ───────────────────────
        _cest_btn_row = QHBoxLayout()
        self.btn_cest_params = QPushButton("CEST parameters")
        self.btn_cest_params.setToolTip(
            "Acquisition type, offsets (.txt) file and the WASSR+CEST option.")
        self.btn_cest_params.clicked.connect(self._open_cest_params_dialog)
        _cest_btn_row.addWidget(self.btn_cest_params)
        self.btn_load_cest = QPushButton("Load CEST Data")
        self.btn_load_cest.clicked.connect(self._load_cest)
        _cest_btn_row.addWidget(self.btn_load_cest)
        g1.addLayout(_cest_btn_row)

        self.lbl_cest_info = QLabel("No data loaded.")
        self.lbl_cest_info.setStyleSheet("font-size: 11px; color: gray;")
        self.lbl_cest_info.setWordWrap(True)
        g1.addWidget(self.lbl_cest_info)

        left_layout.addWidget(grp_cest)

        # ── B0 map (WASSR) group ──────────────────────────────────────────
        grp_b0 = QGroupBox("B0 Map — WASSR")
        g2 = QVBoxLayout(grp_b0)

        # Vendor selector row
        vendor_wassr_row = QHBoxLayout()
        vendor_wassr_row.addWidget(QLabel("Platform:"))
        self.combo_vendor_wassr = QComboBox()
        self.combo_vendor_wassr.addItems(
            ["Bruker", "GE/Siemens", "MR Solutions"]
        )
        vendor_wassr_row.addWidget(self.combo_vendor_wassr, stretch=1)
        g2.addLayout(vendor_wassr_row)

        # ── Bruker sub-widget ──────────────────────────────────────────────
        self._wassr_bruker_w = QWidget()
        bruker_wassr_lay = QVBoxLayout(self._wassr_bruker_w)
        bruker_wassr_lay.setContentsMargins(0, 0, 0, 0)

        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Folder:"))
        self.edit_wassr_dir = QLineEdit()
        self.edit_wassr_dir.setPlaceholderText("Select the scan folder…")
        self.edit_wassr_dir.setReadOnly(True)
        r3.addWidget(self.edit_wassr_dir, stretch=1)
        self.btn_wassr_browse = QPushButton("Browse…")
        self.btn_wassr_browse.clicked.connect(self._browse_wassr)
        r3.addWidget(self.btn_wassr_browse)
        bruker_wassr_lay.addLayout(r3)

        # Bruker version kept in the backend (auto/PV360) but not shown.
        self.combo_pv_wassr = QComboBox()
        self.combo_pv_wassr.addItems(["PV360", "PV6 / PV7"])
        self.combo_pv_wassr.setVisible(False)
        g2.addWidget(self._wassr_bruker_w)

        # ── GE/Siemens combined DICOM + NIfTI sub-widget ──────────────────
        self._wassr_ge_w = QWidget()
        ge_wassr_lay = QVBoxLayout(self._wassr_ge_w)
        ge_wassr_lay.setContentsMargins(0, 0, 0, 0)
        ge_wassr_lay.setSpacing(3)

        # ── Acquisition type ───────────────────────────────────────────────
        gwa = QHBoxLayout()
        gwa.addWidget(QLabel("Acquisition:"))
        self.combo_wassr_acq = QComboBox()
        self.combo_wassr_acq.addItems(["Custom  (Siemens / any)", "SSFSE", "CUBE (3D) — GE"])
        gwa.addWidget(self.combo_wassr_acq, stretch=1)
        ge_wassr_lay.addLayout(gwa)

        # ── Data (file or folder — usually the SAME series as CEST) ────────
        gwd = QHBoxLayout()
        gwd.addWidget(QLabel("Data:"))
        self.edit_wassr_data = QLineEdit(); self.edit_wassr_data.setReadOnly(True)
        self.edit_wassr_data.setPlaceholderText("DICOM/NIfTI — file or folder")
        gwd.addWidget(self.edit_wassr_data, stretch=1)
        _bwf = QPushButton("File…"); _bwf.clicked.connect(lambda: self._browse_wassr_data(False))
        gwd.addWidget(_bwf)
        _bwd = QPushButton("Folder…"); _bwd.clicked.connect(lambda: self._browse_wassr_data(True))
        gwd.addWidget(_bwd)
        ge_wassr_lay.addLayout(gwd)

        # ── Offsets (.txt) — WASSR offsets for this scan ───────────────────
        gw3 = QHBoxLayout()
        gw3.addWidget(QLabel("Offsets (.txt):"))
        self.edit_wassr_ppm_path = QLineEdit()
        self.edit_wassr_ppm_path.setReadOnly(True)
        self.edit_wassr_ppm_path.setPlaceholderText("WASSR offsets — one per line (Hz)")
        gw3.addWidget(self.edit_wassr_ppm_path, stretch=1)
        btn_wassr_ppm = QPushButton("Browse…")
        btn_wassr_ppm.clicked.connect(self._browse_wassr_ppm)
        gw3.addWidget(btn_wassr_ppm)
        ge_wassr_lay.addLayout(gw3)

        self._wassr_ge_w.setVisible(False)
        g2.addWidget(self._wassr_ge_w)

        # ── MR Solutions (.MRD) sub-widget ─────────────────────────────────
        self._wassr_mrd_w = QWidget()
        mrd_wassr_lay = QVBoxLayout(self._wassr_mrd_w)
        mrd_wassr_lay.setContentsMargins(0, 0, 0, 0)
        mrd_wassr_lay.setSpacing(3)

        mw1 = QHBoxLayout()
        mw1.addWidget(QLabel("MRD file:"))
        self.edit_wassr_mrd_path = QLineEdit()
        self.edit_wassr_mrd_path.setReadOnly(True)
        self.edit_wassr_mrd_path.setPlaceholderText("…/<scan>_000_0.MRD  (WASSR/B0, SMIS raw)")
        mw1.addWidget(self.edit_wassr_mrd_path, stretch=1)
        btn_wassr_mrd = QPushButton("Browse…")
        btn_wassr_mrd.clicked.connect(self._browse_wassr_mrd)
        mw1.addWidget(btn_wassr_mrd)
        mrd_wassr_lay.addLayout(mw1)

        mw2 = QHBoxLayout()
        mw2.addWidget(QLabel("Larmor (MHz):"))
        self.spin_wassr_mrd_mhz = QDoubleSpinBox()
        self.spin_wassr_mrd_mhz.setDecimals(4)
        self.spin_wassr_mrd_mhz.setRange(1.0, 1000.0)
        self.spin_wassr_mrd_mhz.setValue(199.7502)
        self.spin_wassr_mrd_mhz.setToolTip(
            "Proton Larmor frequency used to convert saturation offsets (Hz) → ppm.\n"
            "MR Solutions .MRD files do not store this; default 199.7502 MHz (4.7 T)."
        )
        self.spin_wassr_mrd_mhz.setFixedWidth(110)
        mw2.addWidget(self.spin_wassr_mrd_mhz)
        mw2.addStretch()
        mrd_wassr_lay.addLayout(mw2)

        self._wassr_mrd_w.setVisible(False)
        g2.addWidget(self._wassr_mrd_w)

        # Connect vendor combo to show/hide sub-widgets
        self.combo_vendor_wassr.currentIndexChanged.connect(self._on_wassr_vendor_changed)

        # Load button and info label
        self.btn_load_wassr = QPushButton("Load B0 Map")
        self.btn_load_wassr.clicked.connect(self._load_wassr)
        g2.addWidget(self.btn_load_wassr)

        self.lbl_wassr_info = QLabel("Not loaded (B0 correction skipped).")
        self.lbl_wassr_info.setStyleSheet("font-size: 11px; color: gray;")
        g2.addWidget(self.lbl_wassr_info)

        left_layout.addWidget(grp_b0)

        # ── M0 / Unsaturated image source ─────────────────────────────────
        grp_m0 = QGroupBox("M0 / Unsaturated Image")
        g_m0 = QVBoxLayout(grp_m0)
        g_m0.setSpacing(4)
        m0_note = QLabel(
            "M0 is used for z-spec normalisation"
        )
        m0_note.setStyleSheet("font-size: 11px; color: #aaa;")
        m0_note.setWordWrap(True)
        g_m0.addWidget(m0_note)

        # Row 1 — data source (CEST / WASSR)
        m0_src_row = QHBoxLayout()
        m0_src_row.addWidget(QLabel("Source:"))
        self.combo_m0_source = QComboBox()
        self.combo_m0_source.addItems(["CEST", "WASSR"])
        self.combo_m0_source.setToolTip(
            "Choose which loaded dataset to pick the M0 frame from.\n"
            "CEST — uses the full CEST image series.\n"
            "WASSR — uses the full WASSR image series (load it first).\n"
            "Phantom auto-detection in the ROI panel always uses this M0."
        )
        m0_src_row.addWidget(self.combo_m0_source, stretch=1)
        g_m0.addLayout(m0_src_row)

        # Row 2 — frame selector (populated once data is loaded)
        m0_frame_row = QHBoxLayout()
        m0_frame_row.addWidget(QLabel("Frame:"))
        self.combo_m0_frame = QComboBox()
        self.combo_m0_frame.setEnabled(False)
        self.combo_m0_frame.addItem("— load CEST / WASSR data first —")
        m0_frame_row.addWidget(self.combo_m0_frame, stretch=1)
        g_m0.addLayout(m0_frame_row)

        self.lbl_m0_info = QLabel("No M0 loaded yet.")
        self.lbl_m0_info.setStyleSheet("font-size: 11px; color: gray;")
        self.lbl_m0_info.setWordWrap(True)
        g_m0.addWidget(self.lbl_m0_info)

        # Repopulate frame list when source changes; also refresh canvas if in M0 mode
        self.combo_m0_source.currentIndexChanged.connect(
            lambda _: self._on_m0_selection_changed()
        )
        # Refresh info label + canvas whenever a different frame is chosen
        self.combo_m0_frame.currentIndexChanged.connect(
            lambda _: self._on_m0_selection_changed()
        )

        left_layout.addWidget(grp_m0)

        # ── Denoising group ───────────────────────────────────────────────
        # ── Motion correction (rigid, itk-elastix) ─────────────────────────
        from my_gui.motion_correction import elastix_available, REFERENCE_MODES
        grp_moco = QGroupBox("Motion Correction")
        g_mc = QVBoxLayout(grp_moco); g_mc.setSpacing(4)
        mc_row = QHBoxLayout()
        self.chk_moco = QCheckBox("Enable motion correction")
        self.chk_moco.setToolTip(
            "Rigidly register every offset image to a reference frame (Elastix),\n"
            "correcting subject motion between saturation offsets.")
        mc_row.addWidget(self.chk_moco); mc_row.addStretch()
        g_mc.addLayout(mc_row)
        mref_row = QHBoxLayout()
        self.combo_moco_ref = QComboBox(); self.combo_moco_ref.addItems(REFERENCE_MODES)
        mref_row.addWidget(self.combo_moco_ref, stretch=1)
        g_mc.addLayout(mref_row)
        if not elastix_available():
            self.chk_moco.setEnabled(False); self.combo_moco_ref.setEnabled(False)
            _mc_hint = QLabel("itk-elastix not installed — pip install itk-elastix")
            _mc_hint.setStyleSheet("color:#c62828; font-size:10px;")
            _mc_hint.setWordWrap(True)
            g_mc.addWidget(_mc_hint)
        self.chk_moco.toggled.connect(self._on_moco_toggled)
        left_layout.addWidget(grp_moco)

        grp_denoise = QGroupBox("Z-Spectrum Denoising  (applied before fitting)")
        g_dn = QVBoxLayout(grp_denoise)
        g_dn.setSpacing(4)

        dn_method_row = QHBoxLayout()
        dn_method_row.addWidget(QLabel("Method:"))
        self.combo_denoise = QComboBox()
        self.combo_denoise.addItem("None")
        self.combo_denoise.addItem("PCA")
        # BM3D needs the `bm3d` package; NLM needs numba — check availability once.
        try:
            import bm3d as _bm3dlib       # noqa: F401
            _bm3d_ok = True
        except ImportError:
            _bm3d_ok = False
        try:
            import numba as _nb           # noqa: F401
            _numba_ok = True
        except ImportError:
            _numba_ok = False
        self.combo_denoise.addItem("BM3D")
        self.combo_denoise.addItem("NLM")
        _model = self.combo_denoise.model()
        if not _bm3d_ok:
            _it = _model.item(2)
            if _it:
                _it.setEnabled(False)
        if not _numba_ok:
            _it = _model.item(3)
            if _it:
                _it.setEnabled(False)
        self.combo_denoise.setToolTip(
            "<b>None</b> — no denoising (default)<br>"
            "<b>PCA</b> — Principal Component Analysis<br>"
            "<b>BM3D</b> — Block-Matching 3D; gold-standard denoising<br>"
            "<b>NLM</b> — Non-Local Means"
        )
        dn_method_row.addWidget(self.combo_denoise, stretch=1)
        g_dn.addLayout(dn_method_row)

        # PCA has no user options — the number of components to keep is chosen
        # automatically with Malinowski's indicator function (the chemometrics
        # standard / recommended criterion), so there is no criteria selector.

        # BM3D options
        self._dn_bm3d_row = QWidget()
        _bm3d_lay = QHBoxLayout(self._dn_bm3d_row)
        _bm3d_lay.setContentsMargins(0, 0, 0, 0)
        _bm3d_lay.addWidget(QLabel("Strength:"))
        self.spin_bm3d_strength = QDoubleSpinBox()
        self.spin_bm3d_strength.setRange(0.2, 3.0)
        self.spin_bm3d_strength.setValue(1.0)
        self.spin_bm3d_strength.setSingleStep(0.1)
        self.spin_bm3d_strength.setDecimals(1)
        self.spin_bm3d_strength.setToolTip(
            "BM3D denoising strength (× the auto-estimated noise σ).\n"
            "1.0 = use the estimated noise level (recommended).\n"
            ">1 = stronger smoothing;  <1 = gentler."
        )
        _bm3d_lay.addWidget(self.spin_bm3d_strength, stretch=1)
        self._dn_bm3d_row.setVisible(False)
        g_dn.addWidget(self._dn_bm3d_row)

        # NLM options
        self._dn_nlm_row = QWidget()
        _nlm_lay = QHBoxLayout(self._dn_nlm_row)
        _nlm_lay.setContentsMargins(0, 0, 0, 0)
        _nlm_lay.addWidget(QLabel("Search win:"))
        self.spin_nlm_big = QSpinBox()
        self.spin_nlm_big.setRange(5, 51)
        self.spin_nlm_big.setValue(21)
        self.spin_nlm_big.setSingleStep(2)
        self.spin_nlm_big.setToolTip("NLM: size of the large search window (odd number)")
        _nlm_lay.addWidget(self.spin_nlm_big)
        _nlm_lay.addWidget(QLabel("Patch win:"))
        self.spin_nlm_small = QSpinBox()
        self.spin_nlm_small.setRange(3, 21)
        self.spin_nlm_small.setValue(5)
        self.spin_nlm_small.setSingleStep(2)
        self.spin_nlm_small.setToolTip("NLM: size of the small patch comparison window (odd number)")
        _nlm_lay.addWidget(self.spin_nlm_small)
        self._dn_nlm_row.setVisible(False)
        g_dn.addWidget(self._dn_nlm_row)

        # Preview button
        dn_btn_row = QHBoxLayout()
        self.btn_preview_denoise = QPushButton("Denoise")
        self.btn_preview_denoise.setEnabled(False)
        self.btn_preview_denoise.setToolTip(
            "Show a before/after comparison of the denoised Z-spectrum\n"
            "for slice 0 at the first frequency offset.\n"
            "Load CEST data first, then select a method."
        )
        self.btn_preview_denoise.clicked.connect(self._preview_denoise)
        dn_btn_row.addWidget(self.btn_preview_denoise)
        dn_btn_row.addStretch()
        g_dn.addLayout(dn_btn_row)

        def _denoise_method_key() -> str:
            """Return clean method key ('None','PCA','BM3D','NLM') from combo text."""
            txt = self.combo_denoise.currentText()
            for _k in ('None', 'PCA', 'BM3D', 'NLM'):
                if txt.startswith(_k):
                    return _k
            return 'None'

        self._denoise_method_key = _denoise_method_key  # expose for _run_analysis

        def _on_denoise_method_changed(_idx):
            _m = _denoise_method_key()
            self._dn_bm3d_row.setVisible(_m == 'BM3D')
            self._dn_nlm_row.setVisible(_m == 'NLM')
            self.btn_preview_denoise.setEnabled(
                _m != 'None' and self._z_img_full is not None
            )

        self.combo_denoise.currentIndexChanged.connect(_on_denoise_method_changed)
        left_layout.addWidget(grp_denoise)

        # ── Processing options — single "CEST MRI Processing" button ─────────
        # All options live in a persistent dialog; button opens/raises it.
        import os as _os
        _ncpu = min(_os.cpu_count() or 4, 16)

        # Create all option widgets as instance attributes (accessible throughout)
        self.spin_snr = QDoubleSpinBox()
        self.spin_snr.setRange(0.5, 50.0)
        self.spin_snr.setValue(3.0)
        self.spin_snr.setSingleStep(0.5)
        self.spin_snr.setToolTip(
            "<b>SNR Threshold</b><br>"
            "Only voxels where  M0 &gt; threshold × noise_floor  are processed.<br>"
            "Voxels below this level are masked out (set to zero) before fitting.<br><br>"
            "<b>How it works:</b><br>"
            "The noise floor is estimated from the darkest 5 % of M0 values.<br>"
            "A voxel passes if its M0 signal exceeds  threshold × noise_floor.<br><br>"
            "<b>Recommended values:</b><br>"
            "• <b>3.0</b> (default) — good for most phantom experiments<br>"
            "• <b>2.0 – 3.0</b> — for low-SNR or small-FOV acquisitions<br>"
            "• <b>4.0 – 6.0</b> — for in-vivo data where background rejection is important<br><br>"
            "Reduce if too many valid voxels are being excluded.<br>"
            "Increase if background noise voxels appear in the maps."
        )

        self.spin_mtr_ppm = QDoubleSpinBox()
        self.spin_mtr_ppm.setRange(0.5, 10.0)
        self.spin_mtr_ppm.setValue(3.5)
        self.spin_mtr_ppm.setSingleStep(0.5)

        self.spin_larmor_mhz = QDoubleSpinBox()
        self.spin_larmor_mhz.setRange(1.0, 1200.0)
        self.spin_larmor_mhz.setValue(400.0)
        self.spin_larmor_mhz.setSingleStep(10.0)
        self.spin_larmor_mhz.setDecimals(1)
        self.spin_larmor_mhz.setFixedWidth(90)
        self.spin_larmor_mhz.setToolTip(
            "Proton Larmor frequency (MHz).\n"
            "Used to convert the B0 map from ppm to Hz.\n"
            "Common values: 400 MHz (9.4 T), 300 MHz (7 T), 128 MHz (3 T)"
        )
        self.spin_larmor_mhz.valueChanged.connect(self._refresh_display)

        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(1, 64)
        self.spin_workers.setValue(_ncpu)
        self.spin_workers.setToolTip(
            "<b>Parallel Workers</b><br>"
            "Each worker runs on one CPU core and fits a chunk of voxels simultaneously.<br>"
            "More workers = faster fitting, up to the number of physical CPU cores.<br><br>"
            f"<b>This computer:</b> {_os.cpu_count() or '?'} logical CPU cores detected<br>"
            f"<b>Default:</b> {_ncpu} workers (capped at 16 to avoid memory pressure)<br><br>"
            "<b>How to check your CPU:</b><br>"
            "• macOS: Apple menu → About This Mac → Chip / Processor<br>"
            "• Windows: Task Manager → Performance → CPU → Cores<br>"
            "• Linux:  <tt>nproc</tt>  or  <tt>lscpu | grep '^CPU(s)'</tt><br><br>"
            "Tip: set workers = physical cores (not logical/hyper-threaded)<br>"
            "for best throughput without memory contention."
        )

        self.combo_fit_quality = QComboBox()
        self.combo_fit_quality.addItems(["Fast", "Balanced", "Precise"])
        self.combo_fit_quality.setCurrentIndex(0)
        self.combo_fit_quality.setItemData(
            0, "Quick scan / parameter scouting", Qt.ItemDataRole.ToolTipRole)
        self.combo_fit_quality.setItemData(
            1, "Recommended for quantitative results", Qt.ItemDataRole.ToolTipRole)
        self.combo_fit_quality.setItemData(
            2, "Publication quality — tightest convergence", Qt.ItemDataRole.ToolTipRole)
        self.combo_fit_quality.setToolTip(
            "<b>Fit Quality Preset</b><br><br>"
            "<b>Fast</b> — ~0.05–0.2 s/voxel<br>"
            "Good for quick inspection and parameter scouting.<br><br>"
            "<b>Balanced</b> — ~0.3–0.8 s/voxel<br>"
            "Recommended for quantitative results.<br><br>"
            "<b>Precise</b> — ~1–3 s/voxel<br>"
            "Tightest convergence for precise quantitative maps.<br><br>"
            "Multiply time by number of voxels and divide by workers to estimate total run time."
        )

        self.btn_pools = QPushButton("Select pools to fit")
        self.btn_pools.setToolTip(
            "Choose which chemical exchange pools are included in DROF, MPLF,\n"
            "and other pool-based fitting methods in ROI Spectra.\n"
            "Water is always included (B₀ reference). Selection applies to all fitting subtabs."
        )
        self.btn_pools.clicked.connect(self._open_pools_dialog)

        # Build persistent dialog containing all the option widgets
        _proc_dlg = QDialog(self)
        _proc_dlg.setWindowTitle("CEST MRI Processing Settings")
        _proc_dlg.setMinimumWidth(340)
        _proc_dv = QVBoxLayout(_proc_dlg)

        grp_proc = QGroupBox("Processing Options")
        g3 = QVBoxLayout(grp_proc)

        snr_row = QHBoxLayout()
        snr_row.addWidget(QLabel("SNR threshold:"))
        snr_row.addWidget(self.spin_snr)
        snr_row.addStretch()
        g3.addLayout(snr_row)

        mtr_row = QHBoxLayout()
        mtr_row.addWidget(QLabel("MTR asym ppm:"))
        mtr_row.addWidget(self.spin_mtr_ppm)
        mtr_row.addStretch()
        g3.addLayout(mtr_row)

        larmor_row = QHBoxLayout()
        larmor_row.addWidget(QLabel("B₀ (MHz):"))
        larmor_row.addWidget(self.spin_larmor_mhz)
        larmor_row.addStretch()
        g3.addLayout(larmor_row)

        wk_row = QHBoxLayout()
        wk_row.addWidget(QLabel("Workers:"))
        wk_row.addWidget(self.spin_workers)
        wk_row.addStretch()
        g3.addLayout(wk_row)

        spd_row = QHBoxLayout()
        spd_row.addWidget(QLabel("Fit quality:"))
        spd_row.addWidget(self.combo_fit_quality, stretch=1)
        g3.addLayout(spd_row)

        pools_row = QHBoxLayout()
        pools_row.addWidget(QLabel("Pools:"))
        pools_row.addWidget(self.btn_pools, stretch=1)
        g3.addLayout(pools_row)

        _proc_dv.addWidget(grp_proc)
        _proc_close = QPushButton("Close")
        _proc_close.clicked.connect(_proc_dlg.hide)
        _proc_dv.addWidget(_proc_close)

        # The "CEST MRI Processing" button opens / raises the dialog
        btn_cest_proc = QPushButton("CEST MRI Processing")
        btn_cest_proc.setFixedHeight(34)
        btn_cest_proc.setStyleSheet(
            "QPushButton{background:#1565c0;color:white;font-weight:bold;"
            "border:none;border-radius:4px;padding:4px 12px;font-size:12px;}"
            "QPushButton:hover{background:#1976d2;}"
        )
        btn_cest_proc.clicked.connect(_proc_dlg.show)
        btn_cest_proc.clicked.connect(_proc_dlg.raise_)
        left_layout.addWidget(btn_cest_proc)

        # ── Run / Cancel buttons ──────────────────────────────────────────
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Z-Spectroscopy Analysis")
        self.btn_run.setFixedHeight(40)
        self.btn_run.setStyleSheet(
            "QPushButton{background:#27ae60;color:white;font-weight:bold;border-radius:4px;}"
            "QPushButton:hover{background:#2ecc71;}"
            "QPushButton:disabled{background:#555;color:#999;}"
        )
        self.btn_run.clicked.connect(self._run_analysis)
        run_row.addWidget(self.btn_run, stretch=1)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setFixedHeight(40)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet(
            "QPushButton { background: #c0392b; color: white; font-weight: bold; "
            "border: none; border-radius: 6px; padding: 4px 12px; }"
            "QPushButton:hover { background: #e74c3c; }"
            "QPushButton:disabled { background: #888; color: #bbb; }"
        )
        self.btn_cancel.clicked.connect(self._cancel_analysis)
        run_row.addWidget(self.btn_cancel)
        left_layout.addLayout(run_row)

        # ── Quick MTR-asymmetry (model-free, no pool fitting) ─────────────
        qmtr_row = QHBoxLayout()
        self.btn_quick_mtr = QPushButton("MTRasym")
        self.btn_quick_mtr.setFixedHeight(30)
        self.btn_quick_mtr.setToolTip(
            "Compute the MTR-asymmetry map directly from the Z-spectrum at the\n"
            "offset on the right — MTRasym = Z(−ppm) − Z(+ppm).\n"
            "Model-free: applies M0 normalisation + B0 correction (if a B0 map is\n"
            "loaded) but skips the slow PV/Gaussian/MPLF pool fitting.\n"
            "Any offset within the acquired range works — values are cubic-spline\n"
            "interpolated, so it need not match an exact acquired offset.")
        self.btn_quick_mtr.setStyleSheet(
            "QPushButton{background:#2980b9;color:white;font-weight:bold;border-radius:4px;}"
            "QPushButton:hover{background:#3498db;}")
        self.btn_quick_mtr.clicked.connect(self._quick_mtr_asym)
        self.btn_quick_mtr.setFixedWidth(130)
        qmtr_row.addWidget(self.btn_quick_mtr)
        qmtr_row.addWidget(QLabel("@ ppm:"))
        self.spin_quick_mtr_ppm = QDoubleSpinBox()
        self.spin_quick_mtr_ppm.setRange(0.1, 10.0)
        self.spin_quick_mtr_ppm.setDecimals(2)
        self.spin_quick_mtr_ppm.setSingleStep(0.1)
        self.spin_quick_mtr_ppm.setValue(3.5)
        self.spin_quick_mtr_ppm.setToolTip(
            "Offset (ppm) for the quick MTR-asymmetry map. e.g. 1.90 creatine,\n"
            "3.00 taurine, 3.50 amide/PLL, 4.20 & 5.50 iopamidol.\n"
            "Interpolated — does not need to be an exact acquired offset.")
        self.spin_quick_mtr_ppm.setFixedWidth(80)
        qmtr_row.addWidget(self.spin_quick_mtr_ppm)
        qmtr_row.addStretch()
        self._qmtr_w = QWidget(); self._qmtr_w.setLayout(qmtr_row)
        self._qmtr_w.setVisible(False)   # only shown for the MTR-asym display

        # MTR_Rex (inverse-difference: 1/Z(+ppm) − 1/Z(−ppm)) — its own offset
        qrex_row = QHBoxLayout()
        self.btn_quick_mtrrex = QPushButton("MTRrex")
        self.btn_quick_mtrrex.setFixedHeight(30)
        self.btn_quick_mtrrex.setToolTip(
            "Compute the MTR_Rex map at the offset on the right:\n"
            "MTRRex = 1/Z(+ppm) − 1/Z(−ppm) — the R1-independent inverse metric\n"
            "(AREX = R1 × MTRRex). Removes spillover/MT bias vs MTR-asymmetry.\n"
            "Model-free: M0 normalisation + B0 correction, no pool fitting.")
        self.btn_quick_mtrrex.setStyleSheet(
            "QPushButton{background:#8e44ad;color:white;font-weight:bold;border-radius:4px;}"
            "QPushButton:hover{background:#9b59b6;}")
        self.btn_quick_mtrrex.clicked.connect(self._quick_mtrrex)
        self.btn_quick_mtrrex.setFixedWidth(130)
        qrex_row.addWidget(self.btn_quick_mtrrex)
        qrex_row.addWidget(QLabel("@ ppm:"))
        self.spin_quick_rex_ppm = QDoubleSpinBox()
        self.spin_quick_rex_ppm.setRange(0.1, 10.0)
        self.spin_quick_rex_ppm.setDecimals(2)
        self.spin_quick_rex_ppm.setSingleStep(0.1)
        self.spin_quick_rex_ppm.setValue(3.5)
        self.spin_quick_rex_ppm.setToolTip(
            "Offset (ppm) for the MTR_Rex map. e.g. 1.90 creatine, 3.00 taurine,\n"
            "3.50 amide/PLL, 4.20 & 5.50 iopamidol. Interpolated — need not match\n"
            "an exact acquired offset.")
        self.spin_quick_rex_ppm.setFixedWidth(80)
        qrex_row.addWidget(self.spin_quick_rex_ppm)
        qrex_row.addStretch()
        self._qrex_w = QWidget(); self._qrex_w.setLayout(qrex_row)
        self._qrex_w.setVisible(False)   # only shown for the MTR_Rex display

        # ── Session save / load ───────────────────────────────────────────
        sess_row = QHBoxLayout()
        self.btn_save_session = QPushButton("Save Session")
        self.btn_save_session.setFixedHeight(28)
        self.btn_save_session.setToolTip(
            "Save the current fitting results (PV, Gaussian, MPLF maps + ROI spectra)\n"
            "to a .mat file, it will be loaded without re-running the analysis."
        )
        self.btn_save_session.setStyleSheet(
            "QPushButton{background:#2c3e50;color:white;border:none;border-radius:4px;padding:3px 8px;}"
            "QPushButton:hover{background:#34495e;}"
            "QPushButton:disabled{background:#444;color:#888;}"
        )
        self.btn_save_session.setEnabled(False)   # enabled only after a run
        self.btn_save_session.clicked.connect(self._save_session)
        sess_row.addWidget(self.btn_save_session, stretch=1)

        self.btn_load_session = QPushButton("Load Session")
        self.btn_load_session.setFixedHeight(28)
        self.btn_load_session.setToolTip(
            "Load a previously saved .mat session file to restore fitting results."
        )
        self.btn_load_session.setStyleSheet(
            "QPushButton{background:#2c3e50;color:white;border:none;border-radius:4px;padding:3px 8px;}"
            "QPushButton:hover{background:#34495e;}"
        )
        self.btn_load_session.clicked.connect(self._load_session)
        sess_row.addWidget(self.btn_load_session, stretch=1)
        left_layout.addLayout(sess_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        self.lbl_run_status = QLabel("")
        self.lbl_run_status.setStyleSheet("font-size: 11px;")
        left_layout.addWidget(self.lbl_run_status)

        # ── Log ───────────────────────────────────────────────────────────
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        from my_gui.theme import mono_font
        self.log.setFont(mono_font(10))
        self.log.setMaximumHeight(120)
        left_layout.addWidget(self.log)

        splitter.addWidget(left_scroll)

        # ── Right panel ───────────────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(4)

        # Map / display selector
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Display:"))
        self.combo_display = QComboBox()
        self.combo_display.addItems([
            "M0 image (unsaturated)",    # ← first; used by phantom auto-detection
            "CEST Images",
            "WASSR Images",
            "B0 map (ppm)",
            "B0 map (Hz)",
            "AREX map",                  # computed on demand from loaded CEST data
            "MTR asymmetry map",
            "MTR_Rex map",               # 1/Z(+ppm) − 1/Z(−ppm), quick path
            # Pool amplitude maps are added dynamically after fitting via _rebuild_display_combo
        ])
        self.combo_display.currentIndexChanged.connect(self._refresh_display)
        ctrl_row.addWidget(self.combo_display, stretch=1)

        # "ROIs + Bg" display mode — colour the map only inside the drawn ROIs,
        # over the 1st raw acquisition frame as a grayscale background.
        self.chk_roi_bg = QCheckBox("ROIs + Bkg")
        self.chk_roi_bg.setToolTip(
            "Show the colored map only inside the ROIs, over the 1st raw image "
            "as a gray background.")
        self.chk_roi_bg.toggled.connect(self._refresh_display)
        ctrl_row.addWidget(self.chk_roi_bg)

        self.btn_roi_bg = QPushButton("Bkg…")
        self.btn_roi_bg.setToolTip(
            "Pick the grayscale background image (from the Scan Directory) for "
            "the 'ROIs + Bkg' overlay.")
        self.btn_roi_bg.clicked.connect(self._pick_roi_bg)
        ctrl_row.addWidget(self.btn_roi_bg)

        self.btn_export_fig = QPushButton("Export figure…")
        self.btn_export_fig.clicked.connect(self._export_figure)
        ctrl_row.addWidget(self.btn_export_fig)
        # ROI Spectra / ROI Stats / Verify Offsets — created here but placed in a
        # row BELOW the image (added after the canvas), like the T1/T2 tab.
        self.btn_roi_spectra = QPushButton("ROI Spectra")
        self.btn_roi_spectra.clicked.connect(self._show_roi_spectra)
        # ROI Stats Table — beside ROI Spectra (matches the MRF viewer)
        self.btn_roi_table = QPushButton("ROI Statistics")
        self.btn_roi_table.clicked.connect(self._show_roi_table)
        # Diagnostic: verify the CEST offset ordering (esp. GE/Siemens DICOM)
        self.btn_verify_off = QPushButton("Verify Offsets…")
        self.btn_verify_off.setToolTip(
            "Check the offset ordering: plots mean phantom signal vs assigned "
            "offset. The Z-spectrum minimum should fall at 0 ppm (water). If it "
            "doesn't, the loaded offsets are mislabeled/misordered.")
        self.btn_verify_off.clicked.connect(self._verify_offsets)
        self.btn_hide_rois = QPushButton("Hide ROIs")
        self.btn_hide_rois.setCheckable(True)
        self.btn_hide_rois.setToolTip("Toggle ROI overlay visibility on the map")
        self.btn_hide_rois.clicked.connect(self._toggle_hide_rois)
        # Window/Level (brightness–contrast) drag tool — OsiriX-style.
        # When toggled on, drag over the map: horizontal → contrast, vertical → brightness.
        self.btn_contrast = WindowLevelToolButton()
        self.btn_contrast.toggled.connect(self._on_wl_tool_toggled)
        ctrl_row.addWidget(self.btn_contrast)
        ctrl_row.addWidget(self.btn_hide_rois)
        right_layout.addLayout(ctrl_row)

        # ── Offset slider frame (visible only for CEST/WASSR Images) ─────
        # Scrolls from most-positive ppm (index 0, left) → most-negative ppm
        # (index N-1, right) — matching the Bruker/MATLAB acquisition convention.
        self._offset_frame = QFrame()
        _of_lay = QHBoxLayout(self._offset_frame)
        _of_lay.setContentsMargins(0, 0, 0, 0)
        _of_lay.addWidget(QLabel("Offset:"))
        # Left endpoint label — shows most-positive ppm value
        self._offset_lbl_start = QLabel("+? ppm")
        self._offset_lbl_start.setStyleSheet("font-size: 10px; color: #aaa;")
        self._offset_lbl_start.setMinimumWidth(58)
        self._offset_lbl_start.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        _of_lay.addWidget(self._offset_lbl_start)
        self._offset_slider = QSlider(Qt.Orientation.Horizontal)
        self._offset_slider.setMinimum(0)
        self._offset_slider.setMaximum(0)
        self._offset_slider.setValue(0)
        self._offset_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._offset_slider.valueChanged.connect(self._on_img_slider)
        _of_lay.addWidget(self._offset_slider, stretch=1)
        # Right endpoint label — shows most-negative ppm value
        self._offset_lbl_end = QLabel("−? ppm")
        self._offset_lbl_end.setStyleSheet("font-size: 10px; color: #aaa;")
        self._offset_lbl_end.setMinimumWidth(58)
        _of_lay.addWidget(self._offset_lbl_end)
        # Current-offset indicator (bold, larger)
        self._offset_label = QLabel("— ppm")
        self._offset_label.setMinimumWidth(80)
        self._offset_label.setStyleSheet("font-weight: bold;")
        _of_lay.addWidget(self._offset_label)
        self._offset_frame.setVisible(False)
        right_layout.addWidget(self._offset_frame)

        # ── Slice slider (visible only when the volume has >1 slice, e.g. a
        #    de-tiled Siemens mosaic) — scrolls through the slice axis ────────
        self._slice_frame = QFrame()
        _sl_lay = QHBoxLayout(self._slice_frame)
        _sl_lay.setContentsMargins(0, 0, 0, 0)
        _sl_lay.addWidget(QLabel("Slice:"))
        self._slice_slider = QSlider(Qt.Orientation.Horizontal)
        self._slice_slider.setMinimum(0)
        self._slice_slider.setMaximum(0)
        self._slice_slider.setValue(0)
        self._slice_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slice_slider.valueChanged.connect(self._on_slice_slider)
        _sl_lay.addWidget(self._slice_slider, stretch=1)
        self._slice_label = QLabel("1/1")
        self._slice_label.setMinimumWidth(60)
        self._slice_label.setStyleSheet("font-weight: bold;")
        _sl_lay.addWidget(self._slice_label)
        self._slice_frame.setVisible(False)
        right_layout.addWidget(self._slice_frame)

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
        right_layout.addWidget(self.grp_fig_custom)

        # Custom title row
        _ztitle_row = QHBoxLayout()
        _ztitle_row.addWidget(QLabel("Title:"))
        self.edit_map_title = QLineEdit()
        self.edit_map_title.setPlaceholderText("Custom map title (leave blank for default)")
        _ztitle_row.addWidget(self.edit_map_title, stretch=1)
        self._fcp_lay.addLayout(_ztitle_row)

        from my_gui.format_bar import add_title_format_bar, connect_title_debounced
        connect_title_debounced(self.edit_map_title, self._refresh_display)

        # Fonts line on top (with B/I/x²/x₂ title buttons), colour-bar below.
        self.plot_bar = PlotCustomBar(default_cmap="gray", fonts_first=True)
        self.plot_bar.applied.connect(self._refresh_display)
        add_title_format_bar(self.edit_map_title, None,
                             target_row=self.plot_bar.font_row(),
                             default_getter=lambda: getattr(self.canvas, "_last_title", ""))
        # "Bg" toggle — black figure background for slides; inserted just before
        # the B / I / x² / x₂ title buttons.  Only the white surround + labels
        # change colour — the map data/colours are untouched.
        self.chk_dark_bg = QCheckBox("Bg")
        self.chk_dark_bg.setToolTip(
            "Black background for the map figure (for slides / PPT).\n"
            "Only the white surround and labels flip — the maps stay identical.")
        self.chk_dark_bg.toggled.connect(self._refresh_display)
        _frow = self.plot_bar.font_row()
        _b_idx = _frow.count()
        for _i in range(_frow.count()):
            _wd = _frow.itemAt(_i).widget()
            if isinstance(_wd, QPushButton) and _wd.text() == "B":
                _b_idx = _i
                break
        _frow.insertWidget(_b_idx, self.chk_dark_bg)

        # "Log map" — Fuderer perceptual log-like colour scaling (MRM 2025),
        # placed right after "Bg".  Warps the colormap so low values get more
        # contrast; data & colour-bar ticks stay linear; signed maps unchanged.
        self.chk_logmap = QCheckBox("Log map")
        self.chk_logmap.setToolTip(
            "Log-color scaling, redistribute the colormaps so equal color "
            "steps = equal % change in value.")
        self.chk_logmap.toggled.connect(self._refresh_display)
        _frow.insertWidget(_b_idx + 1, self.chk_logmap)
        self._fcp_lay.addWidget(self.plot_bar)
        # MTRasym / MTR_Rex map buttons (with @ ppm) live in the display area,
        # under the plot-customisation bar — each shown only for its own map view.
        right_layout.addWidget(self._qmtr_w)
        right_layout.addWidget(self._qrex_w)

        # ROI Spectra / Stats / Verify Offsets — in a row below the image.
        _roi_btn_row = QHBoxLayout()
        _roi_btn_row.addWidget(self.btn_roi_spectra)
        _roi_btn_row.addWidget(self.btn_roi_table)
        self.btn_verify_off.hide()   # moved into the ROI Spectra → Raw Z sub-tab

        self.canvas = ROICanvas()
        self.canvas.mpl_connect('scroll_event', self._on_canvas_scroll)
        right_layout.addWidget(self.canvas, stretch=1)
        right_layout.addLayout(_roi_btn_row)   # ROI Spectra / Stats / Verify — below image

        # Data cursor checkbox
        from my_gui.plot_custom_bar import DataCursorToolButton
        self.chk_datacursor = DataCursorToolButton()
        self.chk_datacursor.toggled.connect(self._toggle_datacursor)
        ctrl_row.insertWidget(ctrl_row.indexOf(self.btn_contrast) + 1, self.chk_datacursor)

        splitter.addWidget(right)
        splitter.setSizes([340, 660])

        left_layout.addStretch()

    # ─────────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────────

    # ── vendor-combo visibility handlers ──────────────────────────────────

    def _on_cest_vendor_changed(self, idx: int):
        v = self.combo_vendor_cest.currentText()
        self._cest_bruker_w.setVisible(v == "Bruker")
        self._cest_ge_w.setVisible(v.startswith("GE/Siemens"))
        self._cest_mrd_w.setVisible(v.startswith("MR Solutions"))
        # Mirror the platform choice in the "CEST parameters" dialog sections
        if hasattr(self, "_cest_ge_params_w"):
            self._cest_ge_params_w.setVisible(v.startswith("GE/Siemens"))
            self._cest_mrd_params_w.setVisible(v.startswith("MR Solutions"))
            self._cest_bruker_note.setVisible(v == "Bruker")

    def _open_cest_params_dialog(self):
        """Show the CEST acquisition/offsets/options in a dedicated dialog so the
        main CEST panel stays just Platform + Folder."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton
        dlg = getattr(self, "_cest_params_dialog", None)
        if dlg is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("CEST parameters")
            dlg.setMinimumWidth(460)
            v = QVBoxLayout(dlg)
            v.addWidget(self._cest_params_w)
            br = QHBoxLayout(); br.addStretch()
            ok = QPushButton("Close"); ok.clicked.connect(dlg.accept)
            br.addWidget(ok); v.addLayout(br)
            self._cest_params_dialog = dlg
        self._on_cest_vendor_changed(0)   # show only the current platform's options
        dlg.show(); dlg.raise_(); dlg.activateWindow()

    def _on_wassr_vendor_changed(self, idx: int):
        v = self.combo_vendor_wassr.currentText()
        self._wassr_bruker_w.setVisible(v == "Bruker")
        self._wassr_ge_w.setVisible(v.startswith("GE/Siemens"))
        self._wassr_mrd_w.setVisible(v.startswith("MR Solutions"))

    # ── public accessors ──────────────────────────────────────────────────

    def get_mtr_asym_map(self):
        """Return the computed MTR-asymmetry map (2-D ndarray) or None.

        Used by the T1/T2 tab to overlay the CEST contrast on T1/T2 images.
        """
        data = getattr(self, "_results", None)
        if not data:
            return None
        m = data.get("mtr_map")
        if m is None:
            return None
        import numpy as _np
        m = _np.asarray(m)
        return m[..., 0] if m.ndim == 3 else m

    def get_mtr_asym_ppm(self) -> float:
        """ppm offset used for the current MTR-asymmetry map (for labelling)."""
        data = getattr(self, "_results", None)
        return float(data.get("mtr_ppm_used", 3.5)) if data else 3.5

    def get_current_cest_image(self):
        """Return the currently-selected CEST image frame (2-D) for overlay base.

        Falls back to the M0 / unsaturated image when no CEST stack is loaded.
        """
        import numpy as _np
        z = getattr(self, "_z_img_full", None)
        if z is not None:
            idx = 0
            sl = getattr(self, "_offset_slider", None)
            if sl is not None:
                idx = min(sl.value(), z.shape[-1] - 1)
            return _np.asarray(z[:, :, 0, idx])
        m0 = self._get_m0()
        if m0 is not None:
            m0 = _np.asarray(m0)
            sl = m0[:, :, 0] if m0.ndim == 3 else m0
            return sl[:, :, 0] if sl.ndim == 3 else sl
        return None

    # ── browse methods ────────────────────────────────────────────────────

    def _browse_cest(self):
        d = QFileDialog.getExistingDirectory(self, "Select CEST pdata/1/2dseq directory", "")
        if d:
            self._cest_dir = d
            self.edit_cest_dir.setText(d)
            self.lbl_cest_info.setText("Directory selected — click 'Load CEST Data'.")
            self.lbl_cest_info.setStyleSheet("font-size: 11px; color: #555;")

    def _browse_cest_data(self, folder: bool):
        if folder:
            p = QFileDialog.getExistingDirectory(self, "Select CEST DICOM folder", "")
        else:
            p, _ = QFileDialog.getOpenFileName(
                self, "Select CEST image", "",
                "Images (*.nii *.nii.gz *.dcm *.IMA);;All files (*)")
        if p:
            self._cest_data_path = p
            self.edit_cest_data.setText(p)

    def _browse_cest_ppm(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CEST offsets file", "",
            "Text files (*.txt *.csv);;All files (*)"
        )
        if path:
            self._cest_ppm_path = path
            self.edit_cest_ppm_path.setText(path)

    def _browse_wassr(self):
        d = QFileDialog.getExistingDirectory(self, "Select WASSR scan folder", "")
        if d:
            self._wassr_dir = d
            self.edit_wassr_dir.setText(d)

    def _browse_wassr_data(self, folder: bool):
        if folder:
            p = QFileDialog.getExistingDirectory(self, "Select WASSR DICOM folder", "")
        else:
            p, _ = QFileDialog.getOpenFileName(
                self, "Select WASSR image", "",
                "Images (*.nii *.nii.gz *.dcm *.IMA);;All files (*)")
        if p:
            self._wassr_data_path = p
            self.edit_wassr_data.setText(p)

    def _browse_wassr_ppm(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select WASSR offsets file", "",
            "Text files (*.txt *.csv);;All files (*)"
        )
        if path:
            self._wassr_ppm_path = path
            self.edit_wassr_ppm_path.setText(path)

    def _browse_cest_mrd(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CEST .MRD file", "",
            "MR Solutions raw (*.MRD *.mrd);;All files (*)"
        )
        if path:
            self._cest_mrd_path = path
            self.edit_cest_mrd_path.setText(path)

    def _browse_wassr_mrd(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select WASSR .MRD file", "",
            "MR Solutions raw (*.MRD *.mrd);;All files (*)"
        )
        if path:
            self._wassr_mrd_path = path
            self.edit_wassr_mrd_path.setText(path)

    # ── load dispatch methods ─────────────────────────────────────────────

    def _load_cest(self):
        v = self.combo_vendor_cest.currentText()
        if v == "Bruker":
            self._load_cest_bruker()
        elif v.startswith("MR Solutions"):
            self._load_cest_mrd()
        else:
            self._load_cest_ge()

    def _load_wassr(self):
        v = self.combo_vendor_wassr.currentText()
        if v == "Bruker":
            self._load_wassr_bruker()
        elif v.startswith("MR Solutions"):
            self._load_wassr_mrd()
        else:
            self._load_wassr_ge()

    @staticmethod
    def _peek_n_frames(data_path: str) -> "int | None":
        """Cheaply read the number of frames without loading pixel data."""
        try:
            if os.path.isdir(data_path):
                import glob as _g
                n = len([f for f in _g.glob(os.path.join(data_path, "*"))
                         if os.path.isfile(f) and f.lower().endswith((".dcm", ".ima"))])
                return n or None
            if data_path.lower().endswith((".nii", ".nii.gz")):
                import nibabel as nib
                shp = nib.load(data_path).shape
                return int(shp[-1]) if len(shp) >= 4 else 1
        except Exception:
            return None
        return None

    @staticmethod
    def _peek_larmor(data_path: str) -> "float | None":
        """Read proton Larmor frequency (MHz) from a DICOM (ImagingFrequency)."""
        try:
            import glob as _g, pydicom
            f = None
            if os.path.isdir(data_path):
                files = sorted(x for x in _g.glob(os.path.join(data_path, "*"))
                               if os.path.isfile(x) and x.lower().endswith((".dcm", ".ima")))
                f = files[0] if files else None
            elif data_path.lower().endswith((".dcm", ".ima")):
                f = data_path
            if f:
                ds = pydicom.dcmread(f, force=True, stop_before_pixels=True)
                v = float(getattr(ds, "ImagingFrequency", 0.0))
                return v if v > 1.0 else None
        except Exception:
            pass
        return None

    @staticmethod
    def _detect_wassr_split(offs: np.ndarray) -> int:
        """In a combined offsets file [WASSR…, CEST…], find where CEST begins.

        The WASSR block is a low-magnitude ramp that ends where the offset jumps
        back up to start the CEST block. Returns the # of leading WASSR offsets
        (0 if no clear split is found → treat the whole file as CEST)."""
        offs = np.asarray(offs, dtype=float).ravel()
        if len(offs) < 4:
            return 0
        d = np.diff(offs)
        j = int(np.argmax(d))                       # biggest upward jump
        med = np.median(np.abs(d[:max(1, j)])) if j > 0 else np.median(np.abs(d))
        if d[j] > 3.0 * (med + 1e-9):
            return j + 1
        return 0

    def _load_cest_ge(self):
        """GE/Siemens CEST. If the offsets file holds WASSR + CEST together
        (either the box is ticked, or a WASSR block is auto-detected from the
        offsets), the single file is split and BOTH blocks (+B0 map) are loaded;
        otherwise just the CEST block (last N offsets) is loaded."""
        combined = bool(getattr(self, "chk_cest_combined", None)
                        and self.chk_cest_combined.isChecked())
        off_path = self._cest_ppm_path.strip()
        if not combined and off_path:
            # Auto-detect a combined WASSR+CEST offsets file (a low-magnitude
            # WASSR ramp followed by the CEST block restarting at a high value).
            try:
                if self._detect_wassr_split(np.loadtxt(off_path).ravel()) > 0:
                    combined = True
                    self.chk_cest_combined.setChecked(True)   # reflect in the UI
                    self._log("  Combined WASSR+CEST offsets file auto-detected.")
            except Exception:
                pass
        if combined:
            self._load_ge_combined()
        else:
            self._load_ge_block("cest")

    def _load_ge_combined(self):
        """Load CEST + WASSR + B0 from one combined data file & offsets file."""
        from my_gui.ge_cest_reader import read_ge_cest_block, read_ge_cest_cvs
        data_path = self._cest_data_path.strip()
        off_path  = self._cest_ppm_path.strip()
        if not data_path or not off_path:
            self.lbl_cest_info.setText("Browse the combined data + offsets (.txt) first.")
            self.lbl_cest_info.setStyleSheet("font-size: 11px; color: orange;"); return
        try:
            offs = np.loadtxt(off_path).ravel()
            nW   = self._detect_wassr_split(offs)
            wassr_offs, cest_offs = offs[:nW], offs[nW:]
            _at = self.combo_cest_acq.currentText()
            acq = ("custom" if _at.startswith("Custom")
                   else "cube" if _at.startswith("CUBE") else "ssfse")
            larmor = self._peek_larmor(data_path) or self.spin_gecest_mhz.value()
            self.spin_gecest_mhz.setValue(larmor); self.spin_larmor_mhz.setValue(larmor)
            in_hz = self.chk_gecest_hz.isChecked()
            self._log(f"Combined GE: {len(offs)} offsets → {nW} WASSR + "
                      f"{len(cest_offs)} CEST ({acq}).")
            QApplication.processEvents()
            # CEST block (last N)
            rc = read_ge_cest_block(data_path, cest_offs, block="cest", acquisition=acq,
                                    larmor_mhz=larmor, offsets_in_hz=in_hz, log_fn=self._log)
            _finish_cest_load(self, rc["img"], rc["ppm"], source="GE CEST",
                              m0_override=rc["s0"])
            self.lbl_cest_info.setText(
                f"GE CEST loaded: {rc['img'].shape[3]} offsets, {rc['n_slices']} slice(s).")
            self.lbl_cest_info.setStyleSheet("font-size: 11px; color: green;")
            # WASSR block (first nW after prefix) → B0 map
            if nW > 0:
                self._log("Fitting WASSR B0 map (from combined series)…")
                QApplication.processEvents()
                rw = read_ge_cest_block(data_path, wassr_offs, block="wassr", acquisition=acq,
                                        larmor_mhz=larmor, offsets_in_hz=in_hz, log_fn=self._log)
                _finish_wassr_load(self, rw["img"], rw["ppm"], source="GE WASSR",
                                   m0_override=rw["s0"])
            else:
                self._log("  No WASSR block detected in the offsets file (B0 skipped).")
        except Exception as exc:
            self.lbl_cest_info.setText(f"Error: {exc}")
            self.lbl_cest_info.setStyleSheet("font-size: 11px; color: red;")
            self._log(f"ERROR loading combined GE: {exc}")
            import traceback; self._log(traceback.format_exc())

    def _load_wassr_ge(self):
        """GE/Siemens WASSR — extract the WASSR block (first N offsets after the
        idling/S0 prefix) using the acquisition type + WASSR offsets file. Uses
        the CEST data file if no separate WASSR data file was browsed."""
        self._load_ge_block("wassr")

    def _load_ge_block(self, block: str):
        """Shared GE CEST/WASSR block loader (SSFSE or CUBE)."""
        from PyQt6.QtWidgets import QMessageBox
        is_cest = (block == "cest")
        lbl   = self.lbl_cest_info if is_cest else self.lbl_wassr_info
        acq_w = self.combo_cest_acq if is_cest else self.combo_wassr_acq
        data_path = (self._cest_data_path if is_cest else self._wassr_data_path).strip()
        if not is_cest and not data_path:
            data_path = self._cest_data_path.strip()      # reuse CEST data file
        off_path  = (self._cest_ppm_path if is_cest else self._wassr_ppm_path).strip()
        if not data_path:
            lbl.setText("Browse the data (file or folder) first.")
            lbl.setStyleSheet("font-size: 11px; color: orange;"); return
        if not off_path:
            lbl.setText(f"Load the {block.upper()} offsets (.txt) file first.")
            lbl.setStyleSheet("font-size: 11px; color: orange;"); return
        try:
            from my_gui.ge_cest_reader import (read_ge_cest_block, read_ge_cest_cvs)
            offsets = np.loadtxt(off_path).ravel()
            _at = acq_w.currentText()
            acq = ("custom" if _at.startswith("Custom")
                   else "cube" if _at.startswith("CUBE") else "ssfse")
            larmor = self._peek_larmor(data_path) or self.spin_gecest_mhz.value()
            self.spin_gecest_mhz.setValue(larmor)
            self.spin_larmor_mhz.setValue(larmor)
            cvs = read_ge_cest_cvs(data_path)
            if cvs:
                self._log(f"  GE CVs: WASSR B1={cvs.get('wassr_b1_uT')} µT, "
                          f"CEST B1={cvs.get('cest_b1_uT')} µT, "
                          f"dur={cvs.get('duration_ms')} ms")
            self._log(f"Loading GE {block.upper()} ({acq}) — {len(offsets)} offsets…")
            QApplication.processEvents()
            res = read_ge_cest_block(
                data_path, offsets, block=block, acquisition=acq,
                larmor_mhz=larmor, offsets_in_hz=self.chk_gecest_hz.isChecked(),
                log_fn=self._log)
            if is_cest:
                _finish_cest_load(self, res["img"], res["ppm"],
                                  source="GE CEST", m0_override=res["s0"])
                lbl.setText(f"GE CEST loaded: {res['img'].shape[3]} offsets, "
                            f"{res['n_slices']} slice(s).")
                lbl.setStyleSheet("font-size: 11px; color: green;")
            else:
                self._log("Fitting WASSR B0 map…"); QApplication.processEvents()
                _finish_wassr_load(self, res["img"], res["ppm"],
                                   source="GE WASSR", m0_override=res["s0"])
        except Exception as exc:
            lbl.setText(f"Error: {exc}")
            lbl.setStyleSheet("font-size: 11px; color: red;")
            self._log(f"ERROR loading GE {block}: {exc}")
            import traceback; self._log(traceback.format_exc())

    # ── concrete load methods ─────────────────────────────────────────────

    def _load_cest_bruker(self):
        if not self._cest_dir:
            self._log("Please browse to a CEST 2dseq directory first.")
            return
        try:
            from my_gui.bruker_reader import read_2dseq_cest
            pv360 = self.combo_pv_cest.currentText() == "PV360"
            self._log("Loading CEST data…")
            image, M0image, info = read_2dseq_cest(self._cest_dir, pv360=pv360)
            self._z_img_full = image
            self._M0_img = M0image
            self._ppm = info["w_offsetPPM"]
            self._z_img_all = info.get("img_all")
            self._ppm_all   = info.get("ppm_all")
            self._cest_m0_auto_idx = int(info.get("m0_frame_idx", 0))
            try:
                _sp = info.get('satpwr_uT', None)
                if _sp is not None and float(_sp) > 0:
                    self._cest_satpwr_uT = float(_sp)
            except (TypeError, ValueError):
                pass
            sz = info["size"]
            self.lbl_cest_info.setText(
                f"Loaded: {sz[0]}×{sz[1]} px  |  {sz[2]} slice(s)  |  "
                f"{sz[3]} offsets  |  B1: {info.get('satpwr_uT', '?')} µT"
            )
            self.lbl_cest_info.setStyleSheet("font-size: 11px; color: green;")
            self._log(f"CEST data loaded. Shape: {image.shape}, ppm range: "
                      f"[{self._ppm.min():.1f}, {self._ppm.max():.1f}]")
            if self.combo_m0_source.currentText() == "CEST":
                self._populate_m0_frame_combo()
            self.combo_display.setCurrentText("M0 image (unsaturated)")
            self._refresh_display()
            self._update_m0_label()
            self.btn_preview_denoise.setEnabled(self._denoise_method_key() != 'None')
            # New data → force rebuild of the ROI Spectra dialog next open
            self._roi_spectra_dlg_key = ()
        except Exception as exc:
            self.lbl_cest_info.setText(f"Error: {exc}")
            self.lbl_cest_info.setStyleSheet("font-size: 11px; color: red;")
            self._log(f"ERROR loading CEST: {exc}")

    def _load_cest_dicom(self):
        """Load CEST 4-D stack from a DICOM folder (GE/Siemens)."""
        if not self._cest_dicom_dir:
            self._log("Please browse to a DICOM folder first.")
            return
        try:
            import pydicom  # noqa: F401
            self._log("Scanning DICOM folder for CEST images…")
            QApplication.processEvents()
            imgs, ppm_all = _load_dicom_4d(self._cest_dicom_dir, self._cest_ppm_path, self._log)
            _finish_cest_load(self, imgs, ppm_all, source="DICOM")
        except ImportError:
            msg = "pydicom not installed — run:  pip install pydicom"
            self.lbl_cest_info.setText(msg)
            self.lbl_cest_info.setStyleSheet("font-size:11px;color:red;")
            self._log(f"ERROR: {msg}")
        except Exception as exc:
            self.lbl_cest_info.setText(f"Error: {exc}")
            self.lbl_cest_info.setStyleSheet("font-size:11px;color:red;")
            self._log(f"ERROR loading DICOM CEST: {exc}")
            import traceback; self._log(traceback.format_exc())

    def _load_cest_nifti(self):
        """Load CEST 4-D stack from a NIfTI file."""
        if not self._cest_nifti_path:
            self._log("Please browse to a NIfTI file first.")
            return
        try:
            import nibabel as nib
            self._log(f"Loading NIfTI CEST: {self._cest_nifti_path}")
            QApplication.processEvents()
            data = np.array(nib.load(self._cest_nifti_path).get_fdata(), dtype=np.float32)
            if data.ndim == 3:
                data = data[:, :, np.newaxis, :]
            elif data.ndim != 4:
                raise ValueError(f"Expected 4-D NIfTI, got shape {data.shape}")
            ppm_all = _load_ppm_sidecar(self._cest_ppm_path, self._cest_nifti_path, data.shape[3], self._log)
            _finish_cest_load(self, data, ppm_all, source="NIfTI")
        except ImportError:
            msg = "nibabel not installed — run:  pip install nibabel"
            self.lbl_cest_info.setText(msg)
            self.lbl_cest_info.setStyleSheet("font-size:11px;color:red;")
            self._log(f"ERROR: {msg}")
        except Exception as exc:
            self.lbl_cest_info.setText(f"Error: {exc}")
            self.lbl_cest_info.setStyleSheet("font-size:11px;color:red;")
            self._log(f"ERROR loading NIfTI CEST: {exc}")

    def _load_cest_mrd(self):
        """Load CEST stack from an MR Solutions .MRD file (2-D iFFT recon)."""
        if not self._cest_mrd_path:
            self._log("Please browse to a CEST .MRD file first.")
            self.lbl_cest_info.setText("Browse a .MRD file first.")
            self.lbl_cest_info.setStyleSheet("font-size: 11px; color: orange;")
            return
        try:
            from my_gui.mrd_reader import read_mrd_cest
            self._log(f"Loading MR Solutions CEST: {self._cest_mrd_path}")
            QApplication.processEvents()
            imgs, ppm_all, info = read_mrd_cest(
                self._cest_mrd_path, larmor_mhz=self.spin_cest_mrd_mhz.value()
            )
            self._log(
                f"  MRD recon: {imgs.shape[0]}×{imgs.shape[1]} px, "
                f"{imgs.shape[3]} offsets, Larmor {info['larmor_mhz']:.4f} MHz, "
                f"ppm [{ppm_all.min():.2f}, {ppm_all.max():.2f}]"
            )
            _finish_cest_load(self, imgs, ppm_all, source="MR Solutions")
        except Exception as exc:
            self.lbl_cest_info.setText(f"Error: {exc}")
            self.lbl_cest_info.setStyleSheet("font-size:11px;color:red;")
            self._log(f"ERROR loading MR Solutions CEST: {exc}")
            import traceback; self._log(traceback.format_exc())

    def _load_wassr_mrd(self):
        """Load WASSR stack from an MR Solutions .MRD file (2-D iFFT recon)."""
        if not self._wassr_mrd_path:
            self._log("Please browse to a WASSR .MRD file first.")
            self.lbl_wassr_info.setText("Browse a .MRD file first.")
            self.lbl_wassr_info.setStyleSheet("font-size: 11px; color: orange;")
            return
        try:
            from my_gui.mrd_reader import read_mrd_cest
            self._log(f"Loading MR Solutions WASSR: {self._wassr_mrd_path}")
            QApplication.processEvents()
            imgs, ppm_all, info = read_mrd_cest(
                self._wassr_mrd_path, larmor_mhz=self.spin_wassr_mrd_mhz.value()
            )
            self._log(
                f"  MRD recon: {imgs.shape[0]}×{imgs.shape[1]} px, "
                f"{imgs.shape[3]} offsets, Larmor {info['larmor_mhz']:.4f} MHz, "
                f"ppm [{ppm_all.min():.2f}, {ppm_all.max():.2f}]"
            )
            _finish_wassr_load(self, imgs, ppm_all, source="MR Solutions")
        except Exception as exc:
            self.lbl_wassr_info.setText(f"Error: {exc}")
            self.lbl_wassr_info.setStyleSheet("font-size:11px;color:red;")
            self._log(f"ERROR loading MR Solutions WASSR: {exc}")
            import traceback; self._log(traceback.format_exc())

    def _load_wassr_bruker(self):
        if not self._wassr_dir:
            self._log("Please browse to a WASSR directory first.")
            return
        try:
            from my_gui.bruker_reader import read_2dseq_cest
            pv360 = self.combo_pv_wassr.currentText() == "PV360"
            self._log("Loading WASSR data for B0 map…")
            w_img, w_M0, w_info = read_2dseq_cest(self._wassr_dir, pv360=pv360)
            self._wassr_M0_img   = w_M0
            self._wassr_img_full = w_img
            self._wassr_ppm_all  = w_info.get("ppm_all")
            self._wassr_img_all  = w_info.get("img_all")
            self._wassr_m0_auto_idx = int(w_info.get("m0_frame_idx", 0))
            ppm = w_info["w_offsetPPM"]
            self._wassr_ppm = ppm
            z_norm = w_img / (w_M0[:, :, :, np.newaxis] + 1e-9)
            # Compute B0 map by SNR-masked per-voxel Lorentzian fit
            # (matches MATLAB WASSR_load_proc.m; replaces bare argmin, which
            #  left the interior flat and snapped noise voxels to the rails).
            from my_gui.zspec_processing import compute_b0_map_wassr
            _ftol, _max_nfev = {
                0: (1e-3, 400),    # Fast
                1: (1e-5, 800),    # Balanced
                2: (1e-7, 2000),   # Precise
            }.get(self.combo_fit_quality.currentIndex(), (1e-3, 400))
            self._log("Fitting WASSR B0 map (Lorentzian)…")
            QApplication.processEvents()
            self._b0_map_ppm = compute_b0_map_wassr(
                z_norm, ppm,
                m0_img=w_M0,
                snr_thresh=self.spin_snr.value(),
                larmor_mhz=self.spin_larmor_mhz.value(),
                n_workers=self.spin_workers.value(),
                ftol=_ftol,
                max_nfev=_max_nfev,
            )
            if self.combo_m0_source.currentText() == "WASSR":
                self._populate_m0_frame_combo()
            self.lbl_wassr_info.setText("B0 map loaded.")
            self.lbl_wassr_info.setStyleSheet("font-size: 11px; color: green;")
            self._log("WASSR B0 map loaded.")
            self._update_m0_label()
        except Exception as exc:
            self.lbl_wassr_info.setText(f"Error: {exc}")
            self.lbl_wassr_info.setStyleSheet("font-size: 11px; color: red;")
            self._log(f"ERROR loading WASSR: {exc}")

    def _load_wassr_dicom(self):
        if not self._wassr_dicom_dir:
            self._log("Please browse to a WASSR DICOM folder first.")
            return
        try:
            import pydicom  # noqa: F401
            self._log("Loading WASSR DICOM…")
            QApplication.processEvents()
            imgs, ppm_all = _load_dicom_4d(self._wassr_dicom_dir, self._wassr_ppm_path, self._log)
            _finish_wassr_load(self, imgs, ppm_all, source="DICOM")
        except ImportError:
            msg = "pydicom not installed — run:  pip install pydicom"
            self.lbl_wassr_info.setText(msg)
            self.lbl_wassr_info.setStyleSheet("font-size:11px;color:red;")
            self._log(f"ERROR: {msg}")
        except Exception as exc:
            self.lbl_wassr_info.setText(f"Error: {exc}")
            self.lbl_wassr_info.setStyleSheet("font-size:11px;color:red;")
            self._log(f"ERROR loading WASSR DICOM: {exc}")

    def _load_wassr_nifti(self):
        if not self._wassr_nifti_path:
            self._log("Please browse to a WASSR NIfTI file first.")
            return
        try:
            import nibabel as nib
            self._log(f"Loading NIfTI WASSR: {self._wassr_nifti_path}")
            QApplication.processEvents()
            data = np.array(nib.load(self._wassr_nifti_path).get_fdata(), dtype=np.float32)
            if data.ndim == 3:
                data = data[:, :, np.newaxis, :]
            elif data.ndim != 4:
                raise ValueError(f"Expected 4-D NIfTI, got shape {data.shape}")
            ppm_all = _load_ppm_sidecar(self._wassr_ppm_path, self._wassr_nifti_path, data.shape[3], self._log)
            _finish_wassr_load(self, data, ppm_all, source="NIfTI")
        except ImportError:
            msg = "nibabel not installed — run:  pip install nibabel"
            self.lbl_wassr_info.setText(msg)
            self.lbl_wassr_info.setStyleSheet("font-size:11px;color:red;")
            self._log(f"ERROR: {msg}")
        except Exception as exc:
            self.lbl_wassr_info.setText(f"Error: {exc}")
            self.lbl_wassr_info.setStyleSheet("font-size:11px;color:red;")
            self._log(f"ERROR loading NIfTI WASSR: {exc}")

    # ─────────────────────────────────────────────────────────────────────
    # Denoising preview
    # ─────────────────────────────────────────────────────────────────────

    def _preview_denoise(self):
        """Show before/after denoising for slice 0 using a fast cropped region."""
        if self._z_img_full is None:
            return
        method = self._denoise_method_key()
        if method == 'None':
            return

        from my_gui.zspec_processing import denoise_zimg
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from PyQt6.QtWidgets import QDialog, QVBoxLayout

        # ── For preview: crop image to a small centre region to keep it fast ──
        # BM3D / NLM are O(n² · W²) — large images take minutes at full res.
        # We crop to PREVIEW_SZ × PREVIEW_SZ pixels and for BM3D also reduce
        # the search/block window sizes.
        PREVIEW_SZ = 64   # px per side; adjust if phantom is small

        # Use slice 0 only
        z_sl = (self._z_img_full[:, :, 0, :]
                if self._z_img_full.ndim == 4
                else self._z_img_full)          # (H, W, n_off)

        H, W, n_off = z_sl.shape
        # Centre crop
        cy, cx = H // 2, W // 2
        hy, hx = min(PREVIEW_SZ // 2, H // 2), min(PREVIEW_SZ // 2, W // 2)
        r0, r1 = cy - hy, cy + hy
        c0, c1 = cx - hx, cx + hx
        z_crop = z_sl[r0:r1, c0:c1, :]          # small patch, all offsets
        crop_h, crop_w = z_crop.shape[:2]
        cropped = (crop_h < H or crop_w < W)
        crop_note = (f"  |  centre crop {crop_h}×{crop_w} px of {H}×{W}"
                     if cropped else "")

        self._log(
            f"Running {method} denoise preview  "
            f"({crop_h}×{crop_w} px, {n_off} offsets)…"
        )
        QApplication.processEvents()

        try:
            _m = method.lower()
            orig_frame = z_crop[:, :, 0].astype(float)
            if _m == 'pca':
                # PCA is a spectral method (needs every offset) and is fast.
                z_dn_crop = denoise_zimg(z_crop, method='pca',
                                         pca_criteria='malinowski')
                dn_frame = z_dn_crop[:, :, 0].astype(float)
            else:
                # BM3D / NLM denoise each offset independently — for the preview we
                # only need the displayed frame (offset 0), which is far faster.
                _dn0 = denoise_zimg(
                    z_crop[:, :, 0:1],
                    method=_m,
                    bm3d_strength=self.spin_bm3d_strength.value(),
                    nlm_big_window=min(self.spin_nlm_big.value(), crop_h - 1 | 1),
                    nlm_small_window=self.spin_nlm_small.value(),
                )
                dn_frame = _dn0[:, :, 0].astype(float)

            diff_frame = orig_frame - dn_frame

            vmin  = float(np.nanmin(orig_frame))
            vmax  = float(np.nanmax(orig_frame))
            d_abs = float(np.nanmax(np.abs(diff_frame))) or 1.0

            # Noise metric — robust Donoho-MAD noise estimate before/after.  (The
            # plain spatial std over bright voxels is dominated by real signal
            # variation across the phantom, not noise, so it barely changes when you
            # denoise — an unreliable indicator.)  MAD on the Haar diagonal detail
            # uses the median, so it ignores the sparse edges/structure.
            _bright = orig_frame > np.percentile(orig_frame[orig_frame > 0], 60)
            if _bright.sum() == 0:
                _bright = np.ones_like(orig_frame, dtype=bool)

            def _mad_noise(fr):
                hh = 0.5 * (fr[:-1, :-1] - fr[:-1, 1:] - fr[1:, :-1] + fr[1:, 1:])
                m = float(np.median(np.abs(hh)))
                return m / 0.6745 if m > 0 else 0.0

            orig_sd = _mad_noise(orig_frame)          # noise σ before
            dn_sd   = _mad_noise(dn_frame)            # noise σ after
            noise_removed = float(diff_frame[_bright].std())

            # ── Figure layout: 2 rows × 3 cols ───────────────────────────
            # Row 0: Original | Denoised | Difference
            # Row 1: Z-spectrum overlay (spans all 3 cols)
            fig = plt.figure(figsize=(14, 4.6))
            gs  = fig.add_gridspec(1, 3, wspace=0.3)

            fig.suptitle(
                f"Denoising  —  {method}{crop_note}",
                fontsize=11, fontweight='bold',
            )

            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(orig_frame, cmap='gray', vmin=vmin, vmax=vmax)
            ax0.set_title("Original  (offset 0)")
            ax0.axis('off')

            ax1 = fig.add_subplot(gs[0, 1])
            ax1.imshow(dn_frame, cmap='gray', vmin=vmin, vmax=vmax)
            ax1.set_title(f"Denoised  ({method})")
            ax1.axis('off')

            ax2 = fig.add_subplot(gs[0, 2])
            im = ax2.imshow(diff_frame, cmap='bwr', vmin=-d_abs, vmax=d_abs)
            ax2.set_title("Difference  (original − denoised)")
            ax2.axis('off')
            fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

            if cropped:
                fig.text(
                    0.5, 0.005,
                    f"Preview uses centre crop for speed — "
                    f"full {H}×{W} image denoised on Run Analysis.",
                    ha='center', fontsize=8, color='gray',
                )

            dlg = QDialog(self)
            dlg.setWindowTitle(f"Denoising — {method}")
            dlg.resize(1060, 400)
            dlg_lay = QVBoxLayout(dlg)
            dlg_lay.setContentsMargins(4, 4, 4, 4)
            canvas = FigureCanvas(fig)
            dlg_lay.addWidget(canvas)
            dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            dlg.show()
            self._log(
                f"Preview ready.  Noise σ ≈ {orig_sd:.1f} → {dn_sd:.1f}  "
                f"(removed ≈ {noise_removed:.1f})"
            )

        except Exception as ex:
            self._log(f"Denoise preview failed: {ex}")

    # ─────────────────────────────────────────────────────────────────────
    # Analysis
    # ─────────────────────────────────────────────────────────────────────

    def _prep_quick_z(self):
        """Shared preprocessing for the quick MTRasym / MTRRex maps.

        Returns (z_norm (Y,X,sl,off), snr_mask (Y,X,sl)) — M0-normalised and
        B0-corrected (on SNR-masked voxels, if a B0 map is loaded) — or None.
        """
        if self._z_img_full is None or self._ppm is None:
            self._log("Load CEST data first.")
            return None
        M0 = self._get_m0()
        if M0 is None:
            self._log("ERROR: No M0 image available. Load CEST/WASSR first.")
            return None
        from my_gui.zspec_processing import b0_correction
        z_img = self._z_img_full
        rows, cols, slices, n_off = z_img.shape

        # Robust SNR mask (same logic as the full analysis)
        snr_thresh = self.spin_snr.value()
        m0_pos = M0[M0 > 0]
        if m0_pos.size:
            thr = np.percentile(m0_pos, 10)
            low = m0_pos[m0_pos <= thr]
            noise = float(np.mean(low)) if low.size else float(thr)
        else:
            noise = 0.0
        if not np.isfinite(noise):
            noise = 0.0
        snr_mask = M0 > snr_thresh * noise
        if not snr_mask.any():
            snr_mask = M0 > 0

        # Align the M0 / mask slice count with the CEST volume (a single-slice M0
        # source vs a multi-slice de-tiled CEST would otherwise mis-broadcast).
        if M0.shape[2] != slices:
            if M0.shape[2] == 1:
                M0       = np.repeat(M0, slices, axis=2)
                snr_mask = np.repeat(snr_mask, slices, axis=2)
            else:
                _c = min(M0.shape[2], slices)
                z_img = z_img[:, :, :_c, :]
                M0 = M0[:, :, :_c]; snr_mask = snr_mask[:, :, :_c]
                rows, cols, slices, n_off = z_img.shape

        z_norm = z_img / (M0[:, :, :, np.newaxis] + 1e-9)

        b0_map = getattr(self, "_b0_map_ppm", None)
        if b0_map is not None:
            self._log("  Applying B0 correction…")
            QApplication.processEvents()
            mask_idx = np.where(snr_mask.ravel())[0]
            z_flat = z_norm.reshape(-1, n_off)
            z_flat[mask_idx] = b0_correction(b0_map.ravel()[mask_idx], self._ppm,
                                             z_flat[mask_idx])
            z_norm = z_flat.reshape(rows, cols, slices, n_off)
        return z_norm, snr_mask

    def _quick_mtr_asym(self):
        """Quick MTR-asymmetry map (no pool fitting): Z(−ppm) − Z(+ppm)."""
        try:
            sel_ppm = self.spin_quick_mtr_ppm.value()
            self._log(f"MTR-Asymmetry (±{sel_ppm:.2f} ppm)…")
            QApplication.processEvents()
            prep = self._prep_quick_z()
            if prep is None:
                return
            z_norm, snr_mask = prep
            from my_gui.zspec_processing import calc_mtr_map
            mtr_vol, sel_true = calc_mtr_map(z_norm, self._ppm, sel_ppm=sel_ppm)
            mtr_vol = np.where(snr_mask, mtr_vol, np.nan)
            mtr_vol = apply_analysis_mask(mtr_vol)   # global brain/phantom mask
            if not isinstance(self._results, dict):
                self._results = {}
            self._results["mtr_map"]      = mtr_vol
            self._results["mtr_ppm_used"] = sel_true
            self._results["ppm"]          = self._ppm
            self._rebuild_display_combo()
            self.combo_display.setCurrentText("MTR asymmetry map")
            self._refresh_display()
            self.btn_save_session.setEnabled(True)
            self._log(f"Quick MTR-asymmetry done (±{sel_true:.2f} ppm).")
        except Exception as exc:
            import traceback
            self._log(f"ERROR (quick MTRasym): {exc}\n{traceback.format_exc()}")

    def _quick_mtrrex(self):
        """Quick MTR_Rex map (no pool fitting): 1/Z(+ppm) − 1/Z(−ppm)."""
        try:
            sel_ppm = self.spin_quick_rex_ppm.value()
            self._log(f"MTR_Rex (±{sel_ppm:.2f} ppm)…")
            QApplication.processEvents()
            prep = self._prep_quick_z()
            if prep is None:
                return
            z_norm, snr_mask = prep
            from my_gui.zspec_processing import calc_mtrrex_map
            rex_vol, sel_true = calc_mtrrex_map(z_norm, self._ppm, sel_ppm=sel_ppm)
            rex_vol = np.where(snr_mask, rex_vol, np.nan)
            rex_vol = apply_analysis_mask(rex_vol)   # global brain/phantom mask
            if not isinstance(self._results, dict):
                self._results = {}
            self._results["mtrrex_map"]      = rex_vol
            self._results["mtrrex_ppm_used"] = sel_true
            self._results["ppm"]             = self._ppm
            self._rebuild_display_combo()
            self.combo_display.setCurrentText("MTR_Rex map")
            self._refresh_display()
            self.btn_save_session.setEnabled(True)
            self._log(f"Quick MTR_Rex done (±{sel_true:.2f} ppm).")
        except Exception as exc:
            import traceback
            self._log(f"ERROR (quick MTRRex): {exc}\n{traceback.format_exc()}")

    def _run_analysis(self):
        if self._z_img_full is None or self._ppm is None:
            self._log("Load CEST data first.")
            return

        # Use pools selected by the user in "Select pools to fit" dialog,
        # restricted to pools that have defined bounds in the PV/Gaussian models.
        # Falls back to POOL_NAMES if none of the selected pools have known bounds.
        from my_gui.zspec_processing import _PSEUDOVOIGT_BOUNDS as _PVB
        _known_pools = set(_PVB.keys())
        selected_pools = [p for p in self._global_pools if p in _known_pools]
        if not selected_pools:
            selected_pools = list(self.POOL_NAMES)

        # ── SNR thresholding — use user-selected M0 source ────────────────
        M0 = self._get_m0()
        if M0 is None:
            self._log("ERROR: No M0 image available. Load CEST or WASSR data first.")
            return
        snr_thresh = self.spin_snr.value()
        # Noise proxy = mean of the lowest-decile signal. Estimate it from the
        # POSITIVE voxels only — a large exactly-zero background (e.g. GE FOV
        # outside the phantom) otherwise makes percentile(M0,10)=0, the
        # low-signal set empty, and noise_level NaN → an all-False mask.
        m0_pos = M0[M0 > 0]
        if m0_pos.size:
            thr  = np.percentile(m0_pos, 10)
            low  = m0_pos[m0_pos <= thr]
            noise_level = float(np.mean(low)) if low.size else float(thr)
        else:
            noise_level = 0.0
        if not np.isfinite(noise_level):
            noise_level = 0.0
        snr_mask = (M0 > snr_thresh * noise_level)             # bool (Y, X, slices)
        if not snr_mask.any():                                 # never fit nothing
            snr_mask = M0 > 0

        # ── Optional denoising (applied to raw signal before normalization) ─
        _denoise_method = self._denoise_method_key()
        if _denoise_method != 'None':
            from my_gui.zspec_processing import denoise_zimg
            self._log(
                f"Denoising Z-spectrum image with {_denoise_method}…  "
                f"(this may take a moment for BM3D / NLM)"
            )
            QApplication.processEvents()
            try:
                z_img_raw = denoise_zimg(
                    self._z_img_full,
                    method=_denoise_method.lower(),
                    pca_criteria='malinowski',
                    bm3d_strength=self.spin_bm3d_strength.value(),
                    nlm_big_window=self.spin_nlm_big.value(),
                    nlm_small_window=self.spin_nlm_small.value(),
                )
                self._log(f"Denoising complete ({_denoise_method}).")
            except Exception as _de:
                self._log(f"WARNING: Denoising failed — {_de}\n  Proceeding with original data.")
                z_img_raw = self._z_img_full
        else:
            z_img_raw = self._z_img_full

        orig_rows, orig_cols, slices, n_off = z_img_raw.shape
        self._img_shape = (orig_rows, orig_cols, slices)

        # ── Reconcile the M0 / SNR-mask slice count with the CEST volume ──────
        # A separately-loaded M0 source (e.g. a single-slice WASSR) may have a
        # different slice count than a de-tiled multi-slice CEST volume. Align
        # them so the downstream normalise/reshape/boolean-index stays consistent
        # instead of raising a shape/broadcast error.
        if M0.shape[2] != slices:
            if M0.shape[2] == 1:
                self._log(f"  Note: M0 source has 1 slice but CEST has {slices}; "
                          f"broadcasting the single M0 across all slices.")
                M0       = np.repeat(M0, slices, axis=2)
                snr_mask = np.repeat(snr_mask, slices, axis=2)
            else:
                _c = min(M0.shape[2], slices)
                self._log(f"  Warning: M0 has {M0.shape[2]} slice(s) but CEST has "
                          f"{slices}; using the first {_c} slice(s) of each.")
                z_img_raw = z_img_raw[:, :, :_c, :]
                M0        = M0[:, :, :_c]
                snr_mask  = snr_mask[:, :, :_c]
                slices    = _c
                self._img_shape = (orig_rows, orig_cols, slices)

        # ── Speed: high-res CEST recon is usually interpolated/zero-filled from
        # a much smaller acquisition matrix (e.g. GE 512×512 ← 128×128). Fitting
        # every interpolated voxel is wasteful → fit on a reduced grid, then
        # upsample the maps afterwards (keeps ROI overlays at full resolution). ─
        self._fit_orig_rc = None
        _FIT_CAP = 192
        if max(orig_rows, orig_cols) > _FIT_CAP:
            from scipy.ndimage import zoom as _zoom
            f = _FIT_CAP / float(max(orig_rows, orig_cols))
            z_img_raw = _zoom(z_img_raw, (f, f, 1, 1), order=1)
            M0        = _zoom(np.asarray(M0, float), (f, f, 1), order=1)
            snr_mask  = _zoom(snr_mask.astype(float), (f, f, 1), order=1) > 0.5
            self._fit_orig_rc = (orig_rows, orig_cols)
            self._log(f"Fit downsample: {orig_rows}×{orig_cols} → "
                      f"{z_img_raw.shape[0]}×{z_img_raw.shape[1]} "
                      f"(recon interpolated from a smaller matrix); maps upsampled back.")
        rows, cols = z_img_raw.shape[0], z_img_raw.shape[1]

        # Normalise to get z-spectra
        z_norm = z_img_raw / (M0[:, :, :, np.newaxis] + 1e-9)  # (Y,X,sl,off)
        # Apply SNR mask (zero-out below-threshold voxels)
        z_norm_masked = z_norm * snr_mask[:, :, :, np.newaxis]

        # Flatten to (n_vox, n_off) and select SNR-passing voxels
        z_flat = z_norm_masked.reshape(-1, n_off)
        mask_flat = snr_mask.ravel()
        z_sel = z_flat[mask_flat]
        n_vox_sel = z_sel.shape[0]
        self._log(f"SNR mask: {n_vox_sel} / {z_flat.shape[0]} voxels selected.")

        # B0 map if loaded (downsample to match the fit grid if needed)
        b0_map = getattr(self, "_b0_map_ppm", None)
        if self._fit_orig_rc is not None and b0_map is not None:
            from scipy.ndimage import zoom as _zoom
            _b0 = np.asarray(b0_map, float)
            b0_map = _zoom(_b0, (rows / _b0.shape[0], cols / _b0.shape[1], 1), order=1)

        # ── Fit quality preset ────────────────────────────────────────────
        # Tolerance rationale (matches SO insight + imaging noise floor):
        #   MRI signal noise σ ≈ 0.5–2 % → cost floor ≈ σ² ≈ 1e-4.
        #   ftol tighter than 1e-3 chases pure noise — extra iterations, zero gain.
        #   CEST-master (MATLAB) uses TolFun=1e-6 (default) but compiled C is 10–20×
        #   faster per iteration; Python needs looser tolerances to match wall-clock.
        #   gtol=np.inf disables gradient-norm stopping — for bounded TRF it fires
        #   spuriously on constrained parameters; ftol+xtol are sufficient.
        _quality_presets = {
            0: (1e-3,  400),    # Fast     — noise floor ~1e-4; 1e-3 is safe
            1: (1e-5,  800),    # Balanced — matches CEST-master TolFun=1e-6 territory
            2: (1e-7,  2000),   # Precise  — for publication-quality ROI fits
        }
        _ftol, _max_nfev = _quality_presets.get(
            self.combo_fit_quality.currentIndex(), (1e-3, 400))
        _fit_lor = False
        # 3 phases: PV + Gaussian + MPLF — each now uses the same max_nfev budget
        # (~0.14 s/vox each at Fast preset with 1 worker)
        _sec_per_vox = (_max_nfev / 600) * 0.14 * 3
        est_sec = n_vox_sel * _sec_per_vox / max(1, self.spin_workers.value())
        self._log(
            f"Fit quality: {self.combo_fit_quality.currentText().strip()}\n"
            f"  workers={self.spin_workers.value()}   methods: PV + Gaussian + MPLF\n"
            f"  Estimated time: ~{est_sec:.0f} s  ({n_vox_sel} voxels)"
        )

        # ── Launch worker ─────────────────────────────────────────────────
        from my_gui.zspec_worker import ZSpecWorker

        # Three fit phases: PV + Gaussian + MPLF
        _n_phases  = 3
        _total_bar = n_vox_sel * _n_phases

        self._worker = ZSpecWorker(
            z_img=z_sel,
            ppm=self._ppm,
            img_shape=(rows, cols, slices),
            snr_mask=mask_flat,
            pools=selected_pools,
            n_workers=self.spin_workers.value(),
            b0_map_ppm=b0_map,
            sel_mtr_ppm=self.spin_mtr_ppm.value(),
            ftol=_ftol,
            max_nfev=_max_nfev,
            fit_lorentzian=_fit_lor,
            fit_gaussian=True,
            fit_mplf=True,
        )
        self._worker.log.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)

        self._worker.cancelled.connect(self._on_cancelled)

        self._total_vox = _total_bar   # full bar = 3 × n_vox_sel
        self._n_vox_sel = n_vox_sel    # single-phase voxel count (for % display)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, _total_bar)
        self.progress_bar.setFormat(f"0 / {n_vox_sel} voxels (0%)")
        self.progress_bar.setVisible(True)
        self.lbl_run_status.setText("Running…")
        self._worker.start()

    def _on_progress(self, n_done: int):
        self.progress_bar.setValue(n_done)
        if self._total_vox:
            n_vox_sel = getattr(self, '_n_vox_sel', self._total_vox)
            pct_total = int(n_done * 100 / self._total_vox)
            # Phase label: 0→n_vox = PV, n_vox→2n_vox = Gaussian, 2n_vox→3n_vox = MPLF
            phase_num  = min(n_done // max(n_vox_sel, 1), 2)
            phase_name = ["PV", "Gaussian", "MPLF"][phase_num]
            phase_done = n_done - phase_num * n_vox_sel
            phase_pct  = int(phase_done * 100 / max(n_vox_sel, 1))
            self.progress_bar.setFormat(
                f"{phase_name}: {phase_pct}%  (overall {pct_total}%)"
            )
            self.lbl_run_status.setText(f"Running… {phase_name} {phase_pct}%")

    @staticmethod
    def _upsample_results(results: dict, orig_rc: tuple) -> dict:
        """Upsample all spatial maps in a results dict back to (R, C)."""
        from scipy.ndimage import zoom as _zoom
        R, C = orig_rc
        out = {}
        for k, v in results.items():
            if (isinstance(v, np.ndarray) and v.ndim >= 3
                    and (v.shape[0] != R or v.shape[1] != C)):
                order = 0 if (v.dtype == bool or np.issubdtype(v.dtype, np.integer)) else 1
                factors = [R / v.shape[0], C / v.shape[1]] + [1] * (v.ndim - 2)
                out[k] = _zoom(v.astype(float), factors, order=order).astype(v.dtype)
            else:
                out[k] = v
        return out

    def _on_done(self, results: dict):
        # If the fit ran on a downsampled grid, upsample maps to full resolution
        if getattr(self, "_fit_orig_rc", None) is not None:
            results = self._upsample_results(results, self._fit_orig_rc)
            self._fit_orig_rc = None
        # Restrict every fitted map to the global analysis mask (brain / phantom
        # outline), if one is active — composes with the SNR mask already applied.
        if isinstance(results, dict):
            for _k, _v in list(results.items()):
                if isinstance(_v, np.ndarray) and _v.ndim >= 2:
                    results[_k] = apply_analysis_mask(_v)
        self._results = results
        self.progress_bar.setFormat("PV + Gaussian + MPLF complete (100%)")
        self.progress_bar.setValue(self._total_vox)
        self._set_running(False)
        self.lbl_run_status.setText("Analysis complete.")
        self.btn_save_session.setEnabled(True)
        self._log("Done. Select a map in the display dropdown.")
        # Mark voxelwise as the most recent source for Gaussian and MPLF
        self._roi_stats_last_source['Gaussian'] = 'voxelwise'
        self._roi_stats_last_source['MPLF']     = 'voxelwise'
        # Pre-populate ROI Spectra cache from voxelwise curves (no re-fitting needed)
        self._auto_populate_roi_spectra_from_voxelwise(results)
        self._rebuild_display_combo()   # refresh dropdown with actual fitted maps
        self._refresh_display()

    # ─────────────────────────────────────────────────────────────────────
    # Session save / load
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _mat_key(k: str) -> str:
        """Sanitise a results-dict key so it is a valid MATLAB variable name."""
        return k.replace('.', 'pt').replace('-', '_')

    def _save_session(self):
        """Save self._results to a .mat file for later reload."""
        if not self._results:
            QMessageBox.warning(self, "Nothing to save", "Run the analysis first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save CEST Session", "cest_session.mat",
            "MATLAB files (*.mat);;All files (*)"
        )
        if not path:
            return

        try:
            import scipy.io as sio

            save_dict: dict = {}
            orig_keys:      list[str] = []
            sanitized_keys: list[str] = []

            for k, v in self._results.items():
                sk = self._mat_key(k)
                orig_keys.append(k)
                sanitized_keys.append(sk)

                if isinstance(v, np.ndarray):
                    save_dict[sk] = v
                elif isinstance(v, list):
                    # Pool name lists → object array
                    save_dict[sk] = np.array(v, dtype=object)
                elif isinstance(v, (int, float, str)):
                    save_dict[sk] = v
                else:
                    try:
                        save_dict[sk] = np.asarray(v)
                    except Exception:
                        pass  # skip un-serialisable entries

            # Store key mapping so load can reconstruct original names
            save_dict['_orig_keys']      = np.array(orig_keys,      dtype=object)
            save_dict['_sanitized_keys'] = np.array(sanitized_keys, dtype=object)

            sio.savemat(path, save_dict, do_compression=True)
            self._log(f"Session saved → {path}")
            self.lbl_run_status.setText(f"Saved: {path.split('/')[-1]}")

        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            self._log(f"Save error: {exc}")

    def _load_session(self):
        """Load a .mat session file saved by _save_session and restore results."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load CEST Session", "",
            "MATLAB files (*.mat);;All files (*)"
        )
        if not path:
            return

        try:
            import scipy.io as sio

            loaded = sio.loadmat(path, squeeze_me=False)

            # Reconstruct original key mapping
            if '_orig_keys' in loaded and '_sanitized_keys' in loaded:
                orig_keys = [str(s) for s in loaded['_orig_keys'].ravel()]
                san_keys  = [str(s) for s in loaded['_sanitized_keys'].ravel()]
                key_map   = {sk: ok for sk, ok in zip(san_keys, orig_keys)}
            else:
                # Fallback: use sanitized keys directly
                key_map = {k: k for k in loaded if not k.startswith('_')}

            results: dict = {}
            for sk, ok in key_map.items():
                if sk not in loaded:
                    continue
                val = loaded[sk]
                # scipy.io.loadmat wraps scalars and strings in arrays
                if isinstance(val, np.ndarray):
                    # Pool name lists are object arrays of strings
                    if val.dtype == object and val.ndim <= 2:
                        flat = val.ravel()
                        if all(isinstance(x, str) for x in flat):
                            val = list(flat)
                        elif len(flat) == 1:
                            val = flat[0]
                    # Scalar numerics
                    elif val.size == 1:
                        val = float(val.ravel()[0])
                results[ok] = val

            # Restore global pool list
            pools = results.get('pools', [])
            if isinstance(pools, np.ndarray):
                pools = list(pools.ravel().astype(str))
            if pools:
                self._global_pools = list(pools)

            self._results = results
            self.btn_save_session.setEnabled(True)
            self._rebuild_display_combo()
            self._refresh_display()

            n_maps = sum(1 for k in results if 'ampl' in k)
            self._log(f"Session loaded ← {path}  ({n_maps} amplitude maps restored)")
            self.lbl_run_status.setText(f"Loaded: {path.split('/')[-1]}")

        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            self._log(f"Load error: {exc}")

    def _rebuild_display_combo(self):
        """
        Rebuild the Display dropdown after analysis so it shows exactly the maps
        that were computed (PV always; Lorentzian only if dual-fit was run).
        Also updates pool entries for whatever pool set was actually fitted.
        """
        current = self.combo_display.currentText()
        self.combo_display.blockSignals(True)

        # ── Static entries always present ────────────────────────────────
        static = [
            "M0 image (unsaturated)",
            "CEST Images",
            "WASSR Images",
            "B0 map (ppm)",
            "B0 map (Hz)",
            "AREX map",
            "MTR asymmetry map",
            "MTR_Rex map",
        ]
        self.combo_display.clear()
        self.combo_display.addItems(static)

        # ── Dynamic entries based on what was computed (only selected pools) ─
        pools = self._results.get("pools", self._global_pools)
        for prefix, label in (
            ("pv_",    "PV: "),
            ("gauss_", "Gaussian: "),
            ("lor_",   "Lor: "),
        ):
            for pool in pools:
                if f"{prefix}ampl_{pool}" in self._results:
                    self.combo_display.addItem(f"{label}{pool}")

        # MPLF has its own pool list (may include extra catalog pools)
        mplf_pools = self._results.get("mplf_pools", [])
        for pool in mplf_pools:
            if f"mplf_ampl_{pool}" in self._results:
                self.combo_display.addItem(f"MPLF: {pool}")

        # ── ROI spectral fit maps — only if no voxelwise version exists ──────
        # (avoids duplicates when voxelwise Gaussian/MPLF was already run)
        _voxelwise_methods = set()
        _pv_pools = self._results.get("pools", [])
        if any(f"gauss_ampl_{p}" in self._results for p in _pv_pools):
            _voxelwise_methods.add("Gaussian")
        if self._results.get("mplf_pools"):
            _voxelwise_methods.add("MPLF")

        for fit_method, pool_maps in getattr(self, '_roi_spectra_maps', {}).items():
            if fit_method in _voxelwise_methods:
                continue   # voxelwise map already in dropdown — skip ROI-only version
            for pool_name in sorted(pool_maps.keys()):
                self.combo_display.addItem(f"{fit_method}: {pool_name}")

        # ── Restore previous selection if still valid ─────────────────────
        self.combo_display.blockSignals(False)
        idx = self.combo_display.findText(current)
        if idx >= 0:
            self.combo_display.setCurrentIndex(idx)
        else:
            # Default to first PV amplitude map after analysis
            pv_idx = self.combo_display.findText("PV: ", Qt.MatchFlag.MatchStartsWith)
            if pv_idx >= 0:
                self.combo_display.setCurrentIndex(pv_idx)

    # ─────────────────────────────────────────────────────────────────────
    # ROI spectral-fit → 2-D map conversion
    # ─────────────────────────────────────────────────────────────────────

    def _update_roi_spectral_maps(self, method: str, fit_results: dict):
        """
        Convert on-demand ROI spectral fit results into 2-D scalar maps
        (one per pool) and store them in ``self._roi_spectra_maps``.
        Each pixel inside an ROI mask gets the scalar summary value for
        that ROI; pixels outside all ROIs are NaN.

        Then rebuilds the display dropdown so the new maps appear immediately.
        """
        rois = getattr(self, '_last_rois', [])
        img_ref = getattr(self, '_z_img_full', None)
        if img_ref is not None:
            _H, _W = img_ref.shape[0], img_ref.shape[1]
        elif rois:
            _H, _W = rois[0].mask.shape[0], rois[0].mask.shape[1]
        else:
            return  # No spatial reference — cannot build a map

        def _scalar_map(roi_vals: dict):
            """Fill ROI masks with scalar values; return 2-D float32 array."""
            arr = np.full((_H, _W), np.nan, dtype=np.float32)
            for roi in rois:
                val = roi_vals.get(roi.name)
                if val is None:
                    continue
                msk = roi.mask
                if msk.shape[:2] != (_H, _W):
                    try:
                        from scipy.ndimage import zoom as _zoom
                        msk = _zoom(msk.astype(float),
                                    (_H / max(msk.shape[0], 1),
                                     _W / max(msk.shape[1], 1)),
                                    order=1) > 0.5
                    except Exception:
                        continue
                arr[msk] = float(val)
            return arr if not np.all(np.isnan(arr)) else None

        pool_maps: dict = {}

        if method in ('DROF', 'MPLF'):
            all_pools: set = set()
            for res in fit_results.values():
                if res:
                    all_pools.update(res.get('pools', {}).keys())
            for pn in sorted(all_pools):
                if pn == 'water':
                    continue
                roi_vals = {}
                for roi in rois:
                    res = fit_results.get(roi.name)
                    if res:
                        pc = res.get('pools', {}).get(pn)
                        if pc is not None:
                            roi_vals[roi.name] = float(np.nanmax(pc))
                m = _scalar_map(roi_vals)
                if m is not None:
                    pool_maps[pn] = m

        elif method == 'Gaussian':
            all_pools = set()
            for res in fit_results.values():
                if res:
                    all_pools.update(res.get('pool_curves', {}).keys())
            for pn in sorted(all_pools):
                if pn == 'water':
                    continue
                roi_vals = {}
                for roi in rois:
                    res = fit_results.get(roi.name)
                    if res:
                        pc = res.get('pool_curves', {}).get(pn)
                        if pc is not None:
                            roi_vals[roi.name] = float(np.nanmax(pc))
                m = _scalar_map(roi_vals)
                if m is not None:
                    pool_maps[pn] = m

        elif method == 'PLOF':
            all_pools = set()
            for res in fit_results.values():
                if res:
                    all_pools.update(res.get('pools', {}).keys())
            for pn in sorted(all_pools):
                roi_vals = {}
                for roi in rois:
                    res = fit_results.get(roi.name)
                    if res:
                        pp = res.get('pools', {}).get(pn)
                        if pp is not None:
                            roi_vals[roi.name] = float(pp.get('delta_z', 0.0))
                m = _scalar_map(roi_vals)
                if m is not None:
                    pool_maps[pn] = m

        if pool_maps:
            # Use a display-friendly key, e.g. "PLOF ΔZ", "DROF", "MPLF", "Gaussian"
            display_key = "PLOF ΔZ" if method == 'PLOF' else method
            self._roi_spectra_maps[display_key] = pool_maps
            self._rebuild_display_combo()
            # Auto-select the first new map from this method
            first_entry = f"{display_key}: {next(iter(pool_maps))}"
            idx = self.combo_display.findText(first_entry)
            if idx >= 0:
                self.combo_display.setCurrentIndex(idx)

    def _auto_populate_roi_spectra_from_voxelwise(self, results: dict):
        """
        After voxelwise Gaussian/MPLF analysis, pre-populate _roi_spectra_fit_cache
        by averaging the per-voxel fitted curves over each ROI mask.

        This means the ROI Spectra dialog can display curves immediately without
        needing a separate "Run Fit" click for Gaussian/MPLF.
        """
        rois = getattr(self, '_last_rois', [])
        if not rois:
            return

        if not hasattr(self, '_roi_spectra_fit_cache'):
            self._roi_spectra_fit_cache = {}

        pools = results.get("pools", [])

        # ── Pseudo-Voigt voxelwise → ROI cache ───────────────────────────
        pv_pools = [p for p in pools if f"pv_peak_{p}" in results]
        if pv_pools:
            pv_cache: dict = {}
            for roi in rois:
                msk = roi.mask
                pool_curves: dict = {}
                for p in pv_pools:
                    vol = results[f"pv_peak_{p}"]        # (H, W, sl, N_off)
                    sl  = vol[:, :, self._cur_slice(vol.shape[2]), :]
                    if msk.shape[:2] == sl.shape[:2]:
                        vox = sl[msk]
                        if len(vox) > 0:
                            pool_curves[p] = np.nanmean(vox, axis=0)
                if pool_curves:
                    sumcurve = np.sum(list(pool_curves.values()), axis=0)
                    pv_cache[roi.name] = {
                        'fit':         1.0 - sumcurve,
                        'pool_curves': pool_curves,
                    }
            if pv_cache:
                self._roi_spectra_fit_cache['Pseudo-Voigt'] = pv_cache

        # ── Gaussian voxelwise → ROI cache ───────────────────────────────
        gauss_pools = [p for p in pools if f"gauss_peak_{p}" in results]
        if gauss_pools:
            gauss_cache: dict = {}
            for roi in rois:
                msk = roi.mask           # (H, W) bool
                pool_curves: dict = {}
                for p in gauss_pools:
                    vol = results[f"gauss_peak_{p}"]  # (H, W, sl, N_off)
                    sl  = vol[:, :, self._cur_slice(vol.shape[2]), :]  # (H, W, N_off)
                    if msk.shape[:2] == sl.shape[:2]:
                        vox = sl[msk]                  # (n_roi, N_off)
                        if len(vox) > 0:
                            pool_curves[p] = np.nanmean(vox, axis=0)
                if pool_curves:
                    sumcurve = np.sum(list(pool_curves.values()), axis=0)
                    gauss_cache[roi.name] = {
                        'fit':         1.0 - sumcurve,
                        'pool_curves': pool_curves,
                    }
            if gauss_cache:
                self._roi_spectra_fit_cache['Gaussian'] = gauss_cache

        # ── MPLF voxelwise → ROI cache ───────────────────────────────────
        mplf_pools = results.get("mplf_pools", [])
        if mplf_pools:
            mplf_cache: dict = {}
            for roi in rois:
                msk = roi.mask
                pool_curves = {}
                for p in mplf_pools:
                    if f"mplf_peak_{p}" not in results:
                        continue
                    vol = results[f"mplf_peak_{p}"]  # (H, W, sl, N_off)
                    sl  = vol[:, :, self._cur_slice(vol.shape[2]), :]
                    if msk.shape[:2] == sl.shape[:2]:
                        vox = sl[msk]
                        if len(vox) > 0:
                            pool_curves[p] = np.nanmean(vox, axis=0)
                if pool_curves:
                    # Compute approximate total Z-fit from pool ΔZ contributions
                    # (sum of attenuation contributions; approximate for display)
                    _sum_dz = np.sum(list(pool_curves.values()), axis=0)
                    mplf_cache[roi.name] = {
                        'pools': pool_curves,
                        'fit':   1.0 - _sum_dz,
                    }
            if mplf_cache:
                self._roi_spectra_fit_cache['MPLF'] = mplf_cache

    def _on_cancelled(self):
        self._set_running(False)
        self.lbl_run_status.setText("Cancelled.")

    def _on_error(self, msg: str):
        self._set_running(False)
        self.lbl_run_status.setText("Error!")
        self._log(f"ERROR: {msg}")

    def _cancel_analysis(self):
        if self._worker is not None and self._worker.isRunning():
            self._log("Cancelling…")
            self._worker.stop()
        self.btn_cancel.setEnabled(False)

    def _set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.progress_bar.setVisible(running)

    # ─────────────────────────────────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────────────────────────────────

    def _get_m0(self) -> "np.ndarray | None":
        """Return the M0 image for the user-selected source + frame.

        Priority:
        1. User-chosen frame from full dataset (combo_m0_frame index into img_all).
        2. Auto-detected M0 extracted by the reader (fallback if img_all absent).
        """
        src       = self.combo_m0_source.currentText()
        frame_idx = self.combo_m0_frame.currentIndex()

        if src == "WASSR":
            if (self._wassr_img_all is not None
                    and 0 <= frame_idx < self._wassr_img_all.shape[-1]):
                return self._wassr_img_all[:, :, :, frame_idx]
            return self._wassr_M0_img          # fallback
        else:  # CEST
            if (self._z_img_all is not None
                    and 0 <= frame_idx < self._z_img_all.shape[-1]):
                return self._z_img_all[:, :, :, frame_idx]
            return self._M0_img               # fallback

    # ── Motion correction (rigid, itk-elastix) ────────────────────────────────
    def _on_moco_toggled(self, checked: bool):
        if getattr(self, '_z_img_all', None) is None:
            if checked:
                self._log("Motion correction: load a CEST stack first.")
                self.chk_moco.blockSignals(True)
                self.chk_moco.setChecked(False)
                self.chk_moco.blockSignals(False)
            return
        self._apply_moco() if checked else self._restore_moco()

    def _apply_moco(self):
        """Rigidly register every offset image to the chosen reference frame."""
        from PyQt6.QtWidgets import QProgressDialog, QApplication
        from my_gui import motion_correction as mc
        if self._z_img_all is None:
            return
        if getattr(self, '_z_img_all_raw', None) is None:
            self._z_img_all_raw = np.asarray(self._z_img_all).copy()
        ref = self.combo_moco_ref.currentText()
        n = int(self._z_img_all_raw.shape[-1])
        prog = QProgressDialog("Motion correction (Elastix)…", "Cancel", 0, n, self)
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        prog.setMinimumDuration(0); prog.setValue(0); prog.show()

        def _p(done, total):
            prog.setValue(done); QApplication.processEvents()
            if prog.wasCanceled():
                raise RuntimeError("cancelled")

        try:
            corr = mc.moco_4d(self._z_img_all_raw, frame_axis=3,
                              ref_mode=ref, progress=_p).astype(np.float32)
        except RuntimeError:
            prog.close()
            self.chk_moco.blockSignals(True); self.chk_moco.setChecked(False)
            self.chk_moco.blockSignals(False)
            self._log("Motion correction cancelled.")
            return
        except Exception as e:          # noqa: BLE001
            prog.close()
            self.chk_moco.blockSignals(True); self.chk_moco.setChecked(False)
            self.chk_moco.blockSignals(False)
            QMessageBox.critical(self, "Motion correction failed", str(e))
            return
        prog.close()
        self._z_img_all = corr
        self._z_img_full = corr
        idx = int(getattr(self, '_cest_m0_auto_idx', -1))
        try:
            self._M0_img = corr[:, :, :, idx]
        except Exception:
            pass
        self._log(f"Motion correction applied (rigid, ref='{ref}').")
        self._roi_spectra_dlg_key = ()  # force ROI-spectra dialog rebuild
        self._refresh_display()

    def _restore_moco(self):
        raw = getattr(self, '_z_img_all_raw', None)
        if raw is None:
            return
        self._z_img_all = raw.copy()
        self._z_img_full = np.asarray(raw, np.float32)
        idx = int(getattr(self, '_cest_m0_auto_idx', -1))
        try:
            self._M0_img = raw[:, :, :, idx]
        except Exception:
            pass
        self._log("Motion correction removed — original images restored.")
        self._roi_spectra_dlg_key = ()
        self._refresh_display()

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
        self._refresh_display()

    def _refresh_display(self):
        self._dc_annot = None   # reset data cursor (axes rebuilt on show_map)
        # Propagate the "Bg" toggle to the canvas (read by show_map on redraw).
        if hasattr(self, 'chk_dark_bg'):
            self.canvas._dark_bg = self.chk_dark_bg.isChecked()
        if hasattr(self, 'chk_logmap'):
            self.canvas._log_map = self.chk_logmap.isChecked()
        sel = self.combo_display.currentText()

        # Show/hide offset slider for CEST/WASSR image modes
        is_offset_mode = sel in ("CEST Images", "WASSR Images")
        self._offset_frame.setVisible(is_offset_mode)

        # Sync the slice slider to the loaded volume (visible only if >1 slice)
        self._sync_slice_slider()

        # The MTRasym / MTR_Rex compute buttons appear only for their own map view
        if hasattr(self, "_qmtr_w"):
            self._qmtr_w.setVisible(sel == "MTR asymmetry map")
        if hasattr(self, "_qrex_w"):
            self._qrex_w.setVisible(sel == "MTR_Rex map")

        # ── M0 image — available as soon as data is loaded ────────────────
        if sel == "M0 image (unsaturated)":
            m0 = self._get_m0()
            if m0 is not None:
                try:
                    if m0.ndim == 3:
                        sl = m0[:, :, self._cur_slice(m0.shape[2])]
                    else:
                        sl = m0
                    if sl.ndim == 3:
                        sl = sl[:, :, self._cur_slice(sl.shape[2])]
                    src_lbl  = self.combo_m0_source.currentText()
                    frame_idx = self.combo_m0_frame.currentIndex()
                    ppm_lbl   = ""
                    ppm_arr   = self._ppm_all if src_lbl == "CEST" else self._wassr_ppm_all
                    if ppm_arr is not None and 0 <= frame_idx < len(ppm_arr):
                        ppm_lbl = f"  {ppm_arr[frame_idx]:+.2f} ppm"
                    _cm = self.plot_bar.get_cmap()
                    _vmn, _vmx = self.plot_bar.get_clim()
                    self.canvas.show_map(
                        sl,
                        f"M0 / Unsaturated image  [{src_lbl}  frame {frame_idx}{ppm_lbl}]",
                        cmap=_cm, vmin=_vmn, vmax=_vmx, **self.plot_bar.get_font_sizes())
                except Exception as exc:
                    self._log(f"Display error (M0): {exc}")
            else:
                self._log("No M0 image loaded yet. Load CEST or WASSR data first.")
            return

        # ── CEST Images (by offset) — scrolls + ppm → 0 → − ppm ────────────
        if sel == "CEST Images":
            if self._z_img_full is None or self._ppm is None:
                self._log("Load CEST data first.")
                return
            n_offsets = self._z_img_full.shape[-1]
            self._offset_slider.blockSignals(True)
            self._offset_slider.setMaximum(n_offsets - 1)
            self._offset_slider.blockSignals(False)
            # Endpoint labels: index 0 = most-positive ppm, index N-1 = most-negative
            self._offset_lbl_start.setText(f"{self._ppm[0]:+.1f} ppm")
            self._offset_lbl_end.setText(f"{self._ppm[-1]:+.1f} ppm")
            idx = min(self._offset_slider.value(), n_offsets - 1)
            ppm_val = self._ppm[idx]
            self._offset_label.setText(f"{ppm_val:+.2f} ppm")
            _sidx = self._cur_slice(self._z_img_full.shape[2])
            sl = self._z_img_full[:, :, _sidx, idx]
            _vmn, _vmx = self.plot_bar.get_clim()
            self.canvas.show_map(sl,
                                 f"CEST Image \u2014 {ppm_val:+.2f} ppm  "
                                 f"({idx + 1}/{n_offsets})",
                                 cmap=self.plot_bar.get_cmap(),
                                 vmin=_vmn, vmax=_vmx,
                                 **self.plot_bar.get_font_sizes())
            return

        # ── WASSR Images (by offset) — scrolls + ppm → 0 → − ppm ───────────
        if sel == "WASSR Images":
            if self._wassr_img_full is None or self._wassr_ppm is None:
                self._log("Load WASSR data first.")
                return
            n_offsets = self._wassr_img_full.shape[-1]
            self._offset_slider.blockSignals(True)
            self._offset_slider.setMaximum(n_offsets - 1)
            self._offset_slider.blockSignals(False)
            # Endpoint labels: index 0 = most-positive ppm, index N-1 = most-negative
            self._offset_lbl_start.setText(f"{self._wassr_ppm[0]:+.2f} ppm")
            self._offset_lbl_end.setText(f"{self._wassr_ppm[-1]:+.2f} ppm")
            idx = min(self._offset_slider.value(), n_offsets - 1)
            ppm_val = self._wassr_ppm[idx]
            self._offset_label.setText(f"{ppm_val:+.2f} ppm")
            _sidx = self._cur_slice(self._wassr_img_full.shape[2])
            sl = self._wassr_img_full[:, :, _sidx, idx]
            _vmn, _vmx = self.plot_bar.get_clim()
            self.canvas.show_map(sl,
                                 f"WASSR Image \u2014 {ppm_val:+.2f} ppm  "
                                 f"({idx + 1}/{n_offsets})",
                                 cmap=self.plot_bar.get_cmap(),
                                 vmin=_vmn, vmax=_vmx,
                                 **self.plot_bar.get_font_sizes())
            return

        # \u2500\u2500 B0 map \u2014 available as soon as WASSR has been loaded \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if sel == "B0 map (ppm)":
            b0 = getattr(self, "_b0_map_ppm", None)
            if b0 is None:
                self._log("No B0 map available \u2014 load WASSR data first.")
                return
            sl = b0[:, :, self._cur_slice(b0.shape[2])] if b0.ndim == 3 else b0
            if sl.ndim == 3:
                sl = sl[:, :, self._cur_slice(sl.shape[2])]
            # Apply phantom outline mask (same as MTR asymmetry map)
            _b0_ph = next(
                (r for r in getattr(self, '_last_rois', []) if r.name == "Phantom_outline"),
                None,
            )
            if _b0_ph is not None:
                msk = np.asarray(_b0_ph.mask)
                if msk.shape[:2] != sl.shape[:2]:
                    try:
                        from scipy.ndimage import zoom as _zoom
                        msk = _zoom(msk.astype(float),
                                    (sl.shape[0] / max(msk.shape[0], 1),
                                     sl.shape[1] / max(msk.shape[1], 1)), order=0) > 0.5
                    except Exception:
                        msk = None
                if msk is not None and msk.shape[:2] == sl.shape[:2]:
                    sl = np.where(msk, sl, np.nan)
            cmap = self.plot_bar.get_cmap()
            vmin, vmax = self.plot_bar.get_clim()
            fs   = self.plot_bar.get_font_sizes()
            _ct  = getattr(self, 'edit_map_title', None)
            _ctitle = _ct.text().strip() if _ct else ""
            if vmin is None and vmax is None:
                fin = sl[np.isfinite(sl)] if sl is not None else np.array([])
                lim = float(np.nanpercentile(np.abs(fin), 99)) if fin.size else 0.5
                vmin, vmax = -lim, lim
            if self.chk_roi_bg.isChecked():
                _base = None
                if getattr(self, "_z_img_full", None) is not None:
                    _zf = self._z_img_full
                    _si = self._cur_slice(_zf.shape[2])
                    _base = np.asarray(_zf)[:, :, _si, 0]
                if getattr(self, "_roi_bg_img", None) is not None:
                    _base = self._roi_bg_img
                _union = roi_union_mask(getattr(self, "_last_rois", []), sl.shape)
                if _base is not None and _union is not None:
                    _ovc = cmap if str(cmap).lower() not in ("gray", "greys", "greys_r") else "jet"
                    self.canvas.show_map_over_raw(
                        sl, _base, _union, _ctitle or "B0 Field Map  (ppm)",
                        cmap=_ovc, vmin=vmin, vmax=vmax, **fs)
                    return
            self.canvas.show_map(
                sl, _ctitle or "B0 Field Map  (ppm)",
                cmap=cmap,
                vmin=vmin, vmax=vmax, **fs
            )
            return

        # ── B0 map (Hz) — available as soon as WASSR has been loaded ──────
        if sel == "B0 map (Hz)":
            b0_ppm = getattr(self, "_b0_map_ppm", None)
            if b0_ppm is None:
                self._log("No B0 map available — load WASSR data first.")
                return
            larmor_mhz = self.spin_larmor_mhz.value()
            b0_hz = b0_ppm * larmor_mhz   # ppm × MHz = Hz
            sl = b0_hz[:, :, self._cur_slice(b0_hz.shape[2])] if b0_hz.ndim == 3 else b0_hz
            if sl.ndim == 3:
                sl = sl[:, :, self._cur_slice(sl.shape[2])]
            _b0_ph = next(
                (r for r in getattr(self, '_last_rois', []) if r.name == "Phantom_outline"),
                None,
            )
            if _b0_ph is not None:
                msk = np.asarray(_b0_ph.mask)
                if msk.shape[:2] != sl.shape[:2]:
                    try:
                        from scipy.ndimage import zoom as _zoom
                        msk = _zoom(msk.astype(float),
                                    (sl.shape[0] / max(msk.shape[0], 1),
                                     sl.shape[1] / max(msk.shape[1], 1)), order=0) > 0.5
                    except Exception:
                        msk = None
                if msk is not None and msk.shape[:2] == sl.shape[:2]:
                    sl = np.where(msk, sl, np.nan)
            cmap = self.plot_bar.get_cmap()
            vmin, vmax = self.plot_bar.get_clim()
            fs   = self.plot_bar.get_font_sizes()
            _ct  = getattr(self, 'edit_map_title', None)
            _ctitle = _ct.text().strip() if _ct else ""
            if vmin is None and vmax is None:
                fin = sl[np.isfinite(sl)] if sl is not None else np.array([])
                lim = float(np.nanpercentile(np.abs(fin), 99)) if fin.size else 100.0
                vmin, vmax = -lim, lim
            if self.chk_roi_bg.isChecked():
                _base = None
                if getattr(self, "_z_img_full", None) is not None:
                    _zf = self._z_img_full
                    _si = self._cur_slice(_zf.shape[2])
                    _base = np.asarray(_zf)[:, :, _si, 0]
                if getattr(self, "_roi_bg_img", None) is not None:
                    _base = self._roi_bg_img
                _union = roi_union_mask(getattr(self, "_last_rois", []), sl.shape)
                if _base is not None and _union is not None:
                    _ovc = cmap if str(cmap).lower() not in ("gray", "greys", "greys_r") else "jet"
                    self.canvas.show_map_over_raw(
                        sl, _base, _union,
                        _ctitle or f"B0 Field Map  (Hz)  [B₀ = {larmor_mhz:.0f} MHz]",
                        cmap=_ovc, vmin=vmin, vmax=vmax, **fs)
                    return
            self.canvas.show_map(
                sl, _ctitle or f"B0 Field Map  (Hz)  [B₀ = {larmor_mhz:.0f} MHz]",
                cmap=cmap,
                vmin=vmin, vmax=vmax, **fs
            )
            return

        # ── AREX map — computed on demand from CEST data + 1/Z params ────
        if sel == "AREX map":
            _zv  = getattr(self, '_z_img_full', None)
            _cur = self._cur_slice(_zv.shape[2]) if (_zv is not None and _zv.ndim == 4) else 0
            arex = getattr(self, '_arex_map', None)
            # Recompute when there is no map yet OR the displayed slice changed
            if arex is None or getattr(self, '_arex_slice', None) != _cur:
                _ap0 = getattr(self, '_arex_params', None)
                if _ap0:
                    # Re-use the first compute's parameters for the other slices
                    _r1, _ep = _ap0.get('R1', 1.0), _ap0.get('eval_ppm', 3.5)
                else:
                    # Prompt for R1 + eval ppm the first time
                    from PyQt6.QtWidgets import (
                        QDialog as _QDlg, QFormLayout as _QFL,
                        QDoubleSpinBox as _QDSP2, QDialogButtonBox as _QDBBox,
                    )
                    _dlg = _QDlg(self)
                    _dlg.setWindowTitle("AREX Map Parameters")
                    _dlg.setMaximumWidth(320)
                    _fl = _QFL(_dlg)
                    _sp_r1 = _QDSP2(); _sp_r1.setRange(0.01, 20.0); _sp_r1.setValue(1.0)
                    _sp_r1.setSuffix(" s⁻¹"); _sp_r1.setSingleStep(0.05); _sp_r1.setDecimals(3)
                    _fl.addRow("R1 (water):", _sp_r1)
                    _sp_ep = _QDSP2(); _sp_ep.setRange(0.1, 15.0); _sp_ep.setValue(3.5)
                    _sp_ep.setSuffix(" ppm"); _sp_ep.setSingleStep(0.5)
                    _fl.addRow("Eval offset (ppm):", _sp_ep)
                    _bbs = _QDBBox(_QDBBox.StandardButton.Ok | _QDBBox.StandardButton.Cancel)
                    _bbs.accepted.connect(_dlg.accept); _bbs.rejected.connect(_dlg.reject)
                    _fl.addRow(_bbs)
                    if _dlg.exec() != _QDlg.DialogCode.Accepted:
                        return
                    _r1  = _sp_r1.value()
                    _ep  = _sp_ep.value()
                # Compute AREX per voxel for the displayed slice (vectorised)
                try:
                    z_vol = _zv
                    if z_vol is None or self._ppm is None:
                        self._log("AREX: no CEST data loaded."); return
                    _zslice = z_vol[:, :, _cur, :] if z_vol.ndim == 4 else z_vol
                    _m0v = self._get_m0()
                    if _m0v is not None:
                        _m0s = (_m0v[:, :, min(_cur, _m0v.shape[2] - 1)]
                                if _m0v.ndim == 3 else _m0v)
                        _zn = _zslice / (_m0s[:, :, np.newaxis] + 1e-9)
                    else:
                        _zn = _zslice
                    _zsl = np.clip(_zn.astype(float), 1e-9, 1.0)
                    _H, _W, _N = _zsl.shape
                    _si   = np.argsort(self._ppm)
                    _ppm_s = self._ppm[_si]
                    _zf   = _zsl.reshape(-1, _N)[:, _si]
                    _Zp   = np.array([np.interp( _ep, _ppm_s, _zf[v]) for v in range(_H * _W)])
                    _Zn   = np.array([np.interp(-_ep, _ppm_s, _zf[v]) for v in range(_H * _W)])
                    _Zp   = np.clip(_Zp, 1e-9, None); _Zn = np.clip(_Zn, 1e-9, None)
                    arex  = ((1.0 / _Zp) - (1.0 / _Zn)) * _r1
                    arex  = arex.reshape(_H, _W)
                    self._arex_map    = apply_analysis_mask(arex)   # global brain/phantom mask
                    self._arex_params = dict(eval_ppm=_ep, R1=_r1)
                    self._arex_slice  = _cur
                except Exception as _exc:
                    self._log(f"AREX computation error: {_exc}"); return
            # Display stored AREX map
            _ap    = getattr(self, '_arex_params', {})
            _ep_d  = _ap.get('eval_ppm', 3.5)
            _r1_d  = _ap.get('R1', 1.0)
            _b1_d  = _ap.get('B1_uT', '')
            _b1_str = f"  B1={_b1_d:.2f}µT" if _b1_d else ""
            cmap   = self.plot_bar.get_cmap()
            vmin, vmax = self.plot_bar.get_clim()
            fs     = self.plot_bar.get_font_sizes()
            _ct    = getattr(self, 'edit_map_title', None)
            _ctitle = _ct.text().strip() if _ct else ""
            # Phantom mask
            _arex_ph = next(
                (r for r in getattr(self, '_last_rois', []) if r.name == "Phantom_outline"),
                None,
            )
            _amap = arex.copy()
            if _arex_ph is not None:
                _msk = _arex_ph.mask
                if _msk.shape == _amap.shape or _msk.shape[:2] == _amap.shape[:2]:
                    _amap = np.where(_msk, _amap, np.nan)
            _fin  = _amap[np.isfinite(_amap)]
            _lim  = float(np.nanpercentile(np.abs(_fin), 99)) if _fin.size else 1.0
            if self.chk_roi_bg.isChecked():
                _base = None
                if getattr(self, "_z_img_full", None) is not None:
                    _zf = self._z_img_full
                    _si = self._cur_slice(_zf.shape[2])
                    _base = np.asarray(_zf)[:, :, _si, 0]
                if getattr(self, "_roi_bg_img", None) is not None:
                    _base = self._roi_bg_img
                _union = roi_union_mask(getattr(self, "_last_rois", []), _amap.shape)
                if _base is not None and _union is not None:
                    _ovc = cmap if str(cmap).lower() not in ("gray", "greys", "greys_r") else "jet"
                    self.canvas.show_map_over_raw(
                        _amap, _base, _union,
                        _ctitle or f"AREX  (±{_ep_d:.2f} ppm,  R1={_r1_d:.3g} s⁻¹{_b1_str})",
                        cmap=_ovc, vmin=vmin or 0, vmax=vmax or _lim, **fs)
                    return
            self.canvas.show_map(
                _amap,
                _ctitle or f"AREX  (±{_ep_d:.2f} ppm,  R1={_r1_d:.3g} s⁻¹{_b1_str})",
                cmap=cmap if cmap != "gray" else "hot",
                vmin=vmin or 0, vmax=vmax or _lim, **fs,
            )
            return

        # ── ROI spectral fit maps ─────────────────────────────────────────
        _roi_maps = getattr(self, '_roi_spectra_maps', {})
        for _fit_method, _pool_maps in _roi_maps.items():
            _prefix = f"{_fit_method}: "
            if sel.startswith(_prefix):
                _pool_name = sel[len(_prefix):]
                _map_data = _pool_maps.get(_pool_name)
                if _map_data is not None:
                    cmap = self.plot_bar.get_cmap()
                    vmin, vmax = self.plot_bar.get_clim()
                    fs   = self.plot_bar.get_font_sizes()
                    _ct  = getattr(self, 'edit_map_title', None)
                    _ctitle = _ct.text().strip() if _ct else ""
                    _sl = (_map_data[:, :, self._cur_slice(_map_data.shape[2])]
                           if _map_data.ndim == 3 else _map_data)
                    self.canvas.show_map(
                        _sl,
                        _ctitle or f"{_fit_method} — {_pool_name}",
                        cmap=cmap, vmin=vmin, vmax=vmax, **fs,
                    )
                return

        if not self._results:
            return

        cmap = self.plot_bar.get_cmap()
        vmin, vmax = self.plot_bar.get_clim()
        fs = self.plot_bar.get_font_sizes()

        _ct = getattr(self, 'edit_map_title', None)
        _ctitle = _ct.text().strip() if _ct else ""

        # Phantom outline masking
        phantom_roi = next(
            (r for r in getattr(self, '_last_rois', []) if r.name == "Phantom_outline"),
            None,
        )
        def _ph_mask(arr):
            if phantom_roi is None or arr is None:
                return arr
            msk = np.asarray(phantom_roi.mask)
            # Rescale the ROI mask to the map shape if resolutions differ
            # (e.g. ROI drawn on a different-sized image than the GE 512² map).
            if msk.shape[:2] != arr.shape[:2]:
                try:
                    from scipy.ndimage import zoom as _zoom
                    zy = arr.shape[0] / max(msk.shape[0], 1)
                    zx = arr.shape[1] / max(msk.shape[1], 1)
                    msk = _zoom(msk.astype(float), (zy, zx), order=0) > 0.5
                except Exception:
                    return arr
            return np.where(msk, arr, np.nan)

        if sel == "MTR asymmetry map":
            data = self._results.get("mtr_map")
            if data is not None:
                sl = data[..., self._cur_slice(data.shape[2])] if data.ndim == 3 else data
                sl = _ph_mask(sl)
                if vmin is None and vmax is None:
                    fin = sl[np.isfinite(sl)] if sl is not None else np.array([])
                    lim = np.nanpercentile(np.abs(fin), 99) if fin.size else 1.0
                    vmin_use, vmax_use = -lim, lim
                else:
                    vmin_use, vmax_use = vmin, vmax
                default_t = f"MTR Asymmetry (±{self._results.get('mtr_ppm_used', 3.5):.2f} ppm)"
                if self.chk_roi_bg.isChecked():
                    _base = None
                    if getattr(self, "_z_img_full", None) is not None:
                        _zf = self._z_img_full
                        _si = self._cur_slice(_zf.shape[2])
                        _base = np.asarray(_zf)[:, :, _si, 0]
                    if getattr(self, "_roi_bg_img", None) is not None:
                        _base = self._roi_bg_img
                    _union = roi_union_mask(getattr(self, "_last_rois", []), sl.shape)
                    if _base is not None and _union is not None:
                        _ovc = cmap if str(cmap).lower() not in ("gray", "greys", "greys_r") else "jet"
                        self.canvas.show_map_over_raw(
                            sl, _base, _union, _ctitle or default_t,
                            cmap=_ovc, vmin=vmin_use, vmax=vmax_use, **fs)
                        return
                self.canvas.show_map(sl, _ctitle or default_t,
                                     cmap=cmap if cmap != "gray" else "bwr",
                                     vmin=vmin_use, vmax=vmax_use, **fs)

        elif sel == "MTR_Rex map":
            data = self._results.get("mtrrex_map")
            if data is not None:
                sl = data[..., self._cur_slice(data.shape[2])] if data.ndim == 3 else data
                sl = _ph_mask(sl)
                if vmin is None and vmax is None:
                    fin = sl[np.isfinite(sl)] if sl is not None else np.array([])
                    lim = np.nanpercentile(np.abs(fin), 99) if fin.size else 1.0
                    vmin_use, vmax_use = -lim, lim
                else:
                    vmin_use, vmax_use = vmin, vmax
                default_t = f"MTR_Rex (±{self._results.get('mtrrex_ppm_used', 3.5):.2f} ppm)"
                if self.chk_roi_bg.isChecked():
                    _base = None
                    if getattr(self, "_z_img_full", None) is not None:
                        _zf = self._z_img_full
                        _si = self._cur_slice(_zf.shape[2])
                        _base = np.asarray(_zf)[:, :, _si, 0]
                    if getattr(self, "_roi_bg_img", None) is not None:
                        _base = self._roi_bg_img
                    _union = roi_union_mask(getattr(self, "_last_rois", []), sl.shape)
                    if _base is not None and _union is not None:
                        _ovc = cmap if str(cmap).lower() not in ("gray", "greys", "greys_r") else "jet"
                        self.canvas.show_map_over_raw(
                            sl, _base, _union, _ctitle or default_t,
                            cmap=_ovc, vmin=vmin_use, vmax=vmax_use, **fs)
                        return
                self.canvas.show_map(sl, _ctitle or default_t,
                                     cmap=cmap if cmap != "gray" else "bwr",
                                     vmin=vmin_use, vmax=vmax_use, **fs)

        elif (sel.startswith("PV: ")
              or sel.startswith("Gaussian: ")
              or sel.startswith("Lor: ")
              or sel.startswith("MPLF: ")):
            if sel.startswith("PV: "):
                prefix = "pv_"
                pool   = sel[len("PV: "):]
                method = "Pseudo-Voigt"
            elif sel.startswith("Gaussian: "):
                prefix = "gauss_"
                pool   = sel[len("Gaussian: "):]
                method = "Gaussian"
            elif sel.startswith("MPLF: "):
                prefix = "mplf_"
                pool   = sel[len("MPLF: "):]
                method = "MPLF"
            else:
                prefix = "lor_"
                pool   = sel[len("Lor: "):]
                method = "Lorentzian"
            data = self._results.get(f"{prefix}ampl_{pool}")
            if data is not None:
                sl = data[:, :, self._cur_slice(data.shape[2])] if data.ndim == 3 else data
                sl = _ph_mask(sl)
                if self.chk_roi_bg.isChecked():
                    _base = None
                    if getattr(self, "_z_img_full", None) is not None:
                        _zf = self._z_img_full
                        _si = self._cur_slice(_zf.shape[2])
                        _base = np.asarray(_zf)[:, :, _si, 0]
                    if getattr(self, "_roi_bg_img", None) is not None:
                        _base = self._roi_bg_img
                    _union = roi_union_mask(getattr(self, "_last_rois", []), sl.shape)
                    if _base is not None and _union is not None:
                        _ovc = cmap if str(cmap).lower() not in ("gray", "greys", "greys_r") else "jet"
                        self.canvas.show_map_over_raw(
                            sl, _base, _union, _ctitle or f"{method} — {pool}",
                            cmap=_ovc, vmin=vmin, vmax=vmax, **fs)
                        return
                self.canvas.show_map(
                    sl,
                    _ctitle or f"{method} — {pool}",
                    cmap=cmap, vmin=vmin, vmax=vmax, **fs,
                )

    def _on_img_slider(self, _value: int):
        """Called when the offset slider is moved; refreshes the displayed image."""
        self._refresh_display()

    def _on_slice_slider(self, _value: int):
        """Called when the slice slider is moved; refreshes the displayed image."""
        n = self._slice_slider.maximum() + 1
        self._slice_label.setText(f"{self._slice_slider.value() + 1}/{n}")
        self._refresh_display()

    def _cur_slice(self, n_slices: int) -> int:
        """Current slice index from the slice slider, clamped to [0, n_slices-1]."""
        i = self._slice_slider.value() if hasattr(self, "_slice_slider") else 0
        return max(0, min(int(i), int(n_slices) - 1))

    def _sync_slice_slider(self):
        """Set the slice-slider range/visibility from the loaded volume's slice count."""
        n_sl = 1
        z = getattr(self, "_z_img_full", None)
        w = getattr(self, "_wassr_img_full", None)
        if z is not None and getattr(z, "ndim", 0) == 4:
            n_sl = z.shape[2]
        elif w is not None and getattr(w, "ndim", 0) == 4:
            n_sl = w.shape[2]
        if not hasattr(self, "_slice_frame"):
            return n_sl
        self._slice_frame.setVisible(n_sl > 1)
        if self._slice_slider.maximum() != n_sl - 1:
            self._slice_slider.blockSignals(True)
            self._slice_slider.setMaximum(max(0, n_sl - 1))
            self._slice_slider.setValue(min(self._slice_slider.value(), n_sl - 1))
            self._slice_slider.blockSignals(False)
        self._slice_label.setText(f"{self._slice_slider.value() + 1}/{n_sl}")
        return n_sl

    def _on_canvas_scroll(self, event):
        """Mouse wheel over canvas scrolls through CEST/WASSR offset images."""
        sel = self.combo_display.currentText()
        if sel not in ("CEST Images", "WASSR Images"):
            return
        step = 1 if event.step > 0 else -1
        new_val = max(0, min(self._offset_slider.maximum(),
                             self._offset_slider.value() + step))
        self._offset_slider.setValue(new_val)

    def _toggle_hide_rois(self, checked: bool):
        self.canvas.toggle_rois_visible()
        self.btn_hide_rois.setText("Show ROIs" if checked else "Hide ROIs")

    # ── Window/Level (brightness–contrast) drag tool ─────────────────────
    def _on_wl_tool_toggled(self, checked: bool):
        """Activate/deactivate the interactive window/level drag tool on the map."""
        # Only meaningful for single-map displays (not overlays/spectra).
        self.canvas.set_wl_active(bool(checked), on_change=self._on_wl_changed)

    def _on_wl_changed(self, vmin: float, vmax: float):
        """Persist the dragged window/level into the colour-bar widget so the
        contrast survives later redraws (offset slider, ROI edits, etc.)."""
        try:
            # Update the spin values WITHOUT triggering a full redraw mid-adjust;
            # set_clim() emits `applied`, which _refresh_display listens to.
            self.plot_bar.set_clim(vmin, vmax)
        except Exception:
            pass

    def connect_roi_manager(self, roi_manager):
        roi_manager.connect_canvas(self.canvas)
        roi_manager.rois_changed.connect(self._update_roi_stats_zspec)
        self._roi_manager_ref = roi_manager

    def _update_roi_stats_zspec(self, rois: list):
        """Store ROIs — stats shown on demand via the ROI Stats Table button."""
        old_names = tuple(r.name for r in self._last_rois
                          if r.name != "Phantom_outline")
        new_names = tuple(r.name for r in rois if r.name != "Phantom_outline")
        self._last_rois = rois
        # Invalidate the cached dialog when the ROI set changes so it is
        # rebuilt fresh next time (old plots would show stale ROI names).
        if old_names != new_names:
            self._roi_spectra_dlg_key = ()   # mismatch forces rebuild

    def _verify_offsets(self):
        """Diagnostic: verify the CEST (and WASSR) offset ordering, per ROI.

        For each ROI it plots the normalized Z-spectrum
            Z(Δω) = S_sat(Δω) / S0            (S0 = unsaturated M0 signal)
        whose minimum must fall at 0 ppm (water saturation). If it lands
        elsewhere the loaded offsets are mislabeled / in the wrong order.
        Shows one tab per ROI (Phantom_outline included) plus an overview,
        with the CEST and (if loaded) WASSR curves overlaid per ROI.
        """
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                     QTabWidget, QWidget, QTableWidget,
                                     QTableWidgetItem, QPushButton)
        from PyQt6.QtGui import QColor
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        import matplotlib.pyplot as _plt

        zc = getattr(self, '_z_img_full', None)
        pc = getattr(self, '_ppm', None)
        if zc is None or pc is None:
            QMessageBox.information(self, "No CEST data",
                                    "Load CEST data (GE/Siemens DICOM) first.")
            return
        pc = np.asarray(pc, dtype=float).ravel()

        # WASSR is optional (stored already normalized in _wassr_img_full)
        zw = getattr(self, '_wassr_img_full', None)
        pw = getattr(self, '_wassr_ppm', None)
        have_w = zw is not None and pw is not None
        if have_w:
            pw = np.asarray(pw, dtype=float).ravel()

        m0c = self._get_m0()
        m0w = getattr(self, '_wassr_M0_img', None)

        def _slice(img):
            return (img[:, :, self._cur_slice(img.shape[2]), :]
                    if (img is not None and img.ndim == 4) else img)
        zc_sl = _slice(zc)
        zw_sl = _slice(zw) if have_w else None
        HW = zc_sl.shape[:2]

        # ── Z-spectrum for one ROI:  Z = S_sat / S0  (already_norm skips /S0) ──
        def _roi_z(mask, img_sl, ppm, m0, already_norm):
            if img_sl is None:
                return None
            m = np.asarray(mask, dtype=bool)
            if m.shape != img_sl.shape[:2] or not m.any() or ppm.size != img_sl.shape[-1]:
                return None
            S = np.nanmean(img_sl.reshape(-1, img_sl.shape[-1])[m.ravel()], axis=0)
            if already_norm:
                Z = S
            else:
                S0 = None
                if m0 is not None:
                    m0a = np.asarray(m0, dtype=float)
                    m0a = m0a[:, :, 0] if m0a.ndim == 3 else m0a
                    if m0a.shape == m.shape:
                        S0 = float(np.nanmean(m0a.ravel()[m.ravel()]))
                Z = S / (S0 + 1e-9) if (S0 and S0 > 1e-9) else S / (np.nanmax(S) + 1e-12)
            kmin = int(np.nanargmin(Z))
            return dict(ppm=ppm, Z=Z, off_min=float(ppm[kmin]),
                        ok=abs(float(ppm[kmin])) <= 0.5)

        # ── Collect ROIs: Phantom_outline first, then the tubes ───────────────
        allr = getattr(self, '_last_rois', [])
        def _valid(r):
            return (getattr(r, 'mask', None) is not None
                    and np.asarray(r.mask).shape == HW and np.asarray(r.mask).any())
        phantom = [r for r in allr if getattr(r, 'name', '') == 'Phantom_outline' and _valid(r)]
        tubes   = [r for r in allr if getattr(r, 'name', '') != 'Phantom_outline' and _valid(r)]
        rois = phantom + tubes
        used_fallback = False
        if not rois:
            used_fallback = True
            ref = np.asarray(m0c, dtype=float) if m0c is not None else zc_sl.mean(axis=-1)
            ref = ref[:, :, 0] if ref.ndim == 3 else ref
            fin = ref[np.isfinite(ref)]
            thr = 0.15 * (float(np.nanmax(fin)) if fin.size else 1.0)
            _M = type('PhantomMask', (), {})
            fake = _M(); fake.name = 'Phantom (whole)'; fake.mask = ref > thr
            rois = [fake]

        cmap = _plt.cm.get_cmap('tab20', max(len(rois), 2))
        per = []
        for ri, r in enumerate(rois):
            m = np.asarray(r.mask, dtype=bool)
            per.append(dict(
                name=str(getattr(r, 'name', f'ROI_{ri}')), color=cmap(ri), mask=m,
                cest=_roi_z(m, zc_sl, pc, m0c, already_norm=False),
                wassr=_roi_z(m, zw_sl, pw, m0w, already_norm=True) if have_w else None))

        # ── Dialog with a tab per ROI + an overview ───────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle("Verify CEST / WASSR Offsets")
        dlg.resize(880, 720)
        v = QVBoxLayout(dlg)
        note = QLabel("Normalization:  Z(Δω) = S_sat(Δω) / S₀   (S₀ = unsaturated M0). "
                      "Correct offsets → Z-spectrum minimum sits at 0 ppm (water).")
        note.setStyleSheet("color:#555; font-size:11px;")
        v.addWidget(note)
        tabs = QTabWidget(); v.addWidget(tabs, stretch=1)

        # ══ Tab 1 — Frame index: WHICH frame is the water dip? ═══════════════
        # Offset-independent check: plot the mean signal against the acquisition
        # frame number. The deepest frame IS the water (0 ppm) image, so if the
        # loaded offsets are right they must assign 0 ppm to exactly that frame.
        fi = QWidget(); fil = QVBoxLayout(fi)
        fi_note = QLabel(
            "Signal vs <b>frame index</b> — independent of the offsets file. "
            "The deepest frame is the water image, so a correct offsets list must "
            "put <b>0 ppm</b> on that frame.")
        fi_note.setStyleSheet("color:#555; font-size:11px;"); fi_note.setWordWrap(True)
        fil.addWidget(fi_note)

        ffig = Figure(figsize=(7.4, 3.6), facecolor='white')
        fax = ffig.add_subplot(111)
        frame_rows = []
        n_fr = zc_sl.shape[-1]
        for p in per:
            m = np.asarray(p['mask'], dtype=bool)
            S = np.nanmean(zc_sl.reshape(-1, n_fr)[m.ravel()], axis=0)
            if not np.isfinite(S).any():
                continue
            k = int(np.nanargmin(S))                       # dip frame (0-based)
            fax.plot(np.arange(1, n_fr + 1), S, '-', lw=1.1, color=p['color'],
                     label=f"{p['name']} (dip f{k + 1})")
            fax.plot([k + 1], [S[k]], 'v', color=p['color'], ms=7)
            assigned = float(pc[k]) if k < pc.size else float('nan')
            # where the offsets file actually puts 0 ppm
            zero_k = int(np.nanargmin(np.abs(pc))) if pc.size else -1
            frame_rows.append((p['name'], k + 1, assigned, zero_k + 1,
                               abs(assigned) <= 0.5))
        fax.set_xlabel('frame index (1-based, as loaded)')
        fax.set_ylabel('mean signal in ROI')
        fax.set_title('Signal vs frame — the dip marks the 0 ppm (water) frame')
        fax.grid(alpha=0.3); fax.legend(fontsize=8)
        ffig.tight_layout()
        fil.addWidget(FigureCanvas(ffig), stretch=1)

        fhdr = ["ROI", "dip @ frame", "offset assigned there", "file puts 0 ppm @ frame", "verdict"]
        ftbl = QTableWidget(len(frame_rows), len(fhdr))
        ftbl.setHorizontalHeaderLabels(fhdr)
        for i, (nm, dipf, assigned, zerof, ok) in enumerate(frame_rows):
            ftbl.setItem(i, 0, QTableWidgetItem(nm))
            ftbl.setItem(i, 1, QTableWidgetItem(str(dipf)))
            ftbl.setItem(i, 2, QTableWidgetItem(f"{assigned:+.3f} ppm"))
            ftbl.setItem(i, 3, QTableWidgetItem(str(zerof)))
            it = QTableWidgetItem("✓ aligned" if ok else f"⚠ shift {zerof - dipf:+d} frame(s)")
            it.setBackground(QColor(200, 240, 200) if ok else QColor(250, 210, 210))
            ftbl.setItem(i, 4, it)
        ftbl.resizeColumnsToContents(); ftbl.setMaximumHeight(190)
        fil.addWidget(ftbl)

        if frame_rows:
            n_ok = sum(1 for r in frame_rows if r[4])
            sm = QLabel()
            if n_ok == len(frame_rows):
                sm.setText(f"✓ Offsets look CORRECT — the water dip falls on the "
                           f"0 ppm entry for all {n_ok} ROI(s).")
                sm.setStyleSheet("color:#2e7d32; font-weight:bold;")
            else:
                d = frame_rows[0]
                sm.setText(
                    f"⚠ Misaligned: the dip is at frame {d[1]}, but the offsets file "
                    f"assigns {d[2]:+.3f} ppm there and places 0 ppm at frame {d[3]}. "
                    f"Shift the offsets list by {d[3] - d[1]:+d} frame(s) "
                    f"(or check the number of leading idling/S0 frames).")
                sm.setStyleSheet("color:#c62828; font-weight:bold;")
            sm.setWordWrap(True); fil.addWidget(sm)
        tabs.addTab(fi, "Frame index")

        def _spec_canvas(entries, title):
            fig = Figure(figsize=(7.4, 3.6), facecolor='white')
            ax = fig.add_subplot(111)
            for lab, col, res, style in entries:
                if res is None:
                    continue
                o = np.argsort(res['ppm'])
                ax.plot(res['ppm'][o], res['Z'][o], style, color=col, lw=1.2,
                        ms=3, label=f"{lab} (min {res['off_min']:+.2f} ppm)")
            ax.axvline(0.0, color='black', ls='--', lw=1.2)
            ax.set_xlabel('assigned offset Δω (ppm)'); ax.set_ylabel('Z = S/S₀')
            ax.set_title(title); ax.invert_xaxis(); ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            return FigureCanvas(fig)

        # Overview tab — all ROIs' CEST Z overlaid + a per-ROI verdict table
        ov = QWidget(); ovl = QVBoxLayout(ov)
        n_tot = sum(1 for p in per if p['cest'])
        n_ok = sum(1 for p in per if p['cest'] and p['cest']['ok'])
        vlab = QLabel()
        if n_tot and n_ok == n_tot:
            vlab.setText(f"✓ All {n_tot} ROI(s): CEST Z-spectrum minimum at ≈ 0 ppm.")
            vlab.setStyleSheet("color:#2e7d32; font-weight:bold;")
        else:
            bad = [p['name'] for p in per if p['cest'] and not p['cest']['ok']]
            vlab.setText(f"⚠ {n_tot - n_ok}/{n_tot} ROI(s) don't dip at 0 ppm: "
                         f"{', '.join(bad[:8])}{' …' if len(bad) > 8 else ''}")
            vlab.setStyleSheet("color:#c62828; font-weight:bold;")
        ovl.addWidget(vlab)
        fig = Figure(figsize=(7.4, 3.4), facecolor='white'); ax = fig.add_subplot(111)
        for p in per:
            if p['cest']:
                o = np.argsort(p['cest']['ppm'])
                ax.plot(p['cest']['ppm'][o], p['cest']['Z'][o], '-', lw=1.0,
                        color=p['color'], alpha=0.9)
        ax.axvline(0.0, color='black', ls='--', lw=1.2)
        ax.set_xlabel('Δω (ppm)'); ax.set_ylabel('Z = S/S₀')
        ax.set_title('CEST Z-spectra — all ROIs'); ax.invert_xaxis(); ax.grid(alpha=0.3)
        fig.tight_layout(); ovl.addWidget(FigureCanvas(fig), stretch=1)
        hdr = ["ROI", "CEST min (ppm)", "WASSR min (ppm)", "status"]
        tbl = QTableWidget(len(per), len(hdr)); tbl.setHorizontalHeaderLabels(hdr)
        for i, p in enumerate(per):
            col = p['color']
            c0 = QTableWidgetItem(p['name'])
            c0.setBackground(QColor(int(col[0]*255), int(col[1]*255), int(col[2]*255)))
            tbl.setItem(i, 0, c0)
            tbl.setItem(i, 1, QTableWidgetItem(f"{p['cest']['off_min']:+.3f}" if p['cest'] else "—"))
            tbl.setItem(i, 2, QTableWidgetItem(f"{p['wassr']['off_min']:+.3f}" if p['wassr'] else "—"))
            ok = bool(p['cest'] and p['cest']['ok'] and (p['wassr'] is None or p['wassr']['ok']))
            st = QTableWidgetItem("✓ 0 ppm" if ok else "⚠ check")
            st.setBackground(QColor(200, 240, 200) if ok else QColor(250, 210, 210))
            tbl.setItem(i, 3, st)
        tbl.resizeColumnsToContents(); tbl.setMaximumHeight(200)
        ovl.addWidget(tbl)
        if used_fallback:
            h = QLabel("No ROIs drawn — used the whole phantom. Draw tube ROIs in "
                       "the ROI Manager and re-run to get one tab per tube.")
            h.setStyleSheet("color:#888; font-size:10px;"); ovl.addWidget(h)
        tabs.addTab(ov, f"Overview ({n_tot})")

        # One tab per ROI (Phantom_outline first) — CEST + WASSR curves
        for p in per:
            w = QWidget(); wl = QVBoxLayout(w)
            entries = [("CEST", (0.12, 0.4, 0.85), p['cest'], '-o')]
            if p['wassr'] is not None:
                entries.append(("WASSR", (0.90, 0.49, 0.13), p['wassr'], '-s'))
            wl.addWidget(_spec_canvas(entries, f"{p['name']} — Z-spectrum"))
            lines = []
            if p['cest']:
                lines.append(f"CEST minimum @ {p['cest']['off_min']:+.2f} ppm  "
                             f"{'✓' if p['cest']['ok'] else '⚠ should be 0 ppm'}")
            if p['wassr'] is not None:
                lines.append(f"WASSR minimum @ {p['wassr']['off_min']:+.2f} ppm  "
                             f"{'✓' if p['wassr']['ok'] else '⚠ should be 0 ppm'}")
            lab = QLabel("\n".join(lines)); lab.setStyleSheet("font-size:11px; font-weight:bold;")
            wl.addWidget(lab)
            tabs.addTab(w, p['name'])

        br = QHBoxLayout(); br.addStretch()
        bc = QPushButton("Close"); bc.clicked.connect(dlg.accept); br.addWidget(bc)
        v.addLayout(br)
        dlg.show()

    def _show_roi_table(self):
        from my_gui.roi_table_dialog import show_roi_table
        rois = getattr(self, '_last_rois', [])
        map_items = []

        # ── Voxelwise analysis results (PV / Gaussian / MPLF / Lor) ─────────
        # For Gaussian and MPLF: only show voxelwise if it is the most recently
        # run source (last-run-wins vs on-demand ROI spectral fit).
        _last_src = getattr(self, '_roi_stats_last_source', {})
        if self._results:
            pools = self._results.get("pools", self.POOL_NAMES)
            for prefix, method_lbl in (
                ("pv_",    "PV"),
                ("gauss_", "Gaussian"),
                ("lor_",   "Lor"),
            ):
                # Skip Gaussian voxelwise if user ran ROI spectra Gaussian more recently
                if method_lbl == "Gaussian" and _last_src.get("Gaussian") == 'roi_spectra':
                    continue
                for pool in pools:
                    arr = self._results.get(f"{prefix}ampl_{pool}")
                    if arr is not None:
                        sl = arr[:, :, self._cur_slice(arr.shape[2])] if arr.ndim == 3 else arr
                        map_items.append((f"{method_lbl}: {pool}", sl))
            mplf_pools = self._results.get("mplf_pools", [])
            # Skip MPLF voxelwise if user ran ROI spectra MPLF more recently
            if _last_src.get("MPLF") != 'roi_spectra':
                for pool in mplf_pools:
                    arr = self._results.get(f"mplf_ampl_{pool}")
                    if arr is not None:
                        sl = arr[:, :, self._cur_slice(arr.shape[2])] if arr.ndim == 3 else arr
                        map_items.append((f"MPLF: {pool}", sl))
            mtr = self._results.get("mtr_map")
            if mtr is not None:
                sl = mtr[..., self._cur_slice(mtr.shape[2])] if mtr.ndim == 3 else mtr
                map_items.append(("MTR asym", sl))
                # %CEST = MTRasym × 100  (at the ppm the MTR-asym map was computed)
                _mppm = self._results.get("mtr_ppm_used",
                                          self.spin_mtr_ppm.value())
                map_items.append((f"%CEST @{_mppm:.2f}ppm", sl * 100.0))
            rex = self._results.get("mtrrex_map")
            if rex is not None:
                slr = rex[..., self._cur_slice(rex.shape[2])] if rex.ndim == 3 else rex
                _rppm = self._results.get("mtrrex_ppm_used",
                                          self.spin_quick_rex_ppm.value())
                map_items.append((f"MTR_Rex @{_rppm:.2f}ppm", slr))

        # ── B0 map (from WASSR) — per-ROI mean B0 in ppm and Hz ────────────
        _b0 = getattr(self, "_b0_map_ppm", None)
        if _b0 is not None:
            b0sl = _b0[:, :, self._cur_slice(_b0.shape[2])] if _b0.ndim == 3 else _b0
            _larmor = self.spin_larmor_mhz.value()
            map_items.append(("B0 (ppm)", b0sl))
            map_items.append((f"B0 (Hz) [{_larmor:.0f} MHz]", b0sl * _larmor))

        # ── On-demand ROI spectral fit results ────────────────────────────
        fit_cache = getattr(self, '_roi_spectra_fit_cache', {})
        H_c = getattr(self, '_z_img_full', None)
        if H_c is not None:
            _H, _W = H_c.shape[0], H_c.shape[1]
        else:
            _H, _W = 1, 1

        def _scalar_map(roi_vals: dict):
            """Build a 2D map from {roi_name: scalar} by filling each ROI mask."""
            if _H == 1 and _W == 1:
                return None
            arr = np.full((_H, _W), np.nan)
            for roi in rois:
                val = roi_vals.get(roi.name)
                if val is None:
                    continue
                msk = roi.mask
                if msk.shape[:2] != (_H, _W):
                    try:
                        from scipy.ndimage import zoom as _zoom
                        msk = _zoom(msk.astype(float),
                                    (_H / max(msk.shape[0], 1), _W / max(msk.shape[1], 1)),
                                    order=1) > 0.5
                    except Exception:
                        continue
                arr[msk] = float(val)
            return arr if not np.all(np.isnan(arr)) else None

        for fit_method, fit_results in fit_cache.items():
            # Skip on-demand results if voxelwise is the most recently run source
            if fit_method in ('Gaussian', 'MPLF') and _last_src.get(fit_method) == 'voxelwise':
                continue
            if fit_method in ('DROF', 'MPLF'):
                # Collect per-pool peak ΔZ across ROIs
                all_pools = set()
                for res in fit_results.values():
                    if res:
                        all_pools.update(res.get('pools', {}).keys())
                for pn in sorted(all_pools):
                    if pn == 'water':
                        continue
                    roi_vals = {}
                    for roi in rois:
                        res = fit_results.get(roi.name)
                        if res:
                            pc = res.get('pools', {}).get(pn)
                            if pc is not None:
                                roi_vals[roi.name] = float(np.nanmax(pc))
                    m = _scalar_map(roi_vals)
                    if m is not None:
                        map_items.append((f"{fit_method}: {pn}", m))
            elif fit_method == 'Gaussian':
                all_pools = set()
                for res in fit_results.values():
                    if res:
                        all_pools.update(res.get('pool_curves', {}).keys())
                for pn in sorted(all_pools):
                    if pn == 'water':
                        continue
                    roi_vals = {}
                    for roi in rois:
                        res = fit_results.get(roi.name)
                        if res:
                            pc = res.get('pool_curves', {}).get(pn)
                            if pc is not None:
                                roi_vals[roi.name] = float(np.nanmax(pc))
                    m = _scalar_map(roi_vals)
                    if m is not None:
                        map_items.append((f"Gaussian: {pn}", m))
            elif fit_method == 'PLOF':
                all_pools = set()
                for res in fit_results.values():
                    if res:
                        all_pools.update(res.get('pools', {}).keys())
                for pn in sorted(all_pools):
                    roi_vals = {}
                    for roi in rois:
                        res = fit_results.get(roi.name)
                        if res:
                            pp = res.get('pools', {}).get(pn)
                            if pp is not None:
                                roi_vals[roi.name] = float(pp.get('delta_z', 0.0))
                    m = _scalar_map(roi_vals)
                    if m is not None:
                        map_items.append((f"PLOF ΔZ: {pn}", m))

        # ── Fallback: current canvas image ────────────────────────────────
        if not map_items:
            img = self.canvas._img_data
            if img is not None:
                map_items = [("Current Map", img)]

        show_roi_table(self, rois, map_items, title="Z-Spectroscopy — ROI Statistics")

    def _export_figure(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", "zspec_figure",
            FIG_EXPORT_FILTER
        )
        if path:
            save_figure(self.canvas._fig, path, dpi=300)
            self._log(f"Figure saved: {path}")

    def _show_roi_spectra(self):
        """
        ROI Z-Spectra window.

        Tabs: Raw Z | Pseudo-Voigt | Lorentzian | Gaussian | PLOF | DROF
        Features:
          • ROI selection panel — choose which ROIs to display
          • Physics params panel (B0, B1, R1, tsat) for PLOF / DROF
          • Per-tab B0-correction toggle (Raw Z tab)
          • Double-click on any subplot → enlarged single-ROI view
          • On-demand fitting for Gaussian / PLOF / DROF
        """
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox,
            QTabWidget, QWidget as _QW, QLabel as _QL,
            QDoubleSpinBox as _QDSP, QSpinBox as _QSB,
            QGroupBox, QProgressDialog, QSizePolicy,
            QScrollArea as _QSA,
        )
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt import NavigationToolbar2QT
        try:
            from my_gui.fig_theme import apply_fig_dark_theme
        except Exception:
            def apply_fig_dark_theme(_fig, _dark):   # graceful no-op fallback
                return

        # ── Data setup ────────────────────────────────────────────────────
        rois = [r for r in getattr(self, '_last_rois', [])
                if r.name != "Phantom_outline"]
        if not rois:
            QMessageBox.information(self, "No ROIs",
                                    "Draw ROIs in the ROI Manager tab first.")
            return
        if self._ppm is None:
            QMessageBox.information(self, "No Data",
                                    "Load CEST/Z-spec data first.")
            return
        z_raw = getattr(self, '_z_img_full', None)
        if z_raw is None:
            QMessageBox.information(self, "No Z-spec",
                                    "No Z-spectrum image loaded yet.")
            return

        # ── Reuse existing dialog if data/ROIs/pools haven't changed ──────
        # Pool selection is part of the key so changing pools in CEST
        # Processing Options rebuilds the 1/Z sub-tab with the new pools.
        _current_key = (tuple(r.name for r in rois), len(self._ppm),
                        tuple(sorted(getattr(self, '_global_pools', []))))
        _cached_dlg  = getattr(self, '_roi_spectra_dlg', None)
        if (_cached_dlg is not None
                and _current_key == getattr(self, '_roi_spectra_dlg_key', ())
                and not _cached_dlg.isHidden()):
            # Already visible — just raise it to the front
            _cached_dlg.raise_()
            _cached_dlg.activateWindow()
            return
        if (_cached_dlg is not None
                and _current_key == getattr(self, '_roi_spectra_dlg_key', ())):
            # Same data, just hidden — restore it without rebuilding
            _cached_dlg.show()
            _cached_dlg.raise_()
            _cached_dlg.activateWindow()
            return
        # Data or ROIs changed (or first open) — fall through to build fresh dialog

        ppm    = self._ppm
        m0     = self._get_m0()
        b0_map = getattr(self, '_b0_map_ppm', None)

        if m0 is not None and z_raw.ndim == 4:
            z_img = z_raw / (m0[:, :, :, np.newaxis] + 1e-9)
        else:
            z_img = z_raw
        if z_img.ndim == 4:
            z_img = z_img[:, :, self._cur_slice(z_img.shape[2]), :]
        if z_img.ndim == 2:
            z_img = z_img[:, :, np.newaxis]

        H, W, nPPM = z_img.shape[0], z_img.shape[1], z_img.shape[-1]

        COLORS = {
            'raw':           (0.6,  0.6,  0.6),
            'water':         (0.18, 0.63, 0.18),
            'NOE':           (0.09, 0.75, 0.81),
            'MT':            (0.12, 0.47, 0.71),
            'amide':         (0.84, 0.15, 0.16),
            'amine':         (0.75, 0.10, 0.75),
            'OH':            (0.84, 0.15, 0.16),
            'guanidinium':   (0.58, 0.40, 0.74),
            'trp':           (0.55, 0.34, 0.29),
            'ppm7pt3':       (0.89, 0.47, 0.76),
            'ppm9pt8':       (0.50, 0.50, 0.10),
            'ppm4pt4':       (0.94, 0.50, 0.18),
            'poly_l_lysine': (0.70, 0.20, 0.50),
            'glucose':       (0.20, 0.70, 0.20),
            'creatine':      (0.90, 0.60, 0.10),
            'taurine':       (0.30, 0.30, 0.90),
            'iopamidol_4.2': (0.60, 0.20, 0.80),
            'iopamidol_5.5': (0.10, 0.60, 0.80),
            'sum':           (0.0,  0.0,  0.0),
        }
        _DISPLAY_NAMES = {
            'water': 'Water', 'NOE': 'NOE', 'MT': 'MT',
            'amide': 'Amide', 'amine': 'Amine', 'OH': 'OH',
            'trp': 'Trp', 'guanidinium': 'Guan.',
            'ppm4pt4': '4.4 ppm',
            'ppm7pt3': '7.3 ppm', 'ppm9pt8': '9.8 ppm',
            'poly_l_lysine': 'Poly-Lys', 'glucose': 'Glucose',
            'creatine': 'Creatine', 'taurine': 'Taurine',
            'iopamidol_4.2': 'Iop 4.2', 'iopamidol_5.5': 'Iop 5.5',
        }

        _results      = getattr(self, '_results', {}) or {}
        _fitted_pools = _results.get("pools", self.POOL_NAMES)
        POOL_ORDER    = [
            (p, _DISPLAY_NAMES.get(p, p.upper() if p in ('NOE', 'MT', 'OH') else p.capitalize()))
            for p in _fitted_pools
        ] + [('sum', 'Sum')]

        def _get_pool_maps(prefix):
            pm = {}
            for pool, _ in POOL_ORDER[:-1]:
                key = f"{prefix}peak_{pool}"
                if key in _results and _results[key] is not None:
                    pm[pool] = np.array(_results[key])
            skey = f"{prefix}peak_sum"
            if skey in _results and _results[skey] is not None:
                pm['sum'] = np.array(_results[skey])
            elif pm:
                # Some fits (e.g. MPLF) don't store a total-fit curve — the
                # worker passes sumv=None.  Reconstruct the "Sum" line as the
                # element-wise sum of the per-pool peak maps so every method's
                # ROI-spectra tab shows a Sum line (matches the plotted pools).
                pm['sum'] = np.sum(list(pm.values()), axis=0)
            return pm

        lor_maps   = _get_pool_maps("lor_")
        pv_maps    = _get_pool_maps("pv_")
        gauss_maps = _get_pool_maps("gauss_")   # voxelwise Gaussian per-pool curves
        mplf_maps  = _get_pool_maps("mplf_")    # voxelwise MPLF per-pool ΔZ curves

        # ── Dialog ────────────────────────────────────────────────────────
        nROI  = len(rois)
        nCols = min(4, nROI)
        nRows = max(1, (nROI + nCols - 1) // nCols)
        dlg   = QDialog(self)
        dlg.setWindowTitle("ROI Z-Spectra")
        dlg.resize(min(nCols * 380 + 40, 1600), nRows * 290 + 340)
        # Keep dialog alive between opens — store on self for reuse
        self._roi_spectra_dlg     = dlg
        self._roi_spectra_dlg_key = _current_key
        vl = QVBoxLayout(dlg)
        vl.setSpacing(4)

        # ── ROI selection panel ───────────────────────────────────────────
        roi_grp = QGroupBox("ROI Selection")
        roi_scroll = _QSA()
        roi_scroll.setWidgetResizable(True)
        roi_scroll.setMaximumHeight(60)
        roi_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        roi_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        roi_inner = _QW()
        roi_hlay  = QHBoxLayout(roi_inner)
        roi_hlay.setContentsMargins(4, 0, 4, 0)
        roi_checks: dict[str, QCheckBox] = {}
        for roi in rois:
            chk = QCheckBox(roi.name)
            chk.setChecked(True)
            roi_checks[roi.name] = chk
            roi_hlay.addWidget(chk)
        roi_hlay.addStretch()
        btn_sel_all  = QPushButton("All");  btn_sel_all.setFixedWidth(42)
        btn_sel_none = QPushButton("None"); btn_sel_none.setMinimumWidth(65)
        roi_hlay.addWidget(btn_sel_all)
        roi_hlay.addWidget(btn_sel_none)
        roi_scroll.setWidget(roi_inner)
        roi_grp_lay = QHBoxLayout(roi_grp)
        roi_grp_lay.addWidget(roi_scroll)
        # Shared black-background toggle for every spectra figure (for slides).
        chk_dark_bg = QCheckBox("Bg")
        chk_dark_bg.setToolTip(
            "Black background for the figures (for slides). Only the surround\n"
            "and labels flip — the plotted spectra/curves stay identical.")
        roi_grp_lay.addWidget(chk_dark_bg)
        vl.addWidget(roi_grp)

        def _bg_on():
            try:
                return chk_dark_bg.isChecked()
            except Exception:
                return False

        # The Sum/total-fit line is black by default — invisible on a black
        # background.  When "Bg" is on it is drawn in this yellow instead.
        _SUM_DARK = '#ffe000'

        def _sum_color(default):
            return _SUM_DARK if _bg_on() else default

        def _get_sel_rois():
            return [r for r in rois if roi_checks.get(r.name,
                    type('_', (), {'isChecked': lambda s: True})()).isChecked()]

        # ── All-tab redraw callbacks ──────────────────────────────────────
        _redraw_all_cbs: list = []

        # Registry of every tab's figure holder {label: [fig]} so a single
        # "Save" can bundle ALL tabs when exporting to .mat/.npz.
        _all_tab_figs: dict = {}

        def _save_tab_fig(this_fh, default_name, title):
            """Save one tab's figure — but for .mat/.npz bundle EVERY tab's data
            into one file; image formats save only this tab (unchanged)."""
            import os
            from my_gui.fig_export import (save_figure, save_figures,
                                           FIG_EXPORT_FILTER)
            p, _ = QFileDialog.getSaveFileName(dlg, title, default_name,
                                               FIG_EXPORT_FILTER)
            if not p:
                return
            if os.path.splitext(p)[1].lower() in ('.mat', '.npz'):
                save_figures({lbl: fh[0] for lbl, fh in _all_tab_figs.items()
                              if fh[0] is not None}, p)
            elif this_fh[0] is not None:
                save_figure(this_fh[0], p, dpi=300)

        # ── Shared helpers ────────────────────────────────────────────────
        def _roi_meanspec(roi):
            """Return (raw_mean, raw_std, b0corr_mean) for one ROI."""
            msk = roi.mask
            if msk.shape != (H, W):
                from scipy.ndimage import zoom as _zoom
                msk = _zoom(msk.astype(float),
                            (H / max(msk.shape[0], 1),
                             W / max(msk.shape[1], 1)), order=1) > 0.5
            fl    = z_img.reshape(-1, nPPM)
            mfl   = msk.ravel()
            if mfl.sum() == 0:
                return None, None, None
            raw_mean = fl[mfl].mean(axis=0)
            raw_std  = fl[mfl].std(axis=0)
            if b0_map is not None:
                # Rescale the B0 map to the z-image grid if resolutions differ.
                _b0 = np.asarray(b0_map, dtype=float)
                if _b0.ndim == 3:
                    _b0 = _b0[:, :, self._cur_slice(_b0.shape[2])]
                if _b0.shape[:2] != (H, W):
                    from scipy.ndimage import zoom as _zoom
                    _b0 = _zoom(_b0, (H / max(_b0.shape[0], 1),
                                      W / max(_b0.shape[1], 1)), order=1)
                # Correct EACH voxel by its OWN B0 (matches MATLAB B0correction.m
                # and the analysis worker), then average — correcting the mean
                # spectrum by the mean B0 leaves the dip off-centre because
                # averaging voxels with different B0 broadens/shifts it.
                try:
                    from my_gui.zspec_processing import b0_correction
                    b0vox = _b0.ravel()[mfl]
                    corr  = b0_correction(b0vox, ppm, fl[mfl])
                    b0c   = np.nanmean(corr, axis=0)
                except Exception:
                    b0c = raw_mean.copy()
            else:
                b0c = raw_mean.copy()
            return raw_mean, raw_std, b0c

        def _add_dblclick_zoom(canvas, axes_ref, roi_ref, draw_single_fn):
            """Double-click a subplot → enlarged single-ROI dialog."""
            def _handler(event):
                if not getattr(event, 'dblclick', False):
                    return
                axs = axes_ref[0]
                rls = roi_ref[0]
                if axs is None or rls is None:
                    return
                for i, ax in enumerate(axs):
                    if ax == event.inaxes and i < len(rls):
                        _open_enlarged(rls[i], draw_single_fn)
                        return
            canvas.mpl_connect('button_press_event', _handler)

        def _open_enlarged(roi, draw_single_fn):
            sub = QDialog(dlg)
            sub.setWindowTitle(f"ROI: {roi.name}")
            sub.resize(720, 520)
            sv = QVBoxLayout(sub)
            fig = Figure(figsize=(8, 5.5), facecolor='white')
            ax  = fig.add_subplot(1, 1, 1)
            draw_single_fn(ax, roi, large=True)
            fig.tight_layout()
            try:
                apply_fig_dark_theme(fig, _bg_on())
            except Exception:
                pass
            cv = FigureCanvas(fig)
            tb = NavigationToolbar2QT(cv, sub)
            sv.addWidget(tb)
            sv.addWidget(cv, stretch=1)
            br = QHBoxLayout()
            br.addStretch()
            bc = QPushButton("Close"); bc.clicked.connect(sub.accept)
            br.addWidget(bc)
            sv.addLayout(br)
            sub.show()

        from matplotlib.font_manager import FontProperties as _FPz
        def _fp(combo, size, bold=False):
            """FontProperties honouring the chosen family + size (+ bold for titles)."""
            ff = combo.currentText() if combo is not None else "Default"
            kw = {'size': size}
            if ff and ff.lower() != "default":
                kw['family'] = ff
            if bold:
                kw['weight'] = 'bold'
            return _FPz(**kw)

        def _make_ctrl_row(tab_vl, with_b0chk=False):
            """Shared controls row: x-axis range, legend, font sizes."""
            row = QHBoxLayout()
            if with_b0chk:
                chk_b0 = QCheckBox("Apply B0 correction")
                chk_b0.setChecked(False)          # always start unchecked
                chk_b0.setEnabled(b0_map is not None)
                chk_b0.setToolTip("Shift each ROI spectrum by its mean B0 offset\n"
                                  "(requires a loaded WASSR B0 map)")
                row.addWidget(chk_b0)
            else:
                chk_b0 = None
            row.addWidget(_QL("  x-axis: from"))
            sp_xmax = _QDSP(); sp_xmax.setRange(0.5, 25); sp_xmax.setValue(ppm.max() + 0.5)
            sp_xmax.setSuffix(" ppm"); sp_xmax.setSingleStep(0.5); sp_xmax.setFixedWidth(90)
            row.addWidget(sp_xmax)
            row.addWidget(_QL("to"))
            sp_xmin = _QDSP(); sp_xmin.setRange(-25, -0.5); sp_xmin.setValue(ppm.min() - 0.5)
            sp_xmin.setSuffix(" ppm"); sp_xmin.setSingleStep(0.5); sp_xmin.setFixedWidth(90)
            row.addWidget(sp_xmin)
            chk_leg = QCheckBox("Show legend"); chk_leg.setChecked(True)
            row.addWidget(chk_leg)
            row.addWidget(_QL("  Font:"))
            combo_ff = QComboBox(); combo_ff.setFixedWidth(120)
            combo_ff.setToolTip("Font family for titles, axis labels and legends")
            for _ff in ("Default", "Arial", "Times New Roman",
                        "Helvetica", "DejaVu Sans", "DejaVu Serif"):
                combo_ff.addItem(_ff)
            row.addWidget(combo_ff)
            row.addWidget(_QL("Title:"))
            sp_tfs = _QSB(); sp_tfs.setRange(4, 40); sp_tfs.setValue(13); sp_tfs.setFixedWidth(50)
            sp_tfs.setToolTip("Subplot title font size"); row.addWidget(sp_tfs)
            row.addWidget(_QL("Main:"))
            sp_main = _QSB(); sp_main.setRange(4, 40); sp_main.setValue(14); sp_main.setFixedWidth(50)
            sp_main.setToolTip("Overall figure title font size"); row.addWidget(sp_main)
            row.addWidget(_QL("Axes:"))
            sp_axes = _QSB(); sp_axes.setRange(4, 32); sp_axes.setValue(10); sp_axes.setFixedWidth(50)
            sp_axes.setToolTip("X / Y axis label font size"); row.addWidget(sp_axes)
            row.addWidget(_QL("Tick:"))
            sp_tkfs = _QSB(); sp_tkfs.setRange(4, 24); sp_tkfs.setValue(10); sp_tkfs.setFixedWidth(50)
            sp_tkfs.setToolTip("Tick label font size"); row.addWidget(sp_tkfs)
            row.addWidget(_QL("Legend:"))
            sp_legend = _QSB(); sp_legend.setRange(4, 24); sp_legend.setValue(9); sp_legend.setFixedWidth(50)
            sp_legend.setToolTip("Legend font size"); row.addWidget(sp_legend)
            row.addStretch()
            tab_vl.addLayout(row)
            # The new controls piggyback on sp_tfs' existing redraw wiring: every tab
            # connects sp_tfs.valueChanged → its own redraw, so re-emitting that signal
            # rebuilds the figure without editing each tab's wiring block.
            def _bump(*_):
                sp_tfs.valueChanged.emit(sp_tfs.value())
            for _w in (sp_main, sp_axes, sp_legend):
                _w.valueChanged.connect(_bump)
            combo_ff.currentIndexChanged.connect(_bump)
            return (chk_b0, sp_xmax, sp_xmin, chk_leg, sp_tfs, sp_tkfs,
                    sp_main, sp_axes, sp_legend, combo_ff)

        def _ax_decor(ax, roi, sp_xmax, sp_xmin, sp_tfs, sp_tkfs, chk_leg,
                      sp_axes=None, sp_legend=None, combo_ff=None):
            ax.set_xlim([sp_xmax.value(), sp_xmin.value()])
            ax.set_ylim([-0.05, 1.10])
            _afs = sp_axes.value() if sp_axes is not None else sp_tkfs.value()
            _lfs = sp_legend.value() if sp_legend is not None else max(7, sp_tkfs.value() - 1)
            ax.set_xlabel('Δω (ppm)', fontproperties=_fp(combo_ff, _afs), labelpad=4)
            ax.set_ylabel('Z(Δω)',    fontproperties=_fp(combo_ff, _afs))
            ax.set_title(roi.name,    fontproperties=_fp(combo_ff, sp_tfs.value(), bold=True),
                         pad=4)
            ax.tick_params(labelsize=sp_tkfs.value())
            if chk_leg.isChecked() and ax.get_legend_handles_labels()[0]:
                leg = ax.legend(prop=_fp(combo_ff, _lfs),
                                loc='upper right', frameon=True,
                                framealpha=0.85, edgecolor='gray')
                leg.set_draggable(True)

        def _rebuild_canvas(tab_vl, canvas_h, axes_h, roi_h,
                            fig_h, draw_all_fn, draw_single_fn):
            """Replace existing canvas with a fresh one."""
            from PyQt6.QtWidgets import QSizePolicy as _SP
            old = canvas_h[0]
            fig, axes = draw_all_fn()
            if fig is None:
                return
            fig_h[0]    = fig
            axes_h[0]   = axes
            roi_h[0]    = _get_sel_rois()
            try:
                apply_fig_dark_theme(fig, _bg_on())
            except Exception:
                pass
            new_c = FigureCanvas(fig)
            new_c.setSizePolicy(_SP.Policy.Expanding, _SP.Policy.Expanding)
            canvas_h[0] = new_c
            if old is not None:
                # replaceWidget doesn't preserve stretch — remove + reinsert
                idx = tab_vl.indexOf(old)
                tab_vl.removeWidget(old)
                old.deleteLater()
                tab_vl.insertWidget(idx, new_c, stretch=1)
            else:
                # Insert before the last layout item (save row)
                tab_vl.insertWidget(tab_vl.count() - 1, new_c, stretch=1)
            _add_dblclick_zoom(new_c, axes_h, roi_h, draw_single_fn)

        def _add_toolbar(tab_vl, canvas_h, toolbar_h, parent_w):
            """Create/replace NavigationToolbar2QT above the current canvas."""
            from matplotlib.backends.backend_qt import NavigationToolbar2QT as _NTBA
            if toolbar_h[0] is not None:
                tab_vl.removeWidget(toolbar_h[0])
                toolbar_h[0].deleteLater()
                toolbar_h[0] = None
            if canvas_h[0] is not None:
                tb = _NTBA(canvas_h[0], parent_w)
                toolbar_h[0] = tb
                c_idx = tab_vl.indexOf(canvas_h[0])
                if c_idx >= 0:
                    tab_vl.insertWidget(c_idx, tb)

        # ── RAW Z TAB ─────────────────────────────────────────────────────
        def _make_raw_tab():
            tab   = _QW()
            tab_vl = QVBoxLayout(tab)

            chk_b0, sp_xmax, sp_xmin, chk_leg, sp_tfs, sp_tkfs, \
                sp_main, sp_axes, sp_legend, combo_ff = \
                _make_ctrl_row(tab_vl, with_b0chk=True)

            canvas_h = [None]; axes_h = [None]; roi_h = [None]; fig_h = [None]

            def _draw_single(ax, roi, large=False):
                raw_mean, raw_std, b0c = _roi_meanspec(roi)
                ax.set_facecolor('#f8f8f8')
                if raw_mean is None:
                    ax.text(0.5, 0.5, 'No pixels', ha='center', va='center',
                            transform=ax.transAxes)
                    return
                use_data = b0c if (chk_b0 is not None and chk_b0.isChecked()) else raw_mean
                valid    = np.isfinite(use_data)
                lbl = 'Raw Z (B0 corr.)' if (chk_b0 and chk_b0.isChecked()) else 'Raw Z'
                ax.plot(ppm[valid], use_data[valid], 'o', markersize=3,
                        color=COLORS['raw'], markerfacecolor=COLORS['raw'],
                        markeredgewidth=0, label=lbl)
                if chk_b0 and chk_b0.isChecked() and b0_map is not None:
                    ax.plot(ppm, raw_mean, '--', lw=0.8, color='#aaa',
                            label='Raw Z (uncorr.)', alpha=0.55)
                _ax_decor(ax, roi, sp_xmax, sp_xmin, sp_tfs, sp_tkfs, chk_leg, sp_axes, sp_legend, combo_ff)

            def _draw_all():
                sel = _get_sel_rois()
                if not sel:
                    return None, []
                nC  = min(4, len(sel)); nR = max(1, (len(sel)+nC-1)//nC)
                fig = Figure(figsize=(nC*5, nR*4), facecolor='white')
                axes = []
                for i, roi in enumerate(sel):
                    ax = fig.add_subplot(nR, nC, i+1)
                    _draw_single(ax, roi)
                    axes.append(ax)
                tag = '(B0 corr.)' if (chk_b0 and chk_b0.isChecked()) else '(raw)'
                fig.suptitle(f'Raw Z  {tag}'.strip(),
                             fontproperties=_fp(combo_ff, sp_main.value(), bold=True))
                fig.tight_layout(pad=1.2, h_pad=2.8, w_pad=1.5, rect=[0,0,1,0.95])
                return fig, axes

            _rebuild_canvas(tab_vl, canvas_h, axes_h, roi_h, fig_h,
                            _draw_all, _draw_single)

            def _redraw():
                _rebuild_canvas(tab_vl, canvas_h, axes_h, roi_h, fig_h,
                                _draw_all, _draw_single)

            _redraw_all_cbs.append(_redraw)
            if chk_b0:  chk_b0.toggled.connect(lambda _: _redraw())
            sp_xmax.valueChanged.connect(lambda _: _redraw())
            sp_xmin.valueChanged.connect(lambda _: _redraw())
            chk_leg.toggled.connect(lambda _: _redraw())
            sp_tfs.valueChanged.connect(lambda _: _redraw())
            sp_tkfs.valueChanged.connect(lambda _: _redraw())

            sv_row = QHBoxLayout()
            _btn_verify = QPushButton("Verify Offsets…")
            _btn_verify.setToolTip(
                "Check the offset ordering: plots mean phantom signal vs assigned\n"
                "offset. The Z-spectrum minimum should fall at 0 ppm (water). If it\n"
                "doesn't, the loaded offsets are mislabeled/misordered.")
            _btn_verify.clicked.connect(self._verify_offsets)
            sv_row.addWidget(_btn_verify)
            _all_tab_figs['Raw Z'] = fig_h
            btn_sv = QPushButton("Save Raw Z figure…")
            def _sv(c=False, _fh=fig_h):
                _save_tab_fig(_fh, "roi_rawZ.png", "Save Raw Z")
            btn_sv.clicked.connect(_sv)
            sv_row.addWidget(btn_sv); sv_row.addStretch()
            tab_vl.addLayout(sv_row)
            return tab

        # ── PRE-COMPUTED TAB (PV / Lorentzian) ───────────────────────────
        def _make_precomp_tab(label, pool_maps):
            tab    = _QW()
            tab_vl = QVBoxLayout(tab)

            pk_row = QHBoxLayout()
            pk_row.addWidget(_QL("Show pools:"))
            checks: dict[str, QCheckBox] = {}
            # Build display-label map from POOL_ORDER (analysis results)
            _pool_label_map = {pn: dl for pn, dl in POOL_ORDER}
            # Iterate exactly what the user selected in "Select pools to fit"
            for pname in self._global_pools:
                dlbl = _pool_label_map.get(
                    pname,
                    pname.upper() if pname in ('NOE', 'MT', 'OH')
                    else pname.replace('_', ' ').capitalize()
                )
                ck = QCheckBox(dlbl)
                ck.setChecked(pname in pool_maps)
                checks[pname] = ck
                pk_row.addWidget(ck)
            # Always include composite Sum entry
            ck_sum = QCheckBox("Sum")
            ck_sum.setChecked('sum' in pool_maps)
            checks['sum'] = ck_sum
            pk_row.addWidget(ck_sum)
            pk_row.addStretch()
            tab_vl.addLayout(pk_row)

            _, sp_xmax, sp_xmin, chk_leg, sp_tfs, sp_tkfs, \
                sp_main, sp_axes, sp_legend, combo_ff = \
                _make_ctrl_row(tab_vl, with_b0chk=False)

            canvas_h = [None]; axes_h = [None]; roi_h = [None]; fig_h = [None]
            toolbar_h = [None]

            def _draw_single(ax, roi, large=False):
                msk = roi.mask
                if msk.shape != (H, W):
                    from scipy.ndimage import zoom as _zm
                    msk = _zm(msk.astype(float),
                              (H/max(msk.shape[0],1), W/max(msk.shape[1],1)), order=1) > 0.5
                fl  = z_img.reshape(-1, nPPM)
                mfl = msk.ravel()
                ax.set_facecolor('#f8f8f8')
                if mfl.sum() == 0:
                    ax.text(0.5,0.5,'No pixels',ha='center',va='center',
                            transform=ax.transAxes); return
                rmean = fl[mfl].mean(axis=0)
                rstd  = fl[mfl].std(axis=0)
                ax.plot(ppm, rmean, 'o', markersize=3,
                        color=COLORS['raw'], markerfacecolor=COLORS['raw'],
                        markeredgewidth=0, label='Raw Z')
                visible = {k for k,c in checks.items() if c.isChecked()}
                for pname, dlbl in POOL_ORDER:
                    if pname not in visible: continue
                    pm = pool_maps.get(pname)
                    if pm is None: continue
                    pm_arr = np.array(pm)
                    if pm_arr.ndim == 4: pm_arr = pm_arr[:,:,0,:]
                    if pm_arr.ndim != 3: continue
                    if pm_arr.shape[:2] != (H, W):
                        try:
                            from scipy.ndimage import zoom as _zm2
                            pm_arr = _zm2(pm_arr.astype(float),
                                         (H/pm_arr.shape[0], W/pm_arr.shape[1], 1), order=1)
                        except Exception:
                            continue
                    pm_fl = pm_arr.reshape(-1, pm_arr.shape[-1])
                    if pm_fl.shape[0] != mfl.shape[0]: continue
                    pm_m = pm_fl[mfl].mean(axis=0)
                    lw   = 2.0 if pname == 'sum' else 1.4
                    _pcol = COLORS.get(pname, (0.4, 0.4, 0.4))
                    if pname == 'sum':
                        _pcol = _sum_color(_pcol)   # yellow when Bg is on
                    ax.plot(ppm, pm_m, '-', color=_pcol, lw=lw, label=dlbl)
                _ax_decor(ax, roi, sp_xmax, sp_xmin, sp_tfs, sp_tkfs, chk_leg, sp_axes, sp_legend, combo_ff)

            def _draw_all():
                sel = _get_sel_rois()
                if not sel: return None, []
                nC = min(4,len(sel)); nR = max(1,(len(sel)+nC-1)//nC)
                fig = Figure(figsize=(nC*5, nR*4), facecolor='white')
                axes = []
                for i, roi in enumerate(sel):
                    ax = fig.add_subplot(nR, nC, i+1)
                    _draw_single(ax, roi); axes.append(ax)
                fig.suptitle(label,
                             fontproperties=_fp(combo_ff, sp_main.value(), bold=True))
                fig.tight_layout(pad=1.2, h_pad=2.8, w_pad=1.5, rect=[0,0,1,0.95])
                return fig, axes

            _rebuild_canvas(tab_vl, canvas_h, axes_h, roi_h, fig_h,
                            _draw_all, _draw_single)
            _add_toolbar(tab_vl, canvas_h, toolbar_h, tab)

            def _redraw():
                _rebuild_canvas(tab_vl, canvas_h, axes_h, roi_h, fig_h,
                                _draw_all, _draw_single)
                _add_toolbar(tab_vl, canvas_h, toolbar_h, tab)

            _redraw_all_cbs.append(_redraw)
            for ck in checks.values(): ck.toggled.connect(lambda _: _redraw())

            sp_xmax.valueChanged.connect(lambda _: _redraw())
            sp_xmin.valueChanged.connect(lambda _: _redraw())
            chk_leg.toggled.connect(lambda _: _redraw())
            sp_tfs.valueChanged.connect(lambda _: _redraw())
            sp_tkfs.valueChanged.connect(lambda _: _redraw())

            sv_row = QHBoxLayout()
            _all_tab_figs[label] = fig_h
            btn_sv = QPushButton(f"Save {label} figure…")
            def _sv(c=False, _fh=fig_h, _lb=label):
                _save_tab_fig(_fh, f"roi_{_lb.lower().replace(' ','_')}.png",
                              f"Save {_lb}")
            btn_sv.clicked.connect(_sv)
            sv_row.addWidget(btn_sv); sv_row.addStretch()
            tab_vl.addLayout(sv_row)
            return tab

        # ── ON-DEMAND FITTING TAB (Gaussian / PLOF / DROF) ───────────────
        def _make_fitted_tab(method: str):
            tab    = _QW()
            tab_vl = QVBoxLayout(tab)

            _, sp_xmax, sp_xmin, chk_leg, sp_tfs, sp_tkfs, \
                sp_main, sp_axes, sp_legend, combo_ff = \
                _make_ctrl_row(tab_vl, with_b0chk=False)

            btn_run = QPushButton(
                f"{method} Processing && Fitting" if method in ('PLOF', 'DROF')
                else f"▶  Run {method} Fitting"
            )
            btn_run.setStyleSheet(
                "QPushButton{background:#2563eb;color:white;font-weight:bold;"
                "border-radius:4px;padding:4px 14px;}"
                "QPushButton:hover{background:#1d4ed8;}"
            )
            btn_run.setToolTip(
                f"Compute {method} fitting on the mean spectrum of each selected ROI.\n"
                + ("Uses B₀, B₁, R₁, tsat parameters from the physics panel below."
                   if method in ('PLOF', 'DROF') else
                   "Gaussian = full 6-parameter Pseudo-Voigt model (high quality for averaged ROI spectra).\n"
                   "Voxelwise maps use a faster 3-parameter model.")
                if method == 'Gaussian' else
                ("Uses B₀, B₁, R₁, tsat parameters from the physics panel below."
                 if method in ('PLOF', 'DROF') else "")
            )

            # Physics params block — only for PLOF / DROF
            if method in ('PLOF', 'DROF'):
                _phys_row = QHBoxLayout()
                _phys_row.addWidget(_QL("B₀:"))
                _spin_B0 = _QDSP(); _spin_B0.setRange(1, 1200)
                _spin_B0.setValue(self.spin_larmor_mhz.value())
                _spin_B0.setSuffix(" MHz"); _spin_B0.setDecimals(1); _spin_B0.setFixedWidth(90)
                _spin_B0.setToolTip("Proton Larmor frequency in MHz")
                _phys_row.addWidget(_spin_B0)
                _phys_row.addWidget(_QL("  B₁:"))
                _spin_B1 = _QDSP(); _spin_B1.setRange(0.01, 50); _spin_B1.setValue(2.5)
                _spin_B1.setSuffix(" µT"); _spin_B1.setDecimals(2); _spin_B1.setFixedWidth(80)
                _spin_B1.setToolTip("Saturation B1 power in µT")
                _phys_row.addWidget(_spin_B1)
                _phys_row.addWidget(_QL("  R₁:"))
                _spin_R1 = _QDSP(); _spin_R1.setRange(0.01, 10); _spin_R1.setValue(0.5)
                _spin_R1.setSuffix(" s⁻¹"); _spin_R1.setDecimals(3); _spin_R1.setFixedWidth(85)
                _spin_R1.setToolTip("Water R₁ = 1/T₁  (s⁻¹)")
                _phys_row.addWidget(_spin_R1)
                _tsat_lbl = _QL("  t<sub>sat</sub>:")
                _tsat_lbl.setTextFormat(Qt.TextFormat.RichText)
                _phys_row.addWidget(_tsat_lbl)
                _spin_tsat = _QDSP(); _spin_tsat.setRange(0.01, 20); _spin_tsat.setValue(2.0)
                _spin_tsat.setSuffix(" s"); _spin_tsat.setDecimals(2); _spin_tsat.setFixedWidth(75)
                _spin_tsat.setToolTip("Saturation pulse duration in seconds")
                _phys_row.addWidget(_spin_tsat)
                _phys_row.addStretch()
                tab_vl.addLayout(_phys_row)
            else:
                # Placeholders so _run_fitting closure doesn't need method checks for every access
                _spin_B0 = _spin_B1 = _spin_R1 = _spin_tsat = None

            # Insert run button into the ctrl row (last item before stretch)
            # Actually, the ctrl row is already built; add button after it.
            btn_row = QHBoxLayout()
            btn_row.addWidget(btn_run)
            btn_row.addStretch()
            tab_vl.addLayout(btn_row)

            fit_results: dict[str, dict | None] = {}
            canvas_h = [None]; axes_h = [None]; roi_h = [None]; fig_h = [None]
            toolbar_h = [None]

            DROF_COLORS = {
                'water':       '#2ca02c',
                'amide':       '#d62728',
                'NOE':         '#1f77b4',
                'MT':          '#ff7f0e',
                'guanidinium': '#9467bd',
                'amine':       '#e377c2',
                'OH':          '#17becf',
                'glucose':     '#bcbd22',
                'creatine':    '#8c564b',
                'taurine':     '#7f7f7f',
                'trp':         '#8c4a2f',
                'ppm7pt3':     '#e24bc9',
                'ppm9pt8':     '#808000',
            }

            # Pool selection comes from the global "Select pools to fit" button
            # (in Processing Options on the main panel) — nothing to build here.
            drof_pool_chks: dict = {}
            mplf_pool_chks: dict = {}

            # ── Show-pool visibility checkboxes (populated after fitting) ─
            pool_show_grp = QGroupBox("Show Pools")
            pool_show_grp.setStyleSheet(
                "QGroupBox{font-weight:bold;font-size:11px;"
                "border:1px solid #888;border-radius:4px;margin-top:15px;"
                "padding:6px 6px 4px 6px;}"
                "QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;"
                "left:8px;top:-1px;padding:0 4px;font-size:11px;}"
            )
            pool_show_lay = QHBoxLayout(pool_show_grp)
            pool_show_lay.setSpacing(8)
            pool_show_lay.setContentsMargins(6, 2, 6, 2)
            pool_vis_chks: dict = {}
            _pvs_info = _QL("  Click  ▶ Run Fitting  to populate…")
            _pvs_info.setStyleSheet("color:#888;font-size:11px;")
            pool_show_lay.addWidget(_pvs_info)
            pool_show_lay.addStretch()
            tab_vl.addWidget(pool_show_grp)

            def _update_vis_chks(pool_names):
                # Clear existing widgets from pool_show_lay
                while pool_show_lay.count():
                    it = pool_show_lay.takeAt(0)
                    if it.widget():
                        it.widget().deleteLater()
                pool_vis_chks.clear()
                if not pool_names:
                    _lbl = _QL("  No pools")
                    pool_show_lay.addWidget(_lbl)
                    pool_show_lay.addStretch()
                    return
                for _pn in pool_names:
                    _cb = QCheckBox(_pn)
                    _cb.setChecked(True)
                    _cb.toggled.connect(lambda _, _r=None: _refresh() if fig_h[0] is not None else None)
                    pool_vis_chks[_pn] = _cb
                    pool_show_lay.addWidget(_cb)
                _bta = QPushButton("All");  _bta.setFixedWidth(42)
                _btn = QPushButton("None"); _btn.setMinimumWidth(58)
                def _all_v():
                    for _c in pool_vis_chks.values(): _c.setChecked(True)
                def _non_v():
                    for _c in pool_vis_chks.values(): _c.setChecked(False)
                _bta.clicked.connect(_all_v)
                _btn.clicked.connect(_non_v)
                pool_show_lay.addStretch()
                pool_show_lay.addWidget(_bta)
                pool_show_lay.addWidget(_btn)

            placeholder = _QL(
                f"Click  ▶ Run {method} Fitting  to compute fits for all selected ROIs."
                + ("\n(Uses B₀, B₁, R₁, tsat values from the physics row above.)"
                   if method in ('PLOF', 'DROF') else "")
            )
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color:#777;font-size:12px;padding:40px;")
            tab_vl.addWidget(placeholder, stretch=1)
            _ph_gone = [False]   # mutable flag: True once placeholder removed from layout

            def _draw_single(ax, roi, large=False):
                raw_mean, raw_std, _ = _roi_meanspec(roi)
                ax.set_facecolor('#f8f8f8')
                if raw_mean is None:
                    ax.text(0.5,0.5,'No pixels',ha='center',va='center',
                            transform=ax.transAxes); return
                ax.plot(ppm, raw_mean, 'o', markersize=3,
                        color=COLORS['raw'], markerfacecolor=COLORS['raw'],
                        markeredgewidth=0, label='Raw Z')
                res = fit_results.get(roi.name)
                if res:
                    # Draw total fit / sum curve — toggled by "Sum" checkbox
                    _sum_visible = (not pool_vis_chks or
                                    pool_vis_chks.get('Sum',
                                        type('_cb', (), {'isChecked': lambda s: True})()
                                    ).isChecked())
                    if 'fit' in res and _sum_visible:
                        ax.plot(ppm, res['fit'], '-',
                                color=_sum_color('#1a1a1a'), lw=2.0,
                                label='Sum')
                    if method == 'PLOF':
                        pool_res_dict = res.get('pools', {})
                        _bg_ref = None
                        for _pn, _pres in pool_res_dict.items():
                            if _bg_ref is None:
                                _bg_ref = _pres['bg']
                        # Plot background as dashed gray (water Lorentzian)
                        if _bg_ref is not None:
                            if not pool_vis_chks or pool_vis_chks.get('bg', type('_', (), {'isChecked': lambda s: True})()).isChecked():
                                ax.plot(ppm, _bg_ref, '--', color='#888', lw=1.0, alpha=0.7, label='Background')
                        for _pn, _pres in pool_res_dict.items():
                            if _pn in pool_vis_chks and not pool_vis_chks[_pn].isChecked():
                                continue
                            _delta = np.clip(_pres['bg'] - _pres['fit'], 0, None)
                            _c = DROF_COLORS.get(_pn, COLORS.get(_pn, '#888'))
                            ax.plot(ppm, _delta, '-', color=_c, lw=1.4, label=_pn)
                    elif method in ('DROF', 'MPLF'):
                        for pn, pcurve in res.get('pools', {}).items():
                            if pn in pool_vis_chks and not pool_vis_chks[pn].isChecked():
                                continue
                            c = DROF_COLORS.get(pn, '#888')
                            ax.plot(ppm, pcurve, '-', color=c, lw=1.4, label=pn)
                    elif method == 'Gaussian':
                        for pn, pcurve in res.get('pool_curves', {}).items():
                            if pn in pool_vis_chks and not pool_vis_chks[pn].isChecked():
                                continue
                            c = COLORS.get(pn, '#888')
                            ax.plot(ppm, pcurve, '-', color=c, lw=1.4,
                                    label=pn)
                _ax_decor(ax, roi, sp_xmax, sp_xmin, sp_tfs, sp_tkfs, chk_leg, sp_axes, sp_legend, combo_ff)

            def _draw_all():
                sel = _get_sel_rois()
                if not sel: return None, []
                nC = min(4,len(sel)); nR = max(1,(len(sel)+nC-1)//nC)
                fig = Figure(figsize=(nC*5, nR*4), facecolor='white')
                axes = []
                for i, roi in enumerate(sel):
                    ax = fig.add_subplot(nR, nC, i+1)
                    _draw_single(ax, roi); axes.append(ax)
                fig.suptitle(method,
                             fontproperties=_fp(combo_ff, sp_main.value(), bold=True))
                fig.tight_layout(pad=1.2, h_pad=2.8, w_pad=1.5, rect=[0,0,1,0.95])
                return fig, axes

            def _refresh():
                if not _ph_gone[0]:
                    placeholder.hide()
                    placeholder.setParent(None)
                    _ph_gone[0] = True
                _rebuild_canvas(tab_vl, canvas_h, axes_h, roi_h, fig_h,
                                _draw_all, _draw_single)
                _add_toolbar(tab_vl, canvas_h, toolbar_h, tab)

            def _run_fitting():
                sel = _get_sel_rois()
                if not sel:
                    return
                prog = QProgressDialog(
                    f"Running {method} fitting…", "Cancel", 0, len(sel), dlg)
                prog.setWindowTitle(f"{method} Fitting")
                prog.setWindowModality(Qt.WindowModality.WindowModal)
                prog.show()

                for k, roi in enumerate(sel):
                    if prog.wasCanceled():
                        break
                    prog.setValue(k)
                    QApplication.processEvents()
                    raw_mean, _, _ = _roi_meanspec(roi)
                    if raw_mean is None:
                        fit_results[roi.name] = None
                        continue
                    z_spec = np.clip(raw_mean, 0.0, 1.0)
                    try:
                        if method == 'Gaussian':
                            from my_gui.zspec_processing import fit_zspec_single
                            # Use global pool selection (same as MPLF/DROF) so that
                            # newly-added pools (ppm7pt3, trp, …) appear here too
                            pools = list(self._global_pools)
                            # Use full 6-param Pseudo-Voigt model for single-spectrum quality
                            # (voxelwise maps use the faster 3-param model; ROI spectra
                            #  deserve the highest quality fit on averaged data)
                            params, indiv, sumcurve = fit_zspec_single(
                                ppm, z_spec, pools=pools, peak_type='Pseudo-Voigt',
                                ftol=1e-8, max_nfev=1200,
                            )
                            fit_results[roi.name] = {
                                'fit':         1.0 - sumcurve,
                                'pool_curves': indiv,
                            }
                        elif method == 'PLOF':
                            from my_gui.zspec_processing import fit_zspec_plof
                            _PLOF_POOL_PPM = {
                                'amide': 3.5, 'amine': 3.0, 'NOE': -3.5, 'MT': -2.5,
                                'guanidinium': 2.0, 'OH': 0.8, 'glucose': 1.2,
                                'creatine': 1.9, 'taurine': 3.2,
                                'iopamidol_4.2': 4.2, 'iopamidol_5.5': 5.5,
                                'trp': 5.4, 'ppm7pt3': 7.5, 'ppm9pt8': 10.0,
                            }
                            pool_results = {}
                            for _pool in self._global_pools:
                                if _pool == 'water':
                                    continue
                                _pppm = _PLOF_POOL_PPM.get(_pool)
                                if _pppm is None:
                                    continue
                                try:
                                    _pres = fit_zspec_plof(
                                        ppm, z_spec,
                                        satpwr_uT=_spin_B1.value(),
                                        B0_MHz=_spin_B0.value(),
                                        R1=_spin_R1.value(),
                                        tsat=_spin_tsat.value(),
                                        peak_ppm=_pppm,
                                    )
                                    pool_results[_pool] = _pres
                                except Exception as _exc:
                                    self._log(f"PLOF fit error ({roi.name}, {_pool}): {_exc}")
                            fit_results[roi.name] = {'pools': pool_results}
                        elif method == 'DROF':
                            from my_gui.zspec_processing import fit_zspec_drof, DROF_POOL_CATALOG as _DPC
                            _sel_pools = [p for p in self._global_pools if p in _DPC] or None
                            res = fit_zspec_drof(
                                ppm, z_spec,
                                satpwr_uT=_spin_B1.value(),
                                B0_MHz=_spin_B0.value(),
                                R1=_spin_R1.value(),
                                tsat=_spin_tsat.value(),
                                pools=_sel_pools,
                            )
                            fit_results[roi.name] = res
                        elif method == 'MPLF':
                            from my_gui.zspec_processing import fit_zspec_mplf, MPLF_POOL_CATALOG as _MPC
                            _sel_pools = [p for p in self._global_pools if p in _MPC] or None
                            res = fit_zspec_mplf(
                                ppm, z_spec,
                                pools=_sel_pools,
                                n_restarts=3,   # single spectrum: use full restarts for quality
                            )
                            fit_results[roi.name] = res
                    except Exception as exc:
                        fit_results[roi.name] = None
                        self._log(f"{method} fit error ({roi.name}): {exc}")

                prog.setValue(len(sel))
                # Cache results on the tab so _show_roi_table can access them
                if not hasattr(self, '_roi_spectra_fit_cache'):
                    self._roi_spectra_fit_cache = {}
                self._roi_spectra_fit_cache[method] = dict(fit_results)
                # Track that this on-demand fit is now the latest source for this method
                self._roi_stats_last_source[method] = 'roi_spectra'
                # Build 2-D scalar maps and push them into the display dropdown
                self._update_roi_spectral_maps(method, fit_results)
                # Populate show-pool checkboxes from first successful result
                _fr = next((r for r in fit_results.values() if r is not None), None)
                if _fr is not None:
                    if method in ('DROF', 'MPLF'):
                        _pnames = list(_fr.get('pools', {}).keys())
                        # Add "Sum" toggle for the total-fit line
                        if 'fit' in _fr:
                            _pnames = ['Sum'] + _pnames
                    elif method == 'Gaussian':
                        _pnames = ['Sum'] + list(_fr.get('pool_curves', {}).keys())
                    elif method == 'PLOF':
                        _pnames = ['bg'] + list(_fr.get('pools', {}).keys())
                    else:
                        _pnames = []
                    _update_vis_chks(_pnames)
                _refresh()

            btn_run.clicked.connect(_run_fitting)

            # ── Auto-show voxelwise results if already cached ─────────────
            # After running voxelwise analysis, Gaussian/MPLF results are
            # pre-populated into _roi_spectra_fit_cache. Show them immediately
            # without requiring the user to click "Run Fit" again.
            _pre_cached = getattr(self, '_roi_spectra_fit_cache', {}).get(method)
            if _pre_cached:
                for _rn, _rres in _pre_cached.items():
                    fit_results[_rn] = _rres

                def _auto_show(_m=method):
                    _fr = next((r for r in fit_results.values()
                                if r is not None), None)
                    if _fr is not None:
                        if _m in ('DROF', 'MPLF'):
                            _pnames = list(_fr.get('pools', {}).keys())
                        elif _m == 'Gaussian':
                            _pnames = list(_fr.get('pool_curves', {}).keys())
                        elif _m == 'PLOF':
                            _pnames = ['bg'] + list(_fr.get('pools', {}).keys())
                        else:
                            _pnames = []
                        _update_vis_chks(_pnames)
                    _refresh()

                from PyQt6.QtCore import QTimer
                QTimer.singleShot(150, _auto_show)

            def _redraw():
                if fig_h[0] is not None:
                    _refresh()

            _redraw_all_cbs.append(_redraw)
            sp_xmax.valueChanged.connect(lambda _: _redraw())
            sp_xmin.valueChanged.connect(lambda _: _redraw())
            chk_leg.toggled.connect(lambda _: _redraw())
            sp_tfs.valueChanged.connect(lambda _: _redraw())
            sp_tkfs.valueChanged.connect(lambda _: _redraw())

            sv_row = QHBoxLayout()
            _all_tab_figs[method] = fig_h
            btn_sv = QPushButton(f"Save {method} figure…")
            def _sv(c=False, _fh=fig_h, _m=method):
                _save_tab_fig(_fh, f"roi_{_m.lower()}.png", f"Save {_m}")
            btn_sv.clicked.connect(_sv)
            sv_row.addWidget(btn_sv); sv_row.addStretch()
            tab_vl.addLayout(sv_row)
            return tab

        # ── 1/Z sub-tab — full MT+CEST fitting pipeline per ROI ──────────
        def _make_inv_z_tab() -> _QW:
            """
            Self-contained 1/Z CEST analysis sub-tab embedded inside the
            ROI Spectra dialog.  Uses the same CEST data already loaded in
            the CEST Analysis tab.  Physics code imported from inv_zspec_tab.
            """
            from my_gui.tabs.inv_zspec_tab import (
                _run_pipeline, POOL_COLORS as _PLC,
                POOL_NAMES as _PNS_ALL, _pfn as _pfn1z, _GAMMA_HZ_UT as _GHZUT,
                _L_DEFS as _INVZ_LDEFS, _expand_cest_range as _expand_cest,
                _canonical_invz_pool as _canon_pool,
            )
            import matplotlib.pyplot as _plt1z

            # ── Determine active pools from main tab selection ────────────
            # The 1/Z model can fit any pool with bounds in _L_DEFS (water, OH,
            # amine, amide, NOE, MT, Trp, 4.4/7.3/9.8 ppm, glucose, creatine,
            # taurine, iopamidol, 3-OMG, poly-L-lysine, GAG, myo-inositol…).
            _1Z_SUPPORTED = tuple(_INVZ_LDEFS.keys())
            _global = getattr(self, '_global_pools',
                              ['water', 'amide', 'NOE', 'MT', 'amine', 'OH'])
            # Fit water (MT baseline) + every globally-selected pool the model
            # supports — so pools chosen in CEST Processing Options carry over.
            # DROF/CEST-MRI keys are mapped to 1/Z keys (e.g. ppm7pt3 → 7.3ppm).
            _active_pools_1z = ['water']
            for _gp in _global:
                _cp = _canon_pool(_gp)
                if _cp != 'water' and _cp in _1Z_SUPPORTED and _cp not in _active_pools_1z:
                    _active_pools_1z.append(_cp)

            # Initial scan params — seed B1 from the loaded scan and R1 from the
            # T1 map if available (otherwise sensible defaults the user can edit).
            _init_b1 = float(getattr(self, '_cest_satpwr_uT', 2.5) or 2.5)
            _init_r1 = 1.0
            try:
                _r1fn = getattr(self, '_inv_r1_getter', None)
                if _r1fn is not None:
                    _rv = _r1fn()
                    if _rv and _rv > 0:
                        _init_r1 = float(_rv)
            except Exception:
                pass

            # Store current params in a mutable container so the dialog can update them
            _params_store = [dict(
                satpwr_uT        = _init_b1,
                B0_MHz           = self.spin_larmor_mhz.value(),
                R1               = _init_r1,
                ppm_exclude_MT   = (-8.0, 8.0),
                ppm_reinclude_MT = (-0.5, 0.5),
                ppm_include_CEST = (-2.0, 5.0),
                peak_type        = 'lorentzian',
                fit_mt           = True,
            )]

            widget = _QW()
            vl_1z  = QVBoxLayout(widget)
            vl_1z.setSpacing(4)
            vl_1z.setContentsMargins(6, 6, 6, 6)

            # ── Top control row: Processing & Fitting button + pool checkboxes ─
            top_row = QHBoxLayout()

            # Processing & Fitting button
            _btn_pf = QPushButton("Inverse Z Processing && Fitting")
            _btn_pf.setFixedHeight(30)
            _btn_pf.setStyleSheet(
                "QPushButton{background:#37474f;color:white;font-size:11px;"
                "border:none;border-radius:4px;padding:3px 10px;}"
                "QPushButton:hover{background:#546e7a;}"
            )
            top_row.addWidget(_btn_pf)
            top_row.addStretch()
            vl_1z.addLayout(top_row)

            def _open_pf_dialog():
                """Open the Processing & Fitting settings dialog."""
                from PyQt6.QtWidgets import QDialog, QDialogButtonBox
                _d = QDialog(dlg)
                _d.setWindowTitle("1/Z — Processing & Fitting Settings")
                _d.setMinimumWidth(520)
                _dv = QVBoxLayout(_d)

                # Scan Parameters
                _sg = QGroupBox("Scan Parameters")
                _sl = QHBoxLayout(_sg)
                _sl.addWidget(_QL("B1 (µT):"))
                _sb1 = _QDSP(); _sb1.setRange(0.01, 50.0)
                _sb1.setValue(_params_store[0]['satpwr_uT']); _sb1.setDecimals(2)
                _sb1.setFixedWidth(75)
                _sl.addWidget(_sb1)
                _sl.addWidget(_QL("  B0 (MHz):"))
                _sb0 = _QDSP(); _sb0.setRange(50.0, 1500.0)
                _sb0.setValue(_params_store[0]['B0_MHz']); _sb0.setDecimals(1)
                _sb0.setFixedWidth(85)
                _sl.addWidget(_sb0)
                _sl.addWidget(_QL("  R1 (s⁻¹):"))
                _sr1 = _QDSP(); _sr1.setRange(0.01, 20.0)
                _sr1.setValue(_params_store[0]['R1']); _sr1.setDecimals(3)
                _sr1.setSingleStep(0.05); _sr1.setFixedWidth(80)
                _sl.addWidget(_sr1)
                _bt1 = QPushButton("From T1 map")
                _bt1.setFixedWidth(110)
                def _fill_r1_d():
                    fn = getattr(self, '_inv_r1_getter', None)
                    if fn is not None:
                        v = fn()
                        if v and v > 0:
                            _sr1.setValue(float(v))
                    else:
                        QMessageBox.information(_d, "T1 map",
                            "No T1 map available.\nEnter R1 manually.")
                _bt1.clicked.connect(_fill_r1_d)
                _sl.addWidget(_bt1)
                _sl.addStretch()
                _dv.addWidget(_sg)

                # MT Fitting — the group title is a checkbox ("MT Fit").  When
                # unchecked the MT background is treated as flat (Z_MT = 1) and
                # the MT-window parameters are hidden.
                _mg = QGroupBox("MT Fit")
                _mg.setCheckable(True)
                _mg.setChecked(bool(_params_store[0].get('fit_mt', True)))
                _mg.setToolTip(
                    "When checked, the MT/direct-saturation background is fitted "
                    "from the far-offset wings and subtracted before the CEST fit.\n"
                    "Uncheck to skip it (treat MT as flat, Z_MT = 1) — e.g. for a "
                    "phantom with no semisolid pool.")
                _ml = QHBoxLayout(_mg)
                _ml.addWidget(_QL("Exclude MT Fit (ppm)"))
                _smex = _QDSP(); _smex.setRange(1.0, 20.0)
                _smex.setValue(_params_store[0]['ppm_exclude_MT'][1])
                _smex.setSuffix(" ppm"); _smex.setFixedWidth(80)
                _ml.addWidget(_smex)
                _ml.addWidget(_QL("   Water"))
                _smrn = _QDSP(); _smrn.setRange(0.0, 3.0)
                _smrn.setValue(_params_store[0]['ppm_reinclude_MT'][1])
                _smrn.setSuffix(" ppm"); _smrn.setFixedWidth(75)
                _ml.addWidget(_smrn)
                _ml.addStretch()
                # Hide the MT-window controls when MT fitting is off.
                def _sync_mt_vis(_on):
                    for _i in range(_ml.count()):
                        _w = _ml.itemAt(_i).widget()
                        if _w is not None:
                            _w.setVisible(_on)
                _mg.toggled.connect(_sync_mt_vis)
                _sync_mt_vis(_mg.isChecked())
                _dv.addWidget(_mg)

                # CEST Fitting
                _cg = QGroupBox("CEST Fitting")
                _cl = QHBoxLayout(_cg)
                _cl.addWidget(_QL("Fit range:"))
                _sclo = _QDSP(); _sclo.setRange(-20.0, 0.0)
                _sclo.setValue(_params_store[0]['ppm_include_CEST'][0])
                _sclo.setSuffix(" ppm"); _sclo.setFixedWidth(80)
                _cl.addWidget(_sclo)
                _cl.addWidget(_QL("to"))
                _schi = _QDSP(); _schi.setRange(0.0, 20.0)
                _schi.setValue(_params_store[0]['ppm_include_CEST'][1])
                _schi.setSuffix(" ppm"); _schi.setFixedWidth(80)
                _cl.addWidget(_schi)
                _cl.addWidget(_QL("   Peak type:"))
                _cpk = QComboBox()
                _cpk.addItems(["Lorentzian", "Pseudo-Voigt"])
                _cpk.setCurrentText(
                    "Pseudo-Voigt"
                    if _params_store[0]['peak_type'] == 'pseudovoigt'
                    else "Lorentzian"
                )
                _cl.addWidget(_cpk)
                _cl.addStretch()
                _dv.addWidget(_cg)

                # OK / Cancel
                _bb = QDialogButtonBox(
                    QDialogButtonBox.StandardButton.Ok |
                    QDialogButtonBox.StandardButton.Cancel
                )
                def _accept():
                    _params_store[0] = dict(
                        satpwr_uT        = _sb1.value(),
                        B0_MHz           = _sb0.value(),
                        R1               = _sr1.value(),
                        ppm_exclude_MT   = (-_smex.value(), _smex.value()),
                        ppm_reinclude_MT = (-_smrn.value(), _smrn.value()),
                        ppm_include_CEST = (_sclo.value(), _schi.value()),
                        peak_type = ('pseudovoigt'
                                     if _cpk.currentText() == 'Pseudo-Voigt'
                                     else 'lorentzian'),
                        fit_mt = bool(_mg.isChecked()),
                    )
                    _d.accept()
                _bb.accepted.connect(_accept)
                _bb.rejected.connect(_d.reject)
                _dv.addWidget(_bb)
                if _d.exec() == QDialog.DialogCode.Accepted:
                    # The new B1 / R1 / MT-exclude / CEST-range values are now in
                    # _params_store. Re-fit immediately so they take visible effect
                    # (otherwise the displayed result is stale and looks ignored).
                    if _1z_results_store:
                        _run_1z()

            _btn_pf.clicked.connect(_open_pf_dialog)

            # ── Pool visibility checkboxes ────────────────────────────────
            _pool_vis_grp = QGroupBox("Show Pools")
            _pool_vis_grp.setStyleSheet(
                "QGroupBox{font-weight:bold;font-size:11px;"
                "border:1px solid #888;border-radius:4px;margin-top:15px;"
                "padding:6px 6px 4px 6px;}"
                "QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;"
                "left:8px;top:-1px;padding:0 4px;font-size:11px;}"
            )
            _pv_lay = QHBoxLayout(_pool_vis_grp)
            _pv_lay.setSpacing(8)
            _pv_lay.setContentsMargins(6, 2, 6, 2)
            _pv_chks: dict = {}
            _PLC_EXT = dict(_PLC)
            _PLC_EXT.setdefault('amide', '#ff7f0e')
            # Fixed entries (raw data + summed fit) + one checkbox per active pool
            # (water, MT and every CEST pool selected in CEST Processing Options).
            _vis_items = [('raw', 'Raw', '#888888'), ('sum', 'Sum', '#1a1a1a')]
            for _pk in _active_pools_1z:
                _vis_items.append((_pk, _pk, _PLC_EXT.get(_pk, '#9aa0a6')))
            for _pk, _pl, _pc in _vis_items:
                _chkb = QCheckBox(_pl)
                _chkb.setChecked(True)
                _chkb.setStyleSheet(
                    f"QCheckBox::indicator:checked {{background:{_pc};"
                    f"border:2px solid {_pc};border-radius:3px;}}"
                )
                _pv_chks[_pk] = _chkb
                _pv_lay.addWidget(_chkb)
            _pv_lay.addStretch()
            vl_1z.addWidget(_pool_vis_grp)

            # ── Run / status row ──────────────────────────────────────────
            run_row = QHBoxLayout()
            _btn_run1z = QPushButton("▶  Run 1/Z Analysis")
            _btn_run1z.setFixedHeight(34)
            _btn_run1z.setStyleSheet(
                "QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                "stop:0 #1565c0,stop:1 #00897b);color:white;font-size:12px;"
                "font-weight:bold;border:none;border-radius:6px;padding:4px 12px;}"
                "QPushButton:hover{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                "stop:0 #1976d2,stop:1 #00acc1);}"
            )
            run_row.addWidget(_btn_run1z, stretch=1)
            _lbl_1z_st = _QL("Click ▶ Run 1/Z Analysis to compute MT + CEST fitting per ROI.")
            _lbl_1z_st.setStyleSheet("font-size:11px; color:gray;")
            run_row.addWidget(_lbl_1z_st, stretch=2)
            vl_1z.addLayout(run_row)

            # ── 1/Z font controls (mirror the main ROI-spectra tabs) ──────
            _fz_row = QHBoxLayout()
            _fz_row.addWidget(_QL("Font:"))
            _1z_ff = QComboBox(); _1z_ff.setFixedWidth(120)
            _1z_ff.setToolTip("Font family for titles, axis labels and legends")
            for _ff in ("Default", "Arial", "Times New Roman",
                        "Helvetica", "DejaVu Sans", "DejaVu Serif"):
                _1z_ff.addItem(_ff)
            _fz_row.addWidget(_1z_ff)
            def _1z_fs(lbl, val, tip):
                _fz_row.addWidget(_QL(lbl))
                _sp = _QSB(); _sp.setRange(4, 40); _sp.setValue(val); _sp.setFixedWidth(50)
                _sp.setToolTip(tip); _fz_row.addWidget(_sp); return _sp
            _1z_title = _1z_fs("Title:", 10, "Subplot title font size")
            _1z_main  = _1z_fs("Main:",  11, "Overall figure title font size")
            _1z_axes  = _1z_fs("Axes:",   9, "X / Y axis label font size")
            _1z_tick  = _1z_fs("Tick:",   8, "Tick label font size")
            _1z_leg   = _1z_fs("Legend:", 7, "Legend font size")
            _fz_row.addStretch()
            vl_1z.addLayout(_fz_row)

            # ── Per-ROI result tabs (populated after Run) ─────────────────
            _roi_tab_w = QTabWidget()
            _ph_1z     = _QL("<center><br/><b>No results yet.</b><br/>"
                             "Click <b>▶ Run 1/Z Analysis</b> to compute the "
                             "MT + CEST fitting pipeline per ROI.</center>")
            _roi_tab_w.addTab(_ph_1z, "—")
            vl_1z.addWidget(_roi_tab_w, stretch=1)
            _ph_1z_gone = [False]

            # ── Drawing helper ────────────────────────────────────────────
            def _get_vis() -> set:
                return {k for k, c in _pv_chks.items() if c.isChecked()}

            def _draw_1z_figure(rname, res_list, clo, chi, vis):
                # Panels mirror the Inverse-Z tab: "Z-spectrum + MT fit" ONLY when
                # MT is fitted (otherwise Z_MT=1 is flat and useless), then the
                # "CEST Fitting in 1/Z" decomposition, then the reconstructed
                # "Z-spectrum" fit.  Colours match the Inverse-Z single-B1 view
                # (data/curves tab10-blue, summed fit black).
                _fit_mt = bool(_params_store[0].get('fit_mt', True))
                _panel_keys = (['zmt'] if _fit_mt else []) + ['cest', 'zfit']
                _titles  = {'zmt': 'Z-spectrum  +  MT fit',
                            'cest': 'CEST  Fitting  in  1/Z',
                            'zfit': 'Z-spectrum'}
                _ylabels = {'zmt': 'Z',
                            'cest': r'$R_1\cos^2\theta\,(1/Z-1)$',
                            'zfit': 'Z'}
                n_p  = len(_panel_keys)
                n_e  = max(len(res_list), 1)
                _tc  = _plt1z.cm.get_cmap('tab10', max(n_e, 2))
                _rcols = [_tc(i) for i in range(n_e)]
                _single = sum(1 for r in res_list if r is not None) == 1
                _ZBLUE  = _plt1z.cm.get_cmap('tab10', 10)(0)

                fig1z = Figure(figsize=(6.0 * n_p, 4.5), facecolor='white')
                _axmap = {}
                for _i, _pk in enumerate(_panel_keys):
                    _axmap[_pk] = fig1z.add_subplot(1, n_p, _i + 1)
                fig1z.suptitle(f"ROI: {rname} — 1/Z Analysis",
                               fontproperties=_fp(_1z_ff, _1z_main.value(), bold=True))

                for _bi, _res in enumerate(res_list):
                    if _res is None:
                        continue
                    _col     = _ZBLUE if _single else _rcols[_bi]
                    _ppm_s   = _res['ppm']
                    _z_s     = _res['zspec']
                    _mt_Z    = _res['mt_Z']
                    _fmask   = _res['mt_fit_mask']
                    _ppm_fit = _res['cest_ppm_fit']
                    _target  = _res['cest_target']
                    _mt_c2   = _res['cest_mt_cos2']
                    _R1v     = _res['R1']
                    _spwr    = _res['satpwr_uT']
                    _b0mhz   = _res['B0_MHz']
                    _ap      = _res.get('active_pools', _active_pools_1z)

                    _pfine = np.linspace(_ppm_s.min(), _ppm_s.max(), 600)
                    _sHz   = _spwr * _GHZUT
                    _c2f   = (_pfine * _b0mhz)**2 / (
                              (_pfine * _b0mhz)**2 + _sHz**2)
                    _mZf   = np.interp(_pfine, _ppm_s, _mt_Z)
                    _iMTf  = _R1v * (1.0 / np.clip(_mZf, 1e-6, None) - 1.0)
                    _mc2f  = _iMTf * _c2f

                    _cmf   = (_pfine >= clo) & (_pfine <= chi)
                    _ppfc  = _pfine[_cmf]
                    _c2fc  = _c2f[_cmf]
                    _pfnf  = _pfn1z(_res['peak_type'])
                    _coeff = _res['cest_coeffs']
                    _indff: dict = {}
                    for _nm in _ap:
                        if _nm in _coeff:
                            _indff[_nm] = _c2fc * _pfnf(_coeff[_nm], _ppfc)
                    _totf = _mc2f[_cmf] + sum(
                        _indff.get(_nm, np.zeros_like(_ppfc)) for _nm in _ap)

                    # Reconstructed Z-spectrum (same formula as the Inverse-Z tab)
                    _rcos2f = _iMTf * _c2f
                    for _nm in _ap:
                        if _nm in _coeff:
                            _rcos2f = _rcos2f + _c2f * _pfnf(_coeff[_nm], _pfine)
                    _zfitf = np.clip(_R1v * _c2f / (_rcos2f + _R1v * _c2f + 1e-20),
                                     0.0, 1.0)

                    # Panel — Z-spectrum + MT fit overlay (only when MT fitted)
                    if 'zmt' in _axmap:
                        _ax = _axmap['zmt']
                        if 'raw' in vis:
                            _ax.scatter(_ppm_s, _z_s, c=[_col],
                                        s=12, alpha=0.35, zorder=3)
                            _ax.scatter(_ppm_s[_fmask], _z_s[_fmask], c=[_col],
                                        s=18, alpha=0.75, zorder=4,
                                        label=f"B1={_spwr:.2f}µT")
                        _ax.plot(_pfine, _mZf, '-', color=_col, lw=1.8, zorder=5)

                    # Panel — 1/Z CEST decomposition
                    _ax = _axmap['cest']
                    if 'raw' in vis:
                        _raw1z = _target + _mt_c2
                        _ax.scatter(_ppm_fit, _raw1z, c=[_col],
                                    s=12, alpha=0.6, zorder=3)
                    if 'MT' in vis:
                        _ax.plot(_pfine, _mc2f, '-', color=_PLC_EXT.get('MT','#1f77b4'),
                                 lw=1.5, alpha=0.85, label='MT')
                    for _nm in _ap:
                        # Draw every active CEST pool (MT is drawn separately above)
                        if _nm != 'MT' and _nm in vis and _nm in _indff:
                            _cl = _PLC_EXT.get(_nm, '#888')
                            _ax.plot(_ppfc, _indff[_nm], '-', color=_cl,
                                     lw=1.8, alpha=0.9, label=_nm, zorder=5)
                    if 'sum' in vis:
                        _ax.plot(_ppfc, _totf, '-', color=_sum_color('k'),
                                 lw=2.0, alpha=0.9 if _bg_on() else 0.6,
                                 label='sum', zorder=6)

                    # Panel — reconstructed Z-spectrum fit
                    _ax = _axmap['zfit']
                    if 'raw' in vis:
                        _ax.scatter(_ppm_s, _z_s, c=[_col], s=12, alpha=0.75,
                                    zorder=3)
                    _ax.plot(_pfine, _zfitf, '-', color=_col, lw=2,
                             label=f"B1={_spwr:.2f}µT", zorder=5)

                # Axes decoration
                for _pk, _ax in _axmap.items():
                    _ax.set_facecolor('#f9f9f9')
                    _ax.set_xlabel('Δω  (ppm)', fontproperties=_fp(_1z_ff, _1z_axes.value()))
                    _ax.set_ylabel(_ylabels[_pk], fontproperties=_fp(_1z_ff, _1z_axes.value()))
                    _ax.set_title(_titles[_pk], fontproperties=_fp(_1z_ff, _1z_title.value(), bold=True))
                    _ax.tick_params(labelsize=_1z_tick.value())
                    _ax.grid(True, alpha=0.2, ls='--')
                    _vals = [r['ppm'] for r in res_list if r is not None]
                    if _vals:
                        _all_p = np.concatenate(_vals)
                        _ax.set_xlim(_all_p.max() + 0.3, _all_p.min() - 0.3)
                    if _pk in ('zmt', 'zfit'):
                        _ax.set_ylim(0, 1.05)
                    if _pk == 'cest':
                        _ax.set_xlim(chi + 0.5, clo - 0.5)
                    _hh, _ll = _ax.get_legend_handles_labels()
                    if _hh:
                        _seen: dict = {}
                        for _h, _l in zip(_hh, _ll):
                            if _l not in _seen:
                                _seen[_l] = _h
                        _ax.legend(_seen.values(), _seen.keys(), prop=_fp(_1z_ff, _1z_leg.value()),
                                   loc='upper right', framealpha=0.85)

                fig1z.tight_layout(pad=1.2, rect=[0, 0, 1, 0.93])
                try:
                    apply_fig_dark_theme(fig1z, _bg_on())
                except Exception:
                    pass
                return fig1z

            # Store per-ROI results for redraw
            _1z_results_store: dict = {}   # {roi_name: res_dict_or_None}
            _1z_tab_refs: dict = {}        # {roi_name: [fig, canvas, toolbar, inner_widget, layout]}

            def _redraw_1z():
                """Redraw all 1/Z ROI tabs with current pool visibility."""
                if not _1z_results_store:
                    return
                vis = _get_vis()
                p   = _params_store[0]
                # Widen the plot window to show off-resonance pools (e.g. 7.3 ppm)
                clo, chi = _expand_cest(p['ppm_include_CEST'], _active_pools_1z)
                for _rn, _res1z in _1z_results_store.items():
                    if _res1z is None or _rn not in _1z_tab_refs:
                        continue
                    _, old_cv, old_tb, _iw, _il = _1z_tab_refs[_rn]
                    from matplotlib.backends.backend_qt import NavigationToolbar2QT as _NTBz
                    new_fig = _draw_1z_figure(_rn, [_res1z], clo, chi, vis)
                    new_cv  = FigureCanvas(new_fig)
                    new_cv.setMinimumHeight(360)
                    new_tb  = _NTBz(new_cv, _iw)
                    _il.replaceWidget(old_tb, new_tb)
                    _il.replaceWidget(old_cv, new_cv)
                    old_tb.deleteLater(); old_cv.deleteLater()
                    _1z_tab_refs[_rn][0] = new_fig
                    _1z_tab_refs[_rn][1] = new_cv
                    _1z_tab_refs[_rn][2] = new_tb

            for _chkv in _pv_chks.values():
                _chkv.toggled.connect(lambda _: _redraw_1z())
            # Font controls re-render the 1/Z figures (debounced so holding a spin
            # arrow coalesces into one redraw).
            from PyQt6.QtCore import QTimer as _QTimer1z
            _1z_font_timer = _QTimer1z(_roi_tab_w)
            _1z_font_timer.setSingleShot(True); _1z_font_timer.setInterval(180)
            _1z_font_timer.timeout.connect(_redraw_1z)
            def _1z_font_bump(*_):
                _1z_font_timer.start()
            for _w in (_1z_title, _1z_main, _1z_axes, _1z_tick, _1z_leg):
                _w.valueChanged.connect(_1z_font_bump)
            _1z_ff.currentIndexChanged.connect(_1z_font_bump)
            # Register 1/Z redraw so the shared "Bg" toggle re-themes it too
            # (no-op until a 1/Z fit has been run — guarded by empty-store check).
            _redraw_all_cbs.append(_redraw_1z)

            # ── Run button handler ────────────────────────────────────────
            def _run_1z():
                p    = _params_store[0]
                # Widen the plot window to show off-resonance pools (e.g. 7.3 ppm)
                clo, chi = _expand_cest(p['ppm_include_CEST'], _active_pools_1z)
                vis  = _get_vis()

                _lbl_1z_st.setText("Running 1/Z fitting…")
                _lbl_1z_st.setStyleSheet("font-size:11px; color:#1565c0;")
                _btn_run1z.setEnabled(False)
                QApplication.processEvents()

                # Remove placeholder, clear old tabs
                if not _ph_1z_gone[0]:
                    _roi_tab_w.clear()
                    _ph_1z_gone[0] = True
                else:
                    _roi_tab_w.clear()
                _1z_results_store.clear()
                _1z_tab_refs.clear()

                for _roi in rois:
                    _raw, _, _ = _roi_meanspec(_roi)
                    if _raw is None:
                        continue
                    try:
                        _res1z = _run_pipeline(
                            ppm, _raw,
                            satpwr_uT        = p['satpwr_uT'],
                            R1               = p['R1'],
                            B0_MHz           = p['B0_MHz'],
                            peak_type        = p['peak_type'],
                            ppm_exclude_MT   = p['ppm_exclude_MT'],
                            ppm_reinclude_MT = p['ppm_reinclude_MT'],
                            ppm_include_CEST = p['ppm_include_CEST'],
                            pool_names       = tuple(_active_pools_1z),
                            fit_mt           = p.get('fit_mt', True),
                        )
                        # Tag result with active pools for redraw
                        _res1z['active_pools'] = list(_active_pools_1z)
                    except Exception as _exc:
                        _res1z = None

                    _1z_results_store[_roi.name] = _res1z
                    if _res1z is None:
                        continue

                    # Build tab for this ROI
                    _fig1z = _draw_1z_figure(_roi.name, [_res1z], clo, chi, vis)
                    _cv1z  = FigureCanvas(_fig1z)
                    _cv1z.setMinimumHeight(360)
                    from matplotlib.backends.backend_qt import NavigationToolbar2QT as _NTBz
                    _iw1z  = _QW(); _il1z = QVBoxLayout(_iw1z)
                    _tb1z  = _NTBz(_cv1z, _iw1z)
                    _il1z.addWidget(_tb1z)
                    _il1z.addWidget(_cv1z, stretch=1)
                    _sr1z  = QHBoxLayout()
                    def _save1z(c=False, _rn=_roi.name, _refs=_1z_tab_refs):
                        _fg = _refs[_rn][0] if _rn in _refs else None
                        if _fg is None: return
                        _fp, _ = QFileDialog.getSaveFileName(
                            dlg, f"Save — {_rn}", f"1z_{_rn}.png",
                            FIG_EXPORT_FILTER)
                        if _fp:
                            save_figure(_fg, _fp, dpi=300)
                    _sbtn1z = QPushButton(f"Save ({_roi.name})…")
                    _sbtn1z.clicked.connect(_save1z)
                    _sr1z.addWidget(_sbtn1z); _sr1z.addStretch()
                    _il1z.addLayout(_sr1z)
                    _roi_tab_w.addTab(_iw1z, _roi.name)
                    _1z_tab_refs[_roi.name] = [_fig1z, _cv1z, _tb1z, _iw1z, _il1z]
                    QApplication.processEvents()

                _n1z = _roi_tab_w.count()
                if _n1z > 0:
                    _lbl_1z_st.setText(
                        f"Done — {_n1z} ROI(s) fitted.  "
                        f"Pools: {', '.join(_active_pools_1z)}")
                    _lbl_1z_st.setStyleSheet("font-size:11px; color:green;")
                    # Store simple AREX map on self for display combo
                    try:
                        _zv = getattr(self, '_z_img_full', None)
                        _m0v = self._get_m0()
                        if _zv is not None and self._ppm is not None:
                            if _m0v is not None and _zv.ndim == 4:
                                _zv2 = _zv / (_m0v[:, :, :, np.newaxis] + 1e-9)
                            else:
                                _zv2 = _zv
                            _sidx2 = self._cur_slice(_zv2.shape[2]) if _zv2.ndim == 4 else 0
                            _zsl2 = np.clip(_zv2[:, :, _sidx2, :].astype(float),
                                            1e-9, 1.0)
                            _H2, _W2, _N2 = _zsl2.shape
                            _si2  = np.argsort(self._ppm)
                            _pp2  = self._ppm[_si2]
                            _zfl2 = _zsl2.reshape(-1, _N2)[:, _si2]
                            _ep2  = p['ppm_include_CEST'][1]
                            _R12  = p['R1']
                            _Zp2  = np.array([np.interp( _ep2, _pp2, _zfl2[v])
                                              for v in range(_H2 * _W2)])
                            _Zn2  = np.array([np.interp(-_ep2, _pp2, _zfl2[v])
                                              for v in range(_H2 * _W2)])
                            _Zp2  = np.clip(_Zp2, 1e-9, None)
                            _Zn2  = np.clip(_Zn2, 1e-9, None)
                            self._arex_map = apply_analysis_mask(
                                ((1.0 / _Zp2 - 1.0 / _Zn2) * _R12).reshape(_H2, _W2))
                            self._arex_params = dict(
                                eval_ppm=_ep2, R1=_R12,
                                B1_uT=p['satpwr_uT'],
                            )
                    except Exception:
                        pass
                else:
                    _lbl_1z_st.setText("No valid ROI results.")
                    _lbl_1z_st.setStyleSheet("font-size:11px; color:red;")

                _btn_run1z.setEnabled(True)

            _btn_run1z.clicked.connect(_run_1z)
            return widget

        # ── Build tab widget ──────────────────────────────────────────────
        tab_widget = QTabWidget()
        tab_widget.addTab(_make_raw_tab(),                  "Raw Z")
        tab_widget.addTab(_make_precomp_tab("Pseudo-Voigt", pv_maps), "Pseudo-Voigt")
        if lor_maps:
            tab_widget.addTab(_make_precomp_tab("Lorentzian", lor_maps), "Lorentzian")
        # Gaussian / MPLF: show the voxelwise fit immediately (like Pseudo-Voigt)
        # when the analysis has produced their per-pool curves; otherwise fall
        # back to the on-demand fitting tab.
        tab_widget.addTab(
            _make_precomp_tab("MPLF", mplf_maps) if mplf_maps
            else _make_fitted_tab("MPLF"), "MPLF")
        tab_widget.addTab(
            _make_precomp_tab("Gaussian", gauss_maps) if gauss_maps
            else _make_fitted_tab("Gaussian"), "Gaussian")
        tab_widget.addTab(_make_fitted_tab("PLOF"),     "PLOF")
        tab_widget.addTab(_make_fitted_tab("DROF"),     "DROF")
        tab_widget.addTab(_make_inv_z_tab(),             "1/Z")

        vl.addWidget(tab_widget, stretch=1)

        # ── ROI selection → redraw all tabs ───────────────────────────────
        def _on_roi_check_changed():
            for cb in _redraw_all_cbs:
                try:
                    cb()
                except Exception:
                    pass

        for chk in roi_checks.values():
            chk.toggled.connect(lambda _: _on_roi_check_changed())
        # "Bg" toggle → re-theme every tab's figure via the shared redraw path.
        chk_dark_bg.toggled.connect(lambda _: _on_roi_check_changed())

        def _sel_all():
            for c in roi_checks.values():
                c.blockSignals(True); c.setChecked(True); c.blockSignals(False)
            _on_roi_check_changed()

        def _sel_none():
            for c in roi_checks.values():
                c.blockSignals(True); c.setChecked(False); c.blockSignals(False)
            _on_roi_check_changed()

        btn_sel_all.clicked.connect(_sel_all)
        btn_sel_none.clicked.connect(_sel_none)

        # ── Close ─────────────────────────────────────────────────────────
        # Use hide() instead of accept()/close() so all fitted plots survive
        # close/reopen without the user needing to re-run the fitting.
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.hide)
        btn_row.addWidget(btn_close)
        vl.addLayout(btn_row)

        # Also intercept the window-manager ✕ button — hide rather than destroy.
        def _hide_on_wm_close(event):
            event.ignore()
            dlg.hide()
        dlg.closeEvent = _hide_on_wm_close

        dlg.show()

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _on_m0_selection_changed(self):
        """Called when source combo OR frame combo changes.

        • If source changed  → repopulate the frame list first.
        • Always update the info label.
        • If the canvas is currently showing the M0 display, refresh it live
          so the selected frame's image appears immediately.
        """
        # Only repopulate when the SOURCE combo fires; the frame combo fires
        # after blockSignals(False) inside _populate_m0_frame_combo, so guard
        # against double calls by checking the sender text.
        sender = self.sender()
        if sender is self.combo_m0_source:
            self._populate_m0_frame_combo()
        else:
            self._update_m0_label()
            # Auto-refresh canvas when M0 display mode is active
            try:
                if self.combo_display.currentText() == "M0 image (unsaturated)":
                    self._refresh_display()
            except Exception:
                pass

    def _populate_m0_frame_combo(self):
        """Fill combo_m0_frame with all frames from the selected source dataset."""
        src = self.combo_m0_source.currentText()
        if src == "WASSR":
            ppm_arr  = self._wassr_ppm_all
            auto_idx = self._wassr_m0_auto_idx
            available = self._wassr_img_all is not None
        else:
            ppm_arr  = self._ppm_all
            auto_idx = self._cest_m0_auto_idx
            available = self._z_img_all is not None

        self.combo_m0_frame.blockSignals(True)
        self.combo_m0_frame.clear()

        if not available or ppm_arr is None or len(ppm_arr) == 0:
            self.combo_m0_frame.addItem("— load data first —")
            self.combo_m0_frame.setEnabled(False)
            self.combo_m0_frame.blockSignals(False)
            self._update_m0_label()
            return

        for i, ppm in enumerate(ppm_arr):
            self.combo_m0_frame.addItem(f"Frame {i}:  {ppm:+.2f} ppm")

        self.combo_m0_frame.setCurrentIndex(auto_idx)
        self.combo_m0_frame.setEnabled(True)
        self.combo_m0_frame.blockSignals(False)
        self._update_m0_label()
        # Refresh canvas if M0 display is active
        try:
            if self.combo_display.currentText() == "M0 image (unsaturated)":
                self._refresh_display()
        except Exception:
            pass

    def _update_m0_label(self):
        """Refresh the M0 info label to show currently selected frame."""
        src = self.combo_m0_source.currentText()
        frame_idx = self.combo_m0_frame.currentIndex()

        if src == "WASSR":
            img_all  = self._wassr_img_all
            ppm_arr  = self._wassr_ppm_all
            auto_idx = self._wassr_m0_auto_idx
        else:
            img_all  = self._z_img_all
            ppm_arr  = self._ppm_all
            auto_idx = self._cest_m0_auto_idx

        if img_all is not None and ppm_arr is not None and 0 <= frame_idx < len(ppm_arr):
            ppm_val = ppm_arr[frame_idx]
            sh      = img_all.shape
            self.lbl_m0_info.setText(
                f"{src}  |  Frame {frame_idx}: {ppm_val:+.2f} ppm  "
                f"|  {sh[0]}×{sh[1]}"
                + (f"×{sh[2]}" if img_all.ndim > 3 else "")
            )
            self.lbl_m0_info.setStyleSheet("font-size: 11px; color: #4ec9b0;")
        elif self._M0_img is not None or self._wassr_M0_img is not None:
            parts = []
            if self._M0_img is not None:
                sh = self._M0_img.shape
                parts.append(f"CEST auto-M0 {sh[0]}×{sh[1]}")
            if self._wassr_M0_img is not None:
                sh = self._wassr_M0_img.shape
                parts.append(f"WASSR auto-M0 {sh[0]}×{sh[1]}")
            self.lbl_m0_info.setText("Available: " + ",  ".join(parts))
            self.lbl_m0_info.setStyleSheet("font-size: 11px; color: green;")
        else:
            self.lbl_m0_info.setText("No M0 loaded yet.")
            self.lbl_m0_info.setStyleSheet("font-size: 11px; color: gray;")

    def _open_pools_dialog(self):
        """Open a pool-selection dialog and update self._global_pools."""
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QScrollArea as _SA
        try:
            from my_gui.zspec_processing import DROF_POOL_CATALOG as _DPC
            pool_catalog = list(_DPC.keys())
        except Exception:
            pool_catalog = ["water", "amide", "NOE", "MT", "guanidinium",
                            "amine", "OH", "glucose", "creatine", "taurine",
                            "poly_l_lysine"]
        # Add iopamidol peaks if not already present
        for _ip in ["iopamidol_4.2", "iopamidol_5.5"]:
            if _ip not in pool_catalog:
                pool_catalog.append(_ip)

        dlg = QDialog(self)
        dlg.setWindowTitle("Select pools to fit")
        dlg.setMinimumWidth(360)
        vl = QVBoxLayout(dlg)
        info_lbl = QLabel(
            "<b>Select which pools to fit:</b><br>"
            "<small>Water is always enabled (provides the B₀ shift reference).</small>"
        )
        info_lbl.setTextFormat(Qt.TextFormat.RichText)
        info_lbl.setWordWrap(True)
        vl.addWidget(info_lbl)

        _display_names = {
            "iopamidol_4.2":  "Iopamidol 4.2 ppm",
            "iopamidol_5.5":  "Iopamidol 5.5 ppm",
            "poly_l_lysine":  "Poly-L-Lysine (3.7 ppm)",
            "trp":            "Trp indole NH (5.4 ppm)",
            "ppm7pt3":        "7.3 ppm pool",
            "ppm9pt8":        "9.8 ppm pool",
        }
        _tooltips = {
            "iopamidol_4.2":  "Iopamidol amide proton exchange peak at ~4.2 ppm",
            "iopamidol_5.5":  "Iopamidol amide proton exchange peak at ~5.5 ppm",
            "poly_l_lysine":  "Poly-L-Lysine –NH₃⁺ side-chain amine peak at ~3.7 ppm\n"
                              "Broad (FWHM ~2–4 ppm), fast exchange (~5000–9000 s⁻¹), pH-sensitive.",
            "trp":            "Tryptophan indole N-H proton exchange peak at ~5.4 ppm.\n"
                              "Broad pool (FWHM ~0.5–2 ppm); associated with aromatic sidechain\n"
                              "exchange in peptides/proteins.",
            "ppm7pt3":        "Downfield pool at ~7.3 ppm (aromatic proton / unknown exchange).\n"
                              "Bounds: A 0–0.8, FWHM 0.2–5 ppm, offset 7.0–8.0 ppm.",
            "ppm9pt8":        "Far-downfield pool at ~9.8 ppm (aromatic / exchangeable proton).\n"
                              "Bounds: A 0–0.8, FWHM 0.2–5 ppm, offset 9.0–11.0 ppm.",
        }
        chks: dict = {}
        for pname in pool_catalog:
            label_text = _display_names.get(pname, pname)
            cb = QCheckBox(label_text)
            cb.setChecked(pname in self._global_pools)
            if pname == 'water':
                cb.setEnabled(False)
                cb.setChecked(True)
                cb.setToolTip("Water is always included as pool 0 "
                              "(provides the B₀ shift reference).")
            elif pname in _tooltips:
                cb.setToolTip(_tooltips[pname])
            chks[pname] = cb
            vl.addWidget(cb)

        vl.addSpacing(4)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        vl.addWidget(btns)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._global_pools = [pn for pn, cb in chks.items() if cb.isChecked()]
            if 'water' not in self._global_pools:
                self._global_pools.insert(0, 'water')
            n = len(self._global_pools)
            self.btn_pools.setText(f"Select pools to fit  ({n} selected)")

    def _log(self, msg: str):
        self.log.append(msg)

    # ── Data cursor ───────────────────────────────────────────────────────────

    def _toggle_datacursor(self, enabled: bool):
        if enabled:
            if self._dc_cid is None:
                self._dc_cid = self.canvas.mpl_connect(
                    "motion_notify_event", self._on_dc_hover
                )
            self._dc_annot = None
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
                self._dc_annot = None

    def _on_dc_hover(self, event):
        ax = self.canvas._ax
        if event.inaxes is not ax or self.canvas._img_data is None:
            if self._dc_annot is not None:
                self._dc_annot.set_visible(False)
                self.canvas.draw_idle()
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        xi, yi = int(round(x)), int(round(y))
        img = self.canvas._img_data
        H, W = img.shape[:2]
        if 0 <= yi < H and 0 <= xi < W:
            val = img[yi, xi]
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
