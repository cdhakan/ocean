"""
kspace_tab.py
OCEAN GUI — K-space Viewer tab.

Browses any Bruker experiment folder (fid / ser / rawdata.job0 etc.),
auto-detects PV5/6/7 vs PV360 format, parses JCAMP-DX metadata from
acqp + method, reads raw binary k-space data with numpy, and displays:

  • K-space magnitude (log scale) with matplotlib — dark background
  • Reconstructed image via 2-D iFFT + fftshift
  • Sliders for slice / repetition / receiver channel
  • Time-series plot: center-of-k-space |DC| vs frame index
  • Export current view as PNG

Public API used by app.py:
    tab.load_from_path(path: str)
"""
from __future__ import annotations

import math
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QObject,
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QGroupBox, QComboBox,
    QSpinBox, QSlider, QCheckBox, QScrollArea,
    QSizePolicy, QFileDialog, QProgressBar,
    QFrame,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Colour constants matching the OCEAN theme
# ---------------------------------------------------------------------------
_BG        = "#1a1a2e"   # plot background
_ACCENT    = "#4ec9b0"   # teal accent
_FG        = "#e0e0e0"   # foreground text
_GRID      = "#2a2a4e"   # subtle grid lines
_WARN_FG   = "#e06060"   # error / warning text


# ---------------------------------------------------------------------------
# JCAMP-DX parser (no external dependency)
# ---------------------------------------------------------------------------

def _parse_jcamp(filepath: str) -> dict:
    """
    Read a Bruker JCAMP-DX parameter file (acqp or method) and return a
    flat dict mapping parameter names to their string values.

    Multi-line array values (introduced by a dimension line  ##$KEY=( n )) are
    collected until the next ## or $$ token and returned as a single
    space-joined string.
    """
    params: dict[str, str] = {}
    try:
        with open(filepath, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return params

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("##"):
            # Strip leading ## and split on first =
            body = line[2:]
            if "=" not in body:
                i += 1
                continue
            key, _, rest = body.partition("=")
            key = key.strip().lstrip("$")
            rest = rest.strip()

            if rest.startswith("("):
                # Dimension descriptor — value is on following lines
                val_parts: list[str] = []
                i += 1
                while i < len(lines):
                    nl = lines[i].rstrip()
                    if nl.startswith("##") or nl.startswith("$$"):
                        break
                    val_parts.append(nl)
                    i += 1
                params[key] = " ".join(val_parts).strip()
            else:
                params[key] = rest
                i += 1
        else:
            i += 1
    return params


def _ints(s: str | None) -> list[int]:
    """Extract all integers from a parameter string."""
    if not s:
        return []
    import re
    return [int(x) for x in re.findall(r'-?\d+', s)]


def _floats(s: str | None) -> list[float]:
    """Extract all floats (incl. scientific notation) from a parameter string."""
    if not s:
        return []
    import re
    return [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)]


# ---------------------------------------------------------------------------
# Bruker k-space geometry helper
# ---------------------------------------------------------------------------

class KSpaceGeometry:
    """
    Parsed geometry / data-format parameters from acqp + method.

    Attributes
    ----------
    readout_pts   : number of complex points per readout line
    phase_encodes : number of phase-encode steps (ky dimension)
    n_slices      : NI (number of images = slices * echoes)
    n_rep         : NR (number of repetitions / frames)
    n_channels    : number of receiver channels
    dtype         : numpy dtype for raw samples (real part)
    bytes_per_spl : bytes per raw sample (real or imaginary component)
    block_padded  : True when GO_block_size = Standard_KBlock_Format
    block_pts     : padded block size in points (only relevant if block_padded)
    is_pv360      : True for rawdata.job0 format
    rawdata_file  : absolute path to the binary raw-data file
    enc_matrix    : [kx, ky] from PVM_EncMatrix (may equal readout_pts, phase_encodes)
    """

    def __init__(
        self,
        readout_pts:   int,
        phase_encodes: int,
        n_slices:      int,
        n_rep:         int,
        n_channels:    int,
        dtype:         np.dtype,
        bytes_per_spl: int,
        block_padded:  bool,
        block_pts:     int,
        is_pv360:      bool,
        rawdata_file:  str,
        enc_matrix:    list[int],
    ):
        self.readout_pts   = readout_pts
        self.phase_encodes = phase_encodes
        self.n_slices      = n_slices
        self.n_rep         = n_rep
        self.n_channels    = n_channels
        self.dtype         = dtype
        self.bytes_per_spl = bytes_per_spl
        self.block_padded  = block_padded
        self.block_pts     = block_pts
        self.is_pv360      = is_pv360
        self.rawdata_file  = rawdata_file
        self.enc_matrix    = enc_matrix

    @property
    def info_string(self) -> str:
        fmt = "PV360 rawdata.job0" if self.is_pv360 else (
            "PV5/6/7 fid (blocked)" if self.block_padded else "PV5/6/7 fid"
        )
        return (
            f"Format : {fmt}\n"
            f"Matrix : {self.readout_pts} × {self.phase_encodes}\n"
            f"Slices : {self.n_slices}  |  Rep : {self.n_rep}  |  Ch : {self.n_channels}\n"
            f"Dtype  : {self.dtype}"
        )


def detect_and_parse(exp_folder: str) -> KSpaceGeometry:
    """
    Given a Bruker experiment folder (containing acqp, method, and a data file),
    return a KSpaceGeometry describing the raw data.

    Raises ValueError with a user-readable message when parsing fails.
    """
    acqp_path   = os.path.join(exp_folder, "acqp")
    method_path = os.path.join(exp_folder, "method")

    if not os.path.isfile(acqp_path):
        raise ValueError(f"acqp not found in:\n{exp_folder}")

    acqp   = _parse_jcamp(acqp_path)
    method = _parse_jcamp(method_path) if os.path.isfile(method_path) else {}

    # ---- Detect data file ---------------------------------------------------
    candidates = ["fid", "ser", "rawdata.job0", "rawdata.job1"]
    rawdata_file = ""
    is_pv360     = False
    for name in candidates:
        p = os.path.join(exp_folder, name)
        if os.path.isfile(p):
            rawdata_file = p
            is_pv360 = "rawdata.job" in name
            break

    if not rawdata_file:
        raise ValueError(
            f"No raw data file found (looked for {candidates}) in:\n{exp_folder}"
        )

    # ---- ACQ_size -----------------------------------------------------------
    # ACQ_size = ( n ) followed by values; [0] = 2 * readout complex pts, [1] = phase encodes
    acq_size_vals = _ints(acqp.get("ACQ_size"))
    if len(acq_size_vals) < 2:
        raise ValueError("Cannot parse ACQ_size from acqp.")
    readout_pts   = acq_size_vals[0] // 2          # complex points in readout
    phase_encodes = acq_size_vals[1]

    # ---- NI / NR ------------------------------------------------------------
    n_slices = int(acqp.get("NI", "1") or 1)
    n_rep    = int(acqp.get("NR", "1") or 1)

    # ---- Channels -----------------------------------------------------------
    n_channels = 1
    # PVM_EncNReceivers (method file preferred)
    nrec_str = method.get("PVM_EncNReceivers") or acqp.get("ACQ_ReceiverSelect")
    if nrec_str:
        # PVM_EncNReceivers might be just an integer
        nrec_vals = _ints(nrec_str)
        if nrec_vals:
            # Count non-zero / "Yes" values — or simply take the first int
            # In newer PV it's a single integer count
            n_channels = max(1, nrec_vals[0])
    # Also try GO_ReceiverSelect which is a list of Yes/No
    recv_sel = acqp.get("GO_ReceiverSelect", "")
    if recv_sel and n_channels == 1:
        n_channels = max(1, recv_sel.count("Yes"))

    # ---- Dtype from GO_raw_data_format / ACQ_word_size ----------------------
    raw_fmt = acqp.get("GO_raw_data_format", "").strip()
    word    = acqp.get("ACQ_word_size", "").strip()

    if "32BIT_SGN_INT" in raw_fmt or "_32_BIT" in word:
        dtype          = np.dtype("int32")
        bytes_per_spl  = 4
    elif "16BIT_SGN_INT" in raw_fmt:
        dtype          = np.dtype("int16")
        bytes_per_spl  = 2
    elif "32BIT_FLOAT" in raw_fmt:
        dtype          = np.dtype("float32")
        bytes_per_spl  = 4
    else:
        # Default: int32 (most common)
        dtype          = np.dtype("int32")
        bytes_per_spl  = 4

    # ---- Byte order ---------------------------------------------------------
    bytorda = acqp.get("BYTORDA", "little").strip().lower()
    if "big" in bytorda:
        dtype = dtype.newbyteorder(">")
    else:
        dtype = dtype.newbyteorder("<")

    # ---- Block padding (PV6/7) ----------------------------------------------
    block_str = acqp.get("GO_block_size", "Continuous").strip()
    block_padded = "Standard_KBlock_Format" in block_str

    bytes_per_complex = bytes_per_spl * 2     # real + imag interleaved
    raw_line_bytes    = readout_pts * bytes_per_complex
    if block_padded:
        # Pad to next multiple of 1024 bytes
        padded_bytes = math.ceil(raw_line_bytes / 1024) * 1024
        block_pts    = padded_bytes // bytes_per_spl   # total samples in padded block
    else:
        block_pts = readout_pts * 2            # no padding: exactly 2*readout complex floats

    # ---- Encoding matrix from method ----------------------------------------
    enc_vals = _ints(method.get("PVM_EncMatrix", ""))
    enc_matrix = enc_vals[:2] if len(enc_vals) >= 2 else [readout_pts, phase_encodes]

    return KSpaceGeometry(
        readout_pts   = readout_pts,
        phase_encodes = phase_encodes,
        n_slices      = n_slices,
        n_rep         = n_rep,
        n_channels    = n_channels,
        dtype         = dtype,
        bytes_per_spl = bytes_per_spl,
        block_padded  = block_padded,
        block_pts     = block_pts,
        is_pv360      = is_pv360,
        rawdata_file  = rawdata_file,
        enc_matrix    = enc_matrix,
    )


# ---------------------------------------------------------------------------
# Low-level k-space reader
# ---------------------------------------------------------------------------

def read_kspace(geo: KSpaceGeometry) -> np.ndarray:
    """
    Read the raw k-space data described by *geo* and return a complex numpy
    array of shape:

        (n_channels, n_slices, n_rep, kx, ky)

    where kx = readout_pts and ky = phase_encodes.

    Strategy
    --------
    PV360 rawdata.job0 — no block padding; shape in file is approximately
        (2 * readout_pts * n_channels * NR * phase_encodes) samples

    PV5/6/7 fid / ser — when block_padded=True, each readout line is zero-
        padded to the next 1 kB boundary.  Read block_pts samples per line,
        keep first readout_pts*2 (real + imag interleaved).

    For both formats: complex = raw[0::2] + 1j * raw[1::2]
    """
    raw_mm = np.memmap(geo.rawdata_file, dtype=geo.dtype, mode="r")

    kx  = geo.readout_pts
    ky  = geo.phase_encodes
    nch = geo.n_channels
    nsl = geo.n_slices
    nr  = geo.n_rep

    n_lines_total = ky * nch * nsl * nr

    if geo.is_pv360:
        # PV360: contiguous, no block padding
        # Expected total samples = 2 * kx * n_lines_total
        n_expected = 2 * kx * n_lines_total
        if len(raw_mm) < n_expected:
            # Tolerate truncated files — pad with zeros
            raw_flat = np.zeros(n_expected, dtype=geo.dtype)
            raw_flat[:len(raw_mm)] = raw_mm
        else:
            raw_flat = np.asarray(raw_mm[:n_expected])

        # Convert to complex
        cx = raw_flat[0::2].astype(np.float32) + 1j * raw_flat[1::2].astype(np.float32)
        # Shape: (n_lines_total, kx)
        cx = cx.reshape(n_lines_total, kx)

    else:
        # PV5/6/7 — may have block padding
        bpts = geo.block_pts             # total raw samples per padded line
        n_expected = bpts * n_lines_total
        if len(raw_mm) < n_expected:
            raw_flat = np.zeros(n_expected, dtype=geo.dtype)
            raw_flat[:len(raw_mm)] = raw_mm
        else:
            raw_flat = np.asarray(raw_mm[:n_expected])

        # Reshape to (n_lines_total, bpts), then discard padding
        raw_2d = raw_flat.reshape(n_lines_total, bpts)
        useful = raw_2d[:, : kx * 2]           # first kx*2 samples are the real readout

        cx = (useful[:, 0::2].astype(np.float32)
              + 1j * useful[:, 1::2].astype(np.float32))

    # cx shape: (n_lines_total, kx)
    # line ordering (Bruker): [nr, nsl, nch, ky] — innermost ky, then ch, sl, rep
    # Reshape accordingly and transpose to canonical (ch, sl, rep, kx, ky)
    try:
        cx = cx.reshape(nr, nsl, nch, ky, kx)
        # Move axes: (nr, nsl, nch, ky, kx) -> (nch, nsl, nr, kx, ky)
        cx = np.moveaxis(cx, [2, 1, 0, 4, 3], [0, 1, 2, 3, 4])
    except ValueError:
        # Fallback: flatten everything into a single (kx, ky) frame
        total_cx = cx.shape[0] * kx
        needed   = kx * ky
        pad      = np.zeros(needed, dtype=np.complex64)
        avail    = min(cx.size, needed)
        pad[:avail] = cx.ravel()[:avail]
        cx = pad.reshape(1, 1, 1, kx, ky)

    return cx.astype(np.complex64)


# ---------------------------------------------------------------------------
# Background loader (QThread)
# ---------------------------------------------------------------------------

class _LoaderSignals(QObject):
    finished = pyqtSignal(object, object)  # (geo, kspace_array)
    error    = pyqtSignal(str)
    progress = pyqtSignal(str)


class _LoaderThread(QThread):
    """Loads geometry + k-space data in the background."""

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path    = path
        self.signals = _LoaderSignals()

    def run(self):
        try:
            self.signals.progress.emit("Parsing acqp / method …")
            geo = detect_and_parse(self.path)
            self.signals.progress.emit(
                f"Reading {os.path.basename(geo.rawdata_file)} "
                f"({os.path.getsize(geo.rawdata_file) / 1e6:.1f} MB) …"
            )
            ks = read_kspace(geo)
            self.signals.finished.emit(geo, ks)
        except Exception as exc:
            self.signals.error.emit(f"{exc}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Matplotlib canvas helpers
# ---------------------------------------------------------------------------

def _make_figure(nrows: int = 1, ncols: int = 2,
                 figsize=(9, 4.5)) -> tuple[Figure, FigureCanvas]:
    fig = Figure(figsize=figsize, facecolor=_BG, tight_layout=True)
    canvas = FigureCanvas(fig)
    canvas.setStyleSheet(f"background-color: {_BG};")
    return fig, canvas


class ImageCanvas(FigureCanvas):
    """Dual-panel canvas: left = k-space, right = reconstructed image."""

    def __init__(self, parent=None):
        self._fig = Figure(figsize=(9, 4.5), facecolor=_BG)
        super().__init__(self._fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._ax_ksp = self._fig.add_subplot(1, 2, 1)
        self._ax_img = self._fig.add_subplot(1, 2, 2)
        for ax in (self._ax_ksp, self._ax_img):
            ax.set_facecolor(_BG)
            ax.tick_params(colors=_FG, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(_GRID)
        self._ax_ksp.set_title("K-space (log |·|)", color=_ACCENT, fontsize=9)
        self._ax_img.set_title("Reconstructed image", color=_ACCENT, fontsize=9)
        self._im_ksp = None
        self._im_img = None
        self._fig.tight_layout(pad=0.5)

    def update_images(
        self,
        kspace_2d: np.ndarray,
        cmap: str = "gray",
        log_scale: bool = True,
    ):
        """
        kspace_2d: complex array shape (kx, ky)
        """
        # ---- K-space magnitude -------------------------------------------
        mag_ks = np.abs(kspace_2d)
        if log_scale:
            mag_ks = np.log1p(mag_ks)
        vmin_ks, vmax_ks = mag_ks.min(), mag_ks.max()

        # ---- Reconstructed image via 2D iFFT + fftshift -------------------
        img = np.abs(np.fft.fftshift(np.fft.ifft2(kspace_2d)))
        vmin_img, vmax_img = img.min(), img.max()

        if self._im_ksp is None:
            self._im_ksp = self._ax_ksp.imshow(
                mag_ks.T, cmap=cmap, aspect="auto",
                vmin=vmin_ks, vmax=vmax_ks, origin="lower",
            )
        else:
            self._im_ksp.set_data(mag_ks.T)
            self._im_ksp.set_cmap(cmap)
            self._im_ksp.set_clim(vmin_ks, vmax_ks)

        if self._im_img is None:
            self._im_img = self._ax_img.imshow(
                img.T, cmap=cmap, aspect="auto",
                vmin=vmin_img, vmax=vmax_img, origin="lower",
            )
        else:
            self._im_img.set_data(img.T)
            self._im_img.set_cmap(cmap)
            self._im_img.set_clim(vmin_img, vmax_img)

        self.draw_idle()

    def clear(self):
        self._ax_ksp.cla()
        self._ax_img.cla()
        self._ax_ksp.set_facecolor(_BG)
        self._ax_img.set_facecolor(_BG)
        self._ax_ksp.set_title("K-space (log |·|)", color=_ACCENT, fontsize=9)
        self._ax_img.set_title("Reconstructed image", color=_ACCENT, fontsize=9)
        self._im_ksp = None
        self._im_img = None
        self.draw_idle()


class TimeSeriesCanvas(FigureCanvas):
    """
    Single-axis canvas: center-of-k-space |DC| vs frame index.
    Emits frame_clicked(int) when the user clicks on the plot.
    """

    frame_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        self._fig = Figure(figsize=(9, 2), facecolor=_BG)
        super().__init__(self._fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._ax  = self._fig.add_subplot(1, 1, 1)
        self._ax.set_facecolor(_BG)
        self._ax.tick_params(colors=_FG, labelsize=8)
        self._ax.set_xlabel("Frame / Rep index", color=_FG, fontsize=8)
        self._ax.set_ylabel("|DC|", color=_ACCENT, fontsize=8)
        self._ax.set_title("Center of k-space signal", color=_ACCENT, fontsize=9)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(_GRID)
        self._line  = None
        self._vline = None
        self._dc    = None
        self._fig.tight_layout(pad=0.3)
        self._fig.canvas.mpl_connect("button_press_event", self._on_click)

    def set_dc_series(self, dc: np.ndarray):
        """dc: 1-D array of DC magnitudes, one per frame."""
        self._dc = dc
        self._ax.cla()
        self._ax.set_facecolor(_BG)
        self._ax.tick_params(colors=_FG, labelsize=8)
        self._ax.set_xlabel("Frame / Rep index", color=_FG, fontsize=8)
        self._ax.set_ylabel("|DC|", color=_ACCENT, fontsize=8)
        self._ax.set_title("Center of k-space signal", color=_ACCENT, fontsize=9)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(_GRID)
        x = np.arange(len(dc))
        self._line, = self._ax.plot(x, dc, color=_ACCENT, linewidth=1.2)
        self._ax.scatter(x, dc, s=12, color=_ACCENT, zorder=3)
        self._vline = self._ax.axvline(0, color=_WARN_FG, linewidth=1.0, alpha=0.8)
        self._fig.tight_layout(pad=0.3)
        self.draw_idle()

    def set_frame(self, idx: int):
        if self._vline is not None:
            self._vline.set_xdata([idx, idx])
            self.draw_idle()

    def _on_click(self, event):
        if event.inaxes is not self._ax or self._dc is None:
            return
        idx = int(round(event.xdata))
        idx = max(0, min(idx, len(self._dc) - 1))
        self.frame_clicked.emit(idx)

    def clear(self):
        self._ax.cla()
        self._ax.set_facecolor(_BG)
        self._ax.tick_params(colors=_FG, labelsize=8)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(_GRID)
        self._line  = None
        self._vline = None
        self._dc    = None
        self.draw_idle()


# ---------------------------------------------------------------------------
# Main Tab Widget
# ---------------------------------------------------------------------------

class KSpaceTab(QWidget):
    """
    OCEAN K-space Viewer tab.

    Hierarchy
    ---------
    QWidget (root)
    └── QHBoxLayout
        ├── QScrollArea (left panel, max 320 px)
        │   └── panel widget
        │       ├── GroupBox "Data Source"
        │       ├── GroupBox "Navigation"
        │       ├── GroupBox "Display"
        │       └── QPushButton "Export PNG"
        └── QSplitter (right panel, vertical)
            ├── ImageCanvas  (k-space + reconstructed image)
            └── TimeSeriesCanvas  (collapsible)
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Data state ───────────────────────────────────────────────────────
        self._geo:        Optional[KSpaceGeometry] = None
        self._kspace:     Optional[np.ndarray]     = None   # (nch, nsl, nr, kx, ky)
        self._dc_series:  Optional[np.ndarray]     = None   # (nr,) per current ch+sl
        self._loader:     Optional[_LoaderThread]  = None
        self._exp_path:   str = ""

        # ── Build UI ─────────────────────────────────────────────────────────
        self._build_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(6)

        # ---- Left scroll panel -------------------------------------------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMaximumWidth(320)
        left_scroll.setMinimumWidth(220)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.setSpacing(8)

        # -- GroupBox: Data Source ------------------------------------------
        grp_src = QGroupBox("Data Source")
        src_layout = QVBoxLayout(grp_src)
        src_layout.setSpacing(6)

        self._browse_btn = QPushButton("Browse Experiment…")
        self._browse_btn.clicked.connect(self._on_browse)
        src_layout.addWidget(self._browse_btn)

        self._path_label = QLabel("No experiment loaded")
        self._path_label.setWordWrap(True)
        self._path_label.setStyleSheet(f"color: {_FG}; font-size: 10px;")
        src_layout.addWidget(self._path_label)

        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet(
            f"color: {_ACCENT}; font-size: 10px; "
            f"background: #1a1a2e; padding: 4px; border-radius: 4px;"
        )
        self._info_label.setVisible(False)
        src_layout.addWidget(self._info_label)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(f"color: {_WARN_FG}; font-size: 10px;")
        self._status_label.setVisible(False)
        src_layout.addWidget(self._status_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)          # indeterminate
        self._progress.setVisible(False)
        self._progress.setMaximumHeight(10)
        src_layout.addWidget(self._progress)

        panel_layout.addWidget(grp_src)

        # -- GroupBox: Navigation ------------------------------------------
        grp_nav = QGroupBox("Navigation")
        nav_layout = QVBoxLayout(grp_nav)
        nav_layout.setSpacing(6)

        # Slice
        row_sl = QHBoxLayout()
        row_sl.addWidget(QLabel("Slice:"))
        self._slice_spin = QSpinBox()
        self._slice_spin.setRange(0, 0)
        self._slice_spin.setEnabled(False)
        self._slice_spin.valueChanged.connect(self._on_nav_changed)
        row_sl.addWidget(self._slice_spin)
        nav_layout.addLayout(row_sl)

        # Frame / Rep
        row_fr_label = QHBoxLayout()
        row_fr_label.addWidget(QLabel("Frame / Rep:"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, 0)
        self._frame_spin.setEnabled(False)
        self._frame_spin.valueChanged.connect(self._on_frame_spin_changed)
        row_fr_label.addWidget(self._frame_spin)
        nav_layout.addLayout(row_fr_label)

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.setEnabled(False)
        self._frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        nav_layout.addWidget(self._frame_slider)

        # Channel
        row_ch = QHBoxLayout()
        row_ch.addWidget(QLabel("Channel:"))
        self._ch_spin = QSpinBox()
        self._ch_spin.setRange(0, 0)
        self._ch_spin.setEnabled(False)
        self._ch_spin.valueChanged.connect(self._on_nav_changed)
        row_ch.addWidget(self._ch_spin)
        nav_layout.addLayout(row_ch)

        panel_layout.addWidget(grp_nav)

        # -- GroupBox: Display ---------------------------------------------
        grp_disp = QGroupBox("Display")
        disp_layout = QVBoxLayout(grp_disp)
        disp_layout.setSpacing(6)

        row_cmap = QHBoxLayout()
        row_cmap.addWidget(QLabel("Colormap:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["gray", "viridis", "hot", "jet", "plasma"])
        self._cmap_combo.currentTextChanged.connect(self._on_display_changed)
        row_cmap.addWidget(self._cmap_combo)
        disp_layout.addLayout(row_cmap)

        row_scale = QHBoxLayout()
        row_scale.addWidget(QLabel("K-space scale:"))
        self._scale_combo = QComboBox()
        self._scale_combo.addItems(["Log", "Linear"])
        self._scale_combo.currentTextChanged.connect(self._on_display_changed)
        row_scale.addWidget(self._scale_combo)
        disp_layout.addLayout(row_scale)

        self._timeseries_chk = QCheckBox("Show time series")
        self._timeseries_chk.setChecked(True)
        self._timeseries_chk.toggled.connect(self._on_timeseries_toggle)
        disp_layout.addWidget(self._timeseries_chk)

        panel_layout.addWidget(grp_disp)

        # -- Export button ------------------------------------------------
        self._export_btn = QPushButton("Export PNG")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        panel_layout.addWidget(self._export_btn)

        panel_layout.addStretch(1)

        left_scroll.setWidget(panel)
        root_layout.addWidget(left_scroll)

        # ---- Right panel (splitter) --------------------------------------
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Image canvas (top)
        self._img_canvas = ImageCanvas()
        right_splitter.addWidget(self._img_canvas)

        # Time-series canvas (bottom, collapsible)
        self._ts_canvas = TimeSeriesCanvas()
        self._ts_canvas.frame_clicked.connect(self._on_ts_click)
        right_splitter.addWidget(self._ts_canvas)

        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setSizes([450, 150])

        root_layout.addWidget(right_splitter, stretch=1)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def load_from_path(self, path: str):
        """
        Called by ScanDirTab (or app.py) when an experiment folder is assigned.
        Kicks off background loading.
        """
        if not path or not os.path.isdir(path):
            self._set_status(f"Path does not exist or is not a folder:\n{path}", error=True)
            return
        self._exp_path = path
        self._start_loading(path)

    # -----------------------------------------------------------------------
    # Slots
    # -----------------------------------------------------------------------

    def _on_browse(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Bruker Experiment Folder",
            self._exp_path or os.path.expanduser("~"),
        )
        if folder:
            self._exp_path = folder
            self._start_loading(folder)

    def _on_nav_changed(self):
        """Slice or channel changed — redraw."""
        self._refresh_display()

    def _on_frame_spin_changed(self, val: int):
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(val)
        self._frame_slider.blockSignals(False)
        self._refresh_display()

    def _on_frame_slider_changed(self, val: int):
        self._frame_spin.blockSignals(True)
        self._frame_spin.setValue(val)
        self._frame_spin.blockSignals(False)
        self._refresh_display()

    def _on_display_changed(self):
        self._refresh_display()

    def _on_timeseries_toggle(self, checked: bool):
        self._ts_canvas.setVisible(checked)

    def _on_ts_click(self, frame_idx: int):
        """User clicked on the time-series plot — jump to that frame."""
        self._frame_spin.setValue(frame_idx)

    def _on_export(self):
        from my_gui.fig_export import save_figure, FIG_EXPORT_FILTER
        if self._geo is None or self._kspace is None:
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export current view as PNG",
            os.path.join(self._exp_path or os.path.expanduser("~"), "kspace_view.png"),
            FIG_EXPORT_FILTER,
        )
        if not filepath:
            return
        try:
            save_figure(self._img_canvas._fig, filepath, dpi=300, facecolor=_BG)
            self._set_status(f"Exported: {os.path.basename(filepath)}", error=False)
        except Exception as exc:
            self._set_status(f"Export failed: {exc}", error=True)

    # -----------------------------------------------------------------------
    # Background loading
    # -----------------------------------------------------------------------

    def _start_loading(self, path: str):
        # Abort previous load if still running
        if self._loader is not None and self._loader.isRunning():
            self._loader.signals.finished.disconnect()
            self._loader.signals.error.disconnect()
            self._loader.quit()
            self._loader.wait(2000)

        # Reset state
        self._geo    = None
        self._kspace = None
        self._dc_series = None
        self._img_canvas.clear()
        self._ts_canvas.clear()
        self._export_btn.setEnabled(False)
        self._set_navigation_enabled(False)
        self._info_label.setVisible(False)

        # Show progress
        short_path = path if len(path) <= 55 else "…" + path[-52:]
        self._path_label.setText(short_path)
        self._progress.setVisible(True)
        self._set_status("Loading …", error=False)

        self._loader = _LoaderThread(path, parent=self)
        self._loader.signals.progress.connect(self._on_load_progress)
        self._loader.signals.finished.connect(self._on_load_finished)
        self._loader.signals.error.connect(self._on_load_error)
        self._loader.start()

    def _on_load_progress(self, msg: str):
        self._set_status(msg, error=False)

    def _on_load_finished(self, geo: KSpaceGeometry, kspace: np.ndarray):
        self._progress.setVisible(False)
        self._geo    = geo
        self._kspace = kspace

        # Populate info label
        self._info_label.setText(geo.info_string)
        self._info_label.setVisible(True)

        # Configure navigation controls
        self._slice_spin.setRange(0, max(0, geo.n_slices - 1))
        self._slice_spin.setValue(0)
        self._frame_spin.setRange(0, max(0, geo.n_rep - 1))
        self._frame_spin.setValue(0)
        self._frame_slider.setRange(0, max(0, geo.n_rep - 1))
        self._frame_slider.setValue(0)
        self._ch_spin.setRange(0, max(0, geo.n_channels - 1))
        self._ch_spin.setValue(0)
        self._set_navigation_enabled(True)

        # Compute DC (center of k-space) time series for current ch, sl
        self._update_dc_series()

        # First display
        self._refresh_display()

        self._export_btn.setEnabled(True)
        self._set_status("Ready", error=False)

    def _on_load_error(self, msg: str):
        self._progress.setVisible(False)
        self._set_status(msg, error=True)
        self._info_label.setVisible(False)

    # -----------------------------------------------------------------------
    # Display helpers
    # -----------------------------------------------------------------------

    def _current_kspace_2d(self) -> Optional[np.ndarray]:
        """Return the (kx, ky) slice for the current navigation position."""
        if self._kspace is None or self._geo is None:
            return None
        ch  = self._ch_spin.value()
        sl  = self._slice_spin.value()
        rep = self._frame_spin.value()

        # Clamp to valid indices (shape may differ from max values if data is short)
        nch, nsl, nr, kx, ky = self._kspace.shape
        ch  = min(ch,  nch - 1)
        sl  = min(sl,  nsl - 1)
        rep = min(rep, nr  - 1)

        return self._kspace[ch, sl, rep, :, :]    # (kx, ky)

    def _update_dc_series(self):
        """Recompute DC magnitude time series for current channel + slice."""
        if self._kspace is None:
            self._dc_series = None
            return
        ch = self._ch_spin.value()
        sl = self._slice_spin.value()
        nch, nsl, nr, kx, ky = self._kspace.shape
        ch = min(ch, nch - 1)
        sl = min(sl, nsl - 1)

        cx = kx // 2
        cy = ky // 2
        # DC magnitude for each rep
        self._dc_series = np.abs(self._kspace[ch, sl, :, cx, cy])

    def _refresh_display(self):
        """Update both the image canvas and the time-series marker."""
        if self._kspace is None:
            return

        # If nav changed to a new ch/sl, recompute DC series
        self._update_dc_series()

        ks2d = self._current_kspace_2d()
        if ks2d is None:
            return

        cmap      = self._cmap_combo.currentText()
        log_scale = self._scale_combo.currentText() == "Log"

        self._img_canvas.update_images(ks2d, cmap=cmap, log_scale=log_scale)

        # Update time-series
        if self._dc_series is not None:
            if self._ts_canvas._dc is None:
                self._ts_canvas.set_dc_series(self._dc_series)
            else:
                # Only update vertical marker, series data hasn't changed
                pass
            rep = self._frame_spin.value()
            self._ts_canvas.set_frame(rep)

        # If the series just changed (ch/sl switched), refresh the full plot
        if self._dc_series is not None and self._ts_canvas._dc is None:
            self._ts_canvas.set_dc_series(self._dc_series)

    def _set_navigation_enabled(self, enabled: bool):
        self._slice_spin.setEnabled(enabled)
        self._frame_spin.setEnabled(enabled)
        self._frame_slider.setEnabled(enabled)
        self._ch_spin.setEnabled(enabled)

    def _set_status(self, msg: str, *, error: bool):
        self._status_label.setText(msg)
        colour = _WARN_FG if error else _ACCENT
        self._status_label.setStyleSheet(f"color: {colour}; font-size: 10px;")
        self._status_label.setVisible(bool(msg))
