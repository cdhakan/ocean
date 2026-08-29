"""
image_viewer_tab.py
===================
General-purpose image viewer.

Load any DICOM (file or folder) or NIfTI dataset and scroll through every
frame/slice, with the same plot-customisation controls used elsewhere
(colormap, colour-bar limits, title, fonts, export) plus intensity / histogram
adjustment.  A colour-bar is drawn for every map by ROICanvas.show_map().
"""
from __future__ import annotations

import os
import glob

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton, QLabel,
    QLineEdit, QSlider, QFrame, QCheckBox, QGroupBox, QFileDialog, QApplication,
)
from PyQt6.QtCore import Qt

from my_gui.roi_tools import ROICanvas
from my_gui.plot_custom_bar import PlotCustomBar
from my_gui.format_bar import add_title_format_bar, connect_title_debounced


def _to_gray_frames(pix: np.ndarray):
    """Yield 2-D grayscale frames from a DICOM pixel_array of any shape."""
    a = np.asarray(pix)
    if a.ndim == 2:
        yield a.astype(float)
    elif a.ndim == 3:
        if a.shape[-1] in (3, 4):                       # single RGB(A)
            yield a[..., :3].astype(float) @ [0.299, 0.587, 0.114]
        else:                                           # multiframe gray
            for k in range(a.shape[0]):
                yield a[k].astype(float)
    elif a.ndim == 4 and a.shape[-1] in (3, 4):         # multiframe colour
        for k in range(a.shape[0]):
            yield a[k, ..., :3].astype(float) @ [0.299, 0.587, 0.114]
    else:
        yield np.asarray(a).reshape(a.shape[-2], a.shape[-1]).astype(float)


def load_image_stack(path: str, log_fn=print):
    """Return (stack (H, W, N) float, labels list[str]) from a DICOM file/folder
    or NIfTI file. Frames not matching the first frame's shape are skipped.

    Siemens MOSAIC frames are de-tiled so each slice becomes its own frame on
    the viewer's scroll axis instead of one tiled montage."""
    from my_gui.dicom_mosaic import is_mosaic, mosaic_to_volume
    frames, labels = [], []

    def _emit_ds(ds, pix, base):
        mos = is_mosaic(ds)
        for i, g in enumerate(_to_gray_frames(pix)):
            if mos and g.ndim == 2:
                vol = mosaic_to_volume(g, ds)
                if vol.ndim == 3:
                    for s in range(vol.shape[2]):
                        frames.append(vol[:, :, s])
                        labels.append(f"{base} sl{s + 1}")
                    continue
            frames.append(g)
            labels.append(base if i == 0 else f"{base} f{i + 1}")

    if os.path.isdir(path):
        import pydicom
        files = sorted(f for f in glob.glob(os.path.join(path, "*"))
                       if os.path.isfile(f))
        for f in files:
            try:
                ds = pydicom.dcmread(f, force=True)
                if not hasattr(ds, "Rows"):
                    continue
                pix = ds.pixel_array
            except Exception:
                continue
            _emit_ds(ds, pix, os.path.basename(f))
    elif path.lower().endswith((".nii", ".nii.gz")):
        import nibabel as nib
        data = np.asarray(nib.load(path).dataobj, dtype=float)
        if data.ndim == 2:
            data = data[:, :, None]
        flat = data.reshape(data.shape[0], data.shape[1], -1)
        for k in range(flat.shape[2]):
            frames.append(flat[:, :, k])
            labels.append(f"vol {k + 1}")
    else:
        import pydicom
        ds = pydicom.dcmread(path, force=True)
        _emit_ds(ds, ds.pixel_array, "frame")

    if not frames:
        raise ValueError("No readable image frames found.")
    h, w = frames[0].shape
    frames = [f for f in frames if f.shape == (h, w)]
    labels = labels[:len(frames)]
    log_fn(f"  Loaded {len(frames)} frame(s) of {h}×{w}.")
    return np.stack(frames, axis=-1), labels


def _hist_eq(img: np.ndarray) -> np.ndarray:
    """Histogram-equalise a 2-D image into its original [min,max] range."""
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return img
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return img
    hist, bins = np.histogram(finite, bins=256, range=(lo, hi))
    cdf = hist.cumsum().astype(float)
    cdf /= cdf[-1]
    out = np.interp(img, bins[:-1], lo + cdf * (hi - lo))
    return out.reshape(img.shape)


class ImageViewerTab(QWidget):
    """Load and browse any DICOM/NIfTI image stack with full customisation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stack: np.ndarray | None = None     # (H, W, N)
        self._labels: list[str] = []
        self._path: str = ""

        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left controls panel ───────────────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(6)

        grp_load = QGroupBox("Load Image (DICOM / NIfTI)")
        gl = QVBoxLayout(grp_load)
        r = QHBoxLayout()
        self.edit_path = QLineEdit(); self.edit_path.setReadOnly(True)
        self.edit_path.setPlaceholderText("a .dcm / .nii / .nii.gz file  OR  a DICOM folder")
        r.addWidget(self.edit_path, stretch=1)
        btn_file = QPushButton("File…"); btn_file.clicked.connect(lambda: self._browse(folder=False))
        r.addWidget(btn_file)
        btn_dir = QPushButton("Folder…"); btn_dir.clicked.connect(lambda: self._browse(folder=True))
        r.addWidget(btn_dir)
        gl.addLayout(r)
        self.lbl_info = QLabel("No image loaded.")
        self.lbl_info.setStyleSheet("font-size: 11px; color: gray;")
        self.lbl_info.setWordWrap(True)
        gl.addWidget(self.lbl_info)
        left_layout.addWidget(grp_load)

        # Frame scroll slider
        self._frame_frame = QFrame()
        fl = QHBoxLayout(self._frame_frame); fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(QLabel("Image:"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0); self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._refresh)
        fl.addWidget(self.slider, stretch=1)
        self.lbl_frame = QLabel("—"); self.lbl_frame.setFixedWidth(120)
        fl.addWidget(self.lbl_frame)
        self._frame_frame.setVisible(False)
        left_layout.addWidget(self._frame_frame)

        # Title + format bar
        _tr = QHBoxLayout(); _tr.addWidget(QLabel("Title:"))
        self.edit_title = QLineEdit()
        self.edit_title.setPlaceholderText("Custom title (blank = file/frame name)")
        _tr.addWidget(self.edit_title, stretch=1)
        left_layout.addLayout(_tr)
        connect_title_debounced(self.edit_title, self._refresh)

        # Plot customisation — fonts line on top (with B/I/x²/x₂ title buttons),
        # colour-bar line below.
        self.plot_bar = PlotCustomBar(default_cmap="gray", fonts_first=True)
        self.plot_bar.applied.connect(self._refresh)
        add_title_format_bar(self.edit_title, None,
                             target_row=self.plot_bar.font_row(),
                             default_getter=lambda: getattr(self.canvas, "_last_title", ""))
        self.chk_dark_bg = QCheckBox("Bg")
        self.chk_dark_bg.setToolTip("Black background for the figure (for slides). Only the white surround and labels flip - the maps stay identical.")
        self.chk_dark_bg.toggled.connect(self._refresh)
        _frow = self.plot_bar.font_row()
        _b_idx = _frow.count()
        for _i in range(_frow.count()):
            _wd = _frow.itemAt(_i).widget()
            if isinstance(_wd, QPushButton) and _wd.text() == "B":
                _b_idx = _i; break
        _frow.insertWidget(_b_idx, self.chk_dark_bg)
        left_layout.addWidget(self.plot_bar)

        # ── Intensity / histogram adjustment ──────────────────────────────
        grp_int = QGroupBox("Intensity / Histogram")
        il = QVBoxLayout(grp_int); il.setSpacing(3)
        lo_row = QHBoxLayout(); lo_row.addWidget(QLabel("Min %ile:"))
        self.slider_lo = QSlider(Qt.Orientation.Horizontal)
        self.slider_lo.setRange(0, 49); self.slider_lo.setValue(0)
        self.slider_lo.valueChanged.connect(self._on_intensity)
        lo_row.addWidget(self.slider_lo, stretch=1)
        self.lbl_lo = QLabel("0%"); self.lbl_lo.setFixedWidth(40); lo_row.addWidget(self.lbl_lo)
        il.addLayout(lo_row)
        hi_row = QHBoxLayout(); hi_row.addWidget(QLabel("Max %ile:"))
        self.slider_hi = QSlider(Qt.Orientation.Horizontal)
        self.slider_hi.setRange(51, 100); self.slider_hi.setValue(100)
        self.slider_hi.valueChanged.connect(self._on_intensity)
        hi_row.addWidget(self.slider_hi, stretch=1)
        self.lbl_hi = QLabel("100%"); self.lbl_hi.setFixedWidth(40); hi_row.addWidget(self.lbl_hi)
        il.addLayout(hi_row)
        cr = QHBoxLayout()
        self.chk_histeq = QCheckBox("Histogram equalize")
        self.chk_histeq.toggled.connect(self._refresh)
        cr.addWidget(self.chk_histeq)
        self.chk_invert = QCheckBox("Invert")
        self.chk_invert.toggled.connect(self._refresh)
        cr.addWidget(self.chk_invert)
        cr.addStretch()
        il.addLayout(cr)
        _int_hint = QLabel("Percentile window adjusts contrast/intensity. Custom clim "
                           "(above) overrides it when enabled.")
        _int_hint.setStyleSheet("font-size: 10px; color: #888;"); _int_hint.setWordWrap(True)
        il.addWidget(_int_hint)
        left_layout.addWidget(grp_int)

        # Export + data cursor
        btn_export = QPushButton("Export figure…")
        btn_export.clicked.connect(self._export)
        left_layout.addWidget(btn_export)
        from my_gui.plot_custom_bar import DataCursorToolButton
        self.chk_datacursor = DataCursorToolButton()
        self.chk_datacursor.toggled.connect(self._toggle_datacursor)
        left_layout.addWidget(self.chk_datacursor)
        left_layout.addStretch()

        # ── Right canvas ──────────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        self.canvas = ROICanvas()
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        rl.addWidget(self.canvas, stretch=1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([360, 640])

    # ── Loading ───────────────────────────────────────────────────────────
    def _browse(self, folder: bool):
        if folder:
            path = QFileDialog.getExistingDirectory(self, "Select DICOM folder")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select image", "",
                "Images (*.dcm *.IMA *.nii *.nii.gz);;All files (*)")
        if not path:
            return
        self._path = path
        self.edit_path.setText(path)
        self._load()

    def _load(self):
        try:
            self.lbl_info.setText("Loading…"); QApplication.processEvents()
            self._stack, self._labels = load_image_stack(self._path, self._log)
            n = self._stack.shape[2]
            self.slider.blockSignals(True)
            self.slider.setMaximum(max(n - 1, 0)); self.slider.setValue(0)
            self.slider.blockSignals(False)
            self._frame_frame.setVisible(n > 1)
            self.lbl_info.setText(
                f"Loaded: {self._stack.shape[0]}×{self._stack.shape[1]} px  |  {n} image(s)")
            self.lbl_info.setStyleSheet("font-size: 11px; color: green;")
            self._refresh()
        except Exception as exc:
            self.lbl_info.setText(f"Error: {exc}")
            self.lbl_info.setStyleSheet("font-size: 11px; color: red;")

    def _log(self, msg: str):
        # Status-bar style logging is optional here; keep it in the info label.
        pass

    # ── Display ─────────────────────────────────────────────────────────────
    def _on_intensity(self):
        self.lbl_lo.setText(f"{self.slider_lo.value()}%")
        self.lbl_hi.setText(f"{self.slider_hi.value()}%")
        self._refresh()

    def _refresh(self):
        if hasattr(self, "chk_dark_bg"):
            self.canvas._dark_bg = self.chk_dark_bg.isChecked()
        if self._stack is None:
            return
        idx = min(self.slider.value(), self._stack.shape[2] - 1)
        img = self._stack[:, :, idx].astype(float)

        if self.chk_histeq.isChecked():
            img = _hist_eq(img)
        if self.chk_invert.isChecked():
            finite = img[np.isfinite(img)]
            if finite.size:
                img = float(finite.max()) + float(finite.min()) - img

        # Colour-bar limits: custom clim wins; otherwise percentile window
        vmin, vmax = self.plot_bar.get_clim()
        if vmin is None or vmax is None:
            finite = img[np.isfinite(img)]
            if finite.size:
                vmin = float(np.percentile(finite, self.slider_lo.value()))
                vmax = float(np.percentile(finite, self.slider_hi.value()))
                if vmax <= vmin:
                    vmax = vmin + 1e-6

        lbl = self._labels[idx] if idx < len(self._labels) else f"image {idx + 1}"
        self.lbl_frame.setText(f"{idx + 1}/{self._stack.shape[2]}  {lbl}")
        ct = self.edit_title.text().strip()
        title = ct or lbl
        self.canvas.show_map(
            img, title, cmap=self.plot_bar.get_cmap(),
            vmin=vmin, vmax=vmax, **self.plot_bar.get_font_sizes())
        self.canvas._img_data = img

    def _on_scroll(self, event):
        if self._stack is None:
            return
        step = 1 if event.step > 0 else -1
        self.slider.setValue(max(0, min(self.slider.maximum(),
                                        self.slider.value() + step)))

    def _toggle_datacursor(self, on: bool):
        self.canvas.set_datacursor(on)
        self._refresh()

    def _export(self):
        from my_gui.fig_export import save_figure, FIG_EXPORT_FILTER
        if self._stack is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", "",
            FIG_EXPORT_FILTER)
        if path:
            try:
                save_figure(self.canvas._fig, path, dpi=300)
                self.lbl_info.setText(f"Saved: {path}")
            except Exception as exc:
                self.lbl_info.setText(f"Export error: {exc}")
