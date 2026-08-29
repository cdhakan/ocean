"""
roi_tab.py
Dedicated ROI Manager tab.

All ROI drawing and phantom detection happens here.
Drawn ROIs are broadcast via ROIManager to every other display tab.
Users can also save/load ROI sets to/from .mat files.

Reference image can be loaded from:
  • Any other tab that currently has an image displayed (T1 map, T2 map,
    M0/CEST first frame, MRF first frame, QUESP M0, B1 map)
  • A Bruker folder directly (auto-tries CEST → MRF readers)
  • A .mat file
"""
from __future__ import annotations

import os
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QLineEdit, QGroupBox,
    QComboBox, QFileDialog, QSizePolicy, QScrollArea,
    QListWidget, QInputDialog, QMessageBox, QDialog,
    QDialogButtonBox, QFormLayout, QColorDialog,
    QSpinBox, QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

from my_gui.roi_tools import ROICanvas, ROIPanel, ROI
from my_gui.roi_manager import get_roi_manager
from my_gui.plot_custom_bar import WindowLevelToolButton


class ROITab(QWidget):
    """
    Standalone ROI drawing tab.

    • Load a reference image from any tab, a Bruker folder, or a .mat file
    • Draw/edit ROIs using the full ROIPanel
    • Run phantom outline + auto-detect tubes
    • Save/load ROI sets  (.mat)
    • All changes are pushed to ROIManager → broadcast to all other tabs
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ref_img: np.ndarray | None = None
        self._roi_manager = get_roi_manager()
        # Registry: label → callable that returns (img_array, title)
        # Populated by app.py after all tabs are created
        self._tab_sources: dict[str, callable] = {}
        # TE-aware sources: label → (n_te_fn, te_setter)
        # n_te_fn() returns int (number of TEs), te_setter(idx) sets the active TE
        self._te_sources: dict[str, tuple] = {}
        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────
    # Public API called by app.py
    # ─────────────────────────────────────────────────────────────────────

    def register_image_source(self, label: str, getter_fn):
        """
        Register a tab as an image source for the ROI reference.

        Args:
            label     : display name, e.g. "T1 Map", "M0 (CEST)", "MRF frame 1"
            getter_fn : zero-argument callable that returns (np.ndarray, title_str)
                        or None if nothing is currently displayed.
        """
        self._tab_sources[label] = getter_fn
        # Refresh combo
        current = self.combo_tab_src.currentText()
        self.combo_tab_src.clear()
        self.combo_tab_src.addItem("— select tab source —")
        for lbl in self._tab_sources:
            self.combo_tab_src.addItem(lbl)
        # Restore selection if still valid
        idx = self.combo_tab_src.findText(current)
        self.combo_tab_src.setCurrentIndex(max(idx, 0))

    def register_te_source(self, label: str, n_te_fn, te_setter):
        """
        Register TE selection support for an image source (e.g. 'T2 Images').

        Args:
            label     : must match the label used in register_image_source
            n_te_fn   : zero-argument callable returning int (number of TEs available)
            te_setter : callable(int) that sets the active TE index (0-based)
        """
        self._te_sources[label] = (n_te_fn, te_setter)

    def set_reference_image(self, img: np.ndarray, title: str = "Reference"):
        """Called by app.py or any tab to pre-load an image directly."""
        self._ref_img = np.array(img).squeeze()
        if self._ref_img.ndim > 2:
            self._ref_img = self._ref_img[:, :, 0] if self._ref_img.ndim == 3 else self._ref_img[0]
        H, W = self._ref_img.shape[:2]
        self.lbl_ref_status.setText(f"Loaded from {title}: {H}×{W}")
        self.lbl_ref_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")
        self.le_ref_dir.setText(title)
        self.canvas.show_map(self._ref_img, title, cmap="gray")

    def _on_dark_bg_toggled(self, checked: bool):
        """Flip the canvas black-background theme and re-render the reference."""
        self.canvas._dark_bg = bool(checked)
        if self._ref_img is not None:
            self.canvas.show_map(
                self._ref_img, getattr(self.canvas, "_last_title", ""), cmap="gray")

    # ─────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left panel ────────────────────────────────────────────────────
        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.setSpacing(6)
        left_l.setContentsMargins(6, 6, 6, 6)

        scroll = QScrollArea()
        scroll.setWidget(left_w)
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(420)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_l.addWidget(self._build_image_group())
        left_l.addWidget(self._build_persist_group())

        # ROI panel placeholder — added after canvas is created
        self._roi_ph = QVBoxLayout()
        self._roi_ph.setContentsMargins(0, 0, 0, 0)
        left_l.addLayout(self._roi_ph)
        left_l.addStretch()

        # ── Right panel ───────────────────────────────────────────────────
        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setSpacing(4)
        right_l.setContentsMargins(6, 6, 6, 6)

        # Image tools row — Window/Level (brightness–contrast) drag tool
        _tool_row = QHBoxLayout()
        self.btn_contrast = WindowLevelToolButton()
        self.btn_contrast.setToolTip(
            "Contrast tool (window/level) — toggle on, then drag over the image:\n"
            "  • horizontal → contrast (window width)\n"
            "  • vertical → brightness (window level)")
        self.btn_contrast.toggled.connect(
            lambda checked: self.canvas.set_wl_active(checked))
        _tool_row.addWidget(self.btn_contrast)

        # Black-background toggle (for slides). Only the white surround and
        # labels flip — the map/colormap stays identical.
        self.chk_dark_bg = QCheckBox("Bg")
        self.chk_dark_bg.setToolTip(
            "Black background for the figure (for slides). Only the white "
            "surround and labels flip — the maps stay identical.")
        self.chk_dark_bg.toggled.connect(self._on_dark_bg_toggled)
        _tool_row.addWidget(self.chk_dark_bg)

        _tool_row.addStretch()
        right_l.addLayout(_tool_row)

        self.canvas = ROICanvas()
        right_l.addWidget(self.canvas, stretch=1)

        # ROI panel — full drawing tools with Edit button
        self.roi_panel = _EditableROIPanel(self.canvas)
        self._roi_ph.addWidget(self.roi_panel)

        # The "Mask Tubes → Overlay Map" button was removed from the panel at user
        # request (as of now).  The handler stays wired so the feature can be
        # re-enabled later just by re-adding the button in roi_tools.py.
        self.roi_panel._mask_tubes_fn = self._mask_tubes_overlay
        # After a deep skull-strip, re-slice the 3-D mask to the current view.
        self.roi_panel._on_brain_detected = self._redisplay_slice

        # Move the panel's pixel-value (data-cursor) toggle up next to the
        # contrast tool in this canvas toolbar.
        if hasattr(self.roi_panel, "_dc_check"):
            _tool_row.insertWidget(
                _tool_row.indexOf(self.btn_contrast) + 1, self.roi_panel._dc_check)

        # Wire canvas → roi_manager whenever ROIs change
        self.canvas.roi_added.connect(self._push_to_manager)
        # Also wire panel delete/clear buttons
        self.roi_panel.rois_changed.connect(self._push_to_manager_no_arg)

        splitter.addWidget(scroll)
        splitter.addWidget(right_w)
        splitter.setSizes([410, 620])

    # ─────────────────────────────────────────────────────────────────────

    def _build_image_group(self) -> QGroupBox:
        grp = QGroupBox("Reference Image")
        v = QVBoxLayout(grp)
        v.setSpacing(6)

        # ── Option A: pull from another tab ──────────────────────────────
        lbl_src = QLabel("Load…")
        lbl_src.setStyleSheet("font-size: 11px; font-weight: bold; color: #ccc;")
        v.addWidget(lbl_src)

        src_row = QHBoxLayout()
        self.combo_tab_src = QComboBox()
        self.combo_tab_src.addItem("— select tab source —")
        src_row.addWidget(self.combo_tab_src, stretch=1)
        btn_use_tab = QPushButton("Use this image")
        btn_use_tab.setFixedWidth(110)
        btn_use_tab.setStyleSheet(
            "QPushButton { background:#1a4a6a; color:white; border:none; "
            "border-radius:4px; padding:4px 8px; }"
            "QPushButton:hover { background:#2a6a9a; }"
        )
        btn_use_tab.clicked.connect(self._load_from_tab)
        src_row.addWidget(btn_use_tab)
        v.addLayout(src_row)

        # ── TE selector — shown only when a multi-TE source is selected ───
        self._te_row = QWidget()
        te_lay = QHBoxLayout(self._te_row)
        te_lay.setContentsMargins(0, 0, 0, 0)
        te_lay.setSpacing(6)
        te_lay.addWidget(QLabel("TE:"))
        self._spin_te = QSpinBox()
        self._spin_te.setMinimum(1)
        self._spin_te.setMaximum(1)
        self._spin_te.setValue(1)
        self._spin_te.setFixedWidth(60)
        self._spin_te.setToolTip("Select which echo time (TE) to use as the reference image")
        te_lay.addWidget(self._spin_te)
        self._lbl_te_of = QLabel("of 1")
        self._lbl_te_of.setStyleSheet("color: #888; font-size: 10px;")
        te_lay.addWidget(self._lbl_te_of)
        te_lay.addStretch()
        self._te_row.hide()
        v.addWidget(self._te_row)

        # Show/hide TE row whenever the combo changes
        self.combo_tab_src.currentTextChanged.connect(self._on_src_combo_changed)

        # Divider
        div = QLabel("— or browse directly —")
        div.setStyleSheet("color: #666; font-size: 10px;")
        div.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(div)

        # Hidden field kept for the load handlers that record the loaded path
        self.le_ref_dir = QLineEdit()
        self.le_ref_dir.setReadOnly(True)
        self.le_ref_dir.hide()

        # ── Two unified buttons: any file, or any folder ─────────────────
        browse_row = QHBoxLayout()
        btn_load_file = QPushButton("Load File…")
        btn_load_file.clicked.connect(self._load_ref_file)
        browse_row.addWidget(btn_load_file)

        btn_load_folder = QPushButton("Load Folder…")
        btn_load_folder.clicked.connect(self._load_ref_folder)
        browse_row.addWidget(btn_load_folder)
        v.addLayout(browse_row)

        # Status (shared by all load paths)
        self.lbl_ref_status = QLabel("No reference image loaded.")
        self.lbl_ref_status.setStyleSheet("color: gray; font-size: 10px;")
        self.lbl_ref_status.setWordWrap(True)
        v.addWidget(self.lbl_ref_status)

        # ── Slice navigator — shown only when a 3-D volume is loaded ─────────
        from PyQt6.QtWidgets import (QSlider as _QSlider, QWidget as _QW,
                                     QComboBox as _QCombo, QVBoxLayout as _QVB)
        from PyQt6.QtCore import Qt as _Qt
        self._ref_axis_map = [2, 1, 0]          # Axial / Coronal / Sagittal → array axis
        self._ref_rot = 0                        # number of 90° clockwise rotations
        self._slice_row = _QW()
        _sv = _QVB(self._slice_row); _sv.setContentsMargins(0, 2, 0, 0); _sv.setSpacing(3)
        # orientation (slicing plane) + rotate-clockwise
        _or_lay = QHBoxLayout()
        _or_lay.addWidget(QLabel("View:"))
        self._orient_combo = _QCombo()
        self._orient_combo.addItems(["Axial", "Coronal", "Sagittal"])
        self._orient_combo.setToolTip("Slicing plane through the loaded 3-D volume")
        self._orient_combo.currentIndexChanged.connect(self._on_ref_axis_changed)
        _or_lay.addWidget(self._orient_combo, 1)
        self._btn_rotate = QPushButton("⟳")
        self._btn_rotate.setFixedWidth(32)
        self._btn_rotate.setToolTip("Rotate the view 90° clockwise")
        self._btn_rotate.clicked.connect(self._on_rotate_view)
        _or_lay.addWidget(self._btn_rotate)
        _sv.addLayout(_or_lay)
        # slice slider
        _sl_lay = QHBoxLayout()
        _sl_lay.addWidget(QLabel("Slice:"))
        self._slice_slider = _QSlider(_Qt.Orientation.Horizontal)
        self._slice_slider.setMinimum(0); self._slice_slider.setMaximum(0)
        self._slice_slider.setToolTip("Scroll through slices of the loaded 3-D volume")
        self._slice_slider.valueChanged.connect(self._on_ref_slice_changed)
        _sl_lay.addWidget(self._slice_slider, 1)
        self._slice_label = QLabel("—")
        self._slice_label.setStyleSheet("color:#aaa; font-size:10px; min-width:52px;")
        _sl_lay.addWidget(self._slice_label)
        _sv.addLayout(_sl_lay)
        self._slice_row.setVisible(False)
        v.addWidget(self._slice_row)

        return grp

    def _cur_axis(self) -> int:
        try:
            return self._ref_axis_map[self._orient_combo.currentIndex()]
        except Exception:
            return 2

    def _slice2d(self, vol, idx):
        """Extract the 2-D slice of a 3-D array along the current orientation
        axis at ``idx``, applying the current clockwise rotation."""
        ax = self._cur_axis()
        idx = int(max(0, min(int(idx), vol.shape[ax] - 1)))
        sl = np.take(vol, idx, axis=ax)
        rot = int(getattr(self, "_ref_rot", 0)) % 4
        if rot:
            sl = np.rot90(sl, -rot)          # negative k = clockwise
        return sl

    def _show_slice_slider(self, n_slices: int = 0, cur: int = 0):
        """Show the slice navigator for a 3-D volume (reset to Axial, no
        rotation); hide it for 2-D images.  Args are ignored — the range is
        derived from the current orientation axis of ``_ref_vol``."""
        vol = getattr(self, "_ref_vol", None)
        if vol is None or vol.ndim != 3 or min(vol.shape) < 2:
            self._slice_row.setVisible(False)
            return
        self._ref_rot = 0
        if hasattr(self, "_orient_combo"):
            self._orient_combo.blockSignals(True)
            self._orient_combo.setCurrentIndex(0)          # Axial
            self._orient_combo.blockSignals(False)
        n = vol.shape[self._cur_axis()]
        self._slice_slider.blockSignals(True)
        self._slice_slider.setMaximum(n - 1)
        self._slice_slider.setValue(n // 2)
        self._slice_slider.blockSignals(False)
        self._slice_label.setText(f"{n // 2 + 1}/{n}")
        self._slice_row.setVisible(True)

    def _redisplay_slice(self):
        """Show the current slice (orientation + rotation aware) and re-slice the
        3-D brain mask so the Brain_outline follows the view."""
        vol = getattr(self, "_ref_vol", None)
        if vol is None:
            return
        ax = self._cur_axis()
        idx = int(max(0, min(self._slice_slider.value(), vol.shape[ax] - 1)))
        self._ref_slice = idx
        self._slice_label.setText(f"{idx + 1}/{vol.shape[ax]}")
        self._ref_img = self._slice2d(vol, idx)
        m3d = getattr(self.roi_panel, "_brain_mask_3d", None)
        if m3d is not None and getattr(m3d, "ndim", 0) == 3 and m3d.shape == vol.shape:
            for r in self.canvas.get_rois():
                if getattr(r, "name", "") == "Brain_outline":
                    r.mask = np.asarray(self._slice2d(m3d, idx), dtype=bool)
                    break
        title = getattr(self.canvas, "_last_title", None) or "Reference"
        self.canvas.show_map(self._ref_img, title, cmap="gray")

    def _on_ref_slice_changed(self, idx: int):
        self._redisplay_slice()

    def _on_ref_axis_changed(self, _i: int):
        """Switch the slicing plane (Axial/Coronal/Sagittal) and reset to the
        middle slice of the new axis."""
        vol = getattr(self, "_ref_vol", None)
        if vol is None:
            return
        n = vol.shape[self._cur_axis()]
        self._slice_slider.blockSignals(True)
        self._slice_slider.setMaximum(n - 1)
        self._slice_slider.setValue(n // 2)
        self._slice_slider.blockSignals(False)
        self._redisplay_slice()

    def _on_rotate_view(self):
        """Rotate the displayed slice 90° clockwise."""
        self._ref_rot = (int(getattr(self, "_ref_rot", 0)) + 1) % 4
        self._redisplay_slice()

    def _reset_slice_nav(self):
        """Clear any loaded 3-D volume and hide the slice slider — called by the
        2-D reference loaders so a stale volume is never scrolled."""
        self._ref_vol = None
        if hasattr(self, "_slice_row"):
            self._slice_row.setVisible(False)

    # The Global Analysis Mask is always on: whenever a Brain_outline or
    # Phantom_outline ROI exists (from Detect Brain Outline / Detect Outline),
    # every module masks its maps to it automatically. ROIManager._mask_enabled
    # defaults to True, so no user-facing toggle is needed.

    def _build_persist_group(self) -> QGroupBox:
        grp = QGroupBox("Save / Load ROIs")
        v = QVBoxLayout(grp)
        v.setSpacing(4)

        btn_row = QHBoxLayout()
        btn_save = QPushButton("Save ROIs…")
        btn_save.setFixedHeight(32)
        btn_save.setStyleSheet(
            "QPushButton { background:#226622; color:white; border:none; "
            "border-radius:5px; font-weight:bold; padding:4px 10px; }"
            "QPushButton:hover { background:#338833; }"
        )
        btn_save.clicked.connect(self._save_rois)
        btn_row.addWidget(btn_save)

        btn_load = QPushButton("Load ROIs…")
        btn_load.setFixedHeight(32)
        btn_load.setStyleSheet(
            "QPushButton { background:#224466; color:white; border:none; "
            "border-radius:5px; font-weight:bold; padding:4px 10px; }"
            "QPushButton:hover { background:#336699; }"
        )
        btn_load.clicked.connect(self._load_rois)
        btn_row.addWidget(btn_load)
        v.addLayout(btn_row)

        self.lbl_persist_status = QLabel("")
        self.lbl_persist_status.setStyleSheet("font-size: 10px; color: gray;")
        self.lbl_persist_status.setWordWrap(True)
        v.addWidget(self.lbl_persist_status)

        broadcast_info = QLabel(
            "ROIs will be automatically displayed in other tabs."
        )
        broadcast_info.setWordWrap(True)
        broadcast_info.setStyleSheet("font-size: 10px; color: #4ec9b0;")
        v.addWidget(broadcast_info)

        return grp

    # ─────────────────────────────────────────────────────────────────────
    # Reference image loading callbacks
    # ─────────────────────────────────────────────────────────────────────

    def _on_src_combo_changed(self, label: str):
        """Show TE spinner when a TE-aware source is selected."""
        if label in self._te_sources:
            n_te_fn, _ = self._te_sources[label]
            try:
                n_te = int(n_te_fn())
            except Exception:
                n_te = 1
            n_te = max(n_te, 1)
            self._spin_te.setMaximum(n_te)
            self._spin_te.setValue(1)
            self._lbl_te_of.setText(f"of {n_te}")
            self._te_row.show()
        else:
            self._te_row.hide()

    def _load_from_tab(self):
        """Pull the current image from the selected tab source."""
        self._reset_slice_nav()
        label = self.combo_tab_src.currentText()
        if label == "— select tab source —" or label not in self._tab_sources:
            QMessageBox.information(
                self, "No source selected",
                "Select a tab from the dropdown first.\n\n"
                "Available sources are registered automatically when each tab loads data."
            )
            return

        # If this source supports TE selection, apply the chosen TE first
        if label in self._te_sources:
            _, te_setter = self._te_sources[label]
            te_idx = self._spin_te.value() - 1  # convert 1-based UI → 0-based index
            try:
                te_setter(te_idx)
            except Exception:
                pass

        try:
            result = self._tab_sources[label]()
            if result is None:
                QMessageBox.warning(
                    self, "No image available",
                    f"'{label}' doesn't have an image displayed yet.\n"
                    "Load or compute that tab's data first."
                )
                return
            img, title = result
            self.set_reference_image(img, title=f"{label}: {title}")
        except Exception as exc:
            self.lbl_ref_status.setText(f"Error loading from {label}: {exc}")
            self.lbl_ref_status.setStyleSheet("color: red; font-size: 10px;")

    def _load_ref_file(self):
        """One button → any reference FILE (DICOM / NIfTI / .mat / Bruker 2dseq)."""
        import os
        path, _ = QFileDialog.getOpenFileName(
            self, "Load reference image (DICOM / NIfTI / .mat / Bruker)", "",
            "All supported (*.dcm *.IMA *.nii *.nii.gz *.img *.hdr *.mat *.npz 2dseq);;"
            "All files (*)")
        if not path:
            return
        low  = path.lower()
        base = os.path.basename(path).lower()
        try:
            if low.endswith((".mat", ".npz")):
                self._load_ref_from_file(path)
            elif low.endswith((".dcm", ".ima", ".nii", ".nii.gz", ".img", ".hdr")):
                self._load_ref_from_dicom_nifti(path)
            elif base == "2dseq":
                self._load_ref_from_bruker(os.path.dirname(path))
            else:
                # Unknown extension — try DICOM/NIfTI, then .mat
                try:
                    self._load_ref_from_dicom_nifti(path)
                except Exception:
                    self._load_ref_from_file(path)
        except Exception as exc:
            self.lbl_ref_status.setText(f"Could not load file: {exc}")
            self.lbl_ref_status.setStyleSheet("color: #e57373; font-size: 10px;")

    def _load_ref_folder(self):
        """One button → any reference FOLDER (Bruker scan or DICOM series)."""
        import os, glob
        folder = QFileDialog.getExistingDirectory(
            self, "Load reference image folder (Bruker scan or DICOM series)", "")
        if not folder:
            return
        try:
            is_bruker = (os.path.exists(os.path.join(folder, "2dseq"))
                         or bool(glob.glob(os.path.join(folder, "**", "2dseq"),
                                           recursive=True))
                         or "pdata" in folder.lower())
            if is_bruker:
                self._load_ref_from_bruker(folder)
            else:
                self._load_ref_from_dicom_series(folder)
        except Exception as exc:
            self.lbl_ref_status.setText(f"Could not load folder: {exc}")
            self.lbl_ref_status.setStyleSheet("color: #e57373; font-size: 10px;")

    def _browse_bruker_folder(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Bruker scan folder (pdata/1/)", ""
        )
        if d:
            self._load_ref_from_bruker(d)

    def _browse_mat_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load reference image file", "",
            "Data files (*.mat);;All files (*)"
        )
        if path:
            self._load_ref_from_file(path)

    def _load_ref_from_file(self, path: str):
        self._reset_slice_nav()
        try:
            if path.endswith(".npz"):
                d = dict(np.load(path, allow_pickle=True))
            else:
                import scipy.io as sio
                d = sio.loadmat(path)

            # Prefer known keys: M0image, image, acquired_data — fallback to first 2-D array
            preferred_keys = ["M0image", "image", "acquired_data"]
            img_arr = None
            for pk in preferred_keys:
                if pk in d:
                    arr = np.array(d[pk]).squeeze()
                    if arr.ndim >= 2:
                        img_arr = arr
                        break
            if img_arr is None:
                for k, val in d.items():
                    if k.startswith("_"):
                        continue
                    arr = np.array(val).squeeze()
                    if arr.ndim >= 2:
                        img_arr = arr
                        break

            if img_arr is None:
                self.lbl_ref_status.setText("No 2-D array found in file.")
                return

            # Take first 2-D slice from ND array
            while img_arr.ndim > 2:
                img_arr = img_arr[..., 0]

            self._ref_img = img_arr.astype(float)
            H, W = self._ref_img.shape[:2]
            self.le_ref_dir.setText(path)
            self.lbl_ref_status.setText(f"Loaded {H}×{W} from {os.path.basename(path)}")
            self.lbl_ref_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")
            self.canvas.show_map(self._ref_img, os.path.basename(path), cmap="gray")
        except Exception as exc:
            self.lbl_ref_status.setText(f"Error: {exc}")
            self.lbl_ref_status.setStyleSheet("color: red; font-size: 10px;")

    def _load_ref_from_bruker(self, scan_dir: str):
        """Try multiple Bruker readers in order: CEST M0 → MRF first frame → generic."""
        self._reset_slice_nav()
        # Try CEST (returns M0 reference image)
        try:
            from my_gui.bruker_reader import read_2dseq_cest
            imgs, M0img, info = read_2dseq_cest(scan_dir)
            # Use M0 image as reference
            ref = np.array(M0img).squeeze()
            while ref.ndim > 2:
                ref = ref[..., 0]
            H, W = ref.shape[:2]
            self._ref_img = ref.astype(float)
            self.le_ref_dir.setText(scan_dir)
            self.lbl_ref_status.setText(f"Loaded M0 {H}×{W} from CEST scan")
            self.lbl_ref_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")
            self.canvas.show_map(self._ref_img, "M0 (CEST)", cmap="gray")
            return
        except Exception:
            pass

        # Try MRF reader
        try:
            from my_gui.bruker_reader import read_2dseq_mrf
            imgs, info, _ = read_2dseq_mrf(scan_dir)
            ref = np.array(imgs[:, :, 0, 0]).astype(float)
            H, W = ref.shape[:2]
            self._ref_img = ref
            self.le_ref_dir.setText(scan_dir)
            self.lbl_ref_status.setText(f"Loaded frame 1 {H}×{W} from MRF scan")
            self.lbl_ref_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")
            self.canvas.show_map(self._ref_img, "MRF frame 1", cmap="gray")
            return
        except Exception:
            pass

        # Try QUESP reader
        try:
            from my_gui.bruker_reader import read_2dseq_quesp
            image, M0img, info = read_2dseq_quesp(scan_dir)
            ref = np.array(M0img).squeeze()
            while ref.ndim > 2:
                ref = ref[..., 0]
            H, W = ref.shape[:2]
            self._ref_img = ref.astype(float)
            self.le_ref_dir.setText(scan_dir)
            self.lbl_ref_status.setText(f"Loaded M0 {H}×{W} from QUESP scan")
            self.lbl_ref_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")
            self.canvas.show_map(self._ref_img, "M0 (QUESP)", cmap="gray")
            return
        except Exception:
            pass

        # Try T1/T2 reader (generic 2dseq)
        try:
            from my_gui.bruker_reader import read_2dseq_t1_rarevtr
            imgs, trs = read_2dseq_t1_rarevtr(scan_dir)
            ref = np.array(imgs[:, :, 0, -1]).astype(float)
            H, W = ref.shape[:2]
            self._ref_img = ref
            self.le_ref_dir.setText(scan_dir)
            self.lbl_ref_status.setText(f"Loaded {H}×{W} from T1 scan (last TR)")
            self.lbl_ref_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")
            self.canvas.show_map(self._ref_img, "T1 scan last TR", cmap="gray")
            return
        except Exception:
            pass

        self.lbl_ref_status.setText(
            "Could not read from this folder. Try a different folder or use File…"
        )
        self.lbl_ref_status.setStyleSheet("color: red; font-size: 10px;")

    # ── DICOM / NIfTI loading ─────────────────────────────────────────────

    def _browse_dicom_nifti_file(self):
        """Open a single DICOM or NIfTI file as the reference image."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load DICOM or NIfTI file", "",
            "DICOM / NIfTI (*.dcm *.nii *.nii.gz *.img *.hdr);;"
            "DICOM (*.dcm);;"
            "NIfTI (*.nii *.nii.gz *.img);;"
            "All files (*)"
        )
        if path:
            self._load_ref_from_dicom_nifti(path)

    def _browse_dicom_series_folder(self):
        """Browse a folder containing a DICOM series (all .dcm files stacked)."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select DICOM series folder", ""
        )
        if folder:
            self._load_ref_from_dicom_series(folder)

    def _load_ref_from_dicom_nifti(self, path: str):
        """Read a single DICOM or NIfTI file and display it as the reference."""
        self._ref_vol = None          # reset; set below only for a 3-D NIfTI volume
        try:
            lower = path.lower()
            if lower.endswith(".nii") or lower.endswith(".nii.gz") or lower.endswith(".img"):
                # ── NIfTI ────────────────────────────────────────────────
                try:
                    import nibabel as nib
                except ImportError:
                    self.lbl_ref_status.setText(
                        "nibabel not found — run:  pip install nibabel"
                    )
                    self.lbl_ref_status.setStyleSheet("color: red; font-size: 10px;")
                    return
                nii = nib.load(path)
                arr = np.squeeze(np.array(nii.get_fdata()))
                # Retain the full 3-D volume for deep skull-stripping (the U-Net
                # needs the whole volume, not a single slice).
                _vol = arr
                while _vol.ndim > 3:
                    _vol = _vol[..., 0]
                if _vol.ndim == 3 and min(_vol.shape) >= 2:
                    self._ref_vol = np.asarray(_vol, dtype=float)      # (H, W, D)
                    self._ref_slice = int(_vol.shape[2] // 2)          # display mid-slice
                    arr = _vol[:, :, self._ref_slice]
                else:
                    self._ref_vol = None
                    while arr.ndim > 2:
                        arr = arr[..., 0]
                title = os.path.basename(path)

            else:
                # ── DICOM ────────────────────────────────────────────────
                try:
                    import pydicom
                except ImportError:
                    self.lbl_ref_status.setText(
                        "pydicom not found — run:  pip install pydicom"
                    )
                    self.lbl_ref_status.setStyleSheet("color: red; font-size: 10px;")
                    return
                ds  = pydicom.dcmread(path)
                arr = ds.pixel_array.astype(float)
                # Some DICOM files are RGB; convert to grayscale
                if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                    arr = arr[..., 0]       # take red channel as proxy
                # Siemens MOSAIC → de-tile and use the middle slice as reference
                from my_gui.dicom_mosaic import is_mosaic, mosaic_to_volume
                if arr.ndim == 2 and is_mosaic(ds):
                    vol = mosaic_to_volume(arr, ds)
                    if vol.ndim == 3:
                        arr = vol[:, :, vol.shape[2] // 2]
                title = (
                    getattr(ds, "SeriesDescription", None)
                    or getattr(ds, "StudyDescription", None)
                    or os.path.basename(path)
                )

            ref = arr.astype(float)
            H, W = ref.shape[:2]
            self._ref_img = ref
            self.le_ref_dir.setText(path)
            self.lbl_ref_status.setText(f"Loaded {H}×{W}  —  {title}")
            self.lbl_ref_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")
            self.canvas.show_map(self._ref_img, title, cmap="gray")
            self._show_slice_slider(
                self._ref_vol.shape[2] if self._ref_vol is not None else 0,
                getattr(self, "_ref_slice", 0))

        except Exception as exc:
            self.lbl_ref_status.setText(f"Load error: {exc}")
            self.lbl_ref_status.setStyleSheet("color: red; font-size: 10px;")

    def _pick_series(self, vols):
        """Dialog to choose which converted series to load; returns path or None."""
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QListWidget,
                                     QListWidgetItem, QDialogButtonBox)
        from PyQt6.QtCore import Qt as _Qt
        d = QDialog(self); d.setWindowTitle("Choose series to load"); d.resize(480, 360)
        lay = QVBoxLayout(d)
        lay.addWidget(QLabel("Multiple series found — pick the volume to load.\n"
                             "A 3-D T1 (e.g. MPRAGE) is best for skull-stripping:"))
        lst = QListWidget()
        for v in vols:
            shp = "×".join(str(s) for s in v["shape"])
            it = QListWidgetItem(f"{v['name']}    [{shp}]")
            it.setData(_Qt.ItemDataRole.UserRole, v["path"])
            lst.addItem(it)
        lst.setCurrentRow(0)
        lay.addWidget(lst, 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        lay.addWidget(bb)
        bb.accepted.connect(d.accept); bb.rejected.connect(d.reject)
        lst.itemDoubleClicked.connect(lambda _it: d.accept())
        if d.exec() != QDialog.DialogCode.Accepted:
            return None
        it = lst.currentItem()
        return it.data(_Qt.ItemDataRole.UserRole) if it else None

    def _load_ref_from_dicom_series(self, folder: str):
        """Load a DICOM folder → clean per-series 3-D volume via dcm2niix (needed
        for deep skull-stripping); falls back to a middle-slice stack."""
        self._reset_slice_nav()   # dcm2niix path re-shows the slider via the NIfTI loader
        # ── Preferred path: dcm2niix separates the study's series and builds
        #    proper 3-D volumes.  A single-slice stack (below) is the fallback. ─
        try:
            from PyQt6.QtWidgets import QApplication as _QApp
            from my_gui.dicom_to_nifti import is_available as _d2n_ok, convert_folder
            if _d2n_ok():
                self.lbl_ref_status.setText("Converting DICOM → NIfTI (dcm2niix)…")
                self.lbl_ref_status.setStyleSheet("color:#888; font-size:10px;")
                _QApp.processEvents()
                vols, _out = convert_folder(folder)
                if vols:
                    chosen = (vols[0]["path"] if len(vols) == 1
                              else self._pick_series(vols))
                    if chosen is None:
                        self.lbl_ref_status.setText("Load cancelled.")
                        self.lbl_ref_status.setStyleSheet("color:#888; font-size:10px;")
                        return
                    self._load_ref_from_dicom_nifti(chosen)   # retains _ref_vol (3-D)
                    return
        except Exception:
            pass   # dcm2niix missing/failed → classical slice-stack below
        try:
            try:
                import pydicom
            except ImportError:
                self.lbl_ref_status.setText(
                    "pydicom not found — run:  pip install pydicom"
                )
                self.lbl_ref_status.setStyleSheet("color: red; font-size: 10px;")
                return

            dcm_files = sorted([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(".dcm")
            ])
            if not dcm_files:
                self.lbl_ref_status.setText("No .dcm files found in that folder.")
                self.lbl_ref_status.setStyleSheet("color: red; font-size: 10px;")
                return

            # Read all slices and sort by InstanceNumber (slice position)
            slices = []
            for fp in dcm_files:
                try:
                    ds = pydicom.dcmread(fp, stop_before_pixels=False)
                    slices.append(ds)
                except Exception:
                    continue
            if not slices:
                self.lbl_ref_status.setText("Could not read any DICOM files.")
                self.lbl_ref_status.setStyleSheet("color: red; font-size: 10px;")
                return

            slices.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))

            # Use the middle slice as the 2-D reference image
            mid = slices[len(slices) // 2]
            arr = mid.pixel_array.astype(float)
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                arr = arr[..., 0]
            # Siemens MOSAIC → de-tile and use its middle slice
            from my_gui.dicom_mosaic import is_mosaic, mosaic_to_volume
            if arr.ndim == 2 and is_mosaic(mid):
                vol = mosaic_to_volume(arr, mid)
                if vol.ndim == 3:
                    arr = vol[:, :, vol.shape[2] // 2]

            H, W = arr.shape[:2]
            self._ref_img = arr
            self.le_ref_dir.setText(folder)
            series_desc = getattr(slices[0], "SeriesDescription", os.path.basename(folder))
            self.lbl_ref_status.setText(
                f"DICOM series: {len(slices)} slices  |  reference = slice {len(slices)//2 + 1}  "
                f"|  {H}×{W}  —  {series_desc}"
            )
            self.lbl_ref_status.setStyleSheet("color: #4ec9b0; font-size: 10px;")
            self.canvas.show_map(self._ref_img, series_desc, cmap="gray")

        except Exception as exc:
            self.lbl_ref_status.setText(f"Load error: {exc}")
            self.lbl_ref_status.setStyleSheet("color: red; font-size: 10px;")

    # ─────────────────────────────────────────────────────────────────────
    # Tube-mask overlay
    # 

    def _mask_tubes_overlay(self):
        """
        Build a composite mask from all Tube_* ROIs and display a figure
        showing the parametric map (T1 / T2) masked to tube regions only,
        overlaid on the anatomical reference image.

        Replicates the MATLAB approach:
          ax1  — grayscale anatomical background
          ax2  — colormap-scaled parametric map with alpha = (map > 0)
          Colorbar on right, clim from data, map title shown.

        Displayed in a non-blocking Qt dialog (avoids FigureCanvasAgg warning).
        """
        import numpy as _np
        from matplotlib.figure import Figure
        from matplotlib.colors import Normalize
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from PyQt6.QtWidgets import QInputDialog

        # ── 1. Collect tube ROIs ──────────────────────────────────────
        tube_rois = [r for r in self.canvas.get_rois()
                     if r.name.lower().startswith("tube")]
        if not tube_rois:
            QMessageBox.information(
                self, "No tube ROIs",
                "No Tube ROIs found.\n"
                "Run 'Auto-Detect Tubes' first (or draw tube ROIs manually)."
            )
            return

        # ── 2. Reference image (anatomical background) ────────────────
        if self._ref_img is None:
            QMessageBox.information(
                self, "No reference image",
                "Load a reference image first (T1 or T2 anatomical image)."
            )
            return

        H, W = self._ref_img.shape[:2]
        anat = _np.asarray(self._ref_img, dtype=float)

        # ── 3. Build composite tube mask ─────────────────────────────
        combined_mask = _np.zeros((H, W), dtype=bool)
        for r in tube_rois:
            if r.mask is not None:
                m = _np.asarray(r.mask)
                if m.shape == (H, W):
                    combined_mask |= m

        if not combined_mask.any():
            QMessageBox.warning(
                self, "Empty mask",
                "All tube masks are empty or have a different size than the "
                "reference image.  Reload the reference image and re-detect tubes."
            )
            return

        # ── 4. Pick parametric map (T1 / T2) from registered sources ─
        map_candidates = {
            k: v for k, v in self._tab_sources.items()
            if any(kw in k for kw in ("T1 Images", "T2 Images"))
        }
        param_img   = None
        param_title = ""

        if map_candidates:
            preferred = [k for k in map_candidates if "Map" in k] or list(map_candidates)
            chosen = preferred[0] if len(preferred) == 1 else None
            if chosen is None:
                chosen, ok = QInputDialog.getItem(
                    self, "Select parametric map",
                    "Choose which map to overlay on the tube masks:",
                    preferred, 0, False
                )
                if not ok:
                    return
            result = map_candidates[chosen]()
            if result is not None:
                param_img, param_title = result
                param_img = _np.asarray(param_img, dtype=float).squeeze()
                while param_img.ndim > 2:
                    param_img = param_img[..., 0]

        # ── 5. Build composite T1/T2 map over tube regions ───────────
        # MATLAB: compositeT1(tubeMask) = T1tube(tubeMask)  — voxelwise values
        has_map = param_img is not None and param_img.shape == (H, W)
        if has_map:
            composite = _np.where(combined_mask, param_img, 0.0)
            alpha_data = combined_mask.astype(float)   # 1 inside tubes, 0 outside
            vmax = min(8000.0, float(_np.nanpercentile(param_img[combined_mask], 99)))
            vmin = 0.0
        else:
            # Fallback: show anatomy highlighted at tube positions
            composite = _np.where(combined_mask, anat, 0.0)
            alpha_data = combined_mask.astype(float)
            vmin, vmax = anat.min(), anat.max()
            param_title = "Anatomy (tube regions)"

        # ── 6. Figure: two panels ────────────────────────────────────
        ncols = 2 if has_map else 1
        fig = Figure(figsize=(6.5 * ncols, 5.5), tight_layout=True)

        # --- Left: anatomy + tube outlines ---
        ax_anat = fig.add_subplot(1, ncols, 1)
        ax_anat.imshow(anat, cmap="gray", aspect="equal",
                       origin="upper", interpolation="bilinear")
        # Overlay tube boundaries with colour
        import matplotlib.patches as _mp
        outline = _np.zeros((H, W, 4), dtype=float)
        outline[combined_mask, :] = [1, 0.65, 0, 0.5]   # orange, 50 % alpha
        ax_anat.imshow(outline, aspect="equal", origin="upper",
                       interpolation="nearest")
        ax_anat.set_title(f"Anatomy + tube masks  (n={len(tube_rois)})",
                          fontsize=11)
        ax_anat.axis("off")

        # --- Right: parametric map (MATLAB-style two-axes overlay) ---
        if ncols == 2:
            ax_map = fig.add_subplot(1, 2, 2)
            # Layer 1: gray anatomical background
            ax_map.imshow(anat, cmap="gray", aspect="equal",
                          origin="upper", interpolation="bilinear")
            # Layer 2: parametric map, visible only inside tube mask
            # Build RGBA array so alpha follows combined_mask exactly
            cmap = _get_t1_cmap()
            norm = Normalize(vmin=vmin, vmax=vmax)
            rgba = cmap(norm(composite))           # (H, W, 4)
            rgba[..., 3] = alpha_data              # 0 outside tubes, 1 inside
            im = ax_map.imshow(rgba, aspect="equal",
                               origin="upper", interpolation="nearest")
            # Invisible scalar-mappable for colorbar (ScalarMappable)
            import matplotlib.cm as _cm
            sm = _cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax_map, fraction=0.046, pad=0.04)
            cbar.set_label("ms", fontsize=10)
            cbar.ax.tick_params(labelsize=9)
            ax_map.set_title(f"{param_title}  (tube mask)", fontsize=11)
            ax_map.axis("off")

        n_tube_px = int(combined_mask.sum())
        fig.suptitle(
            f"Tube-masked overlay  —  {len(tube_rois)} tube(s)  |  {n_tube_px} px",
            fontsize=12, fontweight="bold",
        )

        # ── 7. Show in a resizable Qt dialog (non-blocking) ──────────
        dlg = QDialog(self)
        dlg.setWindowTitle("Tube-Mask Parametric Overlay")
        dlg.resize(700 * ncols, 580)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        canvas = FigureCanvasQTAgg(fig)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(canvas)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        lay.addWidget(btn_close)
        dlg.show()   # non-blocking — user can interact with GUI while overlay is open

    # 
    # ROI manager push
    # ─────────────────────────────────────────────────────────────────────

    def _push_to_manager(self, roi=None):
        """Push current canvas ROIs to ROIManager (broadcast to all tabs)."""
        if self.canvas._img_shape is not None:
            ref_shape = self.canvas._img_shape
        else:
            ref_shape = (1, 1)
        self._roi_manager.update_rois(self.canvas.get_rois(), ref_shape)

    def _push_to_manager_no_arg(self):
        self._push_to_manager()

    # ─────────────────────────────────────────────────────────────────────
    # Save / Load ROIs
    # ─────────────────────────────────────────────────────────────────────

    def _save_rois(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ROIs", "rois.mat",
            "MAT files (*.mat);;All files (*)"
        )
        if not path:
            return
        try:
            self._push_to_manager()
            self._roi_manager.save(path)
            n = len(self.canvas.get_rois())
            self.lbl_persist_status.setText(f"Saved {n} ROIs → {os.path.basename(path)}")
            self.lbl_persist_status.setStyleSheet("font-size: 10px; color: #4ec9b0;")
        except Exception as exc:
            self.lbl_persist_status.setText(f"Save error: {exc}")
            self.lbl_persist_status.setStyleSheet("font-size: 10px; color: red;")

    def _load_rois(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load ROIs", "",
            "MAT files (*.mat);;All files (*)"
        )
        if not path:
            return
        try:
            self._roi_manager.load(path)
            # Use shape-scaled ROIs when the canvas already has a reference image
            # loaded — this prevents mask/image dimension mismatches on click.
            if self.canvas._img_shape is not None:
                rois = self._roi_manager.get_rois_for_shape(self.canvas._img_shape)
            else:
                rois = self._roi_manager.get_rois()
            # Rebuild canvas ROIs
            self.canvas._rois = []
            self.canvas.add_rois(rois)
            self.roi_panel._roi_list.clear()
            for roi in rois:
                self.roi_panel._roi_list.addItem(roi.name)
            self.lbl_persist_status.setText(
                f"Loaded {len(rois)} ROIs from {os.path.basename(path)}"
            )
            self.lbl_persist_status.setStyleSheet("font-size: 10px; color: #4ec9b0;")
        except Exception as exc:
            self.lbl_persist_status.setText(f"Load error: {exc}")
            self.lbl_persist_status.setStyleSheet("font-size: 10px; color: red;")


# ─────────────────────────────────────────────────────────────────────────────
# Extended ROI panel with Edit button
# ─────────────────────────────────────────────────────────────────────────────

class _EditableROIPanel(ROIPanel):
    """ROIPanel subclass that adds an Edit ROI button and exposes rois_changed signal."""

    rois_changed = pyqtSignal()

    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)

    def _build_ui(self):
        super()._build_ui()
        # Insert Edit button next to Rename/Delete in the btn_row
        self._edit_btn = QPushButton("Edit…")
        self._edit_btn.setToolTip("Rename or change color of selected ROI")
        self._edit_btn.clicked.connect(self._edit_roi)

        try:
            btn_parent = self._rename_btn.parent()
            if btn_parent is not None:
                lay = btn_parent.layout()
                if lay is not None:
                    lay.insertWidget(1, self._edit_btn)
        except Exception:
            pass

    def _edit_roi(self):
        row = self._roi_list.currentRow()
        if row < 0:
            QMessageBox.information(self, "No ROI selected", "Select a ROI first.")
            return
        rois = self._canvas.get_rois()
        if row >= len(rois):
            return
        roi = rois[row]
        dlg = _ROIEditDialog(roi.name, roi.color, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_name  = dlg.get_name()
            new_color = dlg.get_color()
            if new_name and new_name != roi.name:
                self._canvas.rename_roi(roi.name, new_name)
                self._roi_list.item(row).setText(new_name)
            for r in self._canvas.get_rois():
                if r.name == new_name:
                    r.color = new_color
                    break
            self.rois_changed.emit()

    def _delete_roi(self):
        super()._delete_roi()
        self.rois_changed.emit()

    def _clear_all(self):
        super()._clear_all()
        self.rois_changed.emit()


# ─────────────────────────────────────────────────────────────────────────────
# T1 / T2 colormap helper (approximates MATLAB T1cm.mat)
# ─────────────────────────────────────────────────────────────────────────────

def _get_t1_cmap():
    """
    Return a matplotlib colormap that approximates the MATLAB T1cm colormap:
    black (0 ms) → dark blue → cyan → green → yellow → white (8000 ms).
    Falls back to 'turbo' if matplotlib < 3.2.
    """
    import matplotlib.colors as mcolors
    try:
        colors = [
            (0.00, (0.00, 0.00, 0.00)),   # 0    — black
            (0.10, (0.10, 0.00, 0.50)),   # 800  — dark purple
            (0.25, (0.00, 0.20, 0.80)),   # 2000 — blue
            (0.40, (0.00, 0.75, 0.90)),   # 3200 — cyan
            (0.55, (0.20, 0.85, 0.20)),   # 4400 — green
            (0.70, (0.90, 0.90, 0.00)),   # 5600 — yellow
            (0.85, (1.00, 0.60, 0.10)),   # 6800 — orange
            (1.00, (1.00, 1.00, 1.00)),   # 8000 — white
        ]
        return mcolors.LinearSegmentedColormap.from_list(
            "T1cm", [(v, c) for v, c in colors]
        )
    except Exception:
        import matplotlib.pyplot as _plt
        return _plt.get_cmap("turbo")


class _ROIEditDialog(QDialog):
    """Small dialog to rename a ROI and change its color."""

    def __init__(self, name: str, color: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit ROI")
        self._color = color

        form = QFormLayout(self)
        self._name_edit = QLineEdit(name)
        form.addRow("Name:", self._name_edit)

        self._color_btn = QPushButton("  ")
        self._color_btn.setFixedWidth(60)
        self._color_btn.setStyleSheet(f"background: {color};")
        self._color_btn.clicked.connect(self._pick_color)
        form.addRow("Color:", self._color_btn)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def _pick_color(self):
        c = QColorDialog.getColor(QColor(self._color), self)
        if c.isValid():
            self._color = c.name()
            self._color_btn.setStyleSheet(f"background: {self._color};")

    def get_name(self) -> str:
        return self._name_edit.text().strip()

    def get_color(self) -> str:
        return self._color
