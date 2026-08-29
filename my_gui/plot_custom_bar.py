"""
plot_custom_bar.py
Reusable compact plot-customisation bar: colormap selector + colour-limits + font sizes.
Drop one into any display tab above/below its ROICanvas.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton, QDialog, QSlider,
    QToolButton,
)
from PyQt6.QtCore import pyqtSignal, Qt, QPoint, QSize, QTimer
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QPolygon


_BUILTIN_CMAPS = [
    "gray", "viridis", "plasma", "hot", "inferno", "magma",
    "turbo", "RdYlGn", "coolwarm", "jet", "bone", "copper",
    "hsv", "twilight",
]


# ── Window/Level (brightness–contrast) drag tool — OsiriX-style ────────────
_WL_TOOLTIP = (
    "Window/Level tool — click to activate, then drag over the image:\n"
    "  • drag left/right → contrast (window width)\n"
    "  • drag up/down → brightness (window level)"
)


def make_wl_icon(size: int = 18) -> QIcon:
    """OsiriX-style WW/WL icon: a rounded square split diagonally into a
    light (lower-left) and dark (upper-right) half."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    m = 1
    # Light background (whole rounded square)
    p.setBrush(QColor("#e8e8e8"))
    p.drawRoundedRect(m, m, size - 2 * m, size - 2 * m, 3, 3)
    # Dark upper-right triangle
    p.setBrush(QColor("#141414"))
    p.setClipRect(m, m, size - 2 * m, size - 2 * m)
    p.drawPolygon(QPolygon([
        QPoint(m, m), QPoint(size - m, m), QPoint(size - m, size - m),
    ]))
    # Subtle border
    p.setClipping(False)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QColor("#888888"))
    p.drawRoundedRect(m, m, size - 2 * m - 1, size - 2 * m - 1, 3, 3)
    p.end()
    return QIcon(pm)


def make_cursor_icon(size: int = 18) -> QIcon:
    """Red mouse-cursor arrow — the pixel-value (data-cursor) tool icon."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 18.0
    pts = [(2, 1), (2, 14), (5, 11), (8, 17), (10, 16), (7, 10), (13, 10)]
    poly = QPolygon([QPoint(round(x * s), round(y * s)) for x, y in pts])
    p.setPen(QColor("#1a1a1a"))
    p.setBrush(QColor("#e05555"))
    p.drawPolygon(poly)
    p.end()
    return QIcon(pm)


class WindowLevelToolButton(QToolButton):
    """Checkable OsiriX-style Window/Level tool button (icon + tooltip only).

    Wire its ``toggled(bool)`` signal to whatever activates the drag tool on the
    owning tab's canvas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setIcon(make_wl_icon())
        self.setIconSize(QSize(18, 18))
        self.setCheckable(True)
        self.setToolTip(_WL_TOOLTIP)


class DataCursorToolButton(QToolButton):
    """Checkable red-cursor tool button — toggles the pixel-value data cursor.

    Wire its ``toggled(bool)`` signal to the tab's data-cursor handler.  Kept
    API-compatible with the old checkbox (isChecked / setChecked / toggled)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setIcon(make_cursor_icon())
        self.setIconSize(QSize(18, 18))
        self.setCheckable(True)
        self.setToolTip("Pixel values")


class PlotCustomBar(QWidget):
    """
    Two-row customisation bar:

    Row 1:  Colormap [combo]  cmin [spin] ☐ auto  cmax [spin] ☐ auto
    Row 2:  Fonts: [family]  Title [spin]  Cbar [spin]

    Connect `applied` signal for a redraw callback.
    Every control auto-applies — there is no Apply button. Colormap, colour
    limits and font family redraw immediately; the font-size spinboxes are
    debounced so rapid changes coalesce into a single redraw.
    """

    applied = pyqtSignal()

    def __init__(self, default_cmap: str = "viridis", parent=None,
                 fonts_first: bool = False):
        super().__init__(parent)
        self._cmap_list: list[str] = list(_BUILTIN_CMAPS)
        self._initialising: bool = True

        try:
            from my_gui.colormaps_gui import get_cmap_list
            for cname, _ in get_cmap_list():
                if cname not in self._cmap_list:
                    self._cmap_list.append(cname)
        except Exception:
            pass

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 2, 4, 2)
        root.setSpacing(2)

        # ── Row 1: colormap + clim ────────────────────────────────────────────
        row1 = QHBoxLayout()
        row1.setSpacing(5)

        _cmap_lbl = QLabel("Colormap:")
        _cmap_lbl.setToolTip("Colormap")
        row1.addWidget(_cmap_lbl)
        self.combo_cmap = QComboBox()
        self.combo_cmap.setFixedWidth(110)
        self.combo_cmap.setToolTip("Colormap")
        for c in self._cmap_list:
            self.combo_cmap.addItem(c)
        if default_cmap in self._cmap_list:
            self.combo_cmap.setCurrentText(default_cmap)
        row1.addWidget(self.combo_cmap)

        # clim = one label + two values (min, max) + a single 'auto' checkbox.
        row1.addWidget(QLabel("clim"))
        self.spin_vmin = QDoubleSpinBox()
        self.spin_vmin.setRange(-1e9, 1e9)
        self.spin_vmin.setDecimals(4)
        self.spin_vmin.setFixedWidth(82)
        row1.addWidget(self.spin_vmin)
        row1.addWidget(QLabel("–"))
        self.spin_vmax = QDoubleSpinBox()
        self.spin_vmax.setRange(-1e9, 1e9)
        self.spin_vmax.setDecimals(4)
        self.spin_vmax.setFixedWidth(82)
        row1.addWidget(self.spin_vmax)
        self.chk_auto = QCheckBox("auto")
        self.chk_auto.setChecked(True)
        self.chk_auto.toggled.connect(self._on_auto_toggled)
        self.spin_vmin.setEnabled(False)
        self.spin_vmax.setEnabled(False)
        row1.addWidget(self.chk_auto)

        row1.addStretch()

        # ── Row 2: font family + font sizes ──────────────────────────────────
        row2 = QHBoxLayout()
        row2.setSpacing(5)

        _font_lbl = QLabel("Aa")
        _font_lbl.setToolTip("Title font family")
        _font_lbl.setStyleSheet("color:#aaa; font-size:12px; font-weight:bold;")
        row2.addWidget(_font_lbl)

        # Font family picker (manuscript-friendly options)
        self.combo_font_family = QComboBox()
        self.combo_font_family.setFixedWidth(148)
        self.combo_font_family.setToolTip("Title font family")
        for _ff in ("Default", "Arial", "Times New Roman",
                    "Helvetica", "DejaVu Sans", "DejaVu Serif"):
            self.combo_font_family.addItem(_ff)
        row2.addWidget(self.combo_font_family)

        def _fs(symbol: str, default: int, tip: str) -> QSpinBox:
            _l = QLabel(symbol)
            _l.setToolTip(tip)
            _l.setStyleSheet("color:#ccc; font-size:12px; font-weight:bold;")
            row2.addWidget(_l)
            sp = QSpinBox()
            sp.setRange(6, 40)
            sp.setValue(default)
            sp.setFixedWidth(46)
            sp.setToolTip(tip)
            row2.addWidget(sp)
            return sp

        # "T" = title font size;  "C" = colour-bar tick font size.
        self.spin_fs_title = _fs("T", 13, "Title font size")
        self.spin_fs_cbar  = _fs("C", 9, "Colour-bar tick font size")
        # Axes/Ticks spinboxes omitted — axis("off") makes them irrelevant for maps.

        # Whole-title Bold / Italic toggles — applied via FontProperties, so they
        # style the DEFAULT (auto) title as well as a custom one, with no retyping.
        # Added AFTER the stretch so they sit on the right, directly beside the
        # x²/x₂ buttons that the title format bar appends (B I x² x₂ together).
        _tgl_ss = (
            "QPushButton { border:1px solid #555; border-radius:3px;"
            " padding:1px 5px; font-size:12px; min-width:24px; color:#e0e0e0; }"
            "QPushButton:hover   { background:#2a3050; border-color:#7986cb; }"
            "QPushButton:checked { background:#3a3a6a; border-color:#7aa2f7; color:#ffffff; }"
        )
        self.btn_bold = QPushButton("B"); self.btn_bold.setCheckable(True)
        self.btn_bold.setFixedSize(26, 22); self.btn_bold.setToolTip("Bold title")
        self.btn_bold.setStyleSheet(_tgl_ss + " QPushButton { font-weight: bold; }")
        self.btn_italic = QPushButton("I"); self.btn_italic.setCheckable(True)
        self.btn_italic.setFixedSize(26, 22); self.btn_italic.setToolTip("Italic title")
        self.btn_italic.setStyleSheet(_tgl_ss + " QPushButton { font-style: italic; }")

        row2.addStretch()
        row2.addWidget(self.btn_bold)
        row2.addWidget(self.btn_italic)

        # Keep references so callers can append extra controls to either row.
        self._row_clim  = row1   # Colormap · cmin · cmax · auto
        self._row_fonts = row2   # font family · Title size · Cbar size
        # Row order: colour-bar first by default, fonts first when requested.
        if fonts_first:
            root.addLayout(row2)
            root.addLayout(row1)
        else:
            root.addLayout(row1)
            root.addLayout(row2)

        self._initialising = False

        # Everything auto-applies — there is no Apply button. Colormap, limits and
        # font family redraw immediately; font-size spinboxes are debounced by a
        # short timer so holding an arrow key does not trigger a redraw storm.
        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(160)
        self._apply_timer.timeout.connect(self._auto_apply)

        self.combo_cmap.currentIndexChanged.connect(self._auto_apply)
        self.chk_auto.toggled.connect(self._auto_apply)
        self.spin_vmin.editingFinished.connect(self._auto_apply)
        self.spin_vmax.editingFinished.connect(self._auto_apply)
        self.combo_font_family.currentIndexChanged.connect(self._auto_apply)
        self.btn_bold.toggled.connect(self._auto_apply)
        self.btn_italic.toggled.connect(self._auto_apply)
        # Font sizes now auto-apply (debounced) instead of needing an Apply button.
        self.spin_fs_title.valueChanged.connect(self._schedule_apply)
        self.spin_fs_cbar.valueChanged.connect(self._schedule_apply)

    def _on_auto_toggled(self, checked: bool):
        """Single 'auto' checkbox enables/disables both clim spinboxes."""
        self.spin_vmin.setEnabled(not checked)
        self.spin_vmax.setEnabled(not checked)

    def _schedule_apply(self, *_):
        """Debounced auto-apply for the font-size spinboxes."""
        if not self._initialising:
            self._apply_timer.start()

    def _auto_apply(self):
        if not self._initialising:
            self.applied.emit()

    # ── Public API ─────────────────────────────────────────────────────────

    def get_cmap(self) -> str:
        return self.combo_cmap.currentText()

    def get_clim(self) -> tuple[float | None, float | None]:
        if self.chk_auto.isChecked():
            return None, None
        return self.spin_vmin.value(), self.spin_vmax.value()

    def get_font_sizes(self) -> dict:
        """Return dict with keys title_fs, cbar_fs, title_font, title_bold,
        title_italic (axes/ticks omitted)."""
        ff = self.combo_font_family.currentText()
        return {
            'title_fs':     self.spin_fs_title.value(),
            'cbar_fs':      self.spin_fs_cbar.value(),
            'title_font':   ff if ff != "Default" else "default",
            'title_bold':   self.btn_bold.isChecked(),
            'title_italic': self.btn_italic.isChecked(),
        }

    def font_row(self):
        """The fonts QHBoxLayout — callers may append extra controls to it
        (appended after the trailing stretch, i.e. right-aligned)."""
        return self._row_fonts

    def add_to_clim_row(self, w):
        """Append a widget to the right end of the colour-bar (Colormap) row."""
        self._row_clim.addWidget(w)

    def set_clim_from_data(self, data: np.ndarray):
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return
        self.spin_vmin.setValue(float(finite.min()))
        self.spin_vmax.setValue(float(finite.max()))

    def set_clim(self, vmin: float, vmax: float):
        """Set explicit colour limits (turns off auto) and trigger a redraw."""
        self.chk_auto.setChecked(False)
        self.spin_vmin.setValue(float(vmin))
        self.spin_vmax.setValue(float(vmax))
        self.applied.emit()

    def reset_clim_auto(self):
        """Restore automatic colour limits and redraw."""
        self.chk_auto.setChecked(True)
        self.applied.emit()

    def open_contrast_dialog(self, image_getter, parent=None):
        """Open the interactive contrast (window/level) dialog for the current
        image. `image_getter` returns the displayed 2-D array (or None)."""
        ContrastDialog(self, image_getter, parent or self).show()


class ContrastDialog(QDialog):
    """Interactive contrast tool — two sliders set vmin/vmax over the image
    range and push them to a PlotCustomBar (live redraw)."""

    def __init__(self, plot_bar: "PlotCustomBar", image_getter, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Adjust contrast")
        self.setMinimumWidth(360)
        self._bar = plot_bar
        self._get = image_getter
        import numpy as _np
        img = image_getter()
        fin = _np.asarray(img)[_np.isfinite(img)] if img is not None else _np.array([])
        self._lo = float(fin.min()) if fin.size else 0.0
        self._hi = float(fin.max()) if fin.size else 1.0
        if self._hi <= self._lo:
            self._hi = self._lo + 1.0

        v = QVBoxLayout(self)
        info = QLabel(f"Image range: {self._lo:.4g} … {self._hi:.4g}")
        info.setStyleSheet("color:#aaa;font-size:10px;")
        v.addWidget(info)
        self._s_min = self._slider_row(v, "Min:", 0)
        self._s_max = self._slider_row(v, "Max:", 1000)
        self._lbl = QLabel(""); self._lbl.setStyleSheet("color:#ccc;font-size:11px;")
        v.addWidget(self._lbl)
        row = QHBoxLayout()
        btn_auto = QPushButton("Auto"); btn_auto.clicked.connect(self._auto)
        row.addWidget(btn_auto)
        row.addStretch()
        btn_close = QPushButton("Close"); btn_close.clicked.connect(self.close)
        row.addWidget(btn_close)
        v.addLayout(row)
        self._apply()

    def _slider_row(self, parent_layout, label, init):
        from PyQt6.QtCore import Qt as _Qt
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        s = QSlider(_Qt.Orientation.Horizontal)
        s.setRange(0, 1000); s.setValue(init)
        s.valueChanged.connect(self._apply)
        row.addWidget(s, stretch=1)
        parent_layout.addLayout(row)
        return s

    def _val(self, slider):
        return self._lo + (self._hi - self._lo) * slider.value() / 1000.0

    def _apply(self):
        vmin = self._val(self._s_min)
        vmax = self._val(self._s_max)
        if vmax <= vmin:
            vmax = vmin + (self._hi - self._lo) * 1e-3 + 1e-9
        self._lbl.setText(f"cmin = {vmin:.4g}    cmax = {vmax:.4g}")
        self._bar.set_clim(vmin, vmax)

    def _auto(self):
        self._s_min.setValue(0); self._s_max.setValue(1000)
        self._bar.reset_clim_auto()

    def apply_to_canvas(self, canvas, data: np.ndarray, title: str = ""):
        vmin, vmax = self.get_clim()
        if vmin is None and vmax is None:
            self.set_clim_from_data(data)
        canvas.show_map(data, title=title, cmap=self.get_cmap(),
                        vmin=vmin, vmax=vmax, **self.get_font_sizes())
