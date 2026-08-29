from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector, EllipseSelector, LassoSelector, PolygonSelector
from matplotlib.path import Path as MplPath
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QLabel,
    QComboBox, QLineEdit, QGroupBox, QSizePolicy, QDoubleSpinBox, QSpinBox,
    QCheckBox, QInputDialog, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from scipy.ndimage import (
    gaussian_filter, binary_closing, binary_fill_holes, binary_opening,
    binary_erosion, binary_dilation, binary_propagation,
    distance_transform_edt, label as ndlabel,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROI_COLORS = [
    "#ff4444", "#44ff88", "#4488ff", "#ffff44",
    "#ff44ff", "#44ffff", "#ff8844", "#88ff44",
]

# ---------------------------------------------------------------------------
# ROI dataclass
# ---------------------------------------------------------------------------

@dataclass
class ROI:
    name: str
    roi_type: str          # 'rectangle','ellipse','polygon','freehand','auto'
    mask: np.ndarray       # bool H×W
    color: str             # hex string

    def stats(self, data: np.ndarray) -> dict:
        """Return mean/std/min/max/n for pixels selected by the mask."""
        arr = np.asarray(data)
        # Ensure mask matches data spatial dims (last two dims if 3-D)
        msk = self.mask
        if arr.ndim == 3:
            arr2d = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
        else:
            arr2d = arr
        # Guard: resize mask if image has been resized since the ROI was created
        if msk.shape[:2] != arr2d.shape[:2]:
            try:
                from scipy.ndimage import zoom
                zy = arr2d.shape[0] / max(msk.shape[0], 1)
                zx = arr2d.shape[1] / max(msk.shape[1], 1)
                msk = zoom(msk.astype(float), (zy, zx), order=1) > 0.5
            except Exception:
                return {"mean": float("nan"), "std": float("nan"),
                        "min": float("nan"), "max": float("nan"), "n": 0}
        vals = arr2d[msk]
        if vals.size == 0:
            return {"mean": float("nan"), "std": float("nan"),
                    "min": float("nan"), "max": float("nan"), "n": 0}
        return {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
            "n":    int(vals.size),
        }

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _otsu_threshold(img: np.ndarray) -> float:
    """Compute Otsu threshold using only numpy (no sklearn/skimage)."""
    flat = img.ravel().astype(np.float64)
    # Build 256-bin histogram over [0,1]
    nbins = 256
    hist, bin_edges = np.histogram(flat, bins=nbins, range=(0.0, 1.0))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5

    # Cumulative sums
    w0 = np.cumsum(hist) / total
    w1 = 1.0 - w0
    mu0 = np.cumsum(hist * bin_centers) / (np.cumsum(hist) + 1e-12)
    mu_total = np.sum(hist * bin_centers) / total
    mu1 = np.where(w1 > 1e-12,
                   (mu_total - w0 * mu0) / (w1 + 1e-12),
                   0.0)

    sigma_b2 = w0 * w1 * (mu0 - mu1) ** 2
    idx = int(np.argmax(sigma_b2))
    return float(bin_centers[idx])


def _mask_from_rect(eclick, erelease, shape) -> np.ndarray:
    """Bool mask from RectangleSelector eclick/erelease callbacks."""
    H, W = shape[:2]
    x1, x2 = sorted([eclick.xdata, erelease.xdata])
    y1, y2 = sorted([eclick.ydata, erelease.ydata])
    xs = np.arange(W)
    ys = np.arange(H)
    xg, yg = np.meshgrid(xs, ys)
    mask = (xg >= x1) & (xg <= x2) & (yg >= y1) & (yg <= y2)
    return mask.astype(bool)


def _mask_from_ellipse(eclick, erelease, shape) -> np.ndarray:
    """Bool mask from EllipseSelector eclick/erelease callbacks."""
    H, W = shape[:2]
    x1, x2 = sorted([eclick.xdata, erelease.xdata])
    y1, y2 = sorted([eclick.ydata, erelease.ydata])
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rx = (x2 - x1) / 2.0 + 1e-9
    ry = (y2 - y1) / 2.0 + 1e-9
    xs = np.arange(W)
    ys = np.arange(H)
    xg, yg = np.meshgrid(xs, ys)
    mask = ((xg - cx) / rx) ** 2 + ((yg - cy) / ry) ** 2 <= 1.0
    return mask.astype(bool)


def _mask_from_verts(verts, shape) -> np.ndarray:
    """Bool mask from polygon/lasso vertices using matplotlib.path.Path."""
    H, W = shape[:2]
    path = MplPath(verts)
    xs = np.arange(W)
    ys = np.arange(H)
    xg, yg = np.meshgrid(xs, ys)
    points = np.column_stack([xg.ravel(), yg.ravel()])
    mask = path.contains_points(points).reshape(H, W)
    return mask.astype(bool)

# ---------------------------------------------------------------------------
# Phantom detection
# ---------------------------------------------------------------------------

def detect_phantom_outline(img: np.ndarray, gauss_sigma: float = 4.0) -> np.ndarray:
    """
    Port of T1_T2_new.m phantom outline detection.

    Steps:
      1. Gaussian smooth
      2. Normalize to [0, 1]
      3. Otsu threshold → binary image
      4. binary_closing with 11×11 structuring element
      5. binary_fill_holes
      6. Keep largest connected component
      7. binary_opening
      8. Return bool mask
    """
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.mean(axis=0)

    # 1. Gaussian smooth
    smoothed = gaussian_filter(arr, sigma=gauss_sigma)

    # 2. Normalize
    mn, mx = smoothed.min(), smoothed.max()
    if mx - mn < 1e-12:
        return np.zeros(arr.shape[:2], dtype=bool)
    normalized = (smoothed - mn) / (mx - mn)

    # 3. Otsu threshold
    thresh = _otsu_threshold(normalized)
    binary = normalized > thresh

    # 4. binary_closing with 11×11 structuring element
    struct = np.ones((11, 11), dtype=bool)
    closed = binary_closing(binary, structure=struct)

    # 5. Fill holes
    filled = binary_fill_holes(closed)

    # 6. Keep largest connected component
    labeled, n_comp = ndlabel(filled)
    if n_comp == 0:
        return np.zeros(arr.shape[:2], dtype=bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # background
    largest_label = int(np.argmax(sizes))
    largest = labeled == largest_label

    # 7. binary_opening
    opened = binary_opening(largest)

    return opened.astype(bool)


def _disk(radius) -> np.ndarray:
    """Boolean disk structuring element of the given radius (>=0)."""
    r = int(round(radius))
    if r < 1:
        return np.array([[True]])
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r + 1e-9


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    """Boolean mask of the single largest 8-connected component."""
    if not mask.any():
        return mask
    lbl, n = ndlabel(mask, structure=np.ones((3, 3), dtype=bool))
    if n <= 1:
        return mask
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    return lbl == int(counts.argmax())


def detect_brain_outline(img: np.ndarray, gauss_sigma: float = None,
                         erode_radius: int = None) -> np.ndarray:
    """
    Skull-strip a single 2-D brain slice and return a BRAIN-ONLY boolean mask.

    Unlike :func:`detect_phantom_outline` (which traces the whole object), this
    excludes the outer scalp/skull rim and keeps only the intracranial region
    (a single, solid, smoothly-bounded blob — cf. a BET-style extraction).

    It is **contrast-agnostic**: it does not assume the brain is bright.  The
    only anatomical invariant it relies on is that the brain is the region
    *enclosed* by the outer scalp/skull shell (fat and marrow are bright on
    essentially every sequence, so that shell is bright even when the brain is
    dark, e.g. post-contrast T1).  This makes it work on both bright-brain
    (T2 / CEST / pre-contrast T1) and dark-brain (post-contrast T1) slices, and
    across sizes from ~64² CEST maps to 512² anatomicals.

    Strategy (enclosed-cavity):
      1. Build the whole-head silhouette — threshold above air, then CLOSE the
         dark skull gap *before* taking the largest component so a complete dark
         skull ring cannot disconnect the scalp from the brain.
      2. Find the outer scalp/skull shell = bright tissue in the outer band of
         the head, closed into a continuous annulus.  Interior bright vessels
         are excluded because they are not in the outer band.
      3. The brain is the region enclosed by that annulus: fill the annulus hole
         and subtract the annulus.  Hole-filling swallows interior vessels and
         ventricles in one step (no fragmentation).
      4. If the shell failed to close (empty / near-whole-head cavity), fall back
         to a purely geometric peel of the head by the shell thickness.
      5. Fill holes, keep one component, lightly smooth the boundary.

    ``gauss_sigma`` and ``erode_radius`` (the scalp+skull shell thickness in px)
    default to scale-aware values derived from the slice size.
    """
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim > 2:
        arr = arr[..., 0] if arr.shape[-1] < arr.shape[0] else arr.mean(axis=0)
    shape = arr.shape[:2]
    zeros = np.zeros(shape, dtype=bool)
    if arr.size == 0:
        return zeros
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    lo, hi = float(np.percentile(arr, 1.0)), float(np.percentile(arr, 99.0))
    if not np.isfinite(hi) or (hi - lo) < 1e-6:
        return zeros

    # Scale-aware constants.
    dim = float(min(shape))
    sig = gauss_sigma if gauss_sigma is not None else max(1.0, dim / 110.0)
    r = int(erode_radius) if erode_radius is not None else int(round(max(3.0, dim / 18.0)))

    sm = gaussian_filter(arr, sigma=sig)
    norm = np.clip((sm - lo) / (hi - lo), 0.0, 1.0)
    otsu = _otsu_threshold(norm)
    if not np.isfinite(otsu) or otsu <= 0:
        otsu = 0.3

    # 1. Whole-head silhouette (scalp+skull+brain).  Close the dark skull gap
    #    BEFORE the largest-component step (a complete dark ring would otherwise
    #    disconnect the scalp from the brain and the brain alone would win).
    thr_head = max(0.08, 0.35 * otsu)
    head = binary_fill_holes(_largest_cc(binary_closing(norm > thr_head, _disk(r))))
    head = binary_fill_holes(binary_closing(head, _disk(max(2, r // 2))))
    if not head.any():
        return zeros

    # 2. Outer scalp/skull shell = bright tissue in the head's outer band, closed
    #    into a continuous annulus (interior vessels are not in the outer band).
    bright = norm > otsu
    outer_band = head & ~binary_erosion(head, _disk(int(round(1.7 * r))))
    shell = binary_closing(bright & outer_band, _disk(r)) & head

    # 3. Brain = region enclosed by the annulus (fill the hole, drop the annulus).
    cavity = binary_fill_holes(shell) & ~shell & binary_erosion(head, _disk(1))
    brain = _largest_cc(cavity)

    # 4. Fallback (shell didn't close): purely geometric peel of the head.
    ha = int(head.sum())
    if (not brain.any()) or brain.sum() < 0.04 * ha or brain.sum() > 0.97 * ha:
        core = _largest_cc(binary_erosion(head, _disk(r)))
        brain = binary_dilation(core, _disk(max(1, r // 3))) & head
        brain = _largest_cc(brain)
        if not brain.any():
            return zeros

    # 5. Tidy: fill holes, close, lightly smooth, keep one component.
    brain = _largest_cc(binary_fill_holes(brain))
    brain = _largest_cc(binary_fill_holes(binary_closing(brain, _disk(max(2, r // 3)))))
    smooth = gaussian_filter(brain.astype(np.float64), sigma=1.5) > 0.5
    smooth = _largest_cc(binary_fill_holes(smooth))
    if not smooth.any():
        return brain.astype(bool)
    return smooth.astype(bool)


def _clear_border(binary_mask: np.ndarray) -> np.ndarray:
    """Remove connected components that touch any image border (port of imclearborder)."""
    labeled, n = ndlabel(binary_mask)
    if n == 0:
        return binary_mask.copy()
    border_labels: set = set()
    for edge in (labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]):
        border_labels.update(edge.ravel().tolist())
    border_labels.discard(0)
    if not border_labels:
        return binary_mask.copy()
    result = binary_mask.copy()
    for lbl in border_labels:
        result[labeled == lbl] = False
    return result


def detect_phantom_tubes(
    img: np.ndarray,
    gauss_sigma: float = 4.0,
    min_tube_pixels: int = 10,
    roi_name_prefix: str = "Tube",
    sort_circular: bool = True,
) -> List[ROI]:
    """
    Port of autoDetectTubes.m (modified 12/08/2025).

    Algorithm:
      1. Detect phantom outline via Otsu on smoothed image (keep largest CC + fill + open)
      2. Background subtraction: diff = img - gaussian_smooth(img)
      3. Otsu-threshold diff image
      4. Apply phantom mask → keep only inside-phantom candidates
      5. Remove largest connected component (phantom wall artefact)
      6. Remove small components (< min_tube_pixels)
      7. Fill holes in remaining blobs
      8. Clear components touching the image border
      9. Label and sort by angular position from phantom centroid

    Returns a list of ROI objects, one per detected tube.
    """
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.mean(axis=0)

    # ── Step 1: Phantom outline ───────────────────────────────────────────
    smoothed = gaussian_filter(arr, sigma=gauss_sigma)
    mn, mx = smoothed.min(), smoothed.max()
    if mx - mn < 1e-12:
        return []
    normalized = (smoothed - mn) / (mx - mn)
    thresh_ph = _otsu_threshold(normalized)
    bw_phantom = normalized > thresh_ph

    # Keep largest connected component
    labeled_ph, n_ph = ndlabel(bw_phantom)
    if n_ph == 0:
        return []
    sizes_ph = np.bincount(labeled_ph.ravel())
    sizes_ph[0] = 0
    phantom_label = int(np.argmax(sizes_ph))
    phantom_mask = labeled_ph == phantom_label
    phantom_mask = binary_fill_holes(phantom_mask)
    phantom_mask = binary_opening(phantom_mask, structure=np.ones((5, 5), dtype=bool))

    # ── Step 2: Background subtraction ───────────────────────────────────
    bg   = gaussian_filter(arr, sigma=gauss_sigma)
    diff = arr - bg                 # positive where bright tubes stick out

    # ── Step 3: Binarise the difference image (Otsu) ──────────────────────
    d_min, d_max = diff.min(), diff.max()
    if d_max - d_min < 1e-12:
        return []
    diff_norm = (diff - d_min) / (d_max - d_min)
    thresh_t  = _otsu_threshold(diff_norm)
    bw        = diff_norm > thresh_t

    # ── Step 4: Apply phantom mask ────────────────────────────────────────
    bw[~phantom_mask] = False

    # ── Step 5: Remove largest component (phantom-wall artefact) ─────────
    labeled_t, n_t = ndlabel(bw)
    if n_t == 0:
        return []
    sizes_t = np.bincount(labeled_t.ravel())
    sizes_t[0] = 0
    idx_max = int(np.argmax(sizes_t))
    bw[labeled_t == idx_max] = False

    # ── Step 6: Remove small components ───────────────────────────────────
    labeled_t2, n_t2 = ndlabel(bw)
    if n_t2 == 0:
        return []
    sizes_t2 = np.bincount(labeled_t2.ravel())
    sizes_t2[0] = 0
    for lbl in range(1, n_t2 + 1):
        if sizes_t2[lbl] < min_tube_pixels:
            bw[labeled_t2 == lbl] = False

    # ── Step 7: Fill holes ────────────────────────────────────────────────
    bw = binary_fill_holes(bw)

    # ── Step 8: Clear border ──────────────────────────────────────────────
    bw = _clear_border(bw)

    # ── Step 9: Label final tubes ─────────────────────────────────────────
    labeled_final, n_final = ndlabel(bw)
    if n_final == 0:
        return []

    # Compute centroids
    H, W = arr.shape
    centroids = []
    for lbl in range(1, n_final + 1):
        ys, xs = np.where(labeled_final == lbl)
        if len(ys) == 0:
            continue
        centroids.append((lbl, float(ys.mean()), float(xs.mean())))

    # ── Sort by counter-clockwise angle from phantom centroid ─────────────
    if sort_circular and len(centroids) > 1:
        ph_ys, ph_xs = np.where(phantom_mask)
        ph_cy = float(ph_ys.mean())
        ph_cx = float(ph_xs.mean())

        def _ccw_angle(item):
            _, cy, cx = item
            return float(np.arctan2(cy - ph_cy, cx - ph_cx))

        centroids.sort(key=_ccw_angle)

    # ── Build ROI objects ─────────────────────────────────────────────────
    rois: List[ROI] = []
    for idx, (lbl, cy, cx) in enumerate(centroids):
        mask  = (labeled_final == lbl).astype(bool)
        color = _ROI_COLORS[idx % len(_ROI_COLORS)]
        name  = f"{roi_name_prefix}_{idx + 1}"
        rois.append(ROI(name=name, roi_type="auto", mask=mask, color=color))

    return rois


def roi_union_mask(rois, shape, exclude=("Phantom_outline", "Brain_outline")):
    """Boolean OR of every drawn ROI's mask (excluding the whole-phantom / brain
    outline ROIs), nearest-neighbour-resized to ``shape`` (H, W).

    Returns ``None`` when no qualifying ROI has a usable 2-D mask — callers use
    that to fall back to the normal (whole-map) display.  Shared by the
    "ROIs only" and "ROIs + Bg" display modes across every map tab."""
    union = None
    H, W = int(shape[0]), int(shape[1])
    for r in rois or []:
        if getattr(r, "name", "") in exclude:
            continue
        m = getattr(r, "mask", None)
        if m is None:
            continue
        m = np.asarray(m)
        if m.ndim != 2:
            continue
        if m.shape != (H, W):
            ys = np.linspace(0, m.shape[0] - 1, H).round().astype(int)
            xs = np.linspace(0, m.shape[1] - 1, W).round().astype(int)
            m = m[np.ix_(ys, xs)]
        m = m.astype(bool)
        union = m if union is None else (union | m)
    return union


def choose_background_image(parent, scan_paths, pv360: bool = False):
    """Dialog to pick the grayscale background for the 'ROIs + Bkg' overlay.

    Lists the assigned Scan-Directory scans (``scan_paths`` = {modality: path}),
    with 'Browse folder…' for any other Bruker scan and 'Use default'.  Returns:
      ('set', 2-D array)  — a chosen image loaded successfully,
      ('default', None)   — revert to the tab's own 1st raw frame,
      ('keep', None)      — cancelled (no change)."""
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
        QPushButton, QDialogButtonBox, QFileDialog, QMessageBox,
    )
    from my_gui.bruker_reader import load_scan_first_frame

    d = QDialog(parent)
    d.setWindowTitle("Background image")
    d.resize(460, 400)
    lay = QVBoxLayout(d)
    lay.addWidget(QLabel(
        "Choose the image shown in gray behind the ROI colours\n"
        "(from the assigned Scan-Directory scans, or browse to any):"))
    lst = QListWidget()
    for k, p in (scan_paths or {}).items():
        if not p:
            continue
        it = QListWidgetItem(f"{k}   —   {p}")
        it.setData(Qt.ItemDataRole.UserRole, p)
        lst.addItem(it)
    lay.addWidget(lst, 1)
    if lst.count() == 0:
        _hint = QLabel("No scans assigned in the Scan Directory — use Browse folder…")
        _hint.setStyleSheet("color:#999;font-size:11px;")
        lay.addWidget(_hint)

    row = QHBoxLayout()
    btn_browse = QPushButton("Browse folder…")
    btn_default = QPushButton("Use default (this scan)")
    row.addWidget(btn_browse); row.addWidget(btn_default); row.addStretch()
    lay.addLayout(row)
    bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                          QDialogButtonBox.StandardButton.Cancel)
    lay.addWidget(bb)

    out = {"mode": "keep", "img": None}

    def _load(p):
        img = load_scan_first_frame(p, pv360=pv360)
        if img is None:
            QMessageBox.warning(d, "Can't read",
                                f"Could not read a 2dseq image from:\n{p}")
            return
        out["mode"] = "set"; out["img"] = img; d.accept()

    def _browse():
        p = QFileDialog.getExistingDirectory(
            d, "Select a Bruker scan folder (…/pdata/1)")
        if p:
            _load(p)

    def _use_default():
        out["mode"] = "default"; d.accept()

    def _ok():
        it = lst.currentItem()
        if it is None:
            QMessageBox.information(
                d, "Pick a scan", "Select a scan, or use Browse / Use default.")
            return
        _load(it.data(Qt.ItemDataRole.UserRole))

    btn_browse.clicked.connect(_browse)
    btn_default.clicked.connect(_use_default)
    bb.accepted.connect(_ok)
    bb.rejected.connect(d.reject)
    lst.itemDoubleClicked.connect(lambda it: _load(it.data(Qt.ItemDataRole.UserRole)))
    d.exec()
    return out["mode"], out["img"]


# ---------------------------------------------------------------------------
# ROICanvas
# ---------------------------------------------------------------------------

class ROICanvas(FigureCanvas):
    roi_added = pyqtSignal(object)

    def __init__(self, parent=None):
        self._fig = Figure(tight_layout=True)
        super().__init__(self._fig)
        if parent is not None:
            self.setParent(parent)

        self._ax = self._fig.add_subplot(111)
        self._img_data: Optional[np.ndarray] = None
        self._img_shape: Optional[tuple] = None
        self._rois: List[ROI] = []
        self._roi_patches: dict = {}
        self._selector = None
        self._draw_type: Optional[str] = None
        self._next_roi_name: str = "ROI_1"
        self._color_idx: int = 0
        self._dc_active: bool = False
        self._dc_annot = None
        self._dc_cid = None
        self._rois_visible: bool = True          # hide/show ROI overlays
        self._selected_roi_name: Optional[str] = None  # highlighted ROI
        self._dark_bg: bool = False              # black figure background (maps unchanged)
        self._log_map: bool = False              # Fuderer perceptual log-remap of the colormap

        # Store current display params for re-rendering after remove_roi
        self._last_title: str = ""
        self._last_cmap: str = "gray"
        self._last_vmin = None
        self._last_vmax = None

        # ROI shape-editing state
        self._edit_mode: bool = False          # True when drag-edit is active
        self._edit_roi_name: Optional[str] = None
        self._edit_verts: Optional[np.ndarray] = None   # (N,2) [col, row]
        self._edit_handle_artists: list = []
        self._edit_line_artist = None
        self._edit_drag_idx: int = -1          # index of vertex being dragged
        self._edit_cids: list = []             # mpl event connections

        # Window/Level (brightness–contrast) drag-tool state — OsiriX-style
        self._im = None                        # current AxesImage (for live clim)
        self._wl_active: bool = False          # True when the WW/WL tool is on
        self._wl_cids: list = []               # mpl event connections
        self._wl_drag: Optional[dict] = None   # in-progress drag anchor/state
        self._wl_callback = None               # notified with (vmin, vmax) on release

    # ------------------------------------------------------------------
    # Public display methods
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Internal helper — safe matplotlib title (survives partial LaTeX)
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_title(ax, title: str, fontsize: int = 13,
                     title_font: str = "default", pad: int = 4,
                     title_bold: bool = False, title_italic: bool = False) -> None:
        """Set ax title with graceful fallback for incomplete/invalid LaTeX.

        Uses ``FontProperties(family=…)`` for reliable font-family selection.
        Passing ``fontproperties=`` to ``ax.set_title`` fully overrides the
        Text artist's font — confirmed to change both ``get_fontname()`` and
        the rendered glyph width in the Agg backend.

        When the user is mid-typing a mathtext expression (e.g. ``$\\mathbf{T``
        before the closing ``}$``) matplotlib raises a ValueError.  We catch it
        and display the raw string as plain text instead so the GUI never
        crashes during live editing.
        """
        import re
        from matplotlib.font_manager import FontProperties

        use_custom_font = title_font and title_font.lower() not in ("default", "")

        def _make_fp(size):
            if use_custom_font:
                return FontProperties(family=title_font, size=size)
            return FontProperties(size=size)

        def _set(t):
            # fontproperties= embeds family+size (fontsize= is ignored when it
            # is set).  Bold/italic are passed as explicit kwargs, which are
            # applied AFTER fontproperties and so reliably override matplotlib's
            # default axes.titleweight/style.
            kw = {"fontproperties": _make_fp(fontsize), "pad": pad}
            if title_bold:
                kw["fontweight"] = "bold"
            if title_italic:
                kw["fontstyle"] = "italic"
            ax.set_title(t, **kw)

        # Validate mathtext EAGERLY: matplotlib parses $…$ lazily at draw()
        # time, so an invalid span (e.g. a half-typed "$\mathbf{T") would abort
        # the Agg render and crash the app instead of raising here.  safe_mathtext
        # returns valid mathtext unchanged, or a plain-text fallback otherwise.
        from my_gui.format_bar import safe_mathtext
        try:
            _set(safe_mathtext(title))
        except (ValueError, RuntimeError):
            plain = re.sub(r'[\\${}^_]', '', title).strip()
            try:
                _set(plain)
            except Exception:
                pass   # last resort: leave title blank

    def _apply_bg_theme(self):
        """Apply the black/white background theme to the whole figure — called
        LAST (after title, colorbar and ROI overlays are drawn) so nothing
        downstream resets the colours.  Only cosmetic: facecolours + text/edge
        colours flip; the image data and colormap are untouched.  Delegates to
        the shared helper so maps and spectra dialogs theme identically."""
        from my_gui.fig_theme import apply_fig_dark_theme
        apply_fig_dark_theme(self._fig, getattr(self, '_dark_bg', False))

    def _resolve_display_cmap(self, cmap, vmin, vmax, arr):
        """Return the colormap to hand to imshow.  When the "Log map" toggle is
        on (``self._log_map``) the colormap is warped with the Fuderer log-like
        remap over the current window (clim if set, else the data range).  Data
        and colour-bar ticks stay linear.  Signed/invalid windows fall back to
        the unmodified colormap, so this is safe to apply to any map."""
        if not getattr(self, '_log_map', False):
            return cmap
        try:
            lo = vmin if vmin is not None else float(np.nanmin(arr))
            hi = vmax if vmax is not None else float(np.nanmax(arr))
            from my_gui.colormaps_gui import log_remap_cmap
            return log_remap_cmap(cmap, lo, hi)
        except Exception:
            return cmap

    def show_map(
        self,
        data: np.ndarray,
        title: str = "",
        cmap: str = "gray",
        vmin=None,
        vmax=None,
        title_fs: int = 13,
        label_fs: int = 10,
        tick_fs:  int = 9,
        cbar_fs:  int = 9,
        title_font: str = "default",
        title_bold: bool = False,
        title_italic: bool = False,
    ):
        self._fig.clf()
        self._ax = self._fig.add_subplot(111)
        # Optional black background (maps unchanged) — driven by the "Bg" toggle.
        _dark = getattr(self, '_dark_bg', False)
        _bg = 'black' if _dark else 'white'
        _fg = 'white' if _dark else 'black'
        self._fig.patch.set_facecolor(_bg)
        self._ax.set_facecolor(_bg)

        arr = np.asarray(data)
        if arr.ndim == 3:
            arr = arr.squeeze()
        if arr.ndim == 3:
            arr = arr[0]
        self._img_data = arr
        self._img_shape = arr.shape
        self._last_title = title
        self._last_cmap = cmap
        self._last_vmin = vmin
        self._last_vmax = vmax
        # Store font sizes / family for redraws (toggle_rois_visible, etc.)
        self._last_title_fs     = title_fs
        self._last_cbar_fs      = cbar_fs
        self._last_title_font   = title_font
        self._last_title_bold   = title_bold
        self._last_title_italic = title_italic

        im = self._ax.imshow(arr, cmap=self._resolve_display_cmap(cmap, vmin, vmax, arr),
                             vmin=vmin, vmax=vmax, aspect="auto")
        # Masked / NaN pixels (e.g. outside the phantom) take the background
        # colour, so a black bg turns the white surround black without changing
        # any real map value.
        try:
            _cmobj = im.get_cmap().copy(); _cmobj.set_bad(_bg); im.set_cmap(_cmobj)
        except Exception:
            pass
        self._im = im
        cb = self._fig.colorbar(im, ax=self._ax)
        cb.ax.tick_params(labelsize=cbar_fs, colors=_fg)
        try:
            cb.outline.set_edgecolor(_fg)
        except Exception:
            pass
        if title:
            self._apply_title(self._ax, title, fontsize=title_fs,
                              title_font=title_font, pad=4,
                              title_bold=title_bold, title_italic=title_italic)
            try:
                self._ax.title.set_color(_fg)
            except Exception:
                pass
        self._ax.axis("off")

        if self._dc_active:
            self._setup_dc_annot()

        self._redraw_roi_overlays()
        self._apply_bg_theme()          # last: nothing after this resets colours
        self.draw()

    def show_overlay(
        self,
        base: np.ndarray,
        overlay: np.ndarray,
        title: str = "",
        base_cmap: str = "gray",
        overlay_cmap: str = "jet",
        alpha: float = 0.5,
        vmin=None,
        vmax=None,
        title_fs: int = 13,
        label_fs: int = 10,
        tick_fs: int = 9,
        cbar_fs: int = 9,
        title_font: str = "default",
        title_bold: bool = False,
        title_italic: bool = False,
        mask_zero: bool = True,
    ):
        """Render a grayscale *base* image with a semi-transparent colour *overlay*.

        ``overlay`` is shown only where it is finite and non-zero (NaN / 0 voxels
        are fully transparent), with a colorbar for the overlay scale.  If the
        overlay shape differs from the base it is nearest-neighbour resized.
        """
        self._fig.clf()
        self._ax = self._fig.add_subplot(111)

        base = np.asarray(base, dtype=float)
        if base.ndim == 3:
            base = base.squeeze()
        if base.ndim == 3:
            base = base[..., 0]

        ov = np.asarray(overlay, dtype=float)
        if ov.ndim == 3:
            ov = ov.squeeze()
        if ov.ndim == 3:
            ov = ov[..., 0]

        # Resize overlay to base shape (nearest neighbour) if needed
        if ov.shape != base.shape:
            ys = (np.linspace(0, ov.shape[0] - 1, base.shape[0])).round().astype(int)
            xs = (np.linspace(0, ov.shape[1] - 1, base.shape[1])).round().astype(int)
            ov = ov[np.ix_(ys, xs)]

        self._img_data = base
        self._img_shape = base.shape
        self._im = None            # WW/WL tool is disabled for overlay views
        self._last_title = title
        self._last_cmap = base_cmap
        self._last_vmin = None
        self._last_vmax = None
        self._last_title_fs = title_fs
        self._last_cbar_fs = cbar_fs
        self._last_title_font = title_font
        self._last_title_bold = title_bold
        self._last_title_italic = title_italic

        # Base layer
        self._ax.imshow(base, cmap=base_cmap, aspect="auto")

        # Overlay layer — always transparent where not finite; also where == 0
        # unless mask_zero=False (so legitimately-zero map pixels stay coloured).
        _bad = ~np.isfinite(ov)
        masked = np.ma.masked_where((_bad | (ov == 0)) if mask_zero else _bad, ov)
        if vmin is None and vmax is None:
            fin = ov[np.isfinite(ov) & (ov != 0)]
            lim = float(np.nanpercentile(np.abs(fin), 99)) if fin.size else 1.0
            vmin, vmax = -lim, lim
        im = self._ax.imshow(masked, cmap=overlay_cmap, alpha=float(alpha),
                             vmin=vmin, vmax=vmax, aspect="auto")
        cb = self._fig.colorbar(im, ax=self._ax)
        cb.ax.tick_params(labelsize=cbar_fs)

        if title:
            self._apply_title(self._ax, title, fontsize=title_fs,
                              title_font=title_font, pad=4,
                              title_bold=title_bold, title_italic=title_italic)
        self._ax.axis("off")

        if self._dc_active:
            self._setup_dc_annot()

        # Draw ROI contours directly on top of base+overlay. (Do NOT call
        # _redraw_roi_overlays here — it clf()'s and re-renders only the base
        # image, which would erase the colour overlay layer.)
        if getattr(self, "_rois_visible", True):
            for roi in getattr(self, "_rois", []):
                self._draw_single_roi(roi)
        self.draw()

    def show_map_over_raw(
        self,
        param_map: np.ndarray,
        raw_base: np.ndarray,
        roi_mask: np.ndarray,
        title: str = "",
        cmap: str = "viridis",
        vmin=None,
        vmax=None,
        alpha: float = 1.0,
        **font_kwargs,
    ):
        """'ROIs + Bg' view: show the colour *param_map* only inside ``roi_mask``
        (everything outside transparent), laid over a grayscale *raw_base*
        underlay — typically the 1st raw acquisition frame.  ROI outlines are
        drawn on top.

        Uses the map's own colormap/limits; when vmin/vmax are None a *sequential*
        1–99 percentile of the in-ROI values is used (not the symmetric default),
        so positive maps like T1/T2 aren't mis-scaled.  ``font_kwargs`` accepts the
        same title/tick/cbar font keys as :meth:`show_map`."""
        ov = np.asarray(param_map, dtype=float).copy()
        if ov.ndim == 3:
            ov = ov.squeeze()
        if ov.ndim == 3:
            ov = ov[..., 0]
        m = np.asarray(roi_mask, dtype=bool)
        if m.shape != ov.shape:                     # align mask → map grid
            ys = np.linspace(0, m.shape[0] - 1, ov.shape[0]).round().astype(int)
            xs = np.linspace(0, m.shape[1] - 1, ov.shape[1]).round().astype(int)
            m = m[np.ix_(ys, xs)]
        ov[~m] = np.nan                             # colour only ROI pixels
        fin = ov[np.isfinite(ov)]
        if fin.size:
            if vmin is None:
                vmin = float(np.nanpercentile(fin, 1))
            if vmax is None:
                vmax = float(np.nanpercentile(fin, 99))
        self.show_overlay(
            base=raw_base, overlay=ov, title=title,
            base_cmap="gray", overlay_cmap=cmap, alpha=float(alpha),
            vmin=vmin, vmax=vmax, mask_zero=False, **font_kwargs,
        )

    def show_spectrum(
        self,
        ppm: np.ndarray,
        spectra_dict: dict,
        raw=None,
        title: str = "",
    ):
        self._fig.clf()
        self._img_data = None  # spectrum view has no pixel data
        self._im = None
        self._ax = self._fig.add_subplot(111)

        for label, spectrum in spectra_dict.items():
            self._ax.plot(ppm, spectrum, label=label)
        if raw is not None:
            self._ax.plot(ppm, raw, color="gray", alpha=0.5, label="raw")
        if title:
            self._ax.set_title(title)
        self._ax.set_xlabel("ppm")
        self._ax.invert_xaxis()
        if spectra_dict:
            self._ax.legend(fontsize=8)

        # Don't draw ROI overlays on spectrum plots — pixel coords don't map to ppm axis
        self.draw()

    def toggle_rois_visible(self):
        """Toggle ROI overlay visibility — re-renders base image then overlays."""
        self._rois_visible = not self._rois_visible
        if self._img_data is not None:
            # Re-render base image from scratch so old contour patches are gone
            self._fig.clf()
            self._ax = self._fig.add_subplot(111)
            im = self._ax.imshow(
                self._img_data,
                cmap=self._resolve_display_cmap(
                    self._last_cmap, self._last_vmin, self._last_vmax, self._img_data),
                vmin=self._last_vmin,
                vmax=self._last_vmax,
                aspect="auto",
            )
            self._im = im
            cb2 = self._fig.colorbar(im, ax=self._ax)
            cb2.ax.tick_params(labelsize=getattr(self, '_last_cbar_fs', 9))
            if self._last_title:
                self._apply_title(
                    self._ax, self._last_title,
                    fontsize=getattr(self, '_last_title_fs', 13),
                    title_font=getattr(self, '_last_title_font', 'default'),
                    pad=4,
                )
            self._ax.axis("off")
            if self._dc_active:
                self._setup_dc_annot()
            self._redraw_roi_overlays()
            self.draw()
        else:
            # No base image loaded yet — manually clear existing ROI contour
            # artists from the axes so hiding actually takes visual effect.
            if self._ax is not None:
                for coll in list(getattr(self._ax, 'collections', [])):
                    try:
                        coll.remove()
                    except Exception:
                        pass
            self._roi_patches.clear()
            # If toggling back to visible, redraw from the ROI list
            if self._rois_visible:
                for roi in self._rois:
                    self._draw_single_roi(roi)
            self.draw_idle()

    def clear(self):
        self._fig.clf()
        self._img_data = None
        self._img_shape = None
        self.draw()

    # ------------------------------------------------------------------
    # ROI shape editing (vertex drag)
    # ------------------------------------------------------------------

    def start_edit_roi(self, roi_name: str):
        """Enter vertex-drag editing mode for the named ROI."""
        self.stop_edit_roi(commit=False)   # cancel any previous edit
        roi = next((r for r in self._rois if r.name == roi_name), None)
        if roi is None or self._img_shape is None:
            return

        # Extract boundary vertices from the mask using contour finding
        contours = []
        try:
            import skimage.measure as _sm
            contours = _sm.find_contours(roi.mask.astype(float), 0.5)
        except Exception:
            pass
        if not contours:
            # Fallback: use matplotlib contour to get vertices
            try:
                import matplotlib.pyplot as _mplt
                cs = self._ax.contour(roi.mask.astype(float), levels=[0.5])
                # matplotlib 3.8+ removed .collections; use .get_paths() directly
                for path in cs.get_paths():
                    v = path.vertices   # (N, 2) in col,row display coords
                    if len(v) > 4:
                        # Convert back to row,col for consistency
                        contours.append(v[:, ::-1])
                cs.remove()
            except Exception:
                pass
        if not contours:
            return
        # Use the longest contour
        verts_rc = contours[np.argmax([len(c) for c in contours])]
        # Downsample to ≤12 vertices for clean interactive editing
        step = max(1, len(verts_rc) // 12)
        verts_rc = verts_rc[::step]
        # Convert row,col → col,row (matplotlib x=col, y=row)
        self._edit_verts = verts_rc[:, ::-1].copy()   # (N, 2): [col, row]
        self._edit_roi_name = roi_name
        self._edit_mode = True
        self._edit_drag_idx = -1
        self._draw_edit_handles()

        # Connect mouse events
        self._edit_cids = [
            self._fig.canvas.mpl_connect('button_press_event',   self._edit_on_press),
            self._fig.canvas.mpl_connect('motion_notify_event',  self._edit_on_motion),
            self._fig.canvas.mpl_connect('button_release_event', self._edit_on_release),
        ]

    def stop_edit_roi(self, commit: bool = True):
        """Exit editing mode. If commit=True, rebuild mask from final vertices."""
        if not self._edit_mode:
            return
        # Disconnect events
        for cid in self._edit_cids:
            try:
                self._fig.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._edit_cids = []

        if commit and self._edit_verts is not None and self._img_shape is not None:
            # self._edit_verts is already (col, row) = (x, y) format
            # which is exactly what _mask_from_verts expects (same as LassoSelector)
            mask = _mask_from_verts(self._edit_verts, self._img_shape)
            roi = next((r for r in self._rois if r.name == self._edit_roi_name), None)
            if roi is not None:
                roi.mask = mask.astype(bool)

        # Remove handle artists
        for art in self._edit_handle_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._edit_handle_artists = []
        if self._edit_line_artist is not None:
            try:
                self._edit_line_artist.remove()
            except Exception:
                pass
            self._edit_line_artist = None

        self._edit_mode = False
        self._edit_roi_name = None
        self._edit_verts = None
        self._edit_drag_idx = -1

        # Redraw overlays with updated mask
        self._redraw_roi_overlays()
        self.draw()

    def _draw_edit_handles(self):
        """Draw the draggable vertex handles and boundary polyline on the canvas."""
        if self._edit_verts is None:
            return
        # Remove old artists
        for art in self._edit_handle_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._edit_handle_artists = []
        if self._edit_line_artist is not None:
            try:
                self._edit_line_artist.remove()
            except Exception:
                pass
            self._edit_line_artist = None

        # Use the edited ROI's own color (fall back to white if not found)
        roi_color = 'white'
        if self._edit_roi_name:
            roi = next((r for r in self._rois if r.name == self._edit_roi_name), None)
            if roi is not None:
                roi_color = roi.color

        verts = self._edit_verts   # (N, 2): col, row
        xs, ys = verts[:, 0], verts[:, 1]

        # Boundary polyline (closed) — ROI color, solid
        line, = self._ax.plot(
            np.append(xs, xs[0]), np.append(ys, ys[0]),
            '-', color=roi_color, linewidth=2.0, alpha=0.95, zorder=10,
        )
        self._edit_line_artist = line

        # Vertex handles — small white circles with ROI-colored edge
        for i, (x, y) in enumerate(zip(xs, ys)):
            h, = self._ax.plot(x, y, 'o', color='white', markersize=7,
                               markeredgecolor=roi_color, markeredgewidth=1.5,
                               zorder=11, picker=8)
            h._edit_vertex_idx = i
            self._edit_handle_artists.append(h)

        self.draw()

    def _edit_on_press(self, event):
        if event.inaxes != self._ax or self._edit_verts is None:
            return
        if event.button != 1:
            return
        # Find nearest vertex within 10 pixels
        verts = self._edit_verts
        xs, ys = verts[:, 0], verts[:, 1]
        # Convert mouse data coords to display coords
        try:
            disp = self._ax.transData.transform(np.column_stack([xs, ys]))
            mouse_disp = self._ax.transData.transform([[event.xdata, event.ydata]])[0]
            dists = np.hypot(disp[:, 0] - mouse_disp[0], disp[:, 1] - mouse_disp[1])
            nearest = int(np.argmin(dists))
            if dists[nearest] < 12:
                self._edit_drag_idx = nearest
        except Exception:
            pass

    def _edit_on_motion(self, event):
        if event.inaxes != self._ax or self._edit_drag_idx < 0:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._edit_verts[self._edit_drag_idx, 0] = event.xdata
        self._edit_verts[self._edit_drag_idx, 1] = event.ydata
        self._draw_edit_handles()

    def _edit_on_release(self, event):
        self._edit_drag_idx = -1

    def _roi_mask_to_verts(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract boundary vertices from a boolean mask."""
        try:
            import skimage.measure as _sm
            contours = _sm.find_contours(mask.astype(float), 0.5)
            if contours:
                return contours[np.argmax([len(c) for c in contours])]
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # ROI drawing
    # ------------------------------------------------------------------

    def start_draw(self, roi_type: str, name: str):
        self._cancel_selector()
        self._draw_type = roi_type.lower()
        self._next_roi_name = name
        self._finish_roi_guard = False   # reset duplicate-fire guard

        # Preview color = the color this ROI will receive when finished
        preview_color = _ROI_COLORS[self._color_idx % len(_ROI_COLORS)]

        # RectangleSelector/EllipseSelector use Patch props (edgecolor ok)
        patch_props = dict(edgecolor=preview_color, facecolor="none", linewidth=2.0,
                           linestyle="-", alpha=0.9)
        # PolygonSelector / LassoSelector use Line2D props (no edgecolor)
        line_props  = dict(color=preview_color, linewidth=2.0, linestyle="-", alpha=0.9)

        if self._draw_type == "rectangle":
            self._selector = RectangleSelector(
                self._ax, self._on_rect,
                useblit=True, button=[1],
                minspanx=2, minspany=2,
                spancoords="pixels",
                interactive=False,
                props=patch_props,
            )
        elif self._draw_type == "ellipse":
            self._selector = EllipseSelector(
                self._ax, self._on_ellipse,
                useblit=True, button=[1],
                minspanx=2, minspany=2,
                spancoords="pixels",
                interactive=False,
                props=patch_props,
            )
        elif self._draw_type == "circle":
            # Circle uses EllipseSelector but the callback snaps to equal radii
            self._selector = EllipseSelector(
                self._ax, self._on_circle,
                useblit=True, button=[1],
                minspanx=2, minspany=2,
                spancoords="pixels",
                interactive=False,
                props=patch_props,
            )
        elif self._draw_type == "freehand":
            self._selector = LassoSelector(
                self._ax, self._on_lasso,
                useblit=True, button=[1],
                props=line_props,
            )
        elif self._draw_type == "polygon":
            self._selector = PolygonSelector(
                self._ax, self._on_polygon,
                useblit=True,
                props=line_props,
            )
        else:
            warnings.warn(f"Unknown roi_type: {roi_type}")
            return

        self.setFocus()

    def _cancel_selector(self):
        if self._selector is not None:
            try:
                self._selector.set_active(False)
            except Exception:
                pass
            self._selector = None

    # ------------------------------------------------------------------
    # Window / Level (brightness–contrast) drag tool — OsiriX-style
    # ------------------------------------------------------------------
    def set_wl_active(self, active: bool, on_change=None):
        """Enable / disable the interactive window–level (WW/WL) drag tool.

        While active, dragging with the left mouse button over the map adjusts
        the display window live, exactly like the OsiriX WL tool:

          * horizontal drag → window **width**  (contrast)
          * vertical drag   → window **level**   (brightness)

        Dragging right widens the window (less contrast); dragging up lowers the
        level so more pixels saturate to the top of the colormap (brighter).
        ``on_change(vmin, vmax)`` is invoked when the drag ends so the caller can
        persist the new limits (e.g. into the colour-bar widget) and keep them
        across later redraws.
        """
        active = bool(active)
        if on_change is not None:
            self._wl_callback = on_change
        if active == self._wl_active:
            return
        self._wl_active = active
        if active:
            # A live WL drag and ROI drawing must not fight over the mouse.
            self._cancel_selector()
            self._wl_cids = [
                self._fig.canvas.mpl_connect('button_press_event',   self._wl_on_press),
                self._fig.canvas.mpl_connect('motion_notify_event',  self._wl_on_motion),
                self._fig.canvas.mpl_connect('button_release_event', self._wl_on_release),
            ]
            try:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            except Exception:
                pass
        else:
            for cid in self._wl_cids:
                try:
                    self._fig.canvas.mpl_disconnect(cid)
                except Exception:
                    pass
            self._wl_cids = []
            self._wl_drag = None
            try:
                self.unsetCursor()
            except Exception:
                pass

    def _wl_data_range(self):
        """Full finite data range of the current image — sets the drag scale."""
        if self._img_data is None:
            return None
        fin = self._img_data[np.isfinite(self._img_data)]
        if fin.size == 0:
            return None
        lo, hi = float(fin.min()), float(fin.max())
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def _wl_on_press(self, event):
        if not self._wl_active or event.button != 1:
            return
        if event.inaxes is not self._ax or self._im is None:
            return
        rng = self._wl_data_range()
        if rng is None:
            return
        lo, hi = rng
        vmin, vmax = self._im.get_clim()
        if vmin is None:
            vmin = lo
        if vmax is None:
            vmax = hi
        self._wl_drag = {
            'x': event.x, 'y': event.y,
            'w': float(vmax - vmin),
            'l': float((vmax + vmin) / 2.0),
            # ~250 px of drag sweeps the full data range
            'scale': (hi - lo) / 250.0,
        }

    def _wl_on_motion(self, event):
        # event.x / event.y stay valid even when the cursor leaves the axes,
        # so the drag keeps tracking like a proper WL tool.
        if self._wl_drag is None or event.x is None or event.y is None:
            return
        d = self._wl_drag
        width = d['w'] + (event.x - d['x']) * d['scale']
        level = d['l'] - (event.y - d['y']) * d['scale']
        min_w = abs(d['scale']) * 1e-2 + 1e-12   # keep a strictly-positive window
        if width < min_w:
            width = min_w
        vmin = level - width / 2.0
        vmax = level + width / 2.0
        self._last_vmin = vmin
        self._last_vmax = vmax
        try:
            self._im.set_clim(vmin, vmax)
        except Exception:
            return
        self.draw_idle()

    def _wl_on_release(self, event):
        if self._wl_drag is None:
            return
        self._wl_drag = None
        if self._wl_callback is not None and self._last_vmin is not None:
            try:
                self._wl_callback(float(self._last_vmin), float(self._last_vmax))
            except Exception:
                pass

    # Selector callbacks -----------------------------------------------

    def _on_rect(self, eclick, erelease):
        if self._img_shape is None:
            return
        mask = _mask_from_rect(eclick, erelease, self._img_shape)
        self._finish_roi("rectangle", mask)

    def _on_ellipse(self, eclick, erelease):
        if self._img_shape is None:
            return
        mask = _mask_from_ellipse(eclick, erelease, self._img_shape)
        self._finish_roi("ellipse", mask)

    def _on_circle(self, eclick, erelease):
        """Like ellipse but forces equal radii — perfect circle mask."""
        if self._img_shape is None:
            return
        H, W = self._img_shape[:2]
        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        y1, y2 = sorted([eclick.ydata, erelease.ydata])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        # Use the smaller of the two half-spans so the circle fits the drag
        r = min((x2 - x1) / 2.0, (y2 - y1) / 2.0) + 1e-9
        xs = np.arange(W)
        ys = np.arange(H)
        xg, yg = np.meshgrid(xs, ys)
        mask = ((xg - cx) ** 2 + (yg - cy) ** 2) <= r ** 2
        self._finish_roi("circle", mask.astype(bool))

    def _on_lasso(self, verts):
        if len(verts) < 3:
            return
        if self._img_shape is None:
            return
        mask = _mask_from_verts(verts, self._img_shape)
        self._finish_roi("freehand", mask)

    def _on_polygon(self, verts):
        if self._img_shape is None:
            return
        # Guard: PolygonSelector can fire the callback more than once
        # (e.g. on double-click completion AND on a subsequent click).
        # Only create the ROI on the first call.
        if getattr(self, '_finish_roi_guard', False):
            return
        mask = _mask_from_verts(verts, self._img_shape)
        self._finish_roi("polygon", mask)

    def _finish_roi(self, roi_type: str, mask: np.ndarray):
        # Prevent duplicate ROI creation if callback fires twice
        if getattr(self, '_finish_roi_guard', False):
            return
        self._finish_roi_guard = True

        color = _ROI_COLORS[self._color_idx % len(_ROI_COLORS)]
        self._color_idx += 1
        roi = ROI(
            name=self._next_roi_name,
            roi_type=roi_type,
            mask=mask.astype(bool),
            color=color,
        )
        self._rois.append(roi)
        # Cancel selector BEFORE redrawing so the preview line disappears
        self._cancel_selector()
        self._draw_single_roi(roi)
        self.draw()
        self.roi_added.emit(roi)

    # ROI overlay helpers ----------------------------------------------

    def _draw_single_roi(self, roi: ROI):
        if not hasattr(self, "_ax") or self._ax is None:
            return
        if not getattr(self, '_rois_visible', True):
            return
        try:
            is_selected = getattr(self, '_selected_roi_name', None) == roi.name
            lw = 3.5 if is_selected else 2

            # Resize mask to match current display image if shapes differ
            draw_mask = roi.mask
            if self._img_shape is not None:
                ih, iw = self._img_shape[:2]
                mh, mw = draw_mask.shape[:2]
                if (mh, mw) != (ih, iw):
                    try:
                        from scipy.ndimage import zoom
                        draw_mask = zoom(draw_mask.astype(float),
                                         (ih / max(mh, 1), iw / max(mw, 1)),
                                         order=1) > 0.5
                    except Exception:
                        pass  # fall back to original mask

            # Brain skull-strip mask → filled semi-transparent overlay so it
            # reads like the deep-extraction result, not just a boundary line.
            if roi.name == "Brain_outline":
                try:
                    self._ax.contourf(draw_mask.astype(float), levels=[0.5, 1.5],
                                      colors=[roi.color], alpha=0.35)
                except Exception:
                    pass

            # Highlighted ROI: draw bright white outline + thick color contour
            # Store ContourSet objects directly — matplotlib 3.8+ removed .collections
            contour_sets = []
            if is_selected:
                cs_white = self._ax.contour(
                    draw_mask.astype(float), levels=[0.5],
                    colors=['white'], linewidths=lw + 2,
                )
                contour_sets.append(cs_white)
                # Filled semi-transparent overlay
                self._ax.contourf(
                    draw_mask.astype(float), levels=[0.5, 1.5],
                    colors=[roi.color], alpha=0.25,
                )
            cs_color = self._ax.contour(
                draw_mask.astype(float),
                levels=[0.5],
                colors=[roi.color],
                linewidths=lw,
                linestyles='solid' if not is_selected else 'dashed',
            )
            contour_sets.append(cs_color)
            self._roi_patches[roi.name] = contour_sets
        except Exception as exc:
            warnings.warn(f"Could not draw ROI {roi.name}: {exc}")

    def _redraw_roi_overlays(self):
        """Clear all existing contour patches then redraw — prevents stale highlights."""
        self._roi_patches.clear()
        if self._img_data is not None:
            # Re-render base image so matplotlib clears all old contour artists
            self._fig.clf()
            self._ax = self._fig.add_subplot(111)
            im = self._ax.imshow(
                self._img_data,
                cmap=self._resolve_display_cmap(
                    self._last_cmap, self._last_vmin, self._last_vmax, self._img_data),
                vmin=self._last_vmin,
                vmax=self._last_vmax,
                aspect="auto",
            )
            self._im = im
            cb3 = self._fig.colorbar(im, ax=self._ax)
            cb3.ax.tick_params(labelsize=getattr(self, '_last_cbar_fs', 9))
            if self._last_title:
                self._apply_title(
                    self._ax, self._last_title,
                    fontsize=getattr(self, '_last_title_fs', 13),
                    title_font=getattr(self, '_last_title_font', 'default'), pad=4,
                    title_bold=getattr(self, '_last_title_bold', False),
                    title_italic=getattr(self, '_last_title_italic', False))
            self._ax.axis("off")
            if self._dc_active:
                self._setup_dc_annot()
        for roi in self._rois:
            self._draw_single_roi(roi)

    # ------------------------------------------------------------------
    # ROI management
    # ------------------------------------------------------------------

    def remove_roi(self, name: str):
        self._rois = [r for r in self._rois if r.name != name]
        # Re-display
        if self._img_data is not None:
            self._fig.clf()
            self._ax = self._fig.add_subplot(111)
            im = self._ax.imshow(
                self._img_data,
                cmap=self._resolve_display_cmap(
                    self._last_cmap, self._last_vmin, self._last_vmax, self._img_data),
                vmin=self._last_vmin,
                vmax=self._last_vmax,
                aspect="auto",
            )
            self._im = im
            cb4 = self._fig.colorbar(im, ax=self._ax)
            cb4.ax.tick_params(labelsize=getattr(self, '_last_cbar_fs', 9))
            if self._last_title:
                self._apply_title(
                    self._ax, self._last_title,
                    fontsize=getattr(self, '_last_title_fs', 13),
                    title_font=getattr(self, '_last_title_font', 'default'), pad=4,
                    title_bold=getattr(self, '_last_title_bold', False),
                    title_italic=getattr(self, '_last_title_italic', False))
            self._ax.axis("off")
        self._redraw_roi_overlays()
        if self._dc_active:
            self._setup_dc_annot()
        self.draw()

    def rename_roi(self, old_name: str, new_name: str):
        for roi in self._rois:
            if roi.name == old_name:
                roi.name = new_name
                break
        if old_name in self._roi_patches:
            self._roi_patches[new_name] = self._roi_patches.pop(old_name)

    def get_rois(self) -> List[ROI]:
        return list(self._rois)

    def add_rois(self, rois_list: List[ROI]):
        for roi in rois_list:
            self._rois.append(roi)
            self._draw_single_roi(roi)
        self.draw()

    # ------------------------------------------------------------------
    # Data cursor
    # ------------------------------------------------------------------

    def set_datacursor(self, enabled: bool):
        self._dc_active = enabled
        if enabled:
            self._setup_dc_annot()
            if self._dc_cid is None:
                self._dc_cid = self.mpl_connect("motion_notify_event", self._on_hover)
        else:
            if self._dc_cid is not None:
                self.mpl_disconnect(self._dc_cid)
                self._dc_cid = None
            if self._dc_annot is not None:
                try:
                    self._dc_annot.set_visible(False)
                except Exception:
                    pass
                self._dc_annot = None
            self.draw_idle()

    def _setup_dc_annot(self):
        if not hasattr(self, "_ax") or self._ax is None:
            return
        bbox_props = dict(
            boxstyle="round,pad=0.4",
            fc="#001a4d",
            ec="#4488ff",
            lw=1.0,
            alpha=0.85,
        )
        self._dc_annot = self._ax.annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=bbox_props,
            color="white",
            fontsize=8,
            visible=False,
            zorder=10,
        )

    def _on_hover(self, event):
        if self._dc_annot is None:
            return
        if event.inaxes != self._ax or self._img_data is None:
            self._dc_annot.set_visible(False)
            self.draw_idle()
            return

        xi = int(round(event.xdata))
        yi = int(round(event.ydata))
        H, W = self._img_data.shape[:2]
        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            self._dc_annot.set_visible(False)
            self.draw_idle()
            return

        val = self._img_data[yi, xi]
        text = f"x={xi}  y={yi}\nval = {val:.4g}"
        self._dc_annot.set_text(text)
        self._dc_annot.xy = (event.xdata, event.ydata)
        self._dc_annot.set_visible(True)
        self.draw_idle()


# ---------------------------------------------------------------------------
# ROIPanel
# ---------------------------------------------------------------------------

class ROIPanel(QWidget):
    def __init__(self, canvas: ROICanvas, parent=None):
        super().__init__(parent)
        self._canvas = canvas
        self._connected_canvas = None   # prevent double-connect
        # Optional callback set by external code (e.g. ROITab) to show a
        # parametric-map overlay masked to the currently detected tubes.
        self._mask_tubes_fn = None
        if canvas is not None:
            canvas.roi_added.connect(self._on_roi_added)
            self._connected_canvas = canvas
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        # ── ROI Tools ─────────────────────────────────────────────────
        grp_tools = QGroupBox("ROI Tools")
        tools_layout = QVBoxLayout(grp_tools)
        tools_layout.setSpacing(4)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Type:"))
        self._type_combo = QComboBox()
        self._type_combo.addItems(["Rectangle", "Ellipse", "Circle", "Polygon", "Freehand"])
        row1.addWidget(self._type_combo)
        tools_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit("ROI_1")
        row2.addWidget(self._name_edit)
        tools_layout.addLayout(row2)

        self._draw_btn = QPushButton("Draw ROI")
        self._draw_btn.setFixedHeight(30)
        self._draw_btn.setStyleSheet(
            "QPushButton { background-color: #2255bb; color: white; "
            "border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #3366cc; }"
        )
        self._draw_btn.clicked.connect(self._start_draw)
        tools_layout.addWidget(self._draw_btn)

        main_layout.addWidget(grp_tools)

        # ── ROI List ──────────────────────────────────────────────────
        grp_list = QGroupBox("ROI List")
        list_layout = QVBoxLayout(grp_list)
        list_layout.setSpacing(4)

        self._roi_list = QListWidget()
        self._roi_list.setMaximumHeight(130)
        self._roi_list.currentRowChanged.connect(self._on_selection_changed)
        list_layout.addWidget(self._roi_list)

        # Per-ROI pixel stats (n / mean / std / min / max) are no longer shown
        # inline in the ROI manager at user request — the full ROI Stats Table
        # provides these and more.  The label is kept (hidden, not in the
        # layout) so the existing setText() calls remain valid.
        self._stats_label = QLabel("")
        self._stats_label.setWordWrap(True)
        self._stats_label.setStyleSheet("font-size: 10px; color: gray;")
        self._stats_label.hide()

        btn_row = QHBoxLayout()
        self._rename_btn = QPushButton("Rename")
        self._rename_btn.clicked.connect(self._rename_roi)
        btn_row.addWidget(self._rename_btn)

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setStyleSheet(
            "QPushButton { background-color: #bb2222; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #cc3333; }"
        )
        self._delete_btn.clicked.connect(self._delete_roi)
        btn_row.addWidget(self._delete_btn)

        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(self._clear_btn)

        list_layout.addLayout(btn_row)

        # Reorder row — move the selected ROI up / down in the list
        move_row = QHBoxLayout()
        self._move_up_btn = QPushButton("▲ Move Up")
        self._move_up_btn.clicked.connect(lambda: self._move_roi(-1))
        move_row.addWidget(self._move_up_btn)
        self._move_dn_btn = QPushButton("▼ Move Down")
        self._move_dn_btn.clicked.connect(lambda: self._move_roi(+1))
        move_row.addWidget(self._move_dn_btn)
        list_layout.addLayout(move_row)

        # Shape-editing button (enter/exit vertex-drag mode)
        self._edit_shape_btn = QPushButton("Edit Shape")
        self._edit_shape_btn.setCheckable(True)
        self._edit_shape_btn.setFixedHeight(26)
        self._edit_shape_btn.setStyleSheet(
            "QPushButton { border: 1px solid #555; border-radius: 4px; }"
            "QPushButton:checked { background: #aa6600; color: white; font-weight: bold; }"
            "QPushButton:hover   { background: #444; }"
        )
        self._edit_shape_btn.clicked.connect(self._toggle_edit_shape)
        list_layout.addWidget(self._edit_shape_btn)

        main_layout.addWidget(grp_list)

        # ── Auto-Detect ROIs ──────────────────────────────────────────
        grp_phantom = QGroupBox("Auto-Detect ROIs")
        phantom_layout = QVBoxLayout(grp_phantom)
        phantom_layout.setSpacing(4)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("σ:"))
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(1.0, 20.0)
        self._sigma_spin.setValue(4.0)
        self._sigma_spin.setSingleStep(0.5)
        self._sigma_spin.setToolTip(
            "<b>Gaussian smoothing σ</b><br>"
            "Controls how much the image is blurred before edge/region detection.<br>"
            "Higher σ = smoother edges, less noise sensitivity.<br><br>"
            "<b>Optimal value for phantoms:</b>  3 – 6<br>"
            "• Start with <b>4.0</b> (default)<br>"
            "• Increase if the image is very noisy or the phantom boundary is unclear<br>"
            "• Decrease if the phantom has very sharp edges or is small in the FOV"
        )
        param_row.addWidget(self._sigma_spin)
        param_row.addWidget(QLabel("Min px:"))
        self._minpx_spin = QSpinBox()
        self._minpx_spin.setRange(5, 500)
        self._minpx_spin.setValue(30)
        self._minpx_spin.setToolTip(
            "<b>Minimum region size (pixels)</b><br>"
            "Any detected region smaller than this is discarded as noise.<br><br>"
            "<b>Optimal value for phantoms:</b>  20 – 80<br>"
            "• Use <b>30</b> (default) for standard tube phantoms<br>"
            "• Increase if spurious small regions are being detected<br>"
            "• Decrease if small tubes are being missed (e.g. high-res acquisitions)"
        )
        param_row.addWidget(self._minpx_spin)
        phantom_layout.addLayout(param_row)

        self._tubes_btn = QPushButton("Auto-Detect Tubes")
        self._tubes_btn.setFixedHeight(30)
        self._tubes_btn.setStyleSheet(
            "QPushButton { background-color: #226622; color: white; "
            "border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #338833; }"
        )
        self._tubes_btn.clicked.connect(self._detect_tubes)
        phantom_layout.addWidget(self._tubes_btn)

        # Detect Phantom Outline + Detect Brain Outline live in this box, below
        # Auto-Detect Tubes (they use the σ / Min-px parameters above).
        self._outline_btn = QPushButton("Detect Phantom Outline")
        self._outline_btn.setFixedHeight(28)
        self._outline_btn.setToolTip(
            "Detect the outer boundary of the phantom and add it as an ROI.\n"
            "Useful for masking background noise before running analysis.")
        self._outline_btn.clicked.connect(self._detect_outline)
        phantom_layout.addWidget(self._outline_btn)

        self._brain_btn = QPushButton("Detect Brain Outline")
        self._brain_btn.setFixedHeight(28)
        self._brain_btn.setToolTip(
            "Skull-strip the brain and add it as an outline ROI.\n"
            "Uses a deep 3-D U-Net (ONNX) when a 3-D T1 volume is loaded in the\n"
            "T1/T2 tab; otherwise the classical method on the current slice.")
        self._brain_btn.setStyleSheet(
            "QPushButton { border: 1px solid #2a7d6f; border-radius: 4px; }"
            "QPushButton:hover { background: #14493f; }")
        self._brain_btn.clicked.connect(self._detect_brain_outline)
        phantom_layout.addWidget(self._brain_btn)
        # "Detect Brain Outline" skull-strips with the deep 3-D U-Net (ONNX)
        # when a 3-D T1 volume is available; otherwise it falls back to the
        # classical Otsu outline on the current slice.  A tab provides the 3-D
        # volume via set_brain_volume_getter().
        self._brain_volume_getter = None

        # NOTE: "Mask Tubes → Overlay Map" button removed at user request (as of
        # now).  The _mask_tubes / _mask_tubes_fn handlers are kept below so the
        # feature can be re-enabled later without re-wiring.

        main_layout.addWidget(grp_phantom)

        # ── Data cursor — red-cursor icon toggle (tooltip "Pixel values") ──
        from my_gui.plot_custom_bar import DataCursorToolButton
        self._dc_check = DataCursorToolButton()
        if self._canvas is not None:
            self._dc_check.toggled.connect(self._canvas.set_datacursor)
        _dc_row = QHBoxLayout()
        _dc_row.addWidget(self._dc_check)
        _dc_row.addStretch()
        main_layout.addLayout(_dc_row)

        main_layout.addStretch()

    # ------------------------------------------------------------------
    # Late canvas wiring
    # ------------------------------------------------------------------

    def set_canvas(self, canvas: "ROICanvas"):
        """Connect a canvas after the panel was constructed with canvas=None."""
        self._canvas = canvas
        if canvas is not self._connected_canvas:
            canvas.roi_added.connect(self._on_roi_added)
            self._connected_canvas = canvas
        self._dc_check.toggled.connect(canvas.set_datacursor)

    # ------------------------------------------------------------------
    # Slots / handlers
    # ------------------------------------------------------------------

    def _toggle_edit_shape(self, checked: bool):
        if checked:
            row = self._roi_list.currentRow()
            rois = self._canvas.get_rois()
            if row < 0 or row >= len(rois):
                self._edit_shape_btn.setChecked(False)
                return
            self._edit_shape_btn.setText("✅ Done Editing")
            self._canvas.start_edit_roi(rois[row].name)
        else:
            self._edit_shape_btn.setText("Edit Shape")
            self._canvas.stop_edit_roi(commit=True)

    def _start_draw(self):
        name = self._name_edit.text().strip() or "ROI_1"
        roi_type = self._type_combo.currentText()
        self._canvas.start_draw(roi_type, name)
        self._draw_btn.setText("Drawing…")

        # Auto-increment name IMMEDIATELY so the next ROI gets the next number.
        # This prevents the user from accidentally creating a duplicate name if
        # they click "Draw ROI" again before the current draw finishes.
        import re
        m = re.match(r"^(.*?)(\d+)$", name)
        if m:
            prefix, num = m.group(1), int(m.group(2))
            self._name_edit.setText(f"{prefix}{num + 1}")

    def _on_roi_added(self, roi: ROI):
        # Guard against duplicate list entries caused by double-connected signals
        for i in range(self._roi_list.count()):
            if self._roi_list.item(i).text() == roi.name:
                self._draw_btn.setText("Draw ROI")
                return
        self._roi_list.addItem(roi.name)
        self._draw_btn.setText("Draw ROI")

        # Show stats if data available
        self._show_stats_for_roi(roi)

    def _move_roi(self, delta: int):
        """Move the selected ROI up (delta=-1) or down (delta=+1) in the list.

        Reorders the canvas ROI list (which drives table / export / copy order),
        rebuilds the list widget, keeps the moved ROI selected, and notifies
        downstream views.
        """
        row = self._roi_list.currentRow()
        rois = self._canvas._rois
        new = row + delta
        if row < 0 or new < 0 or new >= len(rois):
            return
        # Reorder the underlying ROI list
        rois[row], rois[new] = rois[new], rois[row]
        # Rebuild the list widget to match, keeping the moved ROI selected
        self._roi_list.blockSignals(True)
        self._roi_list.clear()
        for r in rois:
            self._roi_list.addItem(r.name)
        self._roi_list.setCurrentRow(new)
        self._roi_list.blockSignals(False)
        # Refresh overlays + stats for the now-selected ROI
        self._on_selection_changed(new)
        # Notify downstream (ROI manager → other tabs' tables/exports)
        sig = getattr(self, "rois_changed", None)
        if sig is not None:
            try:
                sig.emit()
            except Exception:
                pass

    def _on_selection_changed(self, row: int):
        # Commit any active shape edit before switching selection
        if self._canvas._edit_mode:
            self._canvas.stop_edit_roi(commit=True)
            self._edit_shape_btn.setChecked(False)
            self._edit_shape_btn.setText("Edit Shape")

        if row < 0:
            self._stats_label.setText("")
            self._canvas._selected_roi_name = None
            if self._canvas._img_data is not None:
                self._canvas._redraw_roi_overlays()
                self._canvas.draw()
            return
        rois = self._canvas.get_rois()
        if row >= len(rois):
            return
        roi = rois[row]
        # Highlight selected ROI on canvas
        self._canvas._selected_roi_name = roi.name
        if self._canvas._img_data is not None:
            self._canvas._redraw_roi_overlays()
            self._canvas.draw()
        self._show_stats_for_roi(roi)

    def _show_stats_for_roi(self, roi: ROI):
        if self._canvas._img_data is not None:
            s = roi.stats(self._canvas._img_data)
            txt = (
                f"n={s['n']}  mean={s['mean']:.4g}  std={s['std']:.4g}\n"
                f"min={s['min']:.4g}  max={s['max']:.4g}"
            )
        else:
            n = int(roi.mask.sum())
            txt = f"n={n} pixels (no image loaded)"
        self._stats_label.setText(txt)

    def _rename_roi(self):
        row = self._roi_list.currentRow()
        if row < 0:
            return
        rois = self._canvas.get_rois()
        if row >= len(rois):
            return
        old_name = rois[row].name
        new_name, ok = QInputDialog.getText(
            self, "Rename ROI", "New name:", text=old_name
        )
        if ok and new_name.strip():
            new_name = new_name.strip()
            self._canvas.rename_roi(old_name, new_name)
            self._roi_list.item(row).setText(new_name)

    def _delete_roi(self):
        row = self._roi_list.currentRow()
        if row < 0:
            return
        rois = self._canvas.get_rois()
        if row >= len(rois):
            return
        name = rois[row].name
        self._canvas.remove_roi(name)
        self._roi_list.takeItem(row)
        self._stats_label.setText("")

    def _clear_all(self):
        for roi in list(self._canvas.get_rois()):
            self._canvas.remove_roi(roi.name)
        self._roi_list.clear()
        self._stats_label.setText("")
        # Notify downstream (ROI manager → every other tab) so a Clear All
        # removes the ROIs everywhere, not just on this canvas.
        sig = getattr(self, "rois_changed", None)
        if sig is not None:
            try:
                sig.emit()
            except Exception:
                pass

    def _detect_outline(self):
        if self._canvas._img_data is None:
            QMessageBox.information(self, "No image", "Load an image first.")
            return
        # Remove existing Phantom_outline ROI if present (avoid duplicates)
        existing = [r for r in self._canvas.get_rois() if r.name == "Phantom_outline"]
        for r in existing:
            self._canvas.remove_roi(r.name)
            for i in range(self._roi_list.count()):
                if self._roi_list.item(i).text() == "Phantom_outline":
                    self._roi_list.takeItem(i)
                    break

        sigma = self._sigma_spin.value()
        mask = detect_phantom_outline(self._canvas._img_data, gauss_sigma=sigma)
        roi = ROI(
            name="Phantom_outline",
            roi_type="auto",
            mask=mask,
            color="#ff8800",  # orange
        )
        # Add to canvas and list, then emit signal for ROI Manager broadcast
        self._canvas._rois.append(roi)
        self._canvas._draw_single_roi(roi)
        self._canvas.draw()
        self._roi_list.addItem(roi.name)
        # Emit AFTER adding to list so _on_roi_added guard catches duplicate
        self._canvas.roi_added.emit(roi)
        self._show_stats_for_roi(roi)

    def set_brain_volume_getter(self, fn):
        """Wire a callable returning the 3-D T1 volume (and optional current
        slice index) for deep skull-stripping: fn() -> (volume_HxWxD, slice_idx)
        or volume_HxWxD or None.  Set by app.py from the T1 tab."""
        self._brain_volume_getter = fn

    def _deep_brain_mask(self):
        """Deep 3-D U-Net (ONNX) skull-strip → 2-D mask aligned to the current
        display, or None (unavailable / no 3-D volume / error) so the caller
        falls back to the classical Otsu outline."""
        self._brain_mask_3d = None      # reset; set on a successful deep run
        try:
            from my_gui.deep_extractor import is_available, extract_brain_mask
            if not is_available():
                return None
            getter = getattr(self, "_brain_volume_getter", None)
            if not callable(getter):
                return None
            got = getter()
            if got is None:
                return None
            vol, sl = got if isinstance(got, tuple) else (got, None)
            vol = np.asarray(vol, dtype=float)
            if vol.ndim != 3 or min(vol.shape) < 2:
                return None
            mask3d = extract_brain_mask(vol, 0.5)          # (H, W, D) bool
            self._brain_mask_3d = mask3d                   # cache for slice scrolling
            # Non-empty placeholder (mid axial slice); the owning tab re-slices
            # this to the current view plane via its _on_brain_detected hook.
            sl = int(mask3d.shape[2] // 2)
            m2d = mask3d[:, :, sl]
            tgt = np.asarray(self._canvas._img_data).shape[:2]
            if m2d.shape != tgt:
                from scipy.ndimage import zoom
                m2d = zoom(m2d.astype(float),
                           (tgt[0] / max(m2d.shape[0], 1), tgt[1] / max(m2d.shape[1], 1)),
                           order=0) > 0.5
            return m2d if np.asarray(m2d).any() else None
        except Exception:
            return None

    def _detect_brain_outline(self):
        """Skull-strip the current in-vivo slice and add the brain as an ROI."""
        if self._canvas._img_data is None:
            QMessageBox.information(self, "No image", "Load an image first.")
            return
        # Remove existing Brain_outline ROI if present (avoid duplicates)
        existing = [r for r in self._canvas.get_rois() if r.name == "Brain_outline"]
        for r in existing:
            self._canvas.remove_roi(r.name)
            for i in range(self._roi_list.count()):
                if self._roi_list.item(i).text() == "Brain_outline":
                    self._roi_list.takeItem(i)
                    break

        # Deep 3-D U-Net skull-strip (ONNX) first; on ANY problem — model
        # unavailable, no 3-D volume, or a runtime error — fall back to the
        # classical Otsu outline so the button always works.
        mask = self._deep_brain_mask()
        used_deep = mask is not None
        if mask is None:
            mask = detect_brain_outline(self._canvas._img_data)
        if mask is None or not np.asarray(mask).any():
            QMessageBox.information(
                self, "No brain found",
                "Brain detection did not find a brain region.\n"
                "This works on in-vivo brain slices — for phantoms use "
                "'Detect Outline' instead.")
            return
        roi = ROI(
            name="Brain_outline",
            roi_type="auto",
            mask=mask,
            color="#00c2a8",  # teal — distinct from the orange phantom outline
        )
        self._canvas._rois.append(roi)
        self._canvas._draw_single_roi(roi)
        self._canvas.draw()
        self._roi_list.addItem(roi.name)
        self._canvas.roi_added.emit(roi)
        self._show_stats_for_roi(roi)
        # Let the owning tab re-slice the cached 3-D deep mask to the current
        # view plane (axial / coronal / sagittal + rotation), if wired.
        cb = getattr(self, "_on_brain_detected", None)
        if callable(cb):
            try:
                cb()
            except Exception:
                pass

    def _detect_tubes(self):
        if self._canvas._img_data is None:
            QMessageBox.information(self, "No image", "Load an image first.")
            return
        sigma = self._sigma_spin.value()
        min_px = self._minpx_spin.value()
        rois = detect_phantom_tubes(
            self._canvas._img_data,
            gauss_sigma=sigma,
            min_tube_pixels=min_px,
        )
        if not rois:
            QMessageBox.information(
                self, "No tubes found",
                "Auto-detection did not find any tube-like structures.\n"
                "Try adjusting σ or Min px."
            )
            return
        self._canvas.add_rois(rois)
        # Emit roi_added for each detected tube so ROI Manager broadcasts to all tabs
        for roi in rois:
            self._roi_list.addItem(roi.name)
            self._canvas.roi_added.emit(roi)

    def _mask_tubes(self):
        """Delegate to the external overlay callback set by ROITab (if available)."""
        if self._mask_tubes_fn is not None:
            self._mask_tubes_fn()
        else:
            QMessageBox.information(
                self, "Not available",
                "Load the ROI Manager tab to use this feature.\n"
                "The overlay requires access to the T1/T2 parametric maps."
            )
