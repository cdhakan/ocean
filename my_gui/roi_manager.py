"""
roi_manager.py
Shared ROI store — a module-level singleton QObject that all tabs subscribe to.

Usage:
    from my_gui.roi_manager import get_roi_manager
    mgr = get_roi_manager()
    mgr.rois_changed.connect(my_callback)   # subscribe
    mgr.update_rois(rois, ref_shape)        # publish
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────

class ROIManager(QObject):
    """
    Central ROI store.  All draw operations happen in the dedicated ROI tab;
    all display tabs subscribe to `rois_changed` and re-overlay their canvases.
    """

    rois_changed = pyqtSignal(list)   # emits List[ROI]
    mask_changed = pyqtSignal()       # emits when the global analysis mask changes

    # ROIs whose names mark them as the *global analysis mask* — every module
    # (T1/T2/B1, CEST MRI, CEST MRF, QUESP) restricts its output maps to their
    # union.  "Brain_outline" comes from Detect Brain Outline, "Phantom_outline"
    # from Detect Outline.
    MASK_ROI_NAMES = ("Brain_outline", "Phantom_outline")

    def __init__(self):
        super().__init__()
        self._rois:      list  = []
        self._ref_shape: tuple = (1, 1)
        self._mask_enabled: bool = True   # honour the outline ROI as a global mask

    # ── Write ──────────────────────────────────────────────────────────────

    def update_rois(self, rois: list, ref_shape: tuple):
        """Replace the stored ROI list and broadcast to all subscribers."""
        from my_gui.roi_tools import ROI  # lazy import to avoid circularity
        self._rois      = [r for r in rois if isinstance(r, ROI)]
        self._ref_shape = tuple(ref_shape)[:2]
        self.rois_changed.emit(list(self._rois))

    def clear(self):
        self._rois = []
        self._ref_shape = (1, 1)
        self.rois_changed.emit([])

    # ── Read ───────────────────────────────────────────────────────────────

    def get_rois(self) -> list:
        return list(self._rois)

    def get_rois_for_shape(self, target_shape: tuple) -> list:
        """Return ROIs with masks re-scaled to *target_shape* if needed."""
        from my_gui.roi_tools import ROI
        if not self._rois:
            return []
        rh, rw = self._ref_shape[:2]
        th, tw = tuple(target_shape)[:2]
        if (rh, rw) == (th, tw):
            return list(self._rois)
        scaled: list = []
        for roi in self._rois:
            mh, mw = roi.mask.shape[:2]
            if (mh, mw) == (th, tw):
                scaled.append(roi)
                continue
            try:
                from scipy.ndimage import zoom
                zy = th / max(mh, 1)
                zx = tw / max(mw, 1)
                new_mask = zoom(roi.mask.astype(float), (zy, zx), order=1) > 0.5
                scaled.append(ROI(name=roi.name, roi_type=roi.roi_type,
                                  mask=new_mask, color=roi.color))
            except Exception:
                scaled.append(roi)
        return scaled

    # ── Global analysis mask ─────────────────────────────────────────────────

    def set_mask_enabled(self, enabled: bool):
        """Enable/disable using the outline ROI as a global analysis mask."""
        enabled = bool(enabled)
        if enabled != self._mask_enabled:
            self._mask_enabled = enabled
            self.mask_changed.emit()

    def is_mask_enabled(self) -> bool:
        return self._mask_enabled

    def has_mask_roi(self) -> bool:
        """True if a Brain_outline / Phantom_outline ROI is currently defined."""
        return any(r.name in self.MASK_ROI_NAMES for r in self._rois)

    def get_analysis_mask(self, target_shape) -> "np.ndarray | None":
        """
        Boolean (Y, X) mask = union of every Brain_outline / Phantom_outline ROI,
        rescaled to ``target_shape``.  Returns ``None`` when masking is disabled or
        no such ROI exists (→ analyses run over the full FOV, unchanged behaviour).
        """
        if not self._mask_enabled:
            return None
        th, tw = tuple(target_shape)[:2]
        rois = self.get_rois_for_shape((th, tw))
        out = np.zeros((th, tw), dtype=bool)
        found = False
        for r in rois:
            if r.name not in self.MASK_ROI_NAMES:
                continue
            m = np.asarray(r.mask, dtype=bool)
            if m.shape[:2] != (th, tw):
                continue
            out |= m
            found = True
        return out if (found and out.any()) else None

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save all ROIs to a .mat or .npz file."""
        data: dict = {
            "roi_names":  np.array([r.name      for r in self._rois], dtype=object),
            "roi_types":  np.array([r.roi_type  for r in self._rois], dtype=object),
            "roi_colors": np.array([r.color     for r in self._rois], dtype=object),
            "ref_shape":  np.array(list(self._ref_shape[:2])),
        }
        for i, roi in enumerate(self._rois):
            data[f"mask_{i}"] = roi.mask.astype(np.uint8)
        if path.endswith(".npz"):
            np.savez(path, **data)
        else:
            import scipy.io as sio
            sio.savemat(path, data)

    @staticmethod
    def _unwrap_str(val) -> str:
        """Unwrap nested numpy arrays / scalars and return a plain Python str.

        scipy.io.loadmat wraps MATLAB cell strings in nested arrays, e.g.
        ``array(['#44ffff'], dtype='<U7')`` — str() of that gives the list-repr
        ``"['#44ffff']"`` which is invalid as a matplotlib colour.  This helper
        drills down to the actual scalar before calling str().
        """
        while isinstance(val, np.ndarray):
            if val.size == 0:
                return ""
            val = val.flat[0]
        if isinstance(val, (bytes, np.bytes_)):
            return val.decode("utf-8", errors="replace")
        return str(val)

    def load(self, path: str):
        """Load ROIs from a .mat or .npz file and broadcast."""
        from my_gui.roi_tools import ROI
        if path.endswith(".npz"):
            raw = dict(np.load(path, allow_pickle=True))
        else:
            import scipy.io as sio
            raw = sio.loadmat(path)

        names   = list(np.array(raw.get("roi_names",  [])).ravel())
        types   = list(np.array(raw.get("roi_types",  [])).ravel())
        colors  = list(np.array(raw.get("roi_colors", [])).ravel())
        rs_arr  = np.array(raw.get("ref_shape", [1, 1])).ravel()
        self._ref_shape = (int(rs_arr[0]), int(rs_arr[1]))

        _s = ROIManager._unwrap_str          # shorthand
        rois = []
        for i, name in enumerate(names):
            key = f"mask_{i}"
            if key not in raw:
                continue
            mask      = np.array(raw[key]).astype(bool).squeeze()
            roi_type  = _s(types[i])  if i < len(types)  else "auto"
            color     = _s(colors[i]) if i < len(colors) else "#ff4444"
            rois.append(ROI(name=_s(name), roi_type=roi_type,
                            mask=mask, color=color))
        self._rois = rois
        self.rois_changed.emit(list(self._rois))

    # ── Convenience for tabs ───────────────────────────────────────────────

    def connect_canvas(self, canvas, app_ref=None):
        """
        Subscribe *canvas* to ROI updates.
        When rois_changed fires, canvas ROI overlays are refreshed.
        Only works if *canvas* is an ROICanvas (has _img_shape / _rois attrs).
        Plain FigureCanvasQTAgg canvases are silently skipped.
        """
        # Guard: only ROICanvas instances have the required attributes
        if not (hasattr(canvas, '_img_shape') and hasattr(canvas, '_rois')
                and hasattr(canvas, '_redraw_roi_overlays')):
            return

        def _on_rois_changed(rois: list):
            if not (hasattr(canvas, '_img_shape') and hasattr(canvas, '_rois')):
                return
            if canvas._img_shape is not None:
                scaled = self.get_rois_for_shape(canvas._img_shape)
            else:
                scaled = list(rois)
            canvas._rois = scaled
            canvas._redraw_roi_overlays()
            canvas.draw_idle()

        self.rois_changed.connect(_on_rois_changed)


# ── Module-level singleton ─────────────────────────────────────────────────

_manager: ROIManager | None = None


def get_roi_manager() -> ROIManager:
    global _manager
    if _manager is None:
        _manager = ROIManager()
    return _manager


def apply_analysis_mask(arr, manager: "ROIManager | None" = None, fill=np.nan):
    """
    Restrict a quantitative map to the global analysis mask (brain / phantom
    outline).  Voxels outside the mask are set to ``fill`` (NaN by default, so
    they render transparent and are excluded from ROI statistics).

    ``arr`` may be 2-D ``(Y, X)`` or N-D with the first two axes spatial
    ``(Y, X, …)``.  If masking is disabled or no outline ROI exists, ``arr`` is
    returned unchanged.  Integer maps are promoted to float when NaN is the fill.
    """
    if arr is None:
        return arr
    a = np.asarray(arr)
    if a.ndim < 2:
        return arr
    mgr = manager if manager is not None else get_roi_manager()
    mask = mgr.get_analysis_mask(a.shape[:2])
    if mask is None:
        return arr
    # Promote to float if we need to store NaN in an integer/bool array.
    if isinstance(fill, float) and np.isnan(fill) and not np.issubdtype(a.dtype, np.floating):
        out = a.astype(np.float64)
    else:
        out = a.copy()
    out[~mask] = fill        # ~mask is (Y, X); indexes the leading spatial axes
    return out
