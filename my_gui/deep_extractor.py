"""
deep_extractor.py
Optional deep-learning skull-stripping (brain extraction) for OCEAN.

Runs a compact 3-D U-Net (the DeepBrain "Extractor" model, converted to ONNX)
via onnxruntime — no TensorFlow dependency.  The ~3 MB ONNX model is bundled;
onnxruntime is an *optional* runtime dependency, so this module degrades
gracefully (``is_available()`` → False) when it is not installed, and the caller
falls back to the classical Otsu brain outline.

Model I/O is auto-detected: the 5-D tensor is the image input
(1, 128, 128, 128, 1); any scalar/boolean input is the batch-norm ``training``
flag and is fed ``False``.  Input is resized to 128³, max-normalised, and the
output probability volume is resized back to the caller's shape.

Reference: Itzcovich I. DeepBrain — https://github.com/iitzco/deepbrain (MIT).
"""
from __future__ import annotations

import os
import numpy as np

_MODEL_BASENAME = "deepbrain_extractor.onnx"
_SESSION = None            # cached onnxruntime InferenceSession
_MODEL_PATH_CACHE = None


def _model_path() -> str | None:
    """Locate the bundled ONNX model across dev and packaged (PyInstaller) runs."""
    global _MODEL_PATH_CACHE
    if _MODEL_PATH_CACHE is not None:
        return _MODEL_PATH_CACHE
    candidates = []
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "models", _MODEL_BASENAME))
    try:
        from my_gui.paths import get_resource_dir
        candidates.append(os.path.join(str(get_resource_dir()), "my_gui", "models", _MODEL_BASENAME))
        candidates.append(os.path.join(str(get_resource_dir()), "models", _MODEL_BASENAME))
    except Exception:
        pass
    import sys
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        candidates.append(os.path.join(meipass, "my_gui", "models", _MODEL_BASENAME))
        candidates.append(os.path.join(meipass, "models", _MODEL_BASENAME))
    for p in candidates:
        if p and os.path.isfile(p):
            _MODEL_PATH_CACHE = p
            return p
    return None


def is_available() -> bool:
    """True when onnxruntime is importable AND the bundled model is present."""
    if _model_path() is None:
        return False
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def _get_session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort
        mp = _model_path()
        if mp is None:
            raise FileNotFoundError(f"{_MODEL_BASENAME} not found in the OCEAN bundle")
        _SESSION = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
    return _SESSION


def _resize(vol: np.ndarray, out_shape) -> np.ndarray:
    """Trilinear resize a 3-D volume to ``out_shape`` (scipy, no skimage dep)."""
    from scipy.ndimage import zoom
    vol = np.asarray(vol, dtype=np.float32)
    factors = [o / max(s, 1) for o, s in zip(out_shape, vol.shape)]
    return zoom(vol, factors, order=1)


def extract_brain_prob(volume: np.ndarray) -> np.ndarray:
    """Return a per-voxel brain-tissue probability volume, same shape as ``volume``.

    ``volume`` is a 3-D array (H, W, D) — ideally a T1-weighted anatomical volume.
    """
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"expected a 3-D volume (H, W, D), got shape {vol.shape}")
    shape0 = vol.shape

    img = _resize(vol, (128, 128, 128))
    # Robust intensity normalisation: divide by the 99.5th percentile (of the
    # positive voxels) and clip to [0, 1], instead of the raw max.  This keeps
    # the brain from being under-exposed when a bright outlier is present
    # (e.g. fat or contrast enhancement in post-contrast T1 such as GE BRAVO),
    # which otherwise causes severe under-segmentation.
    pos = img[img > 0]
    scale = float(np.percentile(pos, 99.5)) if pos.size else float(np.max(img))
    if scale > 0:
        img = np.clip(img / scale, 0.0, 1.0)
    img = img.reshape(1, 128, 128, 128, 1).astype(np.float32)

    sess = _get_session()
    feed = {}
    for inp in sess.get_inputs():
        # 5-D tensor = the image; anything else = the boolean 'training' flag.
        rank = len([d for d in inp.shape]) if inp.shape is not None else 0
        if rank == 5:
            feed[inp.name] = img
        else:
            feed[inp.name] = np.array(False)
    prob = np.asarray(sess.run(None, feed)[0]).squeeze()
    prob = _resize(prob, shape0)
    return np.clip(prob, 0.0, 1.0)


def _cleanup_mask(m: np.ndarray) -> np.ndarray:
    """Keep the largest connected component, fill enclosed holes, and lightly
    smooth — removes stray false-positive specks and closes gaps so the brain
    mask is a single clean region."""
    try:
        from scipy import ndimage
        m = np.asarray(m, dtype=bool)
        if not m.any():
            return m
        lab, n = ndimage.label(m)
        if n > 1:
            counts = ndimage.sum(np.ones_like(m, dtype=float), lab,
                                 index=range(1, n + 1))
            m = (lab == int(np.argmax(counts)) + 1)
        m = ndimage.binary_fill_holes(m)
        m = ndimage.binary_closing(m, iterations=1)
        return m
    except Exception:
        return np.asarray(m, dtype=bool)


def extract_brain_mask(volume: np.ndarray, threshold: float = 0.5,
                       cleanup: bool = True) -> np.ndarray:
    """Return a boolean brain mask (same shape as ``volume``) at ``threshold``.

    With ``cleanup`` (default) the mask is reduced to its largest connected
    component with holes filled and edges lightly smoothed."""
    mask = extract_brain_prob(volume) > float(threshold)
    if cleanup:
        mask = _cleanup_mask(mask)
    return mask
