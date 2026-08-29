"""
motion_correction.py
=====================
Rigid **motion correction** for CEST 4-D stacks (CEST-MRF schedule frames or
CEST-MRI offset frames), via **itk-elastix** (the Python bindings to Elastix).

Every frame is registered to a chosen reference frame with a rigid (Euler)
transform and a Mattes mutual-information metric — the same scheme as the
clinical CEST-MRF pipeline (``CEST_MRF_MOCO.m`` / ``Rigid_MMI.txt``), but with no
external executables and no NIfTI temp files.

If ``itk-elastix`` is not installed, :func:`elastix_available` returns False and
the callers disable the feature with a clear message
(``pip install itk-elastix``).
"""
from __future__ import annotations

from typing import Callable, List, Optional
import numpy as np

# Reference-frame choices offered in the GUI dropdown.
REFERENCE_MODES = ["M0", "First frame", "Last frame", "Mean image"]


def elastix_available() -> bool:
    """True if itk-elastix is installed.

    Uses ``importlib.util.find_spec`` so the (slow, ~sec on a cold start) ``import
    itk`` is NOT triggered at GUI startup — it only checks that the package can be
    found.  The real import happens lazily the first time motion correction runs.
    """
    import importlib.util
    return importlib.util.find_spec("itk") is not None


def install_hint() -> str:
    return ("Motion correction needs itk-elastix.\n"
            "Install it with:\n\n    pip install itk-elastix")


def _choose_reference(frames: List[np.ndarray], mode: str):
    """Return ``(reference_array, reference_index)`` for the given mode.
    ``reference_index`` is -1 for the synthetic 'Mean image' reference."""
    if mode.startswith("First"):
        return np.asarray(frames[0], np.float32), 0
    if mode.startswith("Last"):
        return np.asarray(frames[-1], np.float32), len(frames) - 1
    if mode.startswith("Mean"):
        return np.mean(np.stack(frames, 0), 0).astype(np.float32), -1
    # default ('M0'): the brightest (least-saturated / max-signal) frame
    means = [float(np.nanmean(np.abs(f))) for f in frames]
    idx = int(np.argmax(means))
    return np.asarray(frames[idx], np.float32), idx


def _rigid_parameter_object(transform: str = "rigid", n_resolutions: int = 4):
    import itk
    po = itk.ParameterObject.New()
    pm = po.GetDefaultParameterMap(transform)
    # Match the clinical Rigid_MMI.txt intent: auto-init, multi-resolution,
    # Mattes MI (the elastix 'rigid' default already uses AdvancedMattesMI).
    pm["AutomaticTransformInitialization"] = ["true"]
    pm["NumberOfResolutions"] = [str(int(n_resolutions))]
    po.AddParameterMap(pm)
    return po


def _register_pair(fixed: np.ndarray, moving: np.ndarray,
                   param_object) -> np.ndarray:
    """Register ``moving`` onto ``fixed`` (same shape), returning the resampled
    moving image on the fixed grid (identical shape to ``fixed``)."""
    import itk
    f = itk.image_from_array(np.ascontiguousarray(fixed, dtype=np.float32))
    m = itk.image_from_array(np.ascontiguousarray(moving, dtype=np.float32))
    result, _ = itk.elastix_registration_method(
        f, m, parameter_object=param_object, log_to_console=False)
    return np.asarray(itk.array_from_image(result), dtype=np.float32)


def moco_frames(frames: List[np.ndarray], ref_mode: str = "M0 (max signal)",
                transform: str = "rigid",
                progress: Optional[Callable[[int, int], None]] = None
                ) -> List[np.ndarray]:
    """Register every frame in ``frames`` (2-D or 3-D arrays, all same shape) to
    the chosen reference.  The reference frame itself is returned unchanged.  A
    frame that fails to register is returned unmodified so the stack length and
    content are always preserved.  ``progress(done, total)`` is called per frame.
    """
    if not frames:
        return frames
    ref, ref_idx = _choose_reference(frames, ref_mode)
    po = _rigid_parameter_object(transform)
    out: List[np.ndarray] = []
    n = len(frames)
    for i, fr in enumerate(frames):
        fr = np.asarray(fr, np.float32)
        if i == ref_idx:
            out.append(fr)                         # reference → identity
        else:
            try:
                reg = _register_pair(ref, fr, po)
                if reg.shape != fr.shape:
                    reg = reg.reshape(fr.shape)
                out.append(reg)
            except Exception:
                out.append(fr)                     # keep original on failure
        if progress is not None:
            progress(i + 1, n)
    return out


def moco_4d(data: np.ndarray, frame_axis: int, ref_mode: str = "M0 (max signal)",
            transform: str = "rigid",
            progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """Motion-correct a 4-D stack whose frames lie along ``frame_axis``.

    Frames are moved to the front, squeezed of singleton spatial dims for the
    2-D/3-D registration, registered to the reference, restored to their
    original shape, and the axis order is put back.  Returns a new array of the
    same shape/dtype-family as ``data``.
    """
    arr = np.asarray(data, dtype=np.float32)
    moved = np.moveaxis(arr, frame_axis, 0)        # (n_frames, ...)
    orig_frame_shape = moved.shape[1:]
    frames = [np.squeeze(moved[i]) for i in range(moved.shape[0])]
    reg = moco_frames(frames, ref_mode, transform, progress)
    reg = [np.asarray(r, np.float32).reshape(orig_frame_shape) for r in reg]
    out = np.stack(reg, 0)
    return np.moveaxis(out, 0, frame_axis)
