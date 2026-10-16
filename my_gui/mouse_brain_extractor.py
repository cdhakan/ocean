"""
mouse_brain_extractor.py — Mouse / rodent brain extraction for OCEAN.
=====================================================================
Separate from the HUMAN skull-strip (`deep_extractor.py`, deepbrain ONNX),
which is left untouched.  Two backends, tried in order:

  1. **MASMAT** (multi-atlas, NiftyReg) — used when a NiftyReg install + a mouse
     atlas are configured (env vars below).  Registration-based brain extraction:
     affine + non-rigid register the atlas template to the subject, resample the
     atlas brain mask, threshold.
  2. **ONNX deep model** (`my_gui/models/mouse_brain_extractor.onnx`) — a
     cross-platform fallback, run through onnxruntime exactly like the human
     model.  Works the moment such a model is dropped into `my_gui/models/`.

If neither backend is available, `extract_mouse_brain_mask()` raises
`MouseBackendUnavailable` with guidance.

The MASMAT backend here does NOT run masmat-master's bash scripts — you do not
need that repo at all.  It reimplements MASMAT's single-atlas `mas_masking`
directly in Python (affine reg_aladin guided by the atlas brain mask, then
nearest-neighbour reg_resample of that mask), which needs ONLY NiftyReg (no
NiftySeg, FSL, ANTs, or 3D-Slicer) plus one mouse atlas folder.

Configuration (MASMAT backend):
  OCEAN_MOUSE_ATLAS_DIR   folder with an atlas template image and its brain mask
                          (e.g. template/<id>.nii.gz + mask/<id>.nii.gz).
  OCEAN_NIFTYREG_BIN      (optional) dir holding reg_aladin/reg_resample, or the
                          reg_aladin path itself.  Normally unneeded — the reg
                          binaries are auto-discovered across conda envs, so no
                          `conda activate` is required.  Install them with:
                              conda create -n niftyreg -c conda-forge niftyreg
"""
from __future__ import annotations

import os
import sys
import shutil
import numpy as np

_ONNX_MODEL_BASENAME = "mouse_brain_extractor.onnx"
_SESSION = None
_MODEL_PATH_CACHE = None


class MouseBackendUnavailable(RuntimeError):
    """Raised when no mouse-brain backend (MBE / MASMAT / ONNX) is usable."""


# ─────────────────────────────────────────────────────────────────────────────
# MBE backend — MouseBrainExtractor (MONAI SwinUNETR), 2-D invivo_aniso model.
# Runs OUT-OF-PROCESS in the dedicated 'mbe' conda env (keeps its GPL-2.0 code
# and TensorFlow-free PyTorch stack out of OCEAN's runtime env).  Best fit for
# 2-D / few-slice Bruker data (e.g. multi-slice T2 MSME, single-slice CEST).
# ─────────────────────────────────────────────────────────────────────────────
def _mbe_paths():
    """Return (env_python, mbe_dir, aniso_checkpoint) or None if not installed.
    Honors OCEAN_MBE_PY / OCEAN_MBE_DIR / OCEAN_MBE_WEIGHTS, else looks in ~/mbe
    and sibling conda envs (no activation needed)."""
    import glob
    home = os.path.expanduser("~")

    def _first(cands):
        return next((c for c in cands if c and os.path.exists(c)), None)

    py = os.environ.get("OCEAN_MBE_PY") or _first([
        os.path.join(home, "miniforge3_default", "envs", "mbe", "bin", "python"),
        os.path.join(home, "miniforge3", "envs", "mbe", "bin", "python"),
        os.path.join(home, "miniconda3", "envs", "mbe", "bin", "python")])
    mbe_dir = os.environ.get("OCEAN_MBE_DIR") or _first([
        os.path.join(home, "mbe", "MouseBrainExtractor")])
    wt = os.environ.get("OCEAN_MBE_WEIGHTS") or _first([
        os.path.join(home, "mbe", "MBE_weights")])
    if not (py and mbe_dir and wt):
        return None
    ckpt = _first(sorted(glob.glob(
        os.path.join(wt, "**", "invivo_aniso", "checkpoint_best.pth"),
        recursive=True)))
    if not ckpt:
        return None
    return py, mbe_dir, ckpt


def mbe_available() -> bool:
    """True when the MBE env + code + invivo_aniso weights are all present."""
    return _mbe_paths() is not None


def run_mbe_2d(image, threshold: float = 0.5, cleanup: bool = True) -> np.ndarray:
    """Skull-strip with MBE's 2-D invivo_aniso model, out-of-process.

    ``image`` is 2-D (H, W) or a few-slice stack (H, W, S).  Returns a bool mask
    of the same shape.  MBE's SliceInferer uses spatial_dim=1 and does NOT
    reorient, so the slice axis is moved to axis 1 for inference and back after.
    """
    import tempfile, subprocess, nibabel as nib
    paths = _mbe_paths()
    if paths is None:
        raise MouseBackendUnavailable(
            "MBE not installed. Run install_mbe.sh (creates the 'mbe' env + "
            "weights under ~/mbe).")
    py, mbe_dir, ckpt = paths

    arr = np.asarray(image, dtype=np.float32)
    was_2d = arr.ndim == 2
    if was_2d:
        arr = arr[:, :, None]                       # (H, W, 1)
    if arr.ndim != 3:
        raise MouseBackendUnavailable(f"MBE expects 2-D or 3-D, got {arr.shape}")

    vol_mbe = np.transpose(arr, (0, 2, 1))          # (H, S, W): slice -> axis 1
    with tempfile.TemporaryDirectory() as td:
        in_nii = os.path.join(td, "in.nii.gz")
        out_nii = os.path.join(td, "mask.nii.gz")
        pos_nii = os.path.join(td, "pos.nii.gz")
        nib.save(nib.Nifti1Image(vol_mbe, np.diag([0.1, 0.8, 0.1, 1.0])), in_nii)
        cmd = [py, "bin/run_mbe_predict_skullstrip.py",
               "-i", in_nii, "-o", out_nii, "-d", "2", "--patch_size", "128",
               "-n", ckpt, "--gen_posenc", pos_nii, "--device", "cpu"]
        env = dict(os.environ, PYTHONPATH=os.path.join(mbe_dir, "bin"))
        r = subprocess.run(cmd, cwd=mbe_dir, env=env,
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out_nii):
            raise MouseBackendUnavailable(
                "MBE inference failed:\n" + (r.stderr or r.stdout)[-600:])
        m = np.asarray(nib.load(out_nii).get_fdata())      # (H, S, W)
    mask = np.transpose(m, (0, 2, 1)) > float(threshold)   # back to (H, W, S)
    if cleanup:
        for s in range(mask.shape[2]):                     # per-slice (2-D) tidy
            mask[:, :, s] = _cleanup_mask_2d(mask[:, :, s])
    return mask[:, :, 0] if was_2d else mask


def _cleanup_mask_2d(m: np.ndarray) -> np.ndarray:
    """Tidy a single 2-D brain mask: morphological closing (fills edge notches /
    concavities so the boundary follows the brain rather than cutting inside it),
    fill holes, keep the largest component, and lightly smooth the boundary."""
    try:
        from scipy import ndimage
        m = np.asarray(m, dtype=bool)
        if not m.any():
            return m
        r = 3
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk = (xx * xx + yy * yy) <= r * r
        m = ndimage.binary_closing(m, disk)          # fill notches/concavities
        m = ndimage.binary_fill_holes(m)
        lab, n = ndimage.label(m)
        if n > 1:
            counts = ndimage.sum(np.ones_like(m, float), lab, index=range(1, n + 1))
            m = (lab == int(np.argmax(counts)) + 1)
        m = ndimage.gaussian_filter(m.astype(float), 1.0) > 0.5   # smooth boundary
        return m
    except Exception:
        return np.asarray(m, dtype=bool)


# ─────────────────────────────────────────────────────────────────────────────
# Backend availability
# ─────────────────────────────────────────────────────────────────────────────
def _find_reg_binary(name: str):
    """Locate a NiftyReg executable (e.g. 'reg_aladin') WITHOUT needing any
    conda env activated.  Search order: OCEAN_NIFTYREG_BIN, PATH, this
    interpreter's bin, $CONDA_PREFIX/bin, then sibling conda envs (…/envs/*/bin).
    Returns an absolute path or None."""
    import glob
    override = os.environ.get("OCEAN_NIFTYREG_BIN", "")
    if override:
        cand = override if os.path.isfile(override) else os.path.join(override, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    p = shutil.which(name)
    if p:
        return p
    bins = [os.path.join(sys.prefix, "bin")]
    cp = os.environ.get("CONDA_PREFIX", "")
    if cp:
        bins.append(os.path.join(cp, "bin"))
    seen, roots = set(), []
    for base in (sys.prefix, cp):
        d = base
        for _ in range(4):                 # walk up toward the conda root
            if not d or d in seen:
                break
            seen.add(d)
            if os.path.isdir(os.path.join(d, "envs")):
                roots.append(d)
            d = os.path.dirname(d)
    for r in roots:
        bins += glob.glob(os.path.join(r, "envs", "*", "bin"))
    for b in bins:
        cand = os.path.join(b, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _atlas_dir() -> str:
    """Resolve the mouse-atlas folder (one holding ``template/`` + ``mask/``).
    Prefers OCEAN_MOUSE_ATLAS_DIR, then auto-detects under common locations that
    fetch_mouse_atlas.sh uses — so the feature still works when a Finder-launched
    app does NOT inherit the shell env var.  Returns '' if none found."""
    import glob

    def _ok(d):
        return bool(d) and os.path.isdir(os.path.join(d, "template")) \
            and os.path.isdir(os.path.join(d, "mask"))

    env = os.environ.get("OCEAN_MOUSE_ATLAS_DIR", "")
    if _ok(env):
        return env
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roots = [os.path.expanduser("~/mouse_atlas"),
             os.path.join(repo_root, "mouse_atlas"), env]
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        if _ok(root):
            return root
        for d in sorted(glob.glob(os.path.join(root, "*", "*"))
                        + glob.glob(os.path.join(root, "*"))):
            if _ok(d):
                return d
    return ""


def masmat_available() -> bool:
    """True when a NiftyReg reg_aladin is locatable (no activation needed) AND a
    mouse atlas (template/ + mask/) is found.  Requires only NiftyReg — not
    NiftySeg — because brain extraction is single-atlas masking."""
    return _find_reg_binary("reg_aladin") is not None and bool(_atlas_dir())


def _onnx_model_path() -> "str | None":
    global _MODEL_PATH_CACHE
    if _MODEL_PATH_CACHE is not None:
        return _MODEL_PATH_CACHE or None
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.join(here, "models", _ONNX_MODEL_BASENAME)]
    try:
        from my_gui.paths import get_resource_dir
        rd = str(get_resource_dir())
        cands += [os.path.join(rd, "my_gui", "models", _ONNX_MODEL_BASENAME),
                  os.path.join(rd, "models", _ONNX_MODEL_BASENAME)]
    except Exception:
        pass
    import sys
    mp = getattr(sys, "_MEIPASS", "")
    if mp:
        cands += [os.path.join(mp, "my_gui", "models", _ONNX_MODEL_BASENAME),
                  os.path.join(mp, "models", _ONNX_MODEL_BASENAME)]
    for p in cands:
        if p and os.path.isfile(p):
            _MODEL_PATH_CACHE = p
            return p
    _MODEL_PATH_CACHE = ""
    return None


def onnx_available() -> bool:
    if _onnx_model_path() is None:
        return False
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def available_backends() -> list[str]:
    out = []
    if mbe_available():
        out.append("mbe")
    if masmat_available():
        out.append("masmat")
    if onnx_available():
        out.append("onnx")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ONNX backend (mirrors deep_extractor's inference, separate model)
# ─────────────────────────────────────────────────────────────────────────────
def _resize(vol, out_shape):
    from scipy.ndimage import zoom
    vol = np.asarray(vol, dtype=np.float32)
    factors = [o / max(s, 1) for o, s in zip(out_shape, vol.shape)]
    return zoom(vol, factors, order=1)


def _onnx_prob(volume: np.ndarray) -> np.ndarray:
    global _SESSION
    import onnxruntime as ort
    mp = _onnx_model_path()
    if mp is None:
        raise MouseBackendUnavailable("mouse ONNX model not found")
    if _SESSION is None:
        _SESSION = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
    vol = np.asarray(volume, dtype=np.float32)
    shape0 = vol.shape
    img = _resize(vol, (128, 128, 128))
    pos = img[img > 0]
    scale = float(np.percentile(pos, 99.5)) if pos.size else float(np.max(img))
    if scale > 0:
        img = np.clip(img / scale, 0.0, 1.0)
    img = img.reshape(1, 128, 128, 128, 1).astype(np.float32)
    feed = {}
    for inp in _SESSION.get_inputs():
        rank = len(inp.shape) if inp.shape is not None else 0
        feed[inp.name] = img if rank == 5 else np.array(False)
    prob = np.asarray(_SESSION.run(None, feed)[0]).squeeze()
    return np.clip(_resize(prob, shape0), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# MASMAT backend (single-atlas registration-based brain extraction via NiftyReg)
# ─────────────────────────────────────────────────────────────────────────────
def _find_atlas_pair(atlas_dir: str):
    """Return (template_nii, mask_nii) from a MASMAT-style atlas folder.

    Prefers the MASMAT layout ``atlas_dir/template/<id>`` + ``atlas_dir/mask/<id>``
    (matched by PARENT folder, since the files are often named just ``<id>.nii.gz``
    with no keyword), choosing a template/mask pair that share the same id; falls
    back to filename keywords.  ``label/`` (parcellation) is never a brain mask.
    """
    import glob
    if not atlas_dir or not os.path.isdir(atlas_dir):
        return None, None
    nii = sorted(glob.glob(os.path.join(atlas_dir, "**", "*.nii*"), recursive=True))
    if not nii:
        return None, None

    def _parent(f):
        return os.path.basename(os.path.dirname(f)).lower()

    def _base(f):
        return os.path.basename(f).lower()

    templates = [f for f in nii if _parent(f) == "template"]
    masks = [f for f in nii if _parent(f) == "mask"]

    # Prefer a template whose matching mask (same filename / id) exists.
    tmpl = mask = None
    if templates and masks:
        mask_by_name = {os.path.basename(f): f for f in masks}
        for t in templates:
            if os.path.basename(t) in mask_by_name:
                tmpl, mask = t, mask_by_name[os.path.basename(t)]
                break
        if tmpl is None:
            tmpl, mask = templates[0], masks[0]
    else:
        tmpl = templates[0] if templates else None
        mask = masks[0] if masks else None

    # Fallbacks by filename keyword (non-MASMAT layouts).
    if tmpl is None:
        tmpl = next((f for f in nii if any(k in _base(f)
                     for k in ("template", "t2", "t1", "average", "atlas"))
                     and "mask" not in _base(f) and "label" not in _base(f)), None)
    if mask is None:
        mask = next((f for f in nii if ("mask" in _base(f) or "brain" in _base(f))
                     and "template" not in _base(f) and "label" not in _base(f)), None)
    return tmpl, mask


def _run_masmat(volume: np.ndarray, voxel_mm=None) -> np.ndarray:
    """Mouse skull-strip the MASMAT ``mas_masking`` way — single-atlas, NiftyReg
    ONLY (no NiftySeg).  Affine-register the atlas template to the subject with
    the atlas brain mask as the floating mask (``reg_aladin -fmask``), then
    propagate that brain mask into the subject with nearest-neighbour
    (``reg_resample -inter 0``).  Returns an (H, W, D) bool brain mask."""
    import tempfile, subprocess, nibabel as nib
    reg_aladin = _find_reg_binary("reg_aladin")
    reg_resample = _find_reg_binary("reg_resample")
    if not reg_aladin or not reg_resample:
        raise MouseBackendUnavailable(
            "NiftyReg not found (reg_aladin / reg_resample). Install it with:\n"
            "    conda create -n niftyreg -c conda-forge niftyreg")
    atlas_dir = _atlas_dir()
    tmpl, mask = _find_atlas_pair(atlas_dir)
    if not tmpl or not mask:
        raise MouseBackendUnavailable(
            f"atlas template/mask not found (looked under {atlas_dir or '~/mouse_atlas'!r})")
    with tempfile.TemporaryDirectory() as td:
        aff = np.diag(list(voxel_mm) + [1.0]) if voxel_mm else np.eye(4)
        subj = os.path.join(td, "subject.nii.gz")
        nib.save(nib.Nifti1Image(np.asarray(volume, np.float32), aff), subj)
        aff_txt = os.path.join(td, "aff.txt")
        resmask = os.path.join(td, "mask_in_subject.nii.gz")
        # Affine atlas→subject, guided by the atlas brain mask (floating space).
        subprocess.run([reg_aladin, "-ref", subj, "-flo", tmpl, "-fmask", mask,
                        "-aff", aff_txt, "-res", os.path.join(td, "aff.nii.gz")],
                       check=True, capture_output=True)
        # Propagate the atlas brain mask into the subject (nearest-neighbour).
        subprocess.run([reg_resample, "-ref", subj, "-flo", mask, "-aff", aff_txt,
                        "-res", resmask, "-inter", "0"],
                       check=True, capture_output=True)
        m = np.asarray(nib.load(resmask).get_fdata()) > 0.5
    return _cleanup_mask(m)


# ─────────────────────────────────────────────────────────────────────────────
# Shared cleanup + public API
# ─────────────────────────────────────────────────────────────────────────────
def _cleanup_mask(m: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage
        m = np.asarray(m, dtype=bool)
        if not m.any():
            return m
        lab, n = ndimage.label(m)
        if n > 1:
            counts = ndimage.sum(np.ones_like(m, float), lab, index=range(1, n + 1))
            m = (lab == int(np.argmax(counts)) + 1)
        m = ndimage.binary_fill_holes(m)
        return ndimage.binary_closing(m, iterations=1)
    except Exception:
        return np.asarray(m, dtype=bool)


def extract_mouse_brain_mask(volume: np.ndarray, threshold: float = 0.5,
                             voxel_mm=None, cleanup: bool = True) -> np.ndarray:
    """Return a (H, W, D) bool mouse-brain mask.

    Tries MASMAT (if configured), then the ONNX model.  Raises
    MouseBackendUnavailable if neither backend is usable.
    """
    if masmat_available():
        return _run_masmat(volume, voxel_mm=voxel_mm)
    if onnx_available():
        m = _onnx_prob(volume) > float(threshold)
        return _cleanup_mask(m) if cleanup else m
    raise MouseBackendUnavailable(
        "No mouse-brain backend available.\n\n"
        "• MASMAT (recommended — works today, no masmat-master repo needed):\n"
        "    1. conda create -n niftyreg -c conda-forge niftyreg\n"
        "    2. point OCEAN at a mouse atlas folder (template + mask):\n"
        "         export OCEAN_MOUSE_ATLAS_DIR=/path/to/atlas\n"
        "  reg_aladin is auto-found across conda envs (no activate needed).\n\n"
        "• ONNX fallback: drop a trained 'mouse_brain_extractor.onnx' into "
        "my_gui/models/.")
