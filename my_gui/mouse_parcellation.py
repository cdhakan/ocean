"""
mouse_parcellation.py — 2-D atlas-based mouse-brain parcellation for OCEAN.
==========================================================================
Single-atlas, NiftyReg-only (no NiftySeg): affine+nonrigid register an atlas
CORONAL slice to the subject brain slice, then propagate the atlas label slice
(nearest-neighbour) into subject space.  Coarse-grouped into major structures
for a clean, readable overlay / set of ROIs.

IMPORTANT geometry: the dama-lab FVB_NCrl atlas is RAS (shape ~112×128×80):
axis 0 = L-R, **axis 1 = anterior-posterior (coronal index)**, axis 2 =
superior-inferior.  Coronal slices are therefore ``vol[:, k, :]`` oriented
dorsal-up as ``vol[:, k, :].T[::-1]`` — NOT ``vol[:, :, k]`` (that is axial).

Requires NiftyReg on PATH / in a conda env (reg_aladin/reg_f3d/reg_resample,
reused from mouse_brain_extractor) and a mouse atlas with template/ + label/ +
mask/ (OCEAN_MOUSE_ATLAS_DIR or ~/mouse_atlas, same as the brain-extraction
atlas).  2-D single-slice registration is approximate (not research-grade);
GitHub-quality parcellation needs a 3-D volume + multi-atlas fusion.
"""
from __future__ import annotations

import os
import glob
import numpy as np

# Coarse grouping of the 40-label MRM-NeAt scheme (masmat MRM_NeAt_Atlas_Label.txt)
# into major structures.  Anything labelled but not listed → "Other / white matter".
GROUPS = {
    "Cortex":              [14, 34],
    "Striatum":            [3, 23],
    "Hippocampus":         [1, 21],
    "Thalamus/Hypothal":   [7, 27, 11, 31],
    "Ventricles":          [10],
    "Cerebellum":          [8, 28],
    "Midbrain/Colliculi":  [9, 12, 18, 29, 32, 38, 13, 33, 17],
}
_OTHER = "Other/WhiteMatter"
GROUP_ORDER = list(GROUPS) + [_OTHER]
GROUP_COLORS = {
    "Cortex":             "#4da6ff",
    "Striatum":           "#1f3bbf",
    "Hippocampus":        "#2ca02c",
    "Thalamus/Hypothal":  "#ff7f0e",
    "Ventricles":         "#e377c2",
    "Cerebellum":         "#17becf",
    "Midbrain/Colliculi": "#d62728",
    _OTHER:               "#8c564b",
}
# label value → coarse group id (1..8); 8 = other
_LABEL2GID = {}
for _gi, _name in enumerate(GROUPS, start=1):
    for _lab in GROUPS[_name]:
        _LABEL2GID[_lab] = _gi
_OTHER_GID = len(GROUP_ORDER)         # 8
_GID2NAME = {i + 1: n for i, n in enumerate(GROUP_ORDER)}


class ParcellationUnavailable(RuntimeError):
    """Raised when NiftyReg or a mouse atlas (with labels) is not available."""


def _atlas_dir() -> str:
    """Resolve an atlas folder that has template/ + label/ + mask/."""
    from my_gui.mouse_brain_extractor import _atlas_dir as _brain_atlas
    d = _brain_atlas()
    if d and all(os.path.isdir(os.path.join(d, s)) for s in ("template", "label", "mask")):
        return d
    return ""


def available() -> bool:
    """True when reg_aladin is locatable AND an atlas with labels is configured."""
    from my_gui.mouse_brain_extractor import _find_reg_binary
    return _find_reg_binary("reg_aladin") is not None and bool(_atlas_dir())


# Atlas (RAS) axis for each slice plane: coronal→A (axis1), axial→S (axis2),
# sagittal→R (axis0).
_PLANE_AXIS = {"coronal": 1, "axial": 2, "sagittal": 0}


def _slice_plane(vol, k, plane):
    """Atlas slice at index k in the given plane, oriented dorsal/anterior-up."""
    if plane == "coronal":
        return np.ascontiguousarray(vol[:, k, :].T[::-1])   # (S↑, R→)
    if plane == "axial":
        return np.ascontiguousarray(vol[:, :, k].T[::-1])   # (A↑, R→)
    if plane == "sagittal":
        return np.ascontiguousarray(vol[k, :, :].T[::-1])   # (S↑, A→)
    raise ValueError(f"unknown plane {plane!r}")


def _coronal(vol, k):
    """Atlas coronal slice at anterior-posterior index k, oriented dorsal-up."""
    return _slice_plane(vol, k, "coronal")


def _atlas_paths(atlas_dir):
    """First template/label/mask triple sharing an id."""
    t = sorted(glob.glob(os.path.join(atlas_dir, "template", "*.nii*")))
    if not t:
        raise ParcellationUnavailable(f"no atlas template under {atlas_dir}")
    base = os.path.basename(t[0])
    lab = os.path.join(atlas_dir, "label", base)
    msk = os.path.join(atlas_dir, "mask", base)
    if not (os.path.isfile(lab) and os.path.isfile(msk)):
        raise ParcellationUnavailable("atlas label/ or mask/ file missing")
    return t[0], lab, msk


def coarse_from_labels(lab2d: np.ndarray) -> np.ndarray:
    """Map a propagated fine-label image to coarse group ids (0 = background)."""
    out = np.zeros(lab2d.shape, dtype=np.int16)
    for lv, gid in _LABEL2GID.items():
        out[lab2d == lv] = gid
    out[(lab2d > 0) & (out == 0)] = _OTHER_GID
    return out


def region_masks(coarse_map: np.ndarray) -> "dict[str, tuple]":
    """{group_name: (bool_mask, hex_color)} for every coarse region present."""
    out = {}
    for gid in np.unique(coarse_map):
        if gid <= 0:
            continue
        name = _GID2NAME.get(int(gid))
        if not name:
            continue
        out[name] = (coarse_map == gid, GROUP_COLORS[name])
    return out


def parcellate_slice(image, brain_mask, ap_level: int = 0,
                     voxel_mm=None, search_step: int = 3, plane: str = "coronal"):
    """Parcellate one 2-D coronal brain slice via single-atlas NiftyReg.

    Parameters
    ----------
    image      : 2-D slice in DISPLAY orientation (dorsal-up), e.g. the brain the
                 'Detect Mice Brain Outline' button was run on.
    brain_mask : 2-D bool brain mask for ``image`` (same shape).
    ap_level   : 0 → auto-search the best atlas coronal (A-P) level; else use that
                 atlas axis-1 index directly (lets the user fix a bad auto-match).
    voxel_mm   : optional (dy, dx) subject voxel size in mm (default 0.1).

    Returns
    -------
    (coarse_map, info) : coarse group-id map (H, W) and
                         {'ap_level', 'flip', 'corr', 'n_regions'}.

    Raises ParcellationUnavailable when NiftyReg / atlas are missing.
    """
    import tempfile, subprocess, nibabel as nib
    from my_gui.mouse_brain_extractor import _find_reg_binary

    reg_aladin = _find_reg_binary("reg_aladin")
    reg_f3d = _find_reg_binary("reg_f3d")
    reg_resample = _find_reg_binary("reg_resample")
    if not (reg_aladin and reg_f3d and reg_resample):
        raise ParcellationUnavailable(
            "NiftyReg not found (reg_aladin/reg_f3d/reg_resample). Install with:\n"
            "    conda create -n niftyreg -c conda-forge niftyreg")
    atlas_dir = _atlas_dir()
    if not atlas_dir:
        raise ParcellationUnavailable(
            "No mouse atlas with labels found. Set OCEAN_MOUSE_ATLAS_DIR or run "
            "fetch_mouse_atlas.sh (needs template/ + label/ + mask/).")
    tmpl_p, lab_p, msk_p = _atlas_paths(atlas_dir)
    tmpl = nib.load(tmpl_p).get_fdata()
    lab = nib.load(lab_p).get_fdata()
    amsk = nib.load(msk_p).get_fdata()

    img = np.asarray(image, dtype=np.float32)
    bm = np.asarray(brain_mask, dtype=bool)
    if bm.shape != img.shape[:2]:
        raise ValueError("brain_mask shape must match image")
    if not bm.any():
        raise ValueError("empty brain mask")
    subj = np.where(bm, img, 0).astype(np.float32)

    dy, dx = (voxel_mm if voxel_mm else (0.1, 0.1))

    def _run(cmd):
        return subprocess.run(cmd, capture_output=True, text=True)

    def _corr(a, b):
        a = a[bm]; b = b[bm]
        if a.std() < 1e-6 or b.std() < 1e-6:
            return -1.0
        return float(np.corrcoef(a, b)[0, 1])

    with tempfile.TemporaryDirectory() as td:
        sp = os.path.join(td, "s.nii.gz")
        nib.save(nib.Nifti1Image(subj[:, :, None], np.diag([dy, dx, 1, 1])), sp)

        plane = plane if plane in _PLANE_AXIS else "coronal"
        pax = _PLANE_AXIS[plane]
        # Candidate levels along the plane axis: user-set, else the atlas's extent.
        if ap_level and ap_level > 0:
            ks = [min(int(ap_level), amsk.shape[pax] - 1)]
        else:
            cov = [int((np.take(amsk, k, pax) > 0).sum()) for k in range(amsk.shape[pax])]
            kmin = next((i for i, c in enumerate(cov) if c > 0.1 * max(cov)), 0)
            kmax = len(cov) - next((i for i, c in enumerate(cov[::-1]) if c > 0.1 * max(cov)), 0)
            ks = list(range(max(kmin, 1), max(kmax, kmin + 2), max(search_step, 1)))

        def _save_flo(arr):
            nib.save(nib.Nifti1Image(np.ascontiguousarray(arr, np.float32)[:, :, None],
                                     np.diag([0.15, 0.15, 1, 1])), os.path.join(td, "a.nii.gz"))

        best = (-2.0, None, False)
        for k in ks:
            base = _slice_plane(tmpl, k, plane) * (_slice_plane(amsk, k, plane) > 0)
            for fl in (False, True):
                at = np.fliplr(base) if fl else base
                _save_flo(at)
                if _run([reg_aladin, "-ref", sp, "-flo", os.path.join(td, "a.nii.gz"),
                         "-aff", os.path.join(td, "af.txt"),
                         "-res", os.path.join(td, "ar.nii.gz")]).returncode:
                    continue
                ar = np.asarray(nib.load(os.path.join(td, "ar.nii.gz")).get_fdata()).squeeze()
                c = _corr(ar, subj)
                if c > best[0]:
                    best = (c, k, fl)

        corr, k, fl = best
        if k is None:
            raise ParcellationUnavailable("registration failed on all atlas levels")

        at = _slice_plane(tmpl, k, plane) * (_slice_plane(amsk, k, plane) > 0)
        al = _slice_plane(lab, k, plane) * (_slice_plane(amsk, k, plane) > 0)
        if fl:
            at, al = np.fliplr(at), np.fliplr(al)
        _save_flo(at)
        nib.save(nib.Nifti1Image(np.ascontiguousarray(al, np.float32)[:, :, None],
                                 np.diag([0.15, 0.15, 1, 1])), os.path.join(td, "al.nii.gz"))
        _run([reg_aladin, "-ref", sp, "-flo", os.path.join(td, "a.nii.gz"),
              "-aff", os.path.join(td, "af.txt"), "-res", os.path.join(td, "ar.nii.gz")])
        _run([reg_f3d, "-ref", sp, "-flo", os.path.join(td, "a.nii.gz"),
              "-aff", os.path.join(td, "af.txt"), "-cpp", os.path.join(td, "cpp.nii.gz"),
              "-res", os.path.join(td, "f.nii.gz"), "-sx", "-4"])
        _run([reg_resample, "-ref", sp, "-flo", os.path.join(td, "al.nii.gz"),
              "-cpp", os.path.join(td, "cpp.nii.gz"),
              "-res", os.path.join(td, "lab.nii.gz"), "-inter", "0"])
        fine = np.asarray(nib.load(os.path.join(td, "lab.nii.gz")).get_fdata()).squeeze()

    fine = fine * bm
    coarse = coarse_from_labels(fine)
    info = {"ap_level": int(k), "flip": bool(fl), "corr": float(corr),
            "plane": plane, "n_regions": int(np.unique(coarse[coarse > 0]).size)}
    return coarse, info
