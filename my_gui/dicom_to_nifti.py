"""
dicom_to_nifti.py
Convert a DICOM folder to clean per-series NIfTI volumes using dcm2niix.

A DICOM folder exported from a scanner is often a whole *study* — many series
(MPRAGE, scout, CEST-MRF, …) with different orientations and matrix sizes mixed
together.  Naively stacking every ``.dcm`` gives garbage.  dcm2niix
(Chris Rorden) groups them correctly into one NIfTI per series, which is exactly
what OCEAN needs to obtain a genuine 3-D T1 volume for deep skull-stripping.

The dcm2niix binary is located from (1) the PyInstaller bundle, (2) the
``dcm2niix`` pip wheel, or (3) the PATH.  ``is_available()`` is False when none
is found, so callers fall back to the classical single-slice DICOM reader.
"""
from __future__ import annotations

import os
import sys
import glob
import shutil
import tempfile
import subprocess

import numpy as np


def _find_dcm2niix() -> str | None:
    """Locate the dcm2niix executable (bundle → pip wheel → PATH)."""
    cands: list[str] = []
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        cands.append(os.path.join(meipass, "dcm2niix"))
        cands.append(os.path.join(meipass, "dcm2niix", "dcm2niix"))
    try:
        import dcm2niix as _d2n
        d = os.path.dirname(_d2n.__file__)
        cands.append(os.path.join(d, "dcm2niix"))
        cands.append(os.path.join(d, "bin", "dcm2niix"))
    except Exception:
        pass
    w = shutil.which("dcm2niix")
    if w:
        cands.append(w)
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def is_available() -> bool:
    return _find_dcm2niix() is not None


def convert_folder(dicom_dir: str, out_dir: str | None = None):
    """Convert ``dicom_dir`` to NIfTI; return ``(volumes, out_dir)``.

    ``volumes`` is a list of ``{"name", "path", "shape"}`` for every produced
    3-D (or 4-D) NIfTI, sorted by descending in-plane × slice voxel count, so
    the main anatomical volume (e.g. MPRAGE) comes first.  Single-slice scouts
    are dropped.  Raises ``RuntimeError`` if dcm2niix is unavailable.
    """
    exe = _find_dcm2niix()
    if exe is None:
        raise RuntimeError("dcm2niix not found")
    out_dir = out_dir or tempfile.mkdtemp(prefix="ocean_dcm2niix_")
    subprocess.run(
        [exe, "-z", "y", "-f", "%d_%s", "-o", out_dir, dicom_dir],
        check=True, capture_output=True, text=True, timeout=900,
    )
    import nibabel as nib
    vols = []
    for f in sorted(glob.glob(os.path.join(out_dir, "*.nii*"))):
        try:
            shp = tuple(int(s) for s in nib.load(f).shape)
        except Exception:
            continue
        if len(shp) >= 3 and min(shp[:3]) >= 2:          # real volume, not a scout slice
            name = os.path.basename(f)
            for ext in (".nii.gz", ".nii"):
                if name.endswith(ext):
                    name = name[: -len(ext)]
                    break
            vols.append({"name": name, "path": f, "shape": shp})
    vols.sort(key=lambda v: -int(np.prod(v["shape"][:3])))
    return vols, out_dir
