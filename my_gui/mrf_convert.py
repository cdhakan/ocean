"""
mrf_convert.py — any MRF scan → ``acquired_data.mat`` for the CEST-MRF pipeline.
==============================================================================
The CEST-MRF tab, the MRF viewer and the dictionary-matching worker all read
one file layout: ``acquired_data`` as float32 (Y, X, n_slices, n_meas) plus an
``info`` struct (``bruker_reader.save_acquired_data``).  This module produces
that file from whatever the Scan Directory's MR-Fingerprinting card is given:

    Bruker scan folder      → read_2dseq_mrf (unchanged, writes <folder>/acquired_data.mat)
    Siemens TWIX .dat       → siemens_twix_reader (raw k-space reconstructed in OCEAN)
    GE P-file P*.7          → ge_pfile_reader (GE Orchestra SDK, separate env/process)
    MATLAB .mat             → used as-is when it already holds ``acquired_data``,
                              else the largest image stack inside (my_gui.mat_reader)
    NIfTI file / folder     → nibabel; 4-D (Y, X, Z, T) keeps slices, 3-D → one slice
    DICOM folder            → one frame per file, in file order (= measurement order)

Where the file goes (``acquired_mat_path_for``): Bruker keeps its convention
(inside the scan folder); every other source gets a sibling folder named after
the input (``…/meas_MID00054_FID43049_MRF_2/acquired_data.mat`` for a .dat,
``…/<dicom-folder>_mrf/acquired_data.mat`` for a DICOM folder) so the .mat is
never dropped among image files that other readers glob.
"""
from __future__ import annotations

import glob
import os

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────────────────────────────────────

def _has_nifti(folder: str) -> bool:
    return bool(glob.glob(os.path.join(folder, "*.nii")) or
                glob.glob(os.path.join(folder, "*.nii.gz")))


def is_bruker_scan(path: str) -> bool:
    """A ParaVision scan folder (or its pdata/N folder)."""
    if not os.path.isdir(path):
        return False
    p = os.path.normpath(path)
    return (os.path.isfile(os.path.join(p, "method")) or
            os.path.isdir(os.path.join(p, "pdata")) or
            os.path.isfile(os.path.join(p, "2dseq")) or
            "pdata" in p.split(os.sep))


def classify(path: str) -> str:
    """'bruker' | 'twix' | 'mat' | 'nifti' | 'dicom'."""
    low = path.lower()
    if os.path.isfile(path):
        if low.endswith(".dat"):
            return "twix"
        if low.endswith(".7") and os.path.basename(path)[:1].upper() == "P":
            return "pfile"                   # GE raw data (P12345.7) → Orchestra
        if low.endswith(".mat"):
            return "mat"
        if low.endswith((".nii", ".nii.gz")):
            return "nifti"
        return "dicom"                       # a single DICOM file
    if is_bruker_scan(path):
        return "bruker"
    if _has_nifti(path):
        return "nifti"
    return "dicom"


def mat_has_acquired_data(path: str) -> bool:
    """True when a .mat already holds the ``acquired_data`` variable (checked
    from the variable list — the data itself is not read)."""
    try:
        import scipy.io as sio
        return any(v[0] == "acquired_data" for v in sio.whosmat(path))
    except NotImplementedError:              # MATLAB v7.3 (HDF5)
        try:
            import h5py
            with h5py.File(path, "r") as f:
                return "acquired_data" in f
        except Exception:
            return False
    except Exception:
        return False


def acquired_mat_path_for(path: str) -> str:
    """Where ``convert_to_acquired_mat(path)`` writes (or finds) the .mat."""
    path = os.path.abspath(path)
    kind = classify(path)
    if kind == "bruker":
        return os.path.join(path, "acquired_data.mat")
    if kind == "mat" and mat_has_acquired_data(path):
        return path                          # already the acquired file
    if os.path.isfile(path):
        base = os.path.basename(path)
        for ext in (".nii.gz", ".nii", ".dat", ".dcm", ".ima"):
            if base.lower().endswith(ext):
                base = base[:-len(ext)]
                break
        else:
            base = os.path.splitext(base)[0]
        return os.path.join(os.path.dirname(path), base, "acquired_data.mat")
    # DICOM / NIfTI folder → sibling folder, never inside the image folder
    return os.path.join(os.path.dirname(path.rstrip(os.sep)),
                        os.path.basename(path.rstrip(os.sep)) + "_mrf",
                        "acquired_data.mat")


# ──────────────────────────────────────────────────────────────────────────────
# Readers → (Y, X, n_slices, n_meas) float32
# ──────────────────────────────────────────────────────────────────────────────

def _read_nifti(path: str) -> np.ndarray:
    import nibabel as nib
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.nii")) +
                       glob.glob(os.path.join(path, "*.nii.gz")))
        if not files:
            raise FileNotFoundError(f"no NIfTI files in {path}")
        if len(files) == 1:
            path = files[0]
        else:
            # one 2-D/3-D file per measurement
            vols = [np.asarray(nib.load(f).dataobj, dtype=np.float32) for f in files]
            vols = [v[:, :, None] if v.ndim == 2 else v for v in vols]
            return np.stack(vols, axis=-1)                     # (Y, X, Z, T)
    data = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    if data.ndim == 2:
        return data[:, :, None, None]
    if data.ndim == 3:
        return data[:, :, None, :]                              # (Y, X, 1, N)
    if data.ndim == 4:
        return data                                             # (Y, X, Z, T)
    raise ValueError(f"unsupported NIfTI ndim {data.ndim}")


def _read_dicom(path: str) -> np.ndarray:
    from my_gui.tabs.image_viewer_tab import load_image_stack
    stack, _labels = load_image_stack(path, log_fn=lambda *a, **k: None)   # (H, W, N)
    return stack.astype(np.float32)[:, :, None, :]


def _read_mat(path: str) -> "tuple[np.ndarray, bool]":
    """(stack, already_acquired_file)."""
    from my_gui.mat_reader import load_mat, _pick_image, _orient_stack
    mats = load_mat(path)
    if "acquired_data" in mats:
        return None, True
    _, img = _pick_image(mats)
    return _orient_stack(img, want_slices=True).astype(np.float32), False


# ──────────────────────────────────────────────────────────────────────────────
# Converter
# ──────────────────────────────────────────────────────────────────────────────

def convert_to_acquired_mat(path: str, pv360: bool = False) -> str:
    """Convert any supported MRF scan to acquired_data.mat; returns its path."""
    from my_gui.bruker_reader import save_acquired_data

    kind = classify(path)
    out = acquired_mat_path_for(path)

    if kind == "bruker":
        from my_gui.bruker_reader import read_2dseq_mrf
        acq, info, seq = read_2dseq_mrf(path, pv360=pv360)
        return save_acquired_data(os.path.join(path, "acquired_data"), acq, info, seq, fmt="mat")

    if kind == "twix":
        from my_gui.siemens_twix_reader import convert_twix_to_acquired_mat
        return convert_twix_to_acquired_mat(path, out_dir=os.path.dirname(out))

    if kind == "pfile":
        from my_gui.ge_pfile_reader import convert_pfile_to_acquired_mat
        return convert_pfile_to_acquired_mat(path, out_dir=os.path.dirname(out))

    if kind == "mat":
        stack, ready = _read_mat(path)
        if ready:
            return path                                   # already the acquired file
    elif kind == "nifti":
        stack = _read_nifti(path)
    else:
        stack = _read_dicom(path)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    info = {
        "source":    {"mat": "MATLAB (.mat)", "nifti": "NIfTI", "dicom": "DICOM"}[kind],
        "src_path":  os.path.abspath(path),
        "n_meas":    int(stack.shape[3]),
        "n_slices":  int(stack.shape[2]),
        "matrix":    [int(stack.shape[0]), int(stack.shape[1])],
    }
    return save_acquired_data(os.path.join(os.path.dirname(out), "acquired_data"),
                              np.ascontiguousarray(stack, dtype=np.float32), info, None, fmt="mat")
