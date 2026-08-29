"""
ge_siemens_reader.py
====================
Readers for GE / Siemens (and any other vendor) data saved as DICOM or NIfTI.

Supported acquisitions
----------------------
T1 mapping – Variable TR (VTR / VFA)
    DICOM : one .dcm file per TR, RepetitionTime tag extracted automatically
    NIfTI : one .nii file per TR, TR extracted from filename (e.g. "Tr_100")

T2 mapping – Multi-Spin-Multi-Echo (MSME)
    DICOM : one .dcm file per TE, EchoTime tag extracted automatically
    NIfTI : separate echo files named *_e1.nii, *_e2.nii … (TE supplied by user
            OR parsed from optional BIDS sidecar JSON)

All functions return the same shape contract as the Bruker readers:
    T1: (Y, X, n_slices, nTR),  trs_ms  ndarray
    T2: (Y, X, n_slices, nTE),  tes_ms  ndarray

Dependencies
------------
    pydicom  – DICOM loading
    nibabel  – NIfTI loading
    numpy

Both packages are included in the cbdmrf environment.  If either is missing
the relevant function raises ImportError with a clear install hint.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Sequence

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _require_pydicom():
    try:
        import pydicom
        return pydicom
    except ImportError:
        raise ImportError(
            "pydicom is required for DICOM loading.\n"
            "Install it with:  pip install pydicom"
        )


def _require_nibabel():
    try:
        import nibabel as nib
        return nib
    except ImportError:
        raise ImportError(
            "nibabel is required for NIfTI loading.\n"
            "Install it with:  pip install nibabel"
        )


def _find_files(folder: str, extensions: Sequence[str]) -> list[str]:
    """Return sorted list of files in *folder* matching any of *extensions*."""
    files: list[str] = []
    for ext in extensions:
        files += glob.glob(os.path.join(folder, f"*{ext}"))
        files += glob.glob(os.path.join(folder, f"*{ext.upper()}"))
    return sorted(set(files))


def _extract_tr_from_filename(path: str) -> float | None:
    """Try to parse TR (ms) from a filename like 'Tr_100' or 'TR2500'."""
    name = os.path.basename(path)
    # Patterns: Tr_100, TR100, Tr 100, tr_2500, RepetitionTime2500, etc.
    m = re.search(r'[Tt][Rr][_\s]?(\d+)', name)
    if m:
        return float(m.group(1))
    return None


def _extract_echo_index_from_filename(path: str) -> int | None:
    """Parse echo index from filenames like *_e1.nii, *_e2.nii, echo1_, etc."""
    name = os.path.basename(path)
    # _e1, _e2, … (dcm2niix convention)
    m = re.search(r'_e(\d+)[\._]', name) or re.search(r'_e(\d+)$', name.split('.')[0])
    if m:
        return int(m.group(1))
    # echo1, echo_1, …
    m = re.search(r'echo[_]?(\d+)', name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _pixel_array_to_2d(ds) -> np.ndarray:
    """
    Extract a 2-D float image from a pydicom dataset.
    Handles single-frame DICOMs (shape HxW) and multi-frame (shape NxHxW → first frame).
    Applies RescaleSlope / RescaleIntercept if present.
    """
    arr = ds.pixel_array.astype(float)
    slope     = float(getattr(ds, 'RescaleSlope',     1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    arr = arr * slope + intercept

    if arr.ndim == 3:          # multi-frame: take first frame
        arr = arr[0]
    return arr                 # shape (H, W)


def _pixel_array_volume(ds) -> np.ndarray:
    """
    Extract a (H, W, n_slices) float image from a pydicom dataset.

    De-tiles Siemens MOSAIC frames into their constituent slices; for ordinary
    single-slice DICOMs returns (H, W, 1).  Applies RescaleSlope/Intercept.
    """
    from my_gui.dicom_mosaic import is_mosaic, mosaic_to_volume

    arr = ds.pixel_array.astype(float)
    slope     = float(getattr(ds, 'RescaleSlope',     1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    arr = arr * slope + intercept

    if arr.ndim == 2 and is_mosaic(ds):
        vol = mosaic_to_volume(arr, ds)        # (H, W, n_slices)
        return vol if vol.ndim == 3 else vol[:, :, np.newaxis]
    if arr.ndim == 3:                          # non-mosaic multiframe → first frame
        arr = arr[0]
    return arr[:, :, np.newaxis]               # (H, W, 1)


# ──────────────────────────────────────────────────────────────────────────────
# DICOM readers
# ──────────────────────────────────────────────────────────────────────────────

def read_dicom_t1_vtr(folder: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a variable-TR T1 dataset stored as one DICOM file per TR.

    Parameters
    ----------
    folder : str
        Directory containing the .dcm files (one per TR).

    Returns
    -------
    image  : ndarray  shape (Y, X, 1, nTR), float64
    trs_ms : ndarray  shape (nTR,), TR values in milliseconds, sorted ascending
    """
    pydicom = _require_pydicom()

    dcm_files = _find_files(folder, ['.dcm'])
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in: {folder}")

    # Load all datasets
    datasets: list = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f)
            datasets.append(ds)
        except Exception as exc:
            raise IOError(f"Failed to read DICOM file {f}: {exc}")

    # Extract RepetitionTime (ms) — fall back to filename if tag is absent
    def _get_tr(ds, filepath):
        tr = getattr(ds, 'RepetitionTime', None)
        if tr is not None:
            return float(tr)
        tr_fn = _extract_tr_from_filename(filepath)
        if tr_fn is not None:
            return tr_fn
        return None

    tagged: list[tuple[float, np.ndarray]] = []
    for ds, fpath in zip(datasets, dcm_files):
        tr = _get_tr(ds, fpath)
        if tr is None:
            raise ValueError(
                f"Cannot determine TR for {os.path.basename(fpath)}.\n"
                f"The file has no RepetitionTime DICOM tag and the filename "
                f"does not contain 'Tr_<value>'."
            )
        vol = _pixel_array_volume(ds)       # (Y, X, n_slices) — de-tiles Siemens mosaic
        tagged.append((tr, vol))

    # Sort by TR
    tagged.sort(key=lambda x: x[0])
    trs_ms = np.array([t[0] for t in tagged])
    # Stack per-TR volumes → (Y, X, n_slices, nTR); drop any TR whose slice count differs
    n_sl = tagged[0][1].shape[2]
    keep = [(tr, v) for tr, v in tagged if v.shape[2] == n_sl]
    trs_ms = np.array([t[0] for t in keep])
    image  = np.stack([t[1] for t in keep], axis=-1)     # (Y, X, n_slices, nTR)

    return image, trs_ms


def read_dicom_t2_msme(folder: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a multi-echo T2 dataset stored as one DICOM file per TE.

    Parameters
    ----------
    folder : str
        Directory containing the .dcm files (one per TE).

    Returns
    -------
    image  : ndarray  shape (Y, X, 1, nTE), float64
    tes_ms : ndarray  shape (nTE,), TE values in milliseconds, sorted ascending
    """
    pydicom = _require_pydicom()

    dcm_files = _find_files(folder, ['.dcm'])
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in: {folder}")

    datasets: list = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f)
            datasets.append(ds)
        except Exception as exc:
            raise IOError(f"Failed to read DICOM file {f}: {exc}")

    def _get_te(ds, filepath):
        te = getattr(ds, 'EchoTime', None)
        if te is not None:
            return float(te)
        return None

    tagged: list[tuple[float, np.ndarray]] = []
    for ds, fpath in zip(datasets, dcm_files):
        te = _get_te(ds, fpath)
        if te is None:
            raise ValueError(
                f"Cannot determine TE for {os.path.basename(fpath)}.\n"
                f"The file has no EchoTime DICOM tag."
            )
        vol = _pixel_array_volume(ds)       # (Y, X, n_slices) — de-tiles Siemens mosaic
        tagged.append((te, vol))

    tagged.sort(key=lambda x: x[0])
    # Stack per-TE volumes → (Y, X, n_slices, nTE); drop any TE whose slice count differs
    n_sl = tagged[0][1].shape[2]
    keep = [(te, v) for te, v in tagged if v.shape[2] == n_sl]
    tes_ms = np.array([t[0] for t in keep])
    image  = np.stack([t[1] for t in keep], axis=-1)     # (Y, X, n_slices, nTE)

    return image, tes_ms


# ──────────────────────────────────────────────────────────────────────────────
# NIfTI readers
# ──────────────────────────────────────────────────────────────────────────────

def read_nifti_t1_vtr(folder: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a variable-TR T1 dataset stored as one NIfTI file per TR.

    TR values are extracted from filenames (e.g. "Tr_100", "TR2500").
    Files that do not contain a recognisable TR pattern are silently skipped.

    Parameters
    ----------
    folder : str
        Directory containing the .nii or .nii.gz files.

    Returns
    -------
    image  : ndarray  shape (Y, X, 1, nTR), float64
    trs_ms : ndarray  shape (nTR,), TR values in ms, sorted ascending
    """
    nib = _require_nibabel()

    nii_files = _find_files(folder, ['.nii', '.nii.gz'])
    # Drop .gz duplicates when the uncompressed version also exists
    seen_base = set()
    filtered = []
    for f in sorted(nii_files, key=lambda x: (not x.endswith('.nii'), x)):
        base = f.removesuffix('.gz').removesuffix('.nii')
        if base not in seen_base:
            seen_base.add(base)
            filtered.append(f)
    nii_files = filtered

    if not nii_files:
        raise FileNotFoundError(f"No NIfTI files found in: {folder}")

    tagged: list[tuple[float, np.ndarray]] = []
    skipped = []
    for f in nii_files:
        tr = _extract_tr_from_filename(f)
        if tr is None:
            skipped.append(os.path.basename(f))
            continue
        img_nib = nib.load(f)
        arr = img_nib.get_fdata().squeeze()   # remove singleton dims → (Y, X)
        if arr.ndim == 3:
            arr = arr[:, :, 0]                # take first slice if still 3D
        tagged.append((tr, arr.astype(np.float64)))

    if not tagged:
        hint = f"\nSkipped files (no TR found): {skipped[:5]}" if skipped else ""
        raise ValueError(
            f"No NIfTI files with a recognisable TR pattern found in:\n{folder}{hint}\n\n"
            f"Expected filenames like: *Tr_100*.nii, *TR2500*.nii, etc."
        )

    tagged.sort(key=lambda x: x[0])
    trs_ms = np.array([t[0] for t in tagged])
    imgs   = np.stack([t[1] for t in tagged], axis=0)   # (nTR, Y, X)

    image = imgs.transpose(1, 2, 0)[:, :, np.newaxis, :]  # (Y, X, 1, nTR)
    return image, trs_ms


def read_nifti_t2_msme(
    folder: str,
    te_first_ms: float | None = None,
    te_step_ms: float | None  = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a multi-echo T2 dataset stored as separate NIfTI files per echo.

    Echo order is determined by the echo index in the filename (*_e1*, *_e2*, …).
    TE values are resolved in this priority order:
      1. BIDS sidecar JSON (``<stem>.json`` with ``EchoTime`` key, in seconds)
      2. ``te_first_ms`` + ``te_step_ms`` arguments supplied by the caller
      3. Raises ValueError — user must provide TE values

    Parameters
    ----------
    folder      : str    Directory containing echo NIfTI files.
    te_first_ms : float  Echo time of the first echo (ms). Used if no JSON.
    te_step_ms  : float  Echo spacing (ms). Used if no JSON.

    Returns
    -------
    image  : ndarray  shape (Y, X, 1, nTE), float64
    tes_ms : ndarray  shape (nTE,), TE values in ms, sorted by echo index
    """
    nib = _require_nibabel()

    nii_files = _find_files(folder, ['.nii', '.nii.gz'])
    # Deduplicate compressed/uncompressed pairs
    seen_base = set()
    filtered = []
    for f in sorted(nii_files, key=lambda x: (not x.endswith('.nii'), x)):
        base = f.removesuffix('.gz').removesuffix('.nii')
        if base not in seen_base:
            seen_base.add(base)
            filtered.append(f)
    nii_files = filtered

    if not nii_files:
        raise FileNotFoundError(f"No NIfTI files found in: {folder}")

    # Pair each file with its echo index
    echo_pairs: list[tuple[int, str]] = []
    for f in nii_files:
        idx = _extract_echo_index_from_filename(f)
        if idx is not None:
            echo_pairs.append((idx, f))

    if not echo_pairs:
        # Fallback: treat all files as ordered echoes (alphabetical order)
        echo_pairs = [(i + 1, f) for i, f in enumerate(nii_files)]

    echo_pairs.sort(key=lambda x: x[0])
    n_echoes = len(echo_pairs)

    # Build TE array
    tes_ms: np.ndarray | None = None

    # Try BIDS JSON sidecar for the first echo file
    _, first_file = echo_pairs[0]
    json_path = first_file.removesuffix('.gz').removesuffix('.nii') + '.json'
    if os.path.isfile(json_path):
        try:
            import json
            with open(json_path) as jf:
                meta = json.load(jf)
            if 'EchoTime' in meta:
                te1_s = float(meta['EchoTime'])          # BIDS stores in seconds
                te1_ms = te1_s * 1000.0
                # For multi-echo, some JSON list EchoTime per echo; others just first
                if isinstance(meta.get('EchoTime'), list):
                    tes_ms = np.array([float(t) * 1000 for t in meta['EchoTime']])
                elif 'EchoSpacing' in meta:
                    step = float(meta['EchoSpacing']) * 1000.0
                    tes_ms = te1_ms + np.arange(n_echoes) * step
                else:
                    # Only first TE known; need step from user
                    pass
        except Exception:
            pass

    if tes_ms is None:
        if te_first_ms is not None and te_step_ms is not None:
            tes_ms = te_first_ms + np.arange(n_echoes) * te_step_ms
        else:
            raise ValueError(
                f"Cannot determine TE values for the NIfTI T2 data in:\n{folder}\n\n"
                f"Please provide 'First TE (ms)' and 'TE step (ms)' values in the GUI,\n"
                f"or add a BIDS sidecar JSON with an EchoTime field."
            )

    # Load images
    imgs: list[np.ndarray] = []
    for _, f in echo_pairs:
        img_nib = nib.load(f)
        arr = img_nib.get_fdata().squeeze()
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        imgs.append(arr.astype(np.float64))

    stack = np.stack(imgs, axis=0)                        # (nTE, Y, X)
    image = stack.transpose(1, 2, 0)[:, :, np.newaxis, :]  # (Y, X, 1, nTE)

    return image, tes_ms[:n_echoes]


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: auto-detect format from folder contents
# ──────────────────────────────────────────────────────────────────────────────

def detect_folder_format(folder: str) -> str:
    """
    Sniff a folder and return 'dicom', 'nifti', or 'unknown'.
    Looks at file extensions; prefers DICOM if both are present.
    """
    if not os.path.isdir(folder):
        return 'unknown'
    dcm_files = _find_files(folder, ['.dcm'])
    nii_files = _find_files(folder, ['.nii', '.nii.gz'])
    if dcm_files:
        return 'dicom'
    if nii_files:
        return 'nifti'
    return 'unknown'


# ──────────────────────────────────────────────────────────────────────────────
# CEST readers (GE / Siemens)
# ──────────────────────────────────────────────────────────────────────────────

def read_dicom_cest(folder: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load CEST data from a folder of DICOM files.

    Assumes: one .dcm file per ppm offset, each file is a 2D image.
    Files are sorted by InstanceNumber (or filename if tag missing).
    ppm values must be provided externally (user supplies them after loading).

    Returns
    -------
    z_img : np.ndarray, shape (Y, X, 1, N_offsets)
    ppm   : np.ndarray, shape (N_offsets,) — all zeros (user must supply)

    The caller (ZSpecTab) will ask the user for the ppm array separately.
    """
    pydicom = _require_pydicom()
    dcm_files = sorted(
        glob.glob(os.path.join(folder, "*.dcm")) +
        glob.glob(os.path.join(folder, "*.IMA")) +
        glob.glob(os.path.join(folder, "*.ima")),
        key=lambda p: os.path.basename(p)
    )
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {folder}")

    # Sort by InstanceNumber tag if available
    def sort_key(p):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            return int(getattr(ds, 'InstanceNumber', 0))
        except Exception:
            return 0
    dcm_files = sorted(dcm_files, key=sort_key)

    frames = []
    for p in dcm_files:
        ds = pydicom.dcmread(p)
        arr = ds.pixel_array.astype(np.float32)
        if hasattr(ds, 'RescaleSlope'):
            arr = arr * float(ds.RescaleSlope) + float(getattr(ds, 'RescaleIntercept', 0))
        frames.append(arr)

    stack = np.stack(frames, axis=-1)   # (Y, X, N_offsets)
    z_img = stack[:, :, np.newaxis, :]  # → (Y, X, 1, N_offsets)
    ppm   = np.zeros(z_img.shape[-1], dtype=np.float32)
    return z_img, ppm


def read_nifti_cest(nii_path: str, ppm_path: "str | None" = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load CEST data from a 4D NIfTI file.

    Parameters
    ----------
    nii_path : path to .nii or .nii.gz file (shape: Y, X, slices, N_offsets)
    ppm_path : optional path to a plain-text file with one ppm value per line

    Returns
    -------
    z_img : np.ndarray, shape (Y, X, slices, N_offsets)
    ppm   : np.ndarray, shape (N_offsets,) — zeros if ppm_path not supplied
    """
    nib = _require_nibabel()
    img = nib.load(nii_path)
    data = np.asarray(img.dataobj, dtype=np.float32)

    # Ensure 4D: (Y, X, slices, offsets)
    if data.ndim == 3:
        data = data[:, :, :, np.newaxis]   # single offset
    elif data.ndim == 2:
        data = data[:, :, np.newaxis, np.newaxis]

    n_off = data.shape[-1]

    # Load ppm values
    if ppm_path and os.path.isfile(ppm_path):
        ppm = np.loadtxt(ppm_path, dtype=np.float32).ravel()
        if len(ppm) != n_off:
            raise ValueError(
                f"ppm file has {len(ppm)} entries but NIfTI has {n_off} volumes."
            )
    else:
        ppm = np.zeros(n_off, dtype=np.float32)

    return data, ppm
