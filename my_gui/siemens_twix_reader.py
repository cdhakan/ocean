"""
siemens_twix_reader.py — Siemens raw-data (.dat / TWIX) CEST reader for OCEAN.
=============================================================================
Reconstructs a Siemens TWIX ``meas_*.dat`` CEST / WASABI acquisition into an
image stack OCEAN can analyse.  Built for 2-D Cartesian multi-coil scans run
through the Pulseq interpreter, where every saturation offset is followed by a
full image readout and the line counter (``Lin``) is a running ADC index
(``Lin = offset*nPE + stored_line``).

Two things make the naive "fold by the header, iFFT in storage order" recon
wrong for such scans, and both are handled here:

* **Lines per offset** — ``lPhaseEncodingLines`` in the header is the GUI's
  dummy protocol value for a Pulseq scan (64 on a 128-line scan).  The true
  block length is read from the mdh *timestamps*: each saturation block is a
  run of ADCs at the imaging TR separated by a long saturation/recovery gap.
  Priority: ``lines_per_image`` (caller) > timestamp blocks > header.
* **Phase-encode order** — the .seq may acquire the lines centre-out
  (0, +1, −1, +2, −2, …; ``mdh.CenterLin == 0``).  Placing stored lines
  directly into k-space then yields an N/2-ghosted / FOV-folded image.  The
  order is detected from where the k-space energy sits (stored line 0 →
  centric, line nPE/2 → linear) and the lines are re-sorted into k-space rows
  before the 2-D inverse FFT.  The centric sign (0,+1,−1 vs 0,−1,+1) cannot be
  told from the data — it only mirrors the image along the phase axis — so
  ``pe_order='centric-'`` selects the mirrored convention.

Reconstruction per offset: 2-D inverse FFT of the (phase × readout) k-space,
readout oversampling removed, combined across coils by sum-of-squares.  The
ppm offset list is NOT stored in the .dat — supply it from the Pulseq ``.seq``
or an offsets .txt.

Requires ``twixtools`` (``pip install twixtools``); ``is_available()`` degrades
gracefully when it is absent.
"""
from __future__ import annotations

import os

import numpy as np


def is_available() -> bool:
    try:
        import twixtools  # noqa: F401
        return True
    except Exception:
        return False


def _phase_encode_lines(hdr) -> int | None:
    """nPE (lines per image) from the protocol header, else None.  For a Pulseq
    interpreter scan this is only the GUI's dummy value — see _timestamp_blocks."""
    try:
        ks = hdr.get("MeasYaps", {}).get("sKSpace", {})
        n = int(ks.get("lPhaseEncodingLines", 0))
        return n if n > 0 else None
    except Exception:
        return None


def _timestamp_blocks(mdb_list, gap_factor: float = 10.0) -> tuple[int | None, int]:
    """(lines_per_block, n_blocks) from the mdh timestamps (2.5 ms ticks).

    Consecutive ADCs of one image readout are one TR apart; saturation blocks
    are separated by a gap many TRs long.  Returns the common block length when
    every block has the same number of lines, else (None, n_blocks).
    """
    try:
        ts = np.array([int(m.mdh.TimeStamp) for m in mdb_list], dtype=np.int64)
    except Exception:
        return None, 0
    if ts.size < 4:
        return None, 0
    d = np.diff(ts)
    med = float(np.median(d))
    if med <= 0:
        return None, 0
    gaps = np.where(d > gap_factor * med)[0]
    if gaps.size == 0:
        return None, 1
    lens = np.diff(np.r_[-1, gaps, ts.size - 1])
    uniq = np.unique(lens)
    return (int(uniq[0]) if uniq.size == 1 else None), int(lens.size)


def _detect_pe_order(k: np.ndarray) -> str:
    """'centric' when the k-space energy peaks at stored line 0, 'linear' when
    it peaks at the middle line.  k: (n_off, nPE, n_cha, n_col)."""
    E = (np.abs(k) ** 2).sum(axis=(0, 2, 3))
    nPE = E.size
    i_max = int(np.argmax(E))
    tol = max(1, nPE // 16)
    if i_max <= tol:
        return "centric"
    if abs(i_max - nPE // 2) <= tol:
        return "linear"
    return "linear"


def _centric_rows(nPE: int, positive_first: bool = True) -> np.ndarray:
    """k-space row for each stored line of a centre-out acquisition.

    Order 0, +1, −1, +2, −2, … (``positive_first``) or 0, −1, +1, … ; for even
    nPE the single unmatched −nPE/2 line is acquired last.  Row = nPE//2 + ky.
    """
    ky = np.arange(-(nPE // 2), nPE - nPE // 2)
    key = (np.abs(ky), (ky < 0) if positive_first else (ky > 0))
    order = np.lexsort(key[::-1])            # sort by |ky|, then by sign
    return (nPE // 2 + ky[order]).astype(int)


def read_twix_cest(path: str, slice_idx: int = 0,
                   lines_per_image: int | None = None,
                   coil_combine: str = "sos",
                   pe_order: str = "auto") -> tuple:
    """Reconstruct a Siemens .dat CEST scan.

    Parameters
    ----------
    path            : Siemens ``meas_*.dat`` file (VD/VE).
    slice_idx       : slice to use when the acquisition is multi-slice.
    lines_per_image : phase-encode lines per offset image.  None → derived
                      from the mdh timestamps, else the header value.
    coil_combine    : 'sos' (sum-of-squares) or 'mean' (|coil| average).
    pe_order        : 'auto' (detect), 'centric' (0,+1,−1,…), 'centric-'
                      (0,−1,+1,…, the mirrored convention) or 'linear'.

    Returns
    -------
    img   : float32 (Y, X, n_slices, n_offsets) magnitude stack, Y = readout.
    M0    : float32 (Y, X, n_slices) — the highest-signal offset frame.
    info  : dict  { n_offsets, matrix, n_coils, m0_frame_idx, lines_per_image,
                    lines_source, n_blocks, header_npe, pe_order, larmor_mhz,
                    b0_t, source, notes }.
    """
    import twixtools

    tw = twixtools.map_twix(path, verbose=False)
    meas = tw[-1]                        # last block = the actual scan
    im = meas["image"]
    hdr = meas.get("hdr", {}) or {}
    im.flags["remove_os"] = True         # drop readout oversampling
    notes = []

    # ── lines per offset: caller > timestamps > header ───────────────────────
    nPE_hdr = _phase_encode_lines(hdr)
    nPE_ts, n_blocks = _timestamp_blocks(getattr(im, "mdb_list", []))
    if lines_per_image:
        nPE, src = int(lines_per_image), "user"
    elif nPE_ts:
        nPE, src = nPE_ts, "timestamps"
    elif nPE_hdr:
        nPE, src = nPE_hdr, "header"
        notes.append("lines per offset taken from the header (no timestamp "
                     "block structure found) — override with lines_per_image "
                     "if the frame count looks wrong")
    else:
        raise ValueError("could not determine phase-encode lines per image; "
                         "pass lines_per_image explicitly.")
    if nPE_hdr and nPE != nPE_hdr:
        notes.append(f"header lPhaseEncodingLines={nPE_hdr} ignored "
                     f"(scan has {nPE} lines per offset by {src})")

    # ── k-space → (Lin, Cha, Col) for the chosen slice ───────────────────────
    arr = np.asarray(im[:])
    dims = list(im.dims)
    sli_ax = dims.index("Sli") if "Sli" in dims else None
    if sli_ax is not None and arr.shape[sli_ax] > 1:
        arr = np.take(arr, min(slice_idx, arr.shape[sli_ax] - 1), axis=sli_ax)
        dims.pop(sli_ax)
    arr = arr.squeeze()
    if arr.ndim != 3:
        raise ValueError(f"unexpected TWIX image shape after squeeze: {arr.shape}")
    nlin_total, ncha, ncol = arr.shape
    if nlin_total % nPE != 0:
        raise ValueError(f"line count {nlin_total} not divisible by lines per "
                         f"image {nPE} ({src}); pass lines_per_image.")
    noff = nlin_total // nPE
    if nPE_ts and n_blocks and noff != n_blocks and src != "timestamps":
        notes.append(f"timestamps suggest {n_blocks} blocks of {nPE_ts} lines, "
                     f"but {noff} frames of {nPE} were used ({src})")

    k = arr.reshape(noff, nPE, ncha, ncol)      # (offset, stored line, coil, col)

    # ── phase-encode order: stored line → k-space row ────────────────────────
    order = (pe_order or "auto").lower()
    if order == "auto":
        order = _detect_pe_order(k)
    if order.startswith("centric"):
        rows = _centric_rows(nPE, positive_first=not order.endswith("-"))
    elif order == "linear":
        rows = np.arange(nPE)
    else:
        raise ValueError(f"unknown pe_order {pe_order!r}")

    # ── per-offset 2-D iFFT + coil combine ───────────────────────────────────
    frames = []
    k_ord = np.empty((nPE, ncha, ncol), dtype=k.dtype)
    for o in range(noff):
        k_ord[rows] = k[o]                                  # re-sort lines
        img = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(k_ord, axes=(0, 2)), axes=(0, 2)),
            axes=(0, 2))                                    # (row, coil, col)
        if coil_combine == "sos":
            mag = np.sqrt((np.abs(img) ** 2).sum(axis=1))   # (row, col)
        else:
            mag = np.abs(img).mean(axis=1)
        frames.append(mag.T)                                # → (readout, phase)

    stack = np.stack(frames, axis=-1).astype(np.float32)    # (Y, X, n_off)
    stack = stack[:, :, np.newaxis, :]                      # add slice axis

    # M0 = the highest-signal frame.  On a CEST plateau many far-offset frames
    # are within <1 % of each other, so this is an arbitrary plateau member —
    # fine as a normalisation reference, and the ppm list overrides it anyway.
    fs = stack.reshape(-1, noff).mean(axis=0)
    m0_idx = int(np.argmax(fs))
    M0 = stack[:, :, :, m0_idx].copy()

    def _hdr(path_keys, scale=1.0):
        try:
            v = hdr
            for kk in path_keys:
                v = v[kk]
            return float(v) * scale
        except Exception:
            return None

    # Voxel size = header FOV / RECONSTRUCTED matrix.  The header FOV and slice
    # thickness are reliable even for Pulseq-interpreter scans; the header
    # matrix fields are not (dummy GUI values), hence the reconstructed shape.
    def _slice0(key):
        try:
            return float(hdr["MeasYaps"]["sSliceArray"]["asSlice"][0][key])
        except Exception:
            return None
    _fov = (_slice0("dReadoutFOV"), _slice0("dPhaseFOV"), _slice0("dThickness"))
    _vox = ((_fov[0] / stack.shape[0]) if _fov[0] else None,
            (_fov[1] / stack.shape[1]) if _fov[1] else None,
            _fov[2])

    info = {
        "n_offsets":       noff,
        "matrix":          (stack.shape[0], stack.shape[1]),
        "n_coils":         ncha,
        "m0_frame_idx":    m0_idx,
        "lines_per_image": nPE,
        "lines_source":    src,
        "n_blocks":        n_blocks,
        "header_npe":      nPE_hdr,
        "pe_order":        order,
        "larmor_mhz":      _hdr(("Dicom", "lFrequency"), 1e-6),
        "b0_t":            _hdr(("Dicom", "flMagneticFieldStrength")),
        "fov_mm":          _fov,       # (readout, phase, slice thickness)
        "voxel_mm":        _vox,       # (Y, X, slice) — None when unknown
        "source":          "Siemens TWIX (.dat)",
        "notes":           notes,
    }
    return stack, M0, info


# ──────────────────────────────────────────────────────────────────────────────
# CEST-MRF: .dat → acquired_data.mat (same layout as the Bruker converter)
# ──────────────────────────────────────────────────────────────────────────────

def twix_output_dir(dat_path: str) -> str:
    """Folder that receives acquired_data.mat for a .dat: a sibling folder named
    after the file (``…/meas_MID00054_FID43049_MRF_2/``), mirroring the Bruker
    scan-folder convention the MRF pipeline expects."""
    base = os.path.splitext(os.path.basename(dat_path))[0]
    return os.path.join(os.path.dirname(os.path.abspath(dat_path)), base)


def convert_twix_to_acquired_mat(dat_path: str, out_dir: "str | None" = None,
                                 slice_idx: int = 0,
                                 lines_per_image: "int | None" = None) -> str:
    """Reconstruct a Siemens .dat CEST-MRF scan and save ``acquired_data.mat``.

    The file holds ``acquired_data`` as float32 (Y, X, n_slices, n_meas) plus an
    ``info`` struct (voxel size, FOV, Larmor, matrix, offsets, …) — the same
    layout ``bruker_reader.save_acquired_data`` writes, so the CEST-MRF tab,
    the MRF viewer and the dictionary-matching worker consume it unchanged.
    Returns the absolute path of the written .mat.
    """
    from my_gui.bruker_reader import save_acquired_data

    stack, M0, info = read_twix_cest(dat_path, slice_idx=slice_idx,
                                     lines_per_image=lines_per_image)
    out_dir = out_dir or twix_output_dir(dat_path)
    os.makedirs(out_dir, exist_ok=True)

    def _clean(v):
        """savemat cannot store None inside tuples → use NaN."""
        if isinstance(v, (tuple, list)):
            return [np.nan if x is None else x for x in v]
        return np.nan if v is None else v

    meta = {
        "source":          info["source"],
        "dat_path":        os.path.abspath(dat_path),
        "n_offsets":       info["n_offsets"],
        "matrix":          list(info["matrix"]),
        "n_coils":         info["n_coils"],
        "lines_per_image": info["lines_per_image"],
        "pe_order":        info["pe_order"],
        "voxel_mm":        _clean(info.get("voxel_mm")),
        "fov_mm":          _clean(info.get("fov_mm")),
        "larmor_mhz":      _clean(info.get("larmor_mhz")),
        "b0_t":            _clean(info.get("b0_t")),
        "m0_frame_idx":    info["m0_frame_idx"],
    }
    return save_acquired_data(os.path.join(out_dir, "acquired_data"),
                              stack.astype(np.float32), meta, None, fmt="mat")
