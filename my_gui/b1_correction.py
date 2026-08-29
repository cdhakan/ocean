"""
b1_correction.py
================
B1-inhomogeneity correction for CEST by interpolation across **multiple B1
saturation powers** (Windschuh et al., NMR Biomed 2015; CEST-sources.de).

Faithful Python port of CEST_EVAL's two MATLAB routines:

  * :func:`z_b1_correction`        ← ``Z_B1_correction.m``
        Corrects a full Z-spectrum stack ``(Y, X, Z, offset, B1)`` — used by the
        QUESP and Inverse-Z tabs (they load Z-spectra at several B1 powers).

  * :func:`contrast_b1_correction` ← ``contrast_B1_correction.m``
        Corrects a stack of contrast images ``(Y, X, B1)`` — used by the
        Quantitative Z Analysis tab.

Idea: each voxel's *absolute* B1 at sample ``i`` is ``B1_input[i] · rel_B1map``.
For every voxel (and every offset) the measured values are fit / interpolated as
a function of that absolute B1 and re-evaluated at a uniform target B1
(``B1_output``), removing B1 inhomogeneity.

Fit types (matching the MATLAB): ``linear``, ``spline``, ``poly2``…``poly5``,
``smoothingspline``.  With the few B1 samples typical of CEST (3–5) a smoothing
spline ≈ an interpolating cubic, so ``smoothingspline``/``spline`` both use a
cubic here; ``linear`` and the polynomials are exact ports.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence
import numpy as np

FIT_TYPES = ["linear", "poly2", "poly3", "poly4", "smoothingspline", "spline"]


def _fit_eval_multi(x: np.ndarray, Y: np.ndarray, x_out: np.ndarray,
                    fit_type: str) -> np.ndarray:
    """Fit each column of ``Y`` (shape ``(n_samples, n_cols)``) as a function of
    ``x`` (``(n_samples,)``) and evaluate at ``x_out`` (``(n_out,)``).
    Returns ``(n_cols, n_out)``.  Extrapolates like the MATLAB 'extrap'."""
    x = np.asarray(x, float)
    Y = np.asarray(Y, float)
    x_out = np.asarray(x_out, float)
    n = x.size

    ft = fit_type.lower()
    if ft.startswith("poly"):
        deg = int(ft[4:]) if len(ft) > 4 else 2
        deg = min(deg, n - 1)                       # can't exceed n-1
        coeffs = np.polyfit(x, Y, deg)              # (deg+1, n_cols) — all cols at once
        # evaluate: (n_out, n_cols) then → (n_cols, n_out)
        out = np.stack([np.polyval(coeffs[:, c], x_out) for c in range(Y.shape[1])], axis=0)
        return out

    if ft in ("spline", "smoothingspline") and n >= 4:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(x, Y, axis=0, extrapolate=True)   # all cols at once
        return cs(x_out).T                          # (n_cols, n_out)

    # linear (default) — also the fallback when too few points for a spline
    from scipy.interpolate import interp1d
    f = interp1d(x, Y, axis=0, kind="linear",
                 bounds_error=False, fill_value="extrapolate")
    return f(x_out).T                               # (n_cols, n_out)


def _prep_common(rel_b1map, b1_input, b1_output, b1_input_index):
    rel = np.clip(np.asarray(rel_b1map, float), 0.01, 3.0)
    b1_input = np.asarray(b1_input, float).ravel()
    if b1_output is None:
        b1_output = np.array([float(np.mean(b1_input))])
    b1_output = np.atleast_1d(np.asarray(b1_output, float))
    if b1_input_index is None:
        b1_input_index = np.arange(b1_input.size)
    b1_input_index = np.asarray(b1_input_index, int)
    return rel, b1_input, b1_output, b1_input_index


def z_b1_correction(z_stack: np.ndarray, rel_b1map: np.ndarray,
                    b1_input: Sequence[float], b1_output=None,
                    segment: Optional[np.ndarray] = None,
                    fit_type: str = "linear",
                    b1_input_index: Optional[Sequence[int]] = None,
                    progress: Optional[Callable[[int, int], None]] = None
                    ) -> np.ndarray:
    """Port of ``Z_B1_correction.m``.

    ``z_stack``  : ``(Y, X, Z, offset, B1)`` Z-spectra at several B1 powers.
    ``rel_b1map``: relative B1 map, ``(Y, X)`` or ``(Y, X, Z)``.
    ``b1_input`` : the B1 value [µT] of each B1 sample (length = last axis).
    ``b1_output``: target B1 (scalar or vector); default = mean(b1_input).
    Returns ``(Y, X, Z, offset, len(b1_output))``.
    """
    z = np.asarray(z_stack, float)
    if z.ndim == 4:                                 # (Y,X,offset,B1) → add slice axis
        z = z[:, :, None, :, :]
    Y, X, Z, O, B = z.shape
    rel, b1_input, b1_output, idx = _prep_common(rel_b1map, b1_input, b1_output, b1_input_index)
    if rel.ndim == 2:
        rel = np.repeat(rel[:, :, None], Z, axis=2)
    if segment is None:
        segment = np.ones((Y, X, Z), bool)
    elif segment.ndim == 2:
        segment = np.repeat(segment[:, :, None], Z, axis=2)
    segment = segment.astype(bool)

    bi = b1_input[idx]
    out = np.full((Y, X, Z, O, b1_output.size), np.nan)
    for zz in range(Z):
        ys, xs = np.where(segment[:, :, zz])
        for yy, xx in zip(ys, xs):
            zvals = z[yy, xx, zz][:, idx]           # (offset, n_samples)
            if not np.all(np.isfinite(zvals)):
                continue
            abs_b1 = bi * rel[yy, xx, zz]
            if not np.all(np.isfinite(abs_b1)):
                continue
            # fit across B1 (samples) for every offset at once
            out[yy, xx, zz] = _fit_eval_multi(abs_b1, zvals.T, b1_output, fit_type)
        if progress is not None:
            progress(zz + 1, Z)
    return out


def contrast_b1_correction(img: np.ndarray, rel_b1map: np.ndarray,
                           b1_input: Sequence[float], b1_output=None,
                           segment: Optional[np.ndarray] = None,
                           fit_type: str = "linear",
                           b1_input_index: Optional[Sequence[int]] = None,
                           progress: Optional[Callable[[int, int], None]] = None
                           ) -> np.ndarray:
    """Port of ``contrast_B1_correction.m``.

    ``img``      : ``(Y, X, B1)`` — one contrast image per B1 power.
    ``rel_b1map``: relative B1 map ``(Y, X)``.
    ``b1_input`` : B1 value [µT] of each image.
    Returns ``(Y, X, len(b1_output))`` (squeezed to ``(Y, X)`` if scalar target).
    """
    im = np.asarray(img, float)
    Y, X, B = im.shape
    rel, b1_input, b1_output, idx = _prep_common(rel_b1map, b1_input, b1_output, b1_input_index)
    if segment is None:
        segment = np.ones((Y, X), bool)
    segment = segment.astype(bool)

    bi = b1_input[idx]
    out = np.full((Y, X, b1_output.size), np.nan)
    ys, xs = np.where(segment)
    total = ys.size
    for k, (yy, xx) in enumerate(zip(ys, xs)):
        vals = im[yy, xx, idx]                       # (n_samples,)
        r = rel[yy, xx]
        if (not np.all(np.isfinite(vals)) or not np.isfinite(r)
                or np.all(vals == 0)):
            continue
        abs_b1 = bi * r
        out[yy, xx] = _fit_eval_multi(abs_b1, vals[:, None], b1_output, fit_type)[0]
        if progress is not None and (k % 512 == 0 or k == total - 1):
            progress(k + 1, total)
    if b1_output.size == 1:
        return out[:, :, 0]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Integration helpers
# ─────────────────────────────────────────────────────────────────────────────
def _resize_map(m: np.ndarray, shape) -> np.ndarray:
    """Resize a 2-D map to ``shape`` (nearest/linear) if it doesn't already match."""
    m = np.asarray(m, float)
    if m.shape[:2] == tuple(shape):
        return m
    try:
        from scipy.ndimage import zoom
        zy, zx = shape[0] / m.shape[0], shape[1] / m.shape[1]
        return zoom(m, (zy, zx), order=1)
    except Exception:
        return m


def correct_datasets(datasets: list, rel_b1map: np.ndarray,
                     fit_type: str = "linear", b1_output=None,
                     progress: Optional[Callable[[int, int], None]] = None) -> list:
    """B1-correct a list of per-B1 CEST datasets (the QUESP / Inverse-Z model).

    Each dataset is a dict with ``'z_img'`` ``(Y, X, slices, offset)`` (or
    ``(Y, X, offset)``), ``'ppm'`` and ``'b1_ut'``.  All datasets are resampled
    onto the FIRST dataset's ppm grid, stacked along a B1 axis, corrected voxel-
    wise across B1 with :func:`z_b1_correction`, and split back.  ``b1_output``
    defaults to the datasets' nominal B1 vector, so the B1 series is preserved
    (only spatial B1 inhomogeneity is removed).  Returns NEW datasets (copies).
    """
    if len(datasets) < 2:
        return datasets                             # need ≥2 B1 powers to interpolate

    ref_ppm = np.asarray(datasets[0]["ppm"], float)
    b1_input = np.array([float(d.get("b1_ut", 0.0)) for d in datasets])

    def _as_yxso(z):
        z = np.asarray(z, float)
        return z[:, :, None, :] if z.ndim == 3 else z    # → (Y,X,slices,offset)

    z0 = _as_yxso(datasets[0]["z_img"])
    Y, X, S, O = z0.shape

    # Build 5-D stack (Y,X,slices,offset,B1), resampling each dataset to ref_ppm.
    stack = np.empty((Y, X, S, O, len(datasets)), float)
    for bi, d in enumerate(datasets):
        z = _as_yxso(d["z_img"])
        ppm = np.asarray(d["ppm"], float)
        if z.shape[:3] != (Y, X, S) or not np.array_equal(ppm, ref_ppm):
            # spatial reshape is not attempted (assume matched grids); resample ppm
            zr = np.empty((Y, X, S, O), float)
            for yy in range(min(Y, z.shape[0])):
                for xx in range(min(X, z.shape[1])):
                    for ss in range(min(S, z.shape[2])):
                        zr[yy, xx, ss] = np.interp(ref_ppm, ppm, z[yy, xx, ss])
            z = zr
        stack[..., bi] = z

    rel = _resize_map(rel_b1map, (Y, X))
    if b1_output is None:
        b1_output = b1_input                        # keep the nominal B1 series
    corr = z_b1_correction(stack, rel, b1_input, b1_output=b1_output,
                           fit_type=fit_type, progress=progress)   # (Y,X,S,O,len(b1_output))

    out = []
    for bi, d in enumerate(datasets):
        nd = dict(d)
        z = corr[:, :, :, :, bi]
        # restore original rank (drop singleton slice axis if the source was 3-D)
        if np.asarray(d["z_img"]).ndim == 3:
            z = z[:, :, 0, :]
        nd["z_img"] = z
        nd["ppm"] = ref_ppm
        nd["b1_corrected"] = True
        out.append(nd)
    return out
