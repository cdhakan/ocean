"""
fig_export.py
=============
Save a Matplotlib figure either as an image (PNG/PDF/SVG/TIFF) or as its
underlying plotted DATA (.mat / .npz), chosen automatically from the file
extension.  Used by every "Export figure" dialog so users can recover the
numbers behind a plot (measured/simulated fingerprints, Z-spectra, schedule
curves, …), not just a picture.

Usage in a save dialog:
    from my_gui.fig_export import save_figure, FIG_EXPORT_FILTER
    path, _ = QFileDialog.getSaveFileName(self, "Export figure", "plot", FIG_EXPORT_FILTER)
    if path:
        save_figure(fig, path, dpi=300)     # .mat/.npz → data, else image
"""

from __future__ import annotations

import os
import re
import numpy as np

# File-dialog filter offering images + data formats.  PNG / JPEG first (the
# common bitmap formats); images are written at 300 dpi by callers that pass
# dpi=300.
FIG_EXPORT_FILTER = (
    "PNG (*.png);;JPEG (*.jpg *.jpeg);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tif);;"
    "MATLAB data (*.mat);;NumPy data (*.npz)"
)


# Unicode sub-/superscripts used in display titles (kₛw, s⁻¹, fs₂, …) → ASCII,
# so exported variable names read like the parameter (kₛw → ksw, not k_w).
_SUBSUP = str.maketrans({
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6",
    "₇": "7", "₈": "8", "₉": "9", "ₛ": "s", "ₐ": "a", "ₑ": "e", "ₒ": "o",
    "ₓ": "x", "ₕ": "h", "ₖ": "k", "ₗ": "l", "ₘ": "m", "ₙ": "n", "ₚ": "p", "ₜ": "t",
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6",
    "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-", "⁺": "+",
})


def _san(name: str, fallback: str) -> str:
    """Turn a line/axis label into a valid MATLAB/NumPy variable name."""
    s = re.sub(r'[^0-9A-Za-z_]', '_', str(name).translate(_SUBSUP)).strip('_')
    if not s or not s[0].isalpha():
        s = fallback + (('_' + s) if s else '')
    return s[:48]


def figure_to_dict(fig) -> dict:
    """Flatten a figure's plotted line / scatter / image data into a dict of
    arrays with MATLAB-safe keys (``ax{i}_<label>_x`` / ``_y``, ``ax{i}_image{j}``)."""
    out: dict = {}
    meta = []
    for ai, ax in enumerate(fig.axes):
        axkey = f"ax{ai}"
        used: dict = {}

        # Line plots (plot / dashed / markers)
        for ln in ax.get_lines():
            lbl = ln.get_label()
            if not lbl or lbl.startswith('_'):     # skip helper lines (axhline, …)
                continue
            base = _san(lbl, 'line')
            used[base] = used.get(base, 0) + 1
            suf = '' if used[base] == 1 else f'_{used[base]}'
            k = f"{axkey}_{base}{suf}"
            out[f"{k}_x"] = np.asarray(ln.get_xdata(), dtype=float)
            out[f"{k}_y"] = np.asarray(ln.get_ydata(), dtype=float)

        # Scatter collections (Raw Z dots, etc.)
        for ci, col in enumerate(getattr(ax, 'collections', [])):
            try:
                off = np.asarray(col.get_offsets())
            except Exception:
                continue
            if off.ndim == 2 and off.shape[0] and off.shape[1] >= 2:
                out[f"{axkey}_scatter{ci}_x"] = off[:, 0].astype(float)
                out[f"{axkey}_scatter{ci}_y"] = off[:, 1].astype(float)

        # Images (parametric maps, k-space, …) — named by the axis TITLE (the
        # display name) so the exported .mat/.npz variable is human-readable:
        # a map shown as "fs  (mM)" is stored as `fs_mM`, "kₛw" as `k_w`, etc.,
        # instead of a generic `ax0_image0`.  Falls back to a generic key when the
        # axis has no title, and de-duplicates colliding names.  The chosen keys
        # are recorded in axes_info so the .mat/.npz still re-opens as a figure.
        img_keys = []
        _title = ax.get_title()
        for ii, im in enumerate(ax.get_images()):
            try:
                arr = np.asarray(im.get_array())
            except Exception:
                continue
            if _title:
                base = re.sub(r'_+', '_', _san(_title, f"{axkey}_image{ii}"))
            else:
                base = f"{axkey}_image{ii}"
            key, _n = base, 1
            while key in out:
                _n += 1
                key = f"{base}_{_n}"
            out[key] = arr
            img_keys.append(key)

        meta.append(
            f"{axkey}: title={ax.get_title()!r} "
            f"xlabel={ax.get_xlabel()!r} ylabel={ax.get_ylabel()!r} "
            f"images={img_keys!r}")

    if meta:
        out['axes_info'] = np.array(meta, dtype=object)
    return out


_AX_RE = re.compile(r'^(?:(?P<fig>[A-Za-z0-9]+)_)?ax(?P<ax>\d+)_(?P<rest>.+)$')


def _load_any(path: str) -> dict:
    """Load a .mat or .npz into a plain {key: ndarray} dict (dropping meta keys)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npz':
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    from scipy.io import loadmat
    d = loadmat(path, squeeze_me=True)
    return {k: v for k, v in d.items() if not str(k).startswith('__')}


def load_figures(path: str):
    """Reconstruct Matplotlib Figure(s) from a .mat/.npz.

    If the file was written by save_figure/save_figures (keys like
    ``[fig_]ax{i}_<label>_x/_y`` and ``ax{i}_image{j}``) the plots are rebuilt
    faithfully (lines + images + titles/labels).  Otherwise every 2-D/3-D array
    in the file is shown as an image so an arbitrary external file still
    displays something.  Returns a list of Figure objects.
    """
    from matplotlib.figure import Figure
    data = _load_any(path)

    # ── Reconstruct our exported figure-data format ────────────────────────
    groups: dict = {}       # figname -> {axidx -> {'lines':{base:{x,y}}, 'images':[]}}
    info: dict = {}         # figname -> list of axes_info strings
    consumed = set()
    for k, v in data.items():
        ks = str(k)
        if ks.endswith('axes_info'):
            fg = ks[:-len('axes_info')].rstrip('_') or '_'
            info[fg] = [str(s) for s in np.atleast_1d(v).ravel()]
            consumed.add(k); continue
        m = _AX_RE.match(ks)
        if not m:
            continue
        fg = m.group('fig') or '_'
        ax = int(m.group('ax')); rest = m.group('rest')
        g = groups.setdefault(fg, {}).setdefault(ax, {'lines': {}, 'images': []})
        if rest.endswith('_x') or rest.endswith('_y'):
            g['lines'].setdefault(rest[:-2], {})[rest[-1]] = np.asarray(v).ravel()
            consumed.add(k)
        elif rest.startswith('image'):
            g['images'].append(np.asarray(v)); consumed.add(k)

    # Attach title-named images (recorded as `images=[...]` in axes_info). These
    # keys are the display names (e.g. fs_mM) and don't match the ax{i}_image{j}
    # pattern, so pull them into their axis explicitly here.
    for fg, meta_lines in info.items():
        for s in meta_lines:
            am = re.match(r'^ax(\d+):', str(s))
            im_m = re.search(r"images=\[(.*?)\]", str(s))
            if not (am and im_m):
                continue
            axn = int(am.group(1))
            for kk in re.findall(r"['\"]([^'\"]+)['\"]", im_m.group(1)):
                lookup = kk if fg == '_' else f"{fg}_{kk}"
                if lookup in data and lookup not in consumed:
                    g = groups.setdefault(fg, {}).setdefault(
                        axn, {'lines': {}, 'images': []})
                    g['images'].append(np.asarray(data[lookup]))
                    consumed.add(lookup)

    figs = []
    for fg, axes in groups.items():
        n = max(len(axes), 1)
        f = Figure(figsize=(9, 3.2 * n))
        meta = info.get(fg, [])
        for i, (axidx, g) in enumerate(sorted(axes.items())):
            ax = f.add_subplot(n, 1, i + 1)
            for img in g['images']:
                ax.imshow(img)
            for base, xy in g['lines'].items():
                if 'x' in xy and 'y' in xy:
                    ax.plot(xy['x'], xy['y'], label=base)
            if g['lines']:
                ax.legend(fontsize=8)
            # title/labels from axes_info line "axN: title='..' xlabel='..' ylabel='..'"
            for s in meta:
                if s.startswith(f"ax{axidx}:"):
                    for attr, setter in (('title', ax.set_title),
                                         ('xlabel', ax.set_xlabel),
                                         ('ylabel', ax.set_ylabel)):
                        mm = re.search(attr + r"=[\"'](.*?)[\"']", s)
                        if mm and mm.group(1):
                            setter(mm.group(1))
        f.tight_layout()
        figs.append(f)
    if figs:
        return figs

    # ── Fallback: display any 2-D / 3-D arrays as images ───────────────────
    arrs = [(str(k), np.asarray(v)) for k, v in data.items()
            if isinstance(v, np.ndarray) and v.ndim in (2, 3) and v.size > 1]
    if not arrs:
        return []
    ncol = min(len(arrs), 3)
    nrow = (len(arrs) + ncol - 1) // ncol
    f = Figure(figsize=(5 * ncol, 4 * nrow))
    for i, (k, a) in enumerate(arrs):
        ax = f.add_subplot(nrow, ncol, i + 1)
        if a.ndim == 3 and a.shape[2] in (3, 4):
            ax.imshow(a)
        else:
            im = ax.imshow(np.real(a)); f.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(k, fontsize=8); ax.axis('off')
    f.tight_layout()
    return [f]


def render_figure_rgba(fig, dpi: int = 150) -> np.ndarray:
    """Render a Figure to an (H, W, 4) uint8 RGBA array for embedding as an image."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig.set_dpi(dpi)
    c = FigureCanvasAgg(fig)
    c.draw()
    return np.asarray(c.buffer_rgba())


def figures_to_dict(figs) -> dict:
    """Merge several figures' data into one dict, each key prefixed by the figure
    name.  `figs` is a dict {name: fig} or an iterable of figures."""
    items = figs.items() if isinstance(figs, dict) else [
        (f"fig{i}", f) for i, f in enumerate(figs)]
    out: dict = {}
    for name, fig in items:
        pre = _san(name, 'fig')
        for k, v in figure_to_dict(fig).items():
            out[f"{pre}_{k}"] = v
    return out


def save_figures(figs, path: str, dpi: int = 300, facecolor=None) -> str:
    """Save MULTIPLE figures.  For .mat/.npz all figures' data are merged into a
    single file (keys prefixed by figure name); for image extensions only the
    first figure is written (an image file can't hold several plots)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.mat':
        from scipy.io import savemat
        d = figures_to_dict(figs)
        savemat(path, d if d else {'note': np.array(['no data'])})
        return 'mat'
    if ext == '.npz':
        np.savez(path, **figures_to_dict(figs))
        return 'npz'
    first = next(iter(figs.values())) if isinstance(figs, dict) else list(figs)[0]
    return save_figure(first, path, dpi=dpi, facecolor=facecolor)


def save_figure(fig, path: str, dpi: int = 300, bbox_inches='tight',
                facecolor=None) -> str:
    """Save ``fig`` to ``path``.

    ``.mat`` → MATLAB struct of the plotted data; ``.npz`` → NumPy archive of the
    same; any other extension (png/pdf/svg/tif/…) → a rendered image via savefig.
    Returns the format written ('mat' | 'npz' | 'image').
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.mat':
        from scipy.io import savemat
        d = figure_to_dict(fig)
        if not d:
            d = {'note': np.array(['no line/scatter/image data in figure'])}
        savemat(path, d)
        return 'mat'
    if ext == '.npz':
        np.savez(path, **figure_to_dict(fig))
        return 'npz'
    kw = dict(dpi=dpi, bbox_inches=bbox_inches)
    if facecolor is not None:
        kw['facecolor'] = facecolor
    elif ext in ('.jpg', '.jpeg'):
        # JPEG has no alpha channel — pin an opaque facecolor (the figure's own)
        # so a transparent/default background doesn't flatten to black.
        try:
            kw['facecolor'] = fig.get_facecolor()
        except Exception:
            kw['facecolor'] = 'white'
    if ext in ('.jpg', '.jpeg'):
        # Near-lossless JPEG: Pillow defaults to quality≈75 with chroma
        # subsampling, which smears thin plot lines and text even at 300 dpi.
        kw['pil_kwargs'] = {'quality': 95, 'subsampling': 0}
    fig.savefig(path, **kw)
    return 'image'
