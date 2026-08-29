"""
fig_theme.py
============
Shared "Bg" (black background) theming for Matplotlib figures across the GUI.

`apply_fig_dark_theme(fig, dark)` flips a whole figure between a white and a
black background *cosmetically only* — it never touches image data, line data
or colormaps.  When `dark` is True the background goes black and every neutral
(black/grey/white) text element — titles, suptitle, tick numbers, axis labels,
legend text, colorbar numbers, annotations — turns white; spines/edges too.
Explicitly-coloured artists (coloured pool lines, coloured legend entries,
annotations sitting on an opaque light box) are left untouched.

Call it as the LAST step after a figure is drawn so nothing downstream resets
the colours; re-call it whenever the figure is rebuilt.
"""

from __future__ import annotations

import matplotlib.colors as _mcolors


def _is_neutral(color) -> bool:
    """True for black / white / grey (so we recolour it), False for a real hue."""
    try:
        r, g, b, _a = _mcolors.to_rgba(color)
    except Exception:
        return False
    return (max(r, g, b) - min(r, g, b)) < 0.15


def _recolor_axis(ax, bg, fg):
    ax.set_facecolor(bg)
    if ax.title is not None:
        ax.title.set_color(fg)
    ax.tick_params(colors=fg, which='both')
    for lab in (list(ax.get_xticklabels()) + list(ax.get_yticklabels())
                + list(ax.get_xticklabels(minor=True))
                + list(ax.get_yticklabels(minor=True))):
        lab.set_color(fg)
    for axis in (ax.xaxis, ax.yaxis):
        if axis.label is not None:
            axis.label.set_color(fg)
    for spine in ax.spines.values():
        spine.set_edgecolor(fg)
    # 3D axes (e.g. the Pulseq k-space plot): z ticks/label + the three panes.
    if hasattr(ax, 'zaxis'):
        try:
            ax.zaxis.set_tick_params(colors=fg)
            for lab in ax.get_zticklabels():
                lab.set_color(fg)
            if ax.zaxis.label is not None:
                ax.zaxis.label.set_color(fg)
            for _pax in (ax.xaxis, ax.yaxis, ax.zaxis):
                _pax.pane.set_facecolor(bg)
                _pax.pane.set_edgecolor(fg)
        except Exception:
            pass
    # Legend — frame + neutral text
    leg = ax.get_legend()
    if leg is not None:
        try:
            fr = leg.get_frame()
            fr.set_facecolor(bg)
            fr.set_edgecolor(fg)
        except Exception:
            pass
        for t in leg.get_texts():
            if _is_neutral(t.get_color()):
                t.set_color(fg)
    # Free-floating text (annotations, fs/ksw boxes …): recolour only neutral
    # text that isn't already sitting on its own opaque light box.
    for txt in ax.texts:
        box = txt.get_bbox_patch()
        if box is not None:
            try:
                if box.get_facecolor()[3] > 0.1:
                    continue          # readable on its own box → leave it
            except Exception:
                pass
        if _is_neutral(txt.get_color()):
            txt.set_color(fg)


def apply_fig_dark_theme(fig, dark: bool):
    """Theme an entire Matplotlib Figure black (dark=True) or white (dark=False).

    Cosmetic only — image/line data and colormaps are never modified.  Returns
    the figure for convenience.
    """
    bg = 'black' if dark else 'white'
    fg = 'white' if dark else 'black'
    fig.patch.set_facecolor(bg)
    st = getattr(fig, '_suptitle', None)
    if st is not None:
        st.set_color(fg)
    for txt in getattr(fig, 'texts', []):        # figure-level text
        if _is_neutral(txt.get_color()):
            txt.set_color(fg)
    for ax in fig.axes:                          # includes colorbar axes
        _recolor_axis(ax, bg, fg)
    return fig
