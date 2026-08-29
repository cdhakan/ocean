import sys
import os
import warnings

# Silence the benign Qt font-alias warning that prints on startup:
#   qt.qpa.fonts: Populating font family aliases took N ms. Replace uses of
#   missing font family "Courier" with one that exists to avoid this cost.
# Qt/matplotlib probe for a "Courier" monospace family that doesn't exist on
# macOS; the automatic fallback is fine, so we mute just that warning category.
# Must be set before the QApplication (i.e. before Qt initialises its logging).
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.warning=false")

# ── Ensure cest_mrf is importable ────────────────────────────────────────────
# The package lives in open-py-cest-mrf/ relative to the project root.
# When the app is run directly (not via the cbdmrf venv) that directory may
# not be on sys.path, causing "No module named 'cest_mrf'".  Add it once here
# so every subsequent import in the app succeeds.
_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
_cest_src = os.path.join(_here, "open-py-cest-mrf")
if os.path.isdir(_cest_src) and _cest_src not in sys.path:
    sys.path.insert(0, _cest_src)

from PyQt6.QtWidgets import QApplication
from my_gui.paths import ensure_cwd   # set writable CWD before any file I/O
from my_gui.theme import apply_theme  # auto dark / light theme
from my_gui.app import MainWindow

# Suppress scipy curve_fit covariance warning — the fitted values are still
# valid; the covariance matrix simply cannot be estimated (e.g. when a
# parameter sits at its bound or the data is insensitive to it).
warnings.filterwarnings(
    "ignore",
    message="Covariance of the parameters could not be estimated",
    category=UserWarning,
    module="scipy",
)

# Silence matplotlib's benign "Glyph NNN (...) missing from font(s) X" warning:
# a Unicode sub/superscript (e.g. ₁) used in a title may be absent from a chosen
# family (Arial / Times New Roman); matplotlib renders it from a fallback font,
# so the character still shows — no action needed.
warnings.filterwarnings(
    "ignore",
    message=r"Glyph \d+ .*missing from font",
    category=UserWarning,
)

# Every figure the app writes should be publication-grade (300 dpi). The in-app
# "Export figure" dialogs already pass dpi=300 (fig_export.save_figure), but the
# matplotlib toolbar's own "Save" disk-icon and any pyplot save fall back to
# matplotlib's default 100 dpi. Pin the global default to 300 so those paths are
# high-resolution too. (Matplotlib rasterizes at figsize × dpi, independent of
# the on-screen canvas size, so this alone guarantees the output resolution.)
import matplotlib as _mpl
_mpl.rcParams["savefig.dpi"] = 300


def _install_qt_message_filter():
    """Definitively mute the benign Qt font-alias warning
    ('Populating font family aliases … missing font family "Courier"').
    The QT_LOGGING_RULES env var does not always take effect, so we also
    intercept the message text directly and forward everything else."""
    try:
        from PyQt6.QtCore import qInstallMessageHandler
    except Exception:
        return

    def _handler(mode, context, message):
        m = message or ""
        if ("Populating font family aliases" in m
                or 'missing font family "Courier"' in m):
            return
        sys.stderr.write(m + "\n")

    qInstallMessageHandler(_handler)


def _enable_draggable_legends():
    """Make every matplotlib legend draggable with the mouse, app-wide.

    Wraps Axes.legend so the returned legend has set_draggable(True); the user
    can then reposition any legend (ROI spectra, fit curves, map overlays, …)
    by clicking and dragging it. Applied once, before any figures are drawn.
    """
    try:
        from matplotlib.axes import Axes
    except Exception:
        return
    if getattr(Axes.legend, "_ocean_draggable", False):
        return
    _orig_legend = Axes.legend

    def _legend(self, *args, **kwargs):
        leg = _orig_legend(self, *args, **kwargs)
        try:
            if leg is not None:
                leg.set_draggable(True)
        except Exception:
            pass
        return leg

    _legend._ocean_draggable = True
    Axes.legend = _legend


def main():
    # Resolve a writable working directory first.
    # In a bundled .app the default CWD is the read-only bundle root;
    # ensure_cwd() switches to ~/Documents/OCEAN_Output/ so that any
    # legacy relative-path code keeps working without changes.
    ensure_cwd()

    _install_qt_message_filter()   # mute the benign Courier font-alias warning

    app = QApplication(sys.argv)
    app.setStyle("Fusion")   # consistent cross-platform widget rendering

    _enable_draggable_legends()   # any legend can be repositioned by dragging

    # Auto-detect system dark/light mode and apply matching palette +
    # global stylesheet. Works on macOS, Windows and Linux.
    mode = apply_theme(app, "auto")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # Required on Windows/macOS when using ProcessPoolExecutor with 'spawn'
    # (no-op on other platforms; harmless to always call)
    import multiprocessing
    multiprocessing.freeze_support()
    main()