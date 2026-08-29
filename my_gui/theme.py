"""
theme.py  —  Automatic dark / light theme for CEST-MRF GUI
============================================================
Detects the macOS (or any OS) system preference at startup and applies a
consistent QPalette so every widget — styled or unstyled — uses readable
colours.

Usage (called once from main.py before the window is shown):
    from my_gui.theme import apply_theme
    apply_theme(app)          # auto-detects system preference
    apply_theme(app, "dark")  # force dark
    apply_theme(app, "light") # force light

The detected mode is stored so other modules can query it:
    from my_gui.theme import is_dark
    if is_dark():
        ...
"""
from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication, QStyleFactory
from PyQt6.QtCore import Qt

# ── Module-level state ─────────────────────────────────────────────────────────
_current_mode: str = "dark"   # updated by apply_theme()


def mono_font(size: int = 10):
    """Return a QFont using the first available monospace family.

    Avoids Qt's alias-lookup cost that triggers the warning:
      'Replace uses of missing font family "Courier" with one that exists'

    Import and use as:
        from my_gui.theme import mono_font
        widget.setFont(mono_font(10))
    """
    from PyQt6.QtGui import QFont, QFontDatabase
    families = set(QFontDatabase.families())
    for candidate in ("Menlo", "Consolas", "DejaVu Sans Mono",
                      "Liberation Mono", "Courier New"):
        if candidate in families:
            return QFont(candidate, size)
    f = QFont()
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPointSize(size)
    return f


def is_dark() -> bool:
    return _current_mode == "dark"


def detect_system_mode() -> str:
    """
    Return "dark" or "light" based on the system window background colour.
    Works on macOS, Windows and Linux without any platform-specific calls.
    """
    app = QApplication.instance()
    if app is None:
        return "dark"
    # Qt reports the native palette before any override
    bg = app.palette().color(QPalette.ColorRole.Window)
    # Light windows have high lightness; dark ones have low lightness
    return "dark" if bg.lightness() < 128 else "light"


# ── Colour tokens ──────────────────────────────────────────────────────────────

_DARK = dict(
    window          = "#1a1a1a",
    window_text     = "#e8e8e8",
    base            = "#2a2a2a",   # text-edit / list backgrounds
    alt_base        = "#232323",
    text            = "#e0e0e0",
    button          = "#2d2d2d",
    button_text     = "#e0e0e0",
    highlight       = "#1565c0",
    highlight_text  = "#ffffff",
    bright_text     = "#ffffff",
    disabled_text   = "#666666",
    tooltip_base    = "#2d2d2d",
    tooltip_text    = "#e0e0e0",
    mid             = "#3a3a3a",
    dark_shade      = "#111111",
    light_shade     = "#4a4a4a",
)

_LIGHT = dict(
    window          = "#f2f2f2",
    window_text     = "#1a1a1a",
    base            = "#ffffff",
    alt_base        = "#ebebeb",
    text            = "#1a1a1a",
    button          = "#e0e0e0",
    button_text     = "#1a1a1a",
    highlight       = "#1565c0",
    highlight_text  = "#ffffff",
    bright_text     = "#000000",
    disabled_text   = "#999999",
    tooltip_base    = "#fffbe6",
    tooltip_text    = "#222222",
    mid             = "#c0c0c0",
    dark_shade      = "#888888",
    light_shade     = "#f8f8f8",
)


def _build_palette(tokens: dict) -> QPalette:
    p = QPalette()

    def c(key: str) -> QColor:
        return QColor(tokens[key])

    # Normal state
    p.setColor(QPalette.ColorRole.Window,          c("window"))
    p.setColor(QPalette.ColorRole.WindowText,       c("window_text"))
    p.setColor(QPalette.ColorRole.Base,             c("base"))
    p.setColor(QPalette.ColorRole.AlternateBase,    c("alt_base"))
    p.setColor(QPalette.ColorRole.Text,             c("text"))
    p.setColor(QPalette.ColorRole.Button,           c("button"))
    p.setColor(QPalette.ColorRole.ButtonText,       c("button_text"))
    p.setColor(QPalette.ColorRole.Highlight,        c("highlight"))
    p.setColor(QPalette.ColorRole.HighlightedText,  c("highlight_text"))
    p.setColor(QPalette.ColorRole.BrightText,       c("bright_text"))
    p.setColor(QPalette.ColorRole.ToolTipBase,      c("tooltip_base"))
    p.setColor(QPalette.ColorRole.ToolTipText,      c("tooltip_text"))
    p.setColor(QPalette.ColorRole.Mid,              c("mid"))
    p.setColor(QPalette.ColorRole.Dark,             c("dark_shade"))
    p.setColor(QPalette.ColorRole.Light,            c("light_shade"))
    p.setColor(QPalette.ColorRole.Midlight,         c("light_shade"))

    # Disabled state — slightly muted
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, c("disabled_text"))

    return p


# ── Global stylesheet ──────────────────────────────────────────────────────────
# These rules fill in any gap left by widget-specific stylesheets.

def _global_stylesheet(mode: str) -> str:
    t = _DARK if mode == "dark" else _LIGHT

    border   = "#444" if mode == "dark" else "#bbb"
    input_bg = "#2a2a2a" if mode == "dark" else "#ffffff"
    input_fg = "#e0e0e0" if mode == "dark" else "#111111"
    grp_fg   = "#cccccc" if mode == "dark" else "#222222"
    # Subtle outline around dropdown/menu popups (frame suppression handles the
    # native black frame; this is just a thin defined edge).
    popup_border = "#666666" if mode == "dark" else "#999999"
    # Highlight for the item under the cursor as it moves up/down — bright white
    # on dark so the hovered/selected row is obvious; blue on light.
    popup_sel_bg = "#ffffff" if mode == "dark" else "#1565c0"
    popup_sel_fg = "#000000" if mode == "dark" else "#ffffff"
    scroll   = "#444"    if mode == "dark" else "#bbb"
    tab_unsel_bg = "#2d2d2d" if mode == "dark" else "#e0e0e0"
    tab_unsel_fg = "#aaa"    if mode == "dark" else "#555"

    return f"""
        /* ── Base ────────────────────────────────────────────────────────── */
        QWidget {{
            background-color: {t['window']};
            color: {t['window_text']};
            font-size: 12px;
        }}

        /* ── Input widgets ───────────────────────────────────────────────── */
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
            background: {input_bg};
            color: {input_fg};
            border: 1px solid {border};
            border-radius: 4px;
            padding: 3px 6px;
            selection-background-color: #1565c0;
            selection-color: white;
            outline: none;
        }}
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 1px solid #1565c0;
            outline: none;
        }}
        QComboBox {{
            background: {input_bg};
            color: {input_fg};
            border: 1px solid {border};
            border-radius: 4px;
            padding: 3px 8px;
            outline: none;
        }}
        QComboBox:focus {{
            border: 1px solid #1565c0;
            outline: none;
        }}
        QComboBox::drop-down {{
            border-left: 1px solid {border};
            border-top-right-radius: 4px;
            border-bottom-right-radius: 4px;
        }}
        /* ── Dropdown popup ──────────────────────────────────────────────── */
        /* QFrame is the native macOS wrapper around the popup — suppress its
           system-drawn black border so only our blue border shows. */
        QComboBox QFrame {{
            border: none;
        }}
        QComboBox QAbstractScrollArea {{
            border: none;
        }}
        QComboBox QAbstractItemView {{
            background: {input_bg};
            color: {input_fg};
            border: 1px solid {popup_border};
            border-radius: 4px;
            outline: none;
            selection-background-color: {popup_sel_bg};
            selection-color: {popup_sel_fg};
            show-decoration-selected: 1;
        }}
        QComboBox QAbstractItemView::item {{
            padding: 4px 10px;
            border: none;
            outline: none;
            min-height: 22px;
        }}
        QComboBox QAbstractItemView::item:selected,
        QComboBox QAbstractItemView::item:focus {{
            background: {popup_sel_bg};
            color: {popup_sel_fg};
            border: none;
            outline: none;
        }}
        QComboBox QAbstractItemView::item:hover:!selected {{
            background: {popup_sel_bg};
            color: {popup_sel_fg};
            border: none;
            outline: none;
        }}

        /* ── Buttons ─────────────────────────────────────────────────────── */
        QPushButton {{
            background: {t['button']};
            color: {t['button_text']};
            border: 1px solid {border};
            border-radius: 5px;
            padding: 4px 12px;
        }}
        QPushButton:hover  {{ background: {t['highlight']}; color: white; border-color: {t['highlight']}; }}
        QPushButton:pressed {{ background: #0d47a1; color: white; }}
        QPushButton:disabled {{ color: {t['disabled_text']}; background: {t['alt_base']}; }}

        /* ── GroupBox — bigger title, placed INSIDE the box ──────────────── */
        QGroupBox {{
            border: 1px solid {border};
            border-radius: 6px;
            margin-top: 6px;
            padding-top: 22px;
            color: {grp_fg};
            font-weight: bold;
        }}
        QGroupBox::title {{
            subcontrol-origin: padding;
            subcontrol-position: top left;
            left: 10px;
            top: 4px;
            padding: 0 4px;
            color: {grp_fg};
            font-size: 14px;
            font-weight: bold;
        }}

        /* ── Tab bar ─────────────────────────────────────────────────────── */
        QTabWidget::pane {{ border: 1px solid {border}; }}
        QTabBar::tab {{
            background: {tab_unsel_bg};
            color: {tab_unsel_fg};
            padding: 5px 14px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            margin-right: 2px;
        }}
        QTabBar::tab:selected {{ background: #1565c0; color: white; font-weight: bold; }}
        QTabBar::tab:hover:!selected {{ background: {t['highlight']}; color: white; }}

        /* ── Lists / Trees ───────────────────────────────────────────────── */
        QListWidget, QTreeWidget, QTableWidget {{
            background: {input_bg};
            color: {input_fg};
            border: 1px solid {border};
            alternate-background-color: {t['alt_base']};
            outline: none;
        }}
        QListWidget::item, QTreeWidget::item, QTableWidget::item {{
            border: none;
            outline: none;
        }}
        QListWidget::item:selected, QTreeWidget::item:selected,
        QTableWidget::item:selected {{
            background: #1565c0;
            color: white;
            border: none;
            outline: none;
        }}
        QListWidget::item:hover:!selected, QTreeWidget::item:hover:!selected {{
            background: #1976d2;
            color: white;
            border: none;
            outline: none;
        }}
        QListWidget::item:focus, QTreeWidget::item:focus,
        QTableWidget::item:focus {{
            border: none;
            outline: none;
        }}
        QListWidget:focus, QTreeWidget:focus {{
            border: 1px solid #1565c0;
            outline: none;
        }}
        QAbstractItemView::item:focus {{
            border: none;
            outline: none;
        }}

        /* ── Scrollbars ──────────────────────────────────────────────────── */
        QScrollBar:vertical {{
            background: {t['alt_base']};
            width: 10px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical {{
            background: {scroll};
            border-radius: 5px;
            min-height: 20px;
        }}
        QScrollBar:horizontal {{
            background: {t['alt_base']};
            height: 10px;
            border-radius: 5px;
        }}
        QScrollBar::handle:horizontal {{
            background: {scroll};
            border-radius: 5px;
            min-width: 20px;
        }}
        QScrollBar::add-line, QScrollBar::sub-line {{ background: none; }}

        /* ── Progress bar ────────────────────────────────────────────────── */
        QProgressBar {{
            border: 1px solid {border};
            border-radius: 4px;
            background: {t['alt_base']};
            text-align: center;
            color: {t['window_text']};
        }}
        QProgressBar::chunk {{ background: #1565c0; border-radius: 3px; }}

        /* ── Tooltips ────────────────────────────────────────────────────── */
        QToolTip {{
            background: {t['tooltip_base']};
            color: {t['tooltip_text']};
            border: 1px solid {border};
            border-radius: 4px;
            padding: 4px 8px;
        }}

        /* ── Splitter ────────────────────────────────────────────────────── */
        QSplitter::handle {{ background: {border}; }}

        /* ── Menu bar ────────────────────────────────────────────────────── */
        QMenuBar {{
            background: {t['window']};
            color: {t['window_text']};
        }}
        QMenuBar::item:selected {{ background: #1565c0; color: white; }}
        QMenu {{
            background: {input_bg};
            color: {input_fg};
            border: 1px solid {border};
        }}
        QMenu::item:selected {{ background: #1565c0; color: white; }}

        /* ── Status bar ──────────────────────────────────────────────────── */
        QStatusBar {{ background: {t['alt_base']}; color: {t['window_text']}; }}
    """


# ── Public entry point ─────────────────────────────────────────────────────────

def apply_theme(app: QApplication, mode: str = "auto") -> str:
    """
    Apply dark or light theme to the application.

    Parameters
    ----------
    app  : QApplication instance
    mode : "auto" (detect from system), "dark", or "light"

    Returns
    -------
    The mode that was applied: "dark" or "light".
    """
    global _current_mode

    if mode == "auto":
        mode = detect_system_mode()

    _current_mode = mode
    tokens = _DARK if mode == "dark" else _LIGHT

    # Switch to Fusion style before applying any palette / stylesheet.
    # Fusion is Qt's built-in cross-platform renderer — it does NOT call
    # macOS CoreGraphics, so the native focus ring (the black border that
    # appears around hovered combo-box items and focused widgets on macOS)
    # is never drawn.  The Qt stylesheet then has complete control over
    # every pixel, which is exactly what the custom dark theme needs.
    fusion = QStyleFactory.create("Fusion")
    if fusion:
        app.setStyle(fusion)

    app.setPalette(_build_palette(tokens))
    app.setStyleSheet(_global_stylesheet(mode))

    return mode
