"""
equations_tab.py
Equations & References tab — clean card grid layout.

Each section is a plain clickable box showing only the equation name in large
text.  Clicking opens a resizable dialog with:
  • The equation(s) displayed in a selectable monospace text box
  • Copy as Plain Text / Copy as LaTeX buttons
  • Variable legend / description  (selectable)
  • Literature references           (selectable)

Sections
--------
  Bloch–McConnell Equations
  CEST-MRF
  Pseudo-Voigt Peak Model
  Lorentzian Peak Model
  Super-Lorentzian MT Lineshape
  QUESP Fitting
  T1 / T2 / B1 Mapping
  PLOF Fit  (Polynomial and Lorentzian O-Field)
  DROF Fit  (Double-step R1ρ Fitting)
  Inverse Z-Spectroscopy — 1/Z Fitting
  AREX
  Gaussian Peak Fit
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QScrollArea, QLabel, QFrame, QSizePolicy,
    QPushButton, QDialog, QApplication, QTextEdit,
)
from PyQt6.QtCore import Qt, QObject, QEvent
from PyQt6.QtGui import QCursor, QFont, QColor


def _mono_font(size: int) -> QFont:
    """Cross-platform monospace font (avoids Qt alias-lookup warning for 'Courier')."""
    from PyQt6.QtGui import QFontDatabase
    families = set(QFontDatabase.families())
    for candidate in ("Menlo", "Consolas", "DejaVu Sans Mono",
                      "Liberation Mono", "Courier New"):
        if candidate in families:
            return QFont(candidate, size)
    f = QFont()
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPointSize(size)
    return f


# ── Colour palette ─────────────────────────────────────────────────────────────
_BG       = "#1e1e2e"
_CARD_BG  = "#252538"
_CARD_HVR = "#2e2e4a"
_CARD_BDR = "#44446a"
_TEXT     = "#cdd6f4"
_REF_CLR  = "#a6e3a1"
_SEP_CLR  = "#585b70"
_HDR_CLR  = "#cba6f7"
_EQ_BG    = "#ffffff"    # equation-box background (white)
_EQ_TEXT  = "#000000"    # equation text (black)

# ── Universal equation font (applies to EVERY box) ──────────────────────────
# Change these three to restyle all equation boxes at once.
_EQ_FONT_FAMILY = "Arial"
_EQ_FONT_PT     = 15      # base equation font size (points)
_EQ_BOLD        = True

# On-screen display scale for the pre-baked LaTeX equation images (all baked at
# 22 pt / 220 dpi). A single fixed value here keeps EVERY box the same size
# (WASABI, Bloch, …). Raise it to make all typeset equations bigger.
_EQ_IMG_SCALE = 0.50
_EQ_IMG_MAXW  = 1180      # hard cap so an ultra-wide matrix still fits the panel


def _eq_font(size: int) -> QFont:
    """Font for the equation boxes — Arial (bold), with graceful fallback."""
    from PyQt6.QtGui import QFontDatabase
    families = set(QFontDatabase.families())
    fam = _EQ_FONT_FAMILY
    if fam not in families:
        for alt in ("Helvetica", "Helvetica Neue", "Liberation Sans",
                    "DejaVu Sans", "Verdana"):
            if alt in families:
                fam = alt
                break
    f = QFont(fam, size)
    f.setBold(_EQ_BOLD)
    return f
_DLG_CARD = "#252535"
_MUTED    = "#7777aa"
_BTN_COPY = "#2a2a4a"


def _apply_eq_palette():
    """Match the equation cards / dialog chrome to the current app theme:
    dark chrome on a dark OS, light chrome on a light OS.  The equation,
    description and reference panels always stay white (the baked LaTeX images
    have white backgrounds), so only the surrounding chrome changes."""
    global _BG, _CARD_BG, _CARD_HVR, _CARD_BDR, _TEXT, _REF_CLR, _SEP_CLR
    global _HDR_CLR, _DLG_CARD, _MUTED, _BTN_COPY
    try:
        from my_gui.theme import is_dark
        dark = is_dark()
    except Exception:
        dark = True
    if dark:
        _BG="#1e1e2e"; _CARD_BG="#252538"; _CARD_HVR="#2e2e4a"; _CARD_BDR="#44446a"
        _TEXT="#cdd6f4"; _REF_CLR="#a6e3a1"; _SEP_CLR="#585b70"; _HDR_CLR="#cba6f7"
        _DLG_CARD="#252535"; _MUTED="#7777aa"; _BTN_COPY="#2a2a4a"
    else:
        _BG="#e9e9f2"; _CARD_BG="#ffffff"; _CARD_HVR="#eef0f8"; _CARD_BDR="#c2c2d4"
        _TEXT="#1c1c30"; _REF_CLR="#2e7d32"; _SEP_CLR="#b0b0c4"; _HDR_CLR="#6a3fb0"
        _DLG_CARD="#dedeea"; _MUTED="#5a5a78"; _BTN_COPY="#e2e2ee"


def _accent_for_theme(accent: str) -> str:
    """On a light theme the pastel accents are too pale for text on light
    chrome — darken them so titles/headers stay legible."""
    try:
        from my_gui.theme import is_dark
        if is_dark():
            return accent
    except Exception:
        return accent
    return QColor(accent).darker(175).name()


# ── Wheel-event forwarder ──────────────────────────────────────────────────────

class _WheelForwarder(QObject):
    """
    Event filter installed on a content widget so that mouse-wheel events
    over any child directly adjust the parent QScrollArea's vertical scrollbar.
    Using direct scrollbar manipulation is more reliable than re-sending the
    event, which can be silently dropped if the event has already been accepted.
    """
    def __init__(self, scroll_area):
        super().__init__(scroll_area)
        self._sa = scroll_area

    def eventFilter(self, obj, event):          # noqa: N802
        if event.type() == QEvent.Type.Wheel:
            delta = event.angleDelta().y()
            sb = self._sa.verticalScrollBar()
            sb.setValue(sb.value() - delta // 3)
            return True
        return False


# ── Equation text widget ───────────────────────────────────────────────────────

def _make_eq_widget(lines: list[str], font_pt: int = _EQ_FONT_PT) -> QTextEdit:
    """
    Selectable, scrollable text box for equation display (Arial, bold, white bg).

    Style is universal — set by the _EQ_* constants above; every box uses it.
      • fully visible — horizontal scrollbar, no clipping
      • selectable and copyable with mouse / keyboard
      • always sized exactly to their content (no wasted space)

    ``font_pt`` defaults to _EQ_FONT_PT; a box may override via extra["eq_font_pt"].
    """
    te = QTextEdit()
    te.setReadOnly(True)
    te.setFont(_eq_font(font_pt))
    te.setPlainText("\n".join(lines))
    te.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
    te.setStyleSheet(
        f"QTextEdit {{"
        f"  background: {_EQ_BG};"
        f"  color: {_EQ_TEXT};"
        f"  border: 1px solid {_SEP_CLR};"
        f"  border-radius: 8px;"
        f"  padding: 10px 14px;"
        f"  selection-background-color: #3a4a8a;"
        f"  selection-color: #ffffff;"
        f"}}"
    )
    # Size to show every line without a vertical scrollbar (≈1.7 px/pt, bold)
    line_h = round(font_pt * 1.7)
    h = len(lines) * line_h + 32
    te.setMinimumHeight(h)
    te.setMaximumHeight(h)
    te.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return te


def _render_mathtext(tex: str, fontsize: int = 30, dpi: int = 200):
    """Render one LaTeX/mathtext equation to a QPixmap (black text, transparent
    background) using matplotlib's built-in mathtext — no external LaTeX needed."""
    import io
    from PyQt6.QtGui import QPixmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=(0.1, 0.1))
        FigureCanvasAgg(fig)
        fig.text(0.5, 0.5, f"${tex}$", fontsize=fontsize, color="black",
                 ha="center", va="center")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, transparent=True,
                    bbox_inches="tight", pad_inches=0.06)
        buf.seek(0)
        pm = QPixmap()
        pm.loadFromData(buf.getvalue(), "PNG")
        return pm
    except Exception:
        return QPixmap()


# ── Detail dialog ──────────────────────────────────────────────────────────────

class _EqDialog(QDialog):
    """
    Scrollable dialog showing equation, description and references.
    Non-modal so the user can keep it open while working.
    All text is selectable; equations can be copied as plain text or LaTeX.
    """

    def __init__(self, title: str,
                 eq_lines: list[str],
                 latex_eq: str,
                 description: str,
                 references: list[str],
                 accent: str,
                 parent: QWidget | None = None,
                 extra: dict | None = None):
        super().__init__(parent)
        _apply_eq_palette()                    # match chrome to the OS theme
        _acc = _accent_for_theme(accent)       # legible accent on light/dark
        extra = extra or {}
        self.setWindowTitle(f"  {title}")
        # Wide enough for the full-width typeset equation images to fit
        self.setMinimumSize(1080, 620)
        self.resize(1340, 780)
        self.setModal(False)
        self.setStyleSheet(f"background: {_BG};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Title bar ─────────────────────────────────────────────────────────
        title_bar = QLabel(f"  {title}")
        title_bar.setStyleSheet(
            f"color: {_acc}; font-size: 26px; font-weight: bold;"
            f"background: {_DLG_CARD}; padding: 22px 28px;"
            f"border-bottom: 2px solid {_SEP_CLR};"
        )
        root.addWidget(title_bar)

        # ── Scroll area ───────────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {_BG}; }}"
            f"QScrollBar:vertical {{ background: {_DLG_CARD}; width: 12px; }}"
            f"QScrollBar::handle:vertical {{"
            f"  background: {_SEP_CLR}; border-radius: 6px; min-height: 30px; }}"
        )
        self._main_scroll = scroll

        content = QWidget()
        content.setStyleSheet(f"background: {_BG};")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(36, 30, 36, 36)
        cl.setSpacing(26)

        # ── Equation section ──────────────────────────────────────────────────
        if eq_lines or extra.get("math"):
            eq_hdr = QLabel("Equation")
            eq_hdr.setStyleSheet(
                f"color: {_acc}; font-size: 21px; font-weight: bold;"
                f"background: transparent; border: none; padding-bottom: 8px;"
            )
            cl.addWidget(eq_hdr)

            math_items = extra.get("math")
            if math_items:
                # Publication-quality typeset equations on a white panel. Each
                # item is one of:
                #   (tex, "[n]")            → rendered via mathtext
                #   {"img": path,"num":..}  → a pre-baked LaTeX / PDF equation image
                #   {"text": html}          → prose paragraph (inline sub/sup ok)
                from my_gui.paths import get_resource_dir
                from PyQt6.QtGui import QPixmap
                mpanel = QWidget()
                mpanel.setStyleSheet(
                    f"background: #ffffff; border: 1px solid {_SEP_CLR};"
                    f"border-radius: 8px;")
                mpl = QVBoxLayout(mpanel)
                mpl.setContentsMargins(26, 22, 26, 22); mpl.setSpacing(18)
                _MAXW = _EQ_IMG_MAXW
                # Build the row list first (so equation images can be scaled by a
                # single common factor → consistent on-screen font size).
                _rows = []   # ("eq", pixmap, num)  |  ("text", html)
                for _item in math_items:
                    if isinstance(_item, dict):
                        if _item.get("img"):
                            _pm = QPixmap(str(get_resource_dir() / _item["img"]))
                            if not _pm.isNull():
                                _kind = "fig" if _item.get("figure") else "eq"
                                _rows.append((_kind, _pm, _item.get("num", "")))
                        elif _item.get("text") is not None:
                            _rows.append(("text", _item["text"]))
                    else:
                        _tex, _num = _item if isinstance(_item, (tuple, list)) else (_item, "")
                        _pm = _render_mathtext(_tex)
                        if not _pm.isNull():
                            _rows.append(("eq", _pm, _num))
                for _r in _rows:
                    if _r[0] == "eq":
                        _pm, _num = _r[1], _r[2]
                        # Fixed display scale so every box matches; only an
                        # ultra-wide matrix is shrunk further to fit the panel.
                        _disp = min(_EQ_IMG_SCALE, _MAXW / max(1, _pm.width()))
                        _pm = _pm.scaledToWidth(
                            max(1, int(_pm.width() * _disp)),
                            Qt.TransformationMode.SmoothTransformation)
                        _row = QHBoxLayout(); _row.setSpacing(12)
                        _im = QLabel(); _im.setPixmap(_pm)
                        _im.setStyleSheet("background: transparent; border: none;")
                        _row.addWidget(_im); _row.addStretch()
                        if _num:
                            _nl = QLabel(_num)
                            _nl.setStyleSheet(
                                "color: #000000; font-size: 17px; font-weight: bold;"
                                "background: transparent; border: none;")
                            _row.addWidget(_nl, alignment=Qt.AlignmentFlag.AlignVCenter)
                        mpl.addLayout(_row)
                    elif _r[0] == "fig":
                        _pm = _r[1]
                        if _pm.width() > _MAXW:
                            _pm = _pm.scaledToWidth(
                                _MAXW, Qt.TransformationMode.SmoothTransformation)
                        _frow = QHBoxLayout()
                        _fim = QLabel(); _fim.setPixmap(_pm)
                        _fim.setStyleSheet("background: transparent; border: none;")
                        _frow.addStretch(); _frow.addWidget(_fim); _frow.addStretch()
                        mpl.addLayout(_frow)
                    else:
                        _tl = QLabel(_r[1]); _tl.setWordWrap(True)
                        _tl.setTextFormat(Qt.TextFormat.RichText)
                        _tl.setTextInteractionFlags(
                            Qt.TextInteractionFlag.TextSelectableByMouse)
                        _tl.setStyleSheet(
                            "color: #000000; font-size: 16px; background: transparent;"
                            "border: none;")
                        mpl.addWidget(_tl)
                cl.addWidget(mpanel)
            else:
                eq_widget = _make_eq_widget(
                    eq_lines, font_pt=int(extra.get("eq_font_pt", _EQ_FONT_PT)))
                cl.addWidget(eq_widget)

            # Copy buttons — gather the ACTUAL equations.  For sections rendered
            # from ``math_items`` the equations live there (as (tex, num) tuples or
            # {"text": html} prose), NOT in eq_lines/latex_eq — which are empty for
            # those sections — so pull straight from math_items; otherwise fall
            # back to the plain eq_lines / latex_eq.
            if math_items:
                import re as _re
                _txt_parts, _tex_parts = [], []
                for _it in math_items:
                    if isinstance(_it, dict):
                        # Copyable LaTeX for a pre-baked equation image: inline
                        # "tex", else the central _EQ_TEX lookup keyed by img path.
                        _img_tex = _it.get("tex") or _EQ_TEX.get(_it.get("img", ""))
                        if _img_tex:
                            _line = str(_img_tex) + (
                                f"    {_it['num']}" if _it.get("num") else "")
                            _txt_parts.append(_line)
                            _tex_parts.append(_line)
                        elif _it.get("text") is not None:
                            _plain = _re.sub("<[^>]+>", "", str(_it["text"])).strip()
                            if _plain:
                                _txt_parts.append(_plain)
                        # a bare {"img": …} (no "tex") has no copyable source text
                    else:
                        _t, _n = _it if isinstance(_it, (tuple, list)) else (_it, "")
                        _line = str(_t) + (f"    {_n}" if _n else "")
                        _txt_parts.append(_line)
                        _tex_parts.append(_line)
                _eq_text = "\n".join(_txt_parts) if _txt_parts else "\n".join(eq_lines)
                _latex   = "\n".join(_tex_parts) if _tex_parts else (latex_eq or _eq_text)
            else:
                _eq_text = "\n".join(eq_lines)
                _latex   = latex_eq

            copy_row = QHBoxLayout()
            copy_row.setSpacing(10)

            _btn_ss = """
                QPushButton {{
                    background: {bg}; color: {fg};
                    border: 1px solid {bdr};
                    border-radius: 5px; padding: 6px 16px; font-size: 13px;
                }}
                QPushButton:hover {{ background: #3a3a6a; border-color: #7aa2f7; }}
            """
            btn_style = _btn_ss.format(bg=_BTN_COPY, fg=_TEXT, bdr=_SEP_CLR)

            btn_copy_txt = QPushButton("Copy Equations")
            btn_copy_tex = QPushButton("Copy Equations to Latex")
            btn_copy_txt.setStyleSheet(btn_style)
            btn_copy_tex.setStyleSheet(btn_style)

            from PyQt6.QtCore import QTimer as _QTimerCopy
            def _copy_clip(_text, _btn, _label):
                QApplication.clipboard().setText(_text or "")
                _btn.setText("Copied ✓")
                _QTimerCopy.singleShot(1200, lambda: _btn.setText(_label))

            btn_copy_txt.clicked.connect(
                lambda _=False, t=_eq_text, b=btn_copy_txt:
                    _copy_clip(t, b, "Copy Equations"))
            btn_copy_tex.clicked.connect(
                lambda _=False, t=_latex, b=btn_copy_tex:
                    _copy_clip(t, b, "Copy Equations to Latex"))

            copy_row.addWidget(btn_copy_txt)
            copy_row.addWidget(btn_copy_tex)
            copy_row.addStretch()
            cl.addLayout(copy_row)

        # ── Description section ───────────────────────────────────────────────
        if description.strip():
            sep1 = QFrame()
            sep1.setFrameShape(QFrame.Shape.HLine)
            sep1.setStyleSheet(f"color: {_SEP_CLR}; margin: 4px 0;")
            cl.addWidget(sep1)

            desc_hdr = QLabel("Description  &  Variables")
            desc_hdr.setStyleSheet(
                f"color: {_acc}; font-size: 21px; font-weight: bold;"
                f"background: transparent; border: none; padding-bottom: 8px;"
            )
            cl.addWidget(desc_hdr)

            desc_panel = QWidget()
            desc_panel.setStyleSheet(
                f"background: #ffffff; border: 1px solid {_SEP_CLR};"
                f"border-radius: 8px;")
            _dpl = QVBoxLayout(desc_panel)
            _dpl.setContentsMargins(22, 16, 22, 16)
            desc_lbl = QLabel(description)
            desc_lbl.setWordWrap(True)
            desc_lbl.setTextFormat(Qt.TextFormat.RichText)
            desc_lbl.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse |
                Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
            desc_lbl.setStyleSheet(
                "color: #000000; font-size: 17px; font-family: Arial;"
                "background: transparent; border: none; line-height: 190%;"
            )
            _dpl.addWidget(desc_lbl)
            cl.addWidget(desc_panel)

        # ── Figure section (optional) ─────────────────────────────────────────
        if extra.get("figure"):
            from my_gui.paths import get_resource_dir
            from PyQt6.QtGui import QPixmap
            fig_path = str(get_resource_dir() / extra["figure"])
            pm = QPixmap(fig_path)
            if not pm.isNull():
                sep_f = QFrame(); sep_f.setFrameShape(QFrame.Shape.HLine)
                sep_f.setStyleSheet(f"color: {_SEP_CLR}; margin: 4px 0;")
                cl.addWidget(sep_f)
                fig_hdr = QLabel("Figure")
                fig_hdr.setStyleSheet(
                    f"color: {_acc}; font-size: 21px; font-weight: bold;"
                    f"background: transparent; border: none; padding-bottom: 8px;")
                cl.addWidget(fig_hdr)
                # Caption may sit above or below the image (caption_above flag).
                _cap_above = bool(extra.get("caption_above"))
                def _add_fig_caption():
                    if extra.get("figure_caption"):
                        cap = QLabel(extra["figure_caption"]); cap.setWordWrap(True)
                        cap.setStyleSheet(
                            f"color: {_MUTED}; font-size: 13px; font-style: italic;"
                            "background: transparent; border: none;")
                        cl.addWidget(cap)
                if _cap_above:
                    _add_fig_caption()
                fig_lbl = QLabel()
                fig_lbl.setPixmap(pm.scaledToWidth(
                    min(pm.width(), 460), Qt.TransformationMode.SmoothTransformation))
                fig_lbl.setStyleSheet(
                    "background: white; border-radius: 8px; padding: 12px;")
                cl.addWidget(fig_lbl, alignment=Qt.AlignmentFlag.AlignLeft)
                if not _cap_above:
                    _add_fig_caption()

        # ── References section ────────────────────────────────────────────────
        if references:
            sep2 = QFrame()
            sep2.setFrameShape(QFrame.Shape.HLine)
            sep2.setStyleSheet(f"color: {_SEP_CLR}; margin: 4px 0;")
            cl.addWidget(sep2)

            ref_hdr = QLabel("References")
            ref_hdr.setStyleSheet(
                f"color: {_REF_CLR}; font-size: 21px; font-weight: bold;"
                f"background: transparent; border: none; padding-bottom: 8px;"
            )
            cl.addWidget(ref_hdr)

            ref_box = QWidget()
            ref_box.setStyleSheet(
                f"background: #ffffff; border: 1px solid {_SEP_CLR};"
                f"border-radius: 8px;"
            )
            rb = QVBoxLayout(ref_box)
            rb.setContentsMargins(22, 14, 22, 14)
            rb.setSpacing(14)
            # Per-reference rendering: plain citation (no ↗ arrow) followed by
            # [Open PDF] / [Open DOI Link] action buttons.  A reference entry may be
            #   "citation"                      → text only
            #   ("citation", url)               → + [Open DOI Link]
            #   ("citation", url, pdf_relpath)  → + [Open PDF] (if the file exists)
            import os as _os
            from PyQt6.QtGui import QDesktopServices
            from PyQt6.QtCore import QUrl
            from my_gui.paths import get_resource_dir
            _refbtn = (
                "QPushButton { background: %s; color: %s; border: 1px solid %s;"
                " border-radius: 5px; padding: 5px 14px; font-size: 12px;"
                " font-family: Arial; }"
                "QPushButton:hover { background: #3a3a6a; border-color: #7aa2f7; }"
                % (_BTN_COPY, _TEXT, _SEP_CLR))
            for i, ref_item in enumerate(references, 1):
                if isinstance(ref_item, (tuple, list)):
                    ref_text = ref_item[0] if len(ref_item) > 0 else ""
                    ref_url  = ref_item[1] if len(ref_item) > 1 else ""
                    ref_pdf  = ref_item[2] if len(ref_item) > 2 else ""
                else:
                    ref_text, ref_url, ref_pdf = ref_item, "", ""

                # Citation text — plain & selectable, no external-link arrow.
                rl = QLabel(f"[{i}]  {ref_text}")
                rl.setTextFormat(Qt.TextFormat.PlainText)
                rl.setWordWrap(True)
                rl.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse |
                    Qt.TextInteractionFlag.TextSelectableByKeyboard
                )
                rl.setStyleSheet(
                    "color: #000000; font-size: 15px; font-style: italic;"
                    "font-family: Arial; background: transparent; border: none;"
                )

                entry = QVBoxLayout(); entry.setSpacing(6)
                entry.addWidget(rl)

                _file_path = ""
                if ref_pdf:
                    _p = get_resource_dir() / ref_pdf
                    if _os.path.exists(str(_p)):
                        _file_path = str(_p)
                if _file_path or ref_url:
                    hb = QHBoxLayout(); hb.setContentsMargins(20, 0, 0, 0); hb.setSpacing(8)
                    if _file_path:
                        # Label the button by file type (PDF / PPT / generic file).
                        _ext = _os.path.splitext(_file_path)[1].lower()
                        _flabel = ("Open PDF" if _ext == ".pdf"
                                   else "Open PPT" if _ext in (".ppt", ".pptx")
                                   else "Open File")
                        b_pdf = QPushButton(_flabel); b_pdf.setStyleSheet(_refbtn)
                        b_pdf.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                        b_pdf.clicked.connect(
                            lambda _=0, p=_file_path: QDesktopServices.openUrl(QUrl.fromLocalFile(p)))
                        hb.addWidget(b_pdf)
                    if ref_url:
                        b_doi = QPushButton("Open DOI Link"); b_doi.setStyleSheet(_refbtn)
                        b_doi.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                        b_doi.clicked.connect(
                            lambda _=0, u=ref_url: QDesktopServices.openUrl(QUrl(u)))
                        hb.addWidget(b_doi)
                    hb.addStretch()
                    entry.addLayout(hb)
                rb.addLayout(entry)
            cl.addWidget(ref_box)

        cl.addStretch()
        scroll.setWidget(content)

        # Install wheel-event forwarder on ALL content children so scrolling
        # works when the mouse is anywhere inside the dialog.
        _wf = _WheelForwarder(scroll)
        content.installEventFilter(_wf)
        for child in content.findChildren(QWidget):
            child.installEventFilter(_wf)

        root.addWidget(scroll, stretch=1)

        # ── Close button ──────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(18, 10, 18, 16)
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(120)
        close_btn.setStyleSheet(
            f"QPushButton {{ background: {_SEP_CLR}; color: white; border: none;"
            "border-radius: 6px; padding: 9px 0; font-size: 14px; }"
            "QPushButton:hover { background: #7aa2f7; color: white; }"
        )
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def wheelEvent(self, event):           # noqa: N802
        """Catch-all: any wheel event reaching the dialog scrolls the area."""
        sb = self._main_scroll.verticalScrollBar()
        sb.setValue(sb.value() - event.angleDelta().y() // 3)
        event.accept()


# ── Clickable card ─────────────────────────────────────────────────────────────

class _EqCard(QFrame):
    """
    A plain, hoverable box that shows only the section title in large text.
    Clicking opens _EqDialog.
    """

    def __init__(self,
                 title: str,
                 eq_lines: list[str],
                 latex_eq: str,
                 description: str,
                 references: list[str],
                 accent: str = "#7aa2f7",
                 parent: QWidget | None = None,
                 extra: dict | None = None):
        super().__init__(parent)

        self._title  = title
        self._eq     = eq_lines
        self._latex  = latex_eq
        self._desc   = description
        self._refs   = references
        self._accent = accent
        self._extra  = extra
        self._dialog: _EqDialog | None = None

        # Object-name selector keeps the stylesheet from leaking into children.
        self.setObjectName("EqCard")
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setMinimumWidth(200)
        self.setMinimumHeight(90)
        self.setMaximumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._apply_style(hover=False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(0)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_lbl = QLabel(title)
        f = QFont()
        f.setPointSize(16)
        f.setBold(True)
        title_lbl.setFont(f)
        title_lbl.setWordWrap(True)
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_lbl.setStyleSheet(
            f"color: {_accent_for_theme(accent)}; background: transparent;"
            f" border: none; padding: 0;"
        )
        lay.addWidget(title_lbl)

    # ── Styling ───────────────────────────────────────────────────────────────

    def _apply_style(self, hover: bool):
        bg  = _CARD_HVR if hover else _CARD_BG
        bdr = self._accent if hover else _CARD_BDR
        w   = 3 if hover else 2
        self.setStyleSheet(
            f"QFrame#EqCard {{"
            f"  background: {bg};"
            f"  border: {w}px solid {bdr};"
            f"  border-radius: 12px;"
            f"}}"
        )

    def enterEvent(self, event):
        self._apply_style(hover=True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._apply_style(hover=False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._open_dialog()
        super().mousePressEvent(event)

    def _open_dialog(self):
        if self._dialog is None or not self._dialog.isVisible():
            self._dialog = _EqDialog(
                self._title, self._eq, self._latex,
                self._desc, self._refs,
                self._accent, parent=self, extra=self._extra,
            )
        self._dialog.show()
        self._dialog.raise_()
        self._dialog.activateWindow()


# ── Section data ───────────────────────────────────────────────────────────────
# Each entry: (title, accent_colour, eq_lines, latex_eq, description, references)

_EQ_TEX: dict[str, str] = {
    # Copyable LaTeX for the pre-baked equation images (keyed by "img" path)
    # so the "Copy Equations" buttons capture the image-based equations, not
    # just the surrounding prose.  Transcribed from the rendered reference PNGs.
    # -- cestmri --
    "references/cestmri/eq/eq4.png": r"""MTR_{asym}(\Delta\omega) = Z(-\Delta\omega) - Z(+\Delta\omega)""",
    "references/cestmri/eq/eq5.png": r"""\mathrm{CEST} = \frac{k_{sw}\, f_s\, \alpha}{R_{1w} + k_{sw} f_s} \left[ 1 - e^{-(R_{1w}+k_{sw}f_s)\, t_{sat}} \right]""",
    "references/cestmri/eq/eq6.png": r"""\alpha = \frac{\omega_1^2}{\omega_1^2 + k_{sw}^2}, \qquad \omega_1 = \gamma B_1""",
    # -- denoising --
    "references/denoising/eq/eq1.png": r"""D' = PCA_k(D)""",
    "references/denoising/eq/eq2.png": r"""S = M(B_r)""",
    "references/denoising/eq/eq3.png": r"""S' = T^{-1}(F\,(T(S)))""",
    "references/denoising/eq/eq4.png": r"""p' = NLM_W(p)""",
    # -- bloch --
    "references/bloch/eq/eq1.png": r"""\frac{\mathrm{d}}{\mathrm{d}t}\,\vec{M} = A \cdot \vec{M} + \vec{C}""",
    "references/bloch/eq/eq2.png": r"""\vec{M} = \begin{pmatrix} M_{xa} \\ M_{ya} \\ M_{za} \\ M_{xb} \\ M_{yb} \\ M_{zb} \end{pmatrix}""",
    "references/bloch/eq/eq3.png": r"""A = \begin{bmatrix} L_a - K_a & +K_b \\ +K_a & L_b - K_b \end{bmatrix}""",
    "references/bloch/eq/eq4.png": r"""L_i = \begin{pmatrix} -R_{2i} & -\Delta\omega_i & 0 \\ +\Delta\omega_i & -R_{2i} & +\omega_1 \\ 0 & -\omega_1 & -R_{1i} \end{pmatrix}""",
    "references/bloch/eq/eq5.png": r"""\vec{C} = \begin{pmatrix} 0 \\ 0 \\ R_{1a} M_{0a} \\ 0 \\ 0 \\ R_{1b} M_{0b} \end{pmatrix}""",
    "references/bloch/eq/eq6.png": r"""K_a = f_b K_b = f_b \begin{pmatrix} k_b & 0 & 0 \\ 0 & k_b & 0 \\ 0 & 0 & k_b \end{pmatrix}""",
    "references/bloch/eq/eq7.png": r"""f_b = \frac{M_{0b}}{M_{0a}} = \frac{n_b \cdot [b]}{n_a \cdot [a]}""",
    "references/bloch/eq/eq9.png": r"""\begin{bmatrix} \partial M_{wx}/\partial t \\ \partial M_{wy}/\partial t \\ \partial M_{wz}/\partial t \\ \cdot \\ \cdot \\ \cdot \\ \partial M_{s_{n-1}x}/\partial t \\ \partial M_{s_{n-1}y}/\partial t \\ \partial M_{s_{n-1}z}/\partial t \end{bmatrix} = \begin{bmatrix} -k_{ws_1} & \Delta\omega_w & 0 & \cdot & 0 & 0 & k_{s_{n-1}w} & 0 & 0 \\ \Delta\omega_w & -k_{ws_1} & -\omega_1 & 0 & \cdot & 0 & 0 & k_{s_{n-1}w} & 0 \\ 0 & -\omega_1 & -k_{ws_1} & 0 & 0 & \cdot & 0 & 0 & k_{s_{n-1}w} \\ \cdot & 0 & 0 & \cdot & \cdot & 0 & 0 & 0 & 0 \\ 0 & \cdot & 0 & \cdot & \cdot & -\omega_1 & 0 & 0 & 0 \\ 0 & 0 & \cdot & 0 & -\omega_1 & \cdot & 0 & 0 & 0 \\ k_{ws_{n-1}} & 0 & 0 & 0 & 0 & 0 & -k_{s_{n-1}w} & \Delta\omega_{s_{n-1}} & 0 \\ 0 & k_{ws_{n-1}} & 0 & 0 & 0 & 0 & \Delta\omega_{s_{n-1}} & -k_{s_{n-1}w} & -\omega_1 \\ 0 & 0 & k_{ws_{n-1}} & 0 & 0 & 0 & 0 & -\omega_1 & -k_{s_{n-1}w} \end{bmatrix} \begin{bmatrix} M_{wx} \\ M_{wy} \\ M_{wz} \\ \cdot \\ \cdot \\ \cdot \\ M_{s_{n-1}x} \\ M_{s_{n-1}y} \\ M_{s_{n-1}z} \end{bmatrix} - \begin{bmatrix} M_{wx}/T_{2w} \\ M_{wy}/T_{2w} \\ (M_{w0} - M_{wz})/T_{1w} \\ \cdot \\ \cdot \\ \cdot \\ M_{s_{n-1}x}/T_{2s_{n-1}} \\ M_{s_{n-1}y}/T_{2s_{n-1}} \\ (M_{s_{n-1}0} - M_{s_{n-1}z})/T_{1s_{n-1}} \end{bmatrix}""",
    # -- cestmrf --
    "references/cestmrf/eq/dp.png": r"""\mathrm{DP}(\mathbf{e}, \mathbf{d}) = \frac{\langle \mathbf{e}, \mathbf{d} \rangle}{\|\mathbf{e}\| \cdot \|\mathbf{d}\|}""",
    "references/cestmrf/eq/ed.png": r"""\mathrm{ED}(\mathbf{e}, \mathbf{d}) = \frac{1}{\sqrt{N_t}} \|\hat{\mathbf{Z}}_e - \hat{\mathbf{Z}}_d\|""",
    "references/cestmrf/eq/znorm.png": r"""\hat{\mathbf{Z}}_e = \frac{\mathbf{e}}{M_{0e}} \, , \quad \hat{\mathbf{Z}}_d = \frac{\mathbf{d}}{M_{0d}}""",
    "references/cestmrf/eq/fisher.png": r"""F_{ij}(\boldsymbol{\theta}) = \frac{1}{\sigma^2} \sum_{n=1}^{N_t} \frac{\partial s_n}{\partial \theta_i} \frac{\partial s_n}{\partial \theta_j} = \frac{1}{\sigma^2} \mathbf{J}^{\top} \mathbf{J}""",
    "references/cestmrf/eq/crlb.png": r"""\mathrm{Var}\left(\hat{\theta}_i\right) \geq \left[\mathbf{F}^{-1}(\boldsymbol{\theta})\right]_{ii} \equiv \mathrm{CRB}(\theta_i)""",
    "references/cestmrf/eq/ncrb.png": r"""\mathrm{nCRB}(\theta_i) = \frac{\sqrt{\left[\mathbf{F}^{-1}(\boldsymbol{\theta})\right]_{ii}}}{\theta_i}""",
    "references/cestmrf/eq/schedule.png": r"""\boldsymbol{\Lambda}^{*} = \arg\min_{\boldsymbol{\Lambda}} \frac{1}{P} \sum_{i=1}^{P} \frac{\left[\mathbf{F}^{-1}(\boldsymbol{\theta};\boldsymbol{\Lambda})\right]_{ii}}{\theta_i^2}""",
    # -- pseudovoigt --
    "references/pseudovoigt/eq/voigt.png": r"""V(\Delta\omega) \approx \alpha \times L(\Delta\omega) + (1 - \alpha)\, G(\Delta\omega)""",
    "references/pseudovoigt/eq/gauss.png": r"""G(\Delta\omega) = \frac{1}{\sqrt{2\pi}\sigma}\, e^{-\frac{(\omega_1 - \omega)^2}{2(\sigma)^2}}""",
    "references/pseudovoigt/eq/lorentz.png": r"""L(\Delta\omega) = \frac{A}{\pi \left[ 1 + \frac{\omega_1 - \omega}{\sigma} \right]^2}""",
    # -- lorentzian --
    "references/lorentzian/eq/multi.png": r"""L(a_k,\, \omega_k^c,\, \sigma_k) = 1 - \frac{I}{I_0} = \sum_{k=1}^{K} \frac{a_k}{1 + 4\left(\frac{\omega - \omega_k^c}{\sigma_k}\right)^2}""",
    "references/lorentzian/eq/single.png": r"""L_i(\Delta\omega) = \frac{A_i}{1 + \left[\frac{\Delta\omega - (\Delta\delta_i + \Delta\delta)}{\Gamma_i/2}\right]^2}""",
    "references/lorentzian/eq/zaiss.png": r"""L(A, \Gamma, \Delta\omega) = \frac{A \cdot \Gamma^2/4}{\Gamma^2/4 + \Delta\omega^2}""",
    # -- superlorentzian --
    "references/superlorentzian/eq/lorentzian.png": r"""g(2\pi\Delta) = \frac{T_2}{\pi} \frac{1}{1 + (2\pi\Delta\, T_2)^2}""",
    "references/superlorentzian/eq/gaussian.png": r"""g(2\pi\Delta) = \frac{T_2}{\sqrt{2\pi}}\, e^{-\frac{(2\pi\Delta\, T_2)^2}{2}}""",
    "references/superlorentzian/eq/dipolar.png": r"""\overline{\mathcal{H}}_{d,\theta} = \overline{\mathcal{H}}_{d,\theta=0} \left( \frac{3\cos^2\theta - 1}{2} \right)""",
    "references/superlorentzian/eq/sl_general.png": r"""g(2\pi\Delta) = \int_0^1 \frac{3}{|\, 3\cos^2\theta - 1 \,|}\, f\!\left( \frac{4(2\pi\Delta)}{|\, 3\cos^2\theta - 1 \,|} \right) d(\cos\theta)""",
    "references/superlorentzian/eq/superlorentzian.png": r"""g(2\pi\Delta) = \int_0^{\pi/2} d\theta\, \sin\theta\, \sqrt{\frac{2}{\pi}}\, \frac{T_2}{|\, 3\cos^2\theta - 1 \,|}\, \exp\!\left( -2 \left( \frac{2\pi\Delta\, T_2}{|\, 3\cos^2\theta - 1 \,|} \right)^2 \right)""",
    "references/superlorentzian/eq/lor_mod.png": r"""g_i\big(2\pi(w - w_0)\big) = \frac{1}{\left[\, 1 + \{2\pi(w - w_0)\, T_{2m}\}^2 \,\right]} \times A""",
    "references/superlorentzian/eq/gauss_mod.png": r"""g_i\big(2\pi(w - w_0)\big) = e^{-\frac{\{2\pi(w - w_0)\, T_{2m}\}^2}{2}} \times A""",
    "references/superlorentzian/eq/sl_mod.png": r"""g_i\big(2\pi(w - w_0)\big) = \int_0^{\pi/2} \frac{\sin\theta}{|\, 3(\cos\theta)^2 - 1 \,|} \times \exp\!\left( -2 \times \left( \frac{2\pi(w - w_0)\, T_{2m}}{|\, 3(\cos\theta)^2 - 1 \,|} \right)^2 \right) d\theta \times A""",
    # -- t1t2b1 --
    "references/t1t2b1/eq/t1.png": r"""S(\mathrm{TR}) = M_0 \left(1 - e^{-\mathrm{TR}/T_1}\right)""",
    "references/t1t2b1/eq/t2.png": r"""S(\mathrm{TE}) = M_0\, e^{-\mathrm{TE}/T_2}""",
    "references/t1t2b1/eq/b1_ratio.png": r"""\frac{I_2(r)}{I_1(r)} = \frac{\sin\alpha_2(r)\, f_2(T_1, \mathrm{TR})}{\sin\alpha_1(r)\, f_1(T_1, \mathrm{TR})}""",
    "references/t1t2b1/eq/b1_alpha.png": r"""\alpha(r) = \arccos\left(\left|\frac{I_2(r)}{2\, I_1(r)}\right|\right)""",
    "references/t1t2b1/eq/bs_phase.png": r"""\phi_{BS} = \int_0^T \frac{\left(\gamma B_1(t)\right)^2}{2\, \omega_{RF}(t)}\, dt = K_{BS}\, B_{1,\mathrm{peak}}^2""",
    "references/t1t2b1/eq/bs_b1.png": r"""B_{1,\mathrm{peak}} = \sqrt{\frac{\phi_{BS}}{K_{BS}}}, \quad \Delta\phi = \phi(+\omega_{RF}) - \phi(-\omega_{RF}) = 2\, K_{BS}\, B_{1,\mathrm{peak}}^2 \ \Rightarrow\ B_{1,\mathrm{peak}} = \sqrt{\frac{\Delta\phi}{2\, K_{BS}}}""",
    # -- plof --
    "references/plof/eq/zss.png": r"""Z^{ss} = \frac{\cos^2\theta\, R_1}{R_{1\rho}}""",
    "references/plof/eq/r1rho.png": r"""R_{1\rho} = R_{\mathrm{eff}} + R_{\mathrm{back}} + R_{\mathrm{peak1}} + R_{\mathrm{peak2}}""",
    "references/plof/eq/reff.png": r"""R_{\mathrm{eff}} = \cos^2\theta\, R_1 + \sin^2\theta\, R_2, \quad \theta = \tan^{-1}(\omega_1/\Delta)""",
    "references/plof/eq/rpeak1.png": r"""R_{\mathrm{peak1}} = R_{\mathrm{peak1}}^{\max} \frac{(w_{\mathrm{peak1}}/2)^2}{(w_{\mathrm{peak1}}/2)^2 + (\Delta - \Delta_{\mathrm{peak1}})^2}""",
    "references/plof/eq/rpeak2.png": r"""R_{\mathrm{peak2}} = R_{\mathrm{peak2}}^{\max} \frac{(w_{\mathrm{peak2}}/2)^2}{(w_{\mathrm{peak2}}/2)^2 + (\Delta - \Delta_{\mathrm{peak2}})^2}""",
    "references/plof/eq/rback.png": r"""R_{\mathrm{back}} = D_0 + D_1(\Delta - 2) + D_2(\Delta - 2)^2 + D_3(\Delta - 2)^3""",
    # -- drof --
    "references/drof/eq/mz.png": r"""M_z(t) = \left(M_0 \cos^2 \theta - M_{ss}\right) e^{-R_{1\rho} t} + M_{ss}""",
    "references/drof/eq/mss.png": r"""M_{ss} = \frac{M_0 \, R_{1w} \cos^2 \theta}{R_{1\rho}}""",
    "references/drof/eq/zspec.png": r"""Z(\Delta\omega) = \frac{M_z(t_{\mathrm{sat}})}{M_0} = \left(\cos^2 \theta - \frac{R_{1w} \cos^2 \theta}{R_{1\rho}}\right) e^{-R_{1\rho} t_{\mathrm{sat}}} + \frac{R_{1w} \cos^2 \theta}{R_{1\rho}}""",
    "references/drof/eq/plof.png": r"""R_{1\rho}(\Delta\omega) = C_0' + L_{\mathrm{water}}(\Delta\omega) + \sum_{n=0}^{N} C_n \, \Delta\omega^{\,n} + L_{\mathrm{amide}}(\Delta\omega) + L_{\mathrm{creatine}}(\Delta\omega)""",
    "references/drof/eq/drof.png": r"""R_{1\rho} = C_0' + L_{\mathrm{water}} + L_{\mathrm{rNOE}} + L_{\mathrm{MT}} + L_{\mathrm{amide}} + L_{\mathrm{creatine}}""",
    # -- inversez --
    "references/inversez/eq/eq1.png": r"""\alpha(B_1) = \frac{\omega_1^2}{\omega_1^2 + k_{sw}^2}""",
    "references/inversez/eq/eq2.png": r"""MTR_{asym} = Z_{ref}(t_p, B_1) - Z_{lab}(t_p, B_1) = \frac{f_s k_{sw} \alpha}{R_{1a} + f_s k_{sw} \alpha} + (Z_i - 1)\, e^{-R_{1a} t_p} - \left( Z_i - \frac{R_{1a}}{R_{1a} + f_s k_{sw} \alpha} \right)""",
    "references/inversez/eq/eq3.png": r"""Z_i = 1 - e^{-R_{1a} t_{rec}}""",
    "references/inversez/eq/eq4.png": r"""\frac{1}{Z_{total}} = \sum_n \frac{1}{Z_n}""",
    "references/inversez/eq/eq5.png": r"""Z(\Delta\omega) = (1 - Z_{ss})\, e^{-R_{1\rho} t_{sat}} + Z_{ss}""",
    "references/inversez/eq/eq6.png": r"""Z_{ss} = \frac{R_1 \cos^2\theta}{R_{1\rho}}""",
    "references/inversez/eq/eq7.png": r"""\theta = \tan^{-1}\frac{\omega_1}{\Delta\omega}""",
    "references/inversez/eq/eq8.png": r"""R_{1\rho} = R_{eff} + R_{ex}""",
    "references/inversez/eq/eq9.png": r"""R_{eff} = R_1 \cos^2\theta + R_2 \sin^2\theta""",
    "references/inversez/eq/eq10.png": r"""R_{ex} + R_2 \sin^2\theta = R_1 \cos^2\theta \left( \frac{1}{Z_{ss}} - 1 \right)""",
    "references/inversez/eq/eq11.png": r"""R_{1\rho} = R_{eff} + R_a + R_b + R_{MT}""",
    "references/inversez/eq/eq12.png": r"""\sum_n R_n + R_2 \sin^2\theta = R_1 \cos^2\theta \left( \frac{1}{Z_{ss}} - 1 \right)""",
    "references/inversez/eq/eq13.png": r"""\sum_n R_n = R_1 \left( \frac{1}{Z_{ss}} - 1 \right)""",
    "references/inversez/eq/eq14.png": r"""Z_{ss} = \frac{R_1 \cos^2\theta}{R_n + R_1 \cos^2\theta}""",
    # -- arex --
    "references/arex/eq/cestr.png": r"""\mathrm{CESTR} = \frac{S_{\mathrm{ref}} - S_{\mathrm{lab}}}{S_0} = Z_{\mathrm{ref}} - Z_{\mathrm{lab}}""",
    "references/arex/eq/cestrnr.png": r"""\mathrm{CESTR}^{nr} = \frac{S_{\mathrm{ref}} - S_{\mathrm{lab}}}{S_{\mathrm{ref}}} = \frac{Z_{\mathrm{ref}} - Z_{\mathrm{lab}}}{Z_{\mathrm{ref}}}""",
    "references/arex/eq/mtrrex.png": r"""\mathrm{MTR}_{\mathrm{Rex}} = \frac{\left( S_{\mathrm{ref}} - S_{\mathrm{lab}} \right) S_0}{S_{\mathrm{ref}} \, S_{\mathrm{lab}}} = \frac{1}{Z_{\mathrm{lab}}} - \frac{1}{Z_{\mathrm{ref}}} = \frac{Z_{\mathrm{ref}} - Z_{\mathrm{lab}}}{Z_{\mathrm{ref}} \, Z_{\mathrm{lab}}}""",
    "references/arex/eq/arex.png": r"""\mathrm{AREX} = \frac{\mathrm{MTR}_{\mathrm{Rex}}}{T_{1w}} = \frac{\left( S_{\mathrm{ref}} - S_{\mathrm{lab}} \right) S_0}{S_{\mathrm{ref}} \, S_{\mathrm{lab}} \, T_{1w}} = \frac{Z_{\mathrm{ref}} - Z_{\mathrm{lab}}}{Z_{\mathrm{ref}} \, Z_{\mathrm{lab}}} \cdot \frac{1}{T_{1w}}""",
    # -- WASABI --
    "references/WASABI/eq/eq1.png": r"""M_z(t_p) = \cos\left( \alpha(t_p) \right) = M_0 \left| 1 - 2 \cdot \sin^2(\theta) \cdot \sin^2\left( \frac{\omega_{\mathrm{eff}} \cdot t_p}{2} \right) \right|""",
    "references/WASABI/eq/with.png": r"""\text{With} \quad \tan\theta = \frac{\gamma B_1}{\Delta\omega}, \quad \omega_{\mathrm{eff}} = \sqrt{(\gamma \cdot B_1)^2 + (\Delta\omega)^2}, \quad \text{and} \quad Z(\Delta\omega) \equiv \frac{M_z(\Delta\omega)}{M_0} \quad \text{we obtain}""",
    "references/WASABI/eq/eq2.png": r"""Z(\Delta\omega) = \left| 1 - 2 \cdot \sin^2\!\left( \tan^{-1}\!\left( \frac{\gamma \cdot B_1}{\Delta\omega} \right) \right) \cdot \sin^2\!\left( \sqrt{(\gamma \cdot B_1)^2 + (\Delta\omega)^2} \cdot \frac{t_p}{2} \right) \right|""",
    "references/WASABI/eq/fit.png": r"""\text{With fit parameters} \quad c,\, d \quad \text{and a per-voxel } B_0 \text{ shift } \delta\omega:""",
    "references/WASABI/eq/eq3.png": r"""Z(\Delta\omega) = c - d \cdot \sin^2\!\left( \tan^{-1}\!\left( \frac{\gamma \cdot B_1}{\Delta\omega - \delta\omega} \right) \right) \cdot \sin^2\!\left( \sqrt{(\gamma \cdot B_1)^2 + (\Delta\omega - \delta\omega)^2} \cdot \frac{t_p}{2} \right)""",
    # -- wassr --
    "references/wassr/eq/mscf.png": r"""\mathrm{MSCF} = \arg\min_{C} \left\langle \left( f(x_i) - \tilde{f}(2C - x_i) \right)^2 \right\rangle_{x_{(1)} \leq 2C - x_i \leq x_{(N)}}""",
}

_SECTIONS: list[tuple] = [

    (
        "CEST MRI",
        "#74c7ec",
        [],
        r"",
        (
            "Chemical Exchange Saturation Transfer (CEST) detects dilute, "
            "exchangeable-proton solutes <i>indirectly</i>, through the water signal. A "
            "frequency-selective RF pulse saturates a solute's labile protons (amide "
            "–NH, amine –NH<sub>2</sub>, hydroxyl –OH); chemical exchange relays that "
            "saturation to the huge water pool, where it builds up into a small, "
            "measurable loss of water signal — amplifying millimolar solutes by orders "
            "of magnitude."
        ),
        [
            ("Liu G., Song X., Chan K.W.Y., McMahon M.T. (2013). A review of optimization "
             "and quantification techniques for chemical exchange saturation transfer (CEST) "
             "MRI toward sensitive in vivo imaging. Contrast Media Mol. Imaging 8, 526–540.",
             "https://doi.org/10.1002/cmmi.1628",
             "references/cestmri/A review of optimization and quantification techniques for chemical exchange saturation transfer (CEST) MRI toward sensitive in vivo imaging.pdf"),
            ("Zu Z., Janve V.A., Li K., Does M.D., Gore J.C., Gochberg D.F. (2011). "
             "Optimizing pulsed-chemical exchange saturation transfer imaging sequences. "
             "Magn. Reson. Med. 66, 1100–1108.",
             "https://doi.org/10.1002/mrm.22884",
             "references/cestmri/Optimizing Pulsed-Chemical Exchange Saturation Transfer (CEST) Imaging Sequences.pdf"),
            ("van Zijl P.C.M., Yadav N.N. (2011). Chemical exchange saturation transfer "
             "(CEST): what is in a name and what isn't? Magn. Reson. Med. 65, 927–948.",
             "https://doi.org/10.1002/mrm.22761",
             "references/cestmri/Chemical Exchange Saturation Transfer what is in a name and what isnt.pdf"),
            ("Eng J., Ceckler T.L., Balaban R.S. (1991). Quantitative ¹H magnetization "
             "transfer imaging in vivo. Magn. Reson. Med. 17, 304–314.",
             "https://doi.org/10.1002/mrm.1910170206",
             "references/cestmri/Quantitative 1H magnetization transfer imaging in vivo.pdf"),
        ],
        {
            "math": [
                {"text": "<b>How CEST works.</b>  A frequency-selective pulse saturates the "
                         "solute's exchangeable protons; chemical exchange relays that "
                         "saturation to water, where it builds up. Plotting the water signal "
                         "S<sub>sat</sub>/S<sub>0</sub> against offset gives the "
                         "<b>Z-spectrum</b> — a large water dip at 0 ppm and a small solute "
                         "dip a few ppm away."},
                {"img": "references/cestmri/fig/cest_principles.png", "figure": True},

                {"text": "<b>Types of CEST agent.</b>  Classified by exchange type: "
                         "atom (proton) exchange — diaCEST, some paraCEST, glycoCEST, "
                         "gagCEST, APT; molecular exchange — paraCEST / Ln(III) complexes; "
                         "and compartmental exchange — lipoCEST."},
                {"img": "references/cestmri/fig/classification.png", "figure": True},

                {"text": "<b>Exchange rate, temperature and field.</b>  CEST needs "
                         "slow-to-intermediate exchange, not an NMR-visible peak. Faster "
                         "exchange (warmer / higher pH) can even strengthen the effect, and a "
                         "higher B<sub>0</sub> resolves the solute better from water."},
                {"img": "references/cestmri/fig/exchange_regime.png", "figure": True},

                {"text": "<b>The pulse sequence.</b>  A long saturation block at the solute "
                         "offset precedes the imaging readout. Because exchange is slow, the "
                         "protons can instead be tagged by repeated label-transfer modules — "
                         "inversion, gradient dephasing or frequency labelling."},
                {"img": "references/cestmri/fig/transfer_schemes.png", "figure": True},

                {"text": "<b>Continuous vs pulsed saturation.</b>  Ideally one long "
                         "continuous-wave block or spin-lock (SL); clinical scanners use a "
                         "pulsed train (duty cycle DC = t<sub>p</sub>/(t<sub>p</sub>+"
                         "t<sub>d</sub>)) to respect SAR limits."},
                {"img": "references/cestmri/fig/saturation.png", "figure": True},

                {"text": "<b>Basic quantification.</b>  MTR asymmetry between mirror offsets:"},
                {"img": "references/cestmri/eq/eq4.png", "num": "(1)"},
                {"text": "the steady-state CEST effect (proton-transfer ratio):"},
                {"img": "references/cestmri/eq/eq5.png", "num": "(2)"},
                {"text": "with saturation efficiency (ω<sub>1</sub> = γB<sub>1</sub>):"},
                {"img": "references/cestmri/eq/eq6.png", "num": "(3)"},
            ],
        },
    ),

    (
        "Denoising",
        "#94e2d5",
        [],
        r"",
        (
            "<b>Denoising the Z-spectrum.</b>  CEST effects are small, so image noise "
            "directly degrades MTR<sub>asym</sub> and line-shape fits. OCEAN offers three "
            "analytical denoisers that exploit the redundancy of the 4-D CEST data "
            "(x, y, slice, offset): <b>PCA</b>, <b>BM3D</b> and <b>NLM</b>. Each trades "
            "noise removal against a little spatial / spectral blurring."
        ),
        [
            ("Radke K.L., Kamp B., Adriaenssens V., Stabinska J., Gallinnis P., Wittsack "
             "H.-J., Antoch G., Müller-Lutz A. (2023). Deep Learning-Based Denoising of "
             "CEST MR Data: A Feasibility Study on Applying Synthetic Phantoms in Medical "
             "Imaging. Diagnostics 13, 3326.",
             "https://doi.org/10.3390/diagnostics13213326",
             "references/denoising/diagnostics-13-03326.pdf"),
        ],
        {
            "math": [
                {"text": "<b>PCA — Principal Component Analysis.</b>  The 4-D dataset D is "
                         "decomposed into orthogonal components ordered by variance; keeping "
                         "only the first <i>k</i> (signal) components and discarding the rest "
                         "(noise) reconstructs a denoised dataset D′. The number k is chosen "
                         "automatically by an indicator criterion (OCEAN uses Malinowski's):"},
                {"img": "references/denoising/eq/eq1.png", "num": "(1)"},
                {"text": "<b>BM3D — Block-Matching and 3-D filtering.</b>  For each reference "
                         "block B<sub>r</sub> the algorithm searches the image for similar "
                         "blocks and stacks them into a 3-D group S:"},
                {"img": "references/denoising/eq/eq2.png", "num": "(2)"},
                {"text": "The group is transformed (T = 3-D discrete cosine transform), its "
                         "coefficients are filtered (F) to suppress noise, and an inverse "
                         "transform returns the cleaned blocks S′ to image space. Running "
                         "this twice — a hard-threshold estimate, then a Wiener filter — "
                         "gives the final result:"},
                {"img": "references/denoising/eq/eq3.png", "num": "(3)"},
                {"text": "BM3D is tuned by the 2-D block size and the search-window size "},
                {"text": "<b>NLM — Non-Local Means.</b>  Each pixel p is replaced by a "
                         "weighted average of pixels inside a search window W, the weights "
                         "set by the similarity of their local neighbourhoods — so repeating "
                         "structure is reinforced while noise cancels:"},
                {"img": "references/denoising/eq/eq4.png", "num": "(4)"},
                {"text": "A larger search window W removes more noise but blurs more; NLM "
                         "uses a big <i>search</i> window and a small <i>patch</i> window "},
            ],
        },
    ),

    (
        "Bloch–McConnell Equations",
        "#7aa2f7",
        [
            r"── (1)  Original BM System  (Zaiss & Bachert 2013, Eqs. 1–7) ──────",
            r"",
            r"  d/dt M⃗  =  A · M⃗  +  C⃗                                   ",
            r"",
            r"  M⃗ = [ Mxa, Mya, Mza, Mxb, Myb, Mzb ]ᵀ                    ",
            r"",
            r"  A = |  La      -Ka+Kb |                                  ",
            r"      | +Ka       Lb-Kb |",
            r"",
            r"  Li = | -R2i   -Δωi     0  |                              ",
            r"       | +Δωi   -R2i   +ω₁ |   i = a, b",
            r"       |   0     -ω₁   -R1i|",
            r"",
            r"  C⃗ = [ 0, 0, R1a·M0a, 0, 0, R1b·M0b ]ᵀ                    ",
            r"",
            r"  Ka = fb·Kb = fb·kb·I₃                                    ",
            r"  fb = M0b/M0a = nb·[b] / (na·[a])                         ",
            r"",
            r"── (2)  Numerical Solution  ───────────────────────────────",
            r"",
            r"  M⃗(tsat) = (M⃗₀ + A⁻¹·C⃗)·exp(A·tsat) − A⁻¹·C⃗               ",
            r"",
            r"  (matrix exponentiation; stepwise-constant A for pulsed CEST)",
            r"",
            r"── (3)  Analytical Eigenspace Solution      ───────────────",
            r"",
            r"  λ₁  =  −R1ρ    (smallest eigenvalue in modulus)          ",
            r"",
            r"  Z(Δω,tsat) = (Pzeff·Pz·Zi − Zss)·exp(−R1ρ·tsat) + Zss    ",
            r"",
            r"  Zss(Δω) = Pz · R1,res / R1ρ(Δω)                          ",
            r"  R1,res ≈ cosθ · R1a ,   θ = arctan(ω₁ / Δω)",
        ],
        # LaTeX 
        r"""% (1) Original BM System
\frac{d\vec{M}}{dt} = \mathbf{A}\,\vec{M} + \vec{C} \tag{1}

\vec{M} = \bigl[M_{xa},M_{ya},M_{za},M_{xb},M_{yb},M_{zb}\bigr]^{\!\top},\quad
\mathbf{A} = \begin{pmatrix}\mathbf{L}_a & -\mathbf{K}_a+\mathbf{K}_b \\ +\mathbf{K}_a & \mathbf{L}_b-\mathbf{K}_b\end{pmatrix} \tag{2,3}

\mathbf{L}_i = \begin{pmatrix}-R_{2i} & -\Delta\omega_i & 0 \\ +\Delta\omega_i & -R_{2i} & +\omega_1 \\ 0 & -\omega_1 & -R_{1i}\end{pmatrix},\quad i=a,b \tag{4}

\vec{C} = \bigl[0,\,0,\,R_{1a}M_{0a},\,0,\,0,\,R_{1b}M_{0b}\bigr]^{\!\top},\quad
\mathbf{K}_a = f_b\mathbf{K}_b = f_b k_b \mathbf{I}_3 \tag{5,6}

f_b = \frac{M_{0b}}{M_{0a}} = \frac{n_b[b]}{n_a[a]} \tag{7}

% (2) Numerical Solution
\vec{M}(t_\text{sat}) =
  \bigl(\vec{M}_0 + \mathbf{A}^{-1}\vec{C}\bigr)\exp(\mathbf{A}\,t_\text{sat})
  - \mathbf{A}^{-1}\vec{C} \tag{16}

% (3) Analytical Eigenspace Solution
\lambda_1 = -R_{1\rho} \tag{17}

Z(\Delta\omega,t_\text{sat}) =
  \bigl(P_{z_\text{eff}}P_z Z_i - Z_\text{ss}\bigr)
  e^{-R_{1\rho}(\Delta\omega)\,t_\text{sat}} + Z_\text{ss}(\Delta\omega) \tag{18}

Z_\text{ss}(\Delta\omega) = P_z\,\frac{R_{1,\text{res}}}{R_{1\rho}(\Delta\omega)},\quad
R_{1,\text{res}} \approx \cos\theta\,R_{1a},\quad
\theta = \arctan\!\left(\frac{\omega_1}{\Delta\omega}\right) \tag{19}""",
        # Description
        (
            "<b>Pool a</b> = water pool (measured signal);  "
            "<b>Pool b</b> = dilute solute / CEST pool<br>"
            "<b>M⃗</b> = 6-component magnetization vector [Mx, My, Mz] per pool<br>"
            "<b>A</b> = system matrix (block: relaxation sub-matrices L + coupling matrices K)<br>"
            "<b>L<sub>i</sub></b> = 3×3 relaxation sub-matrix for pool i "
            "(R1, R2, RF precession at ω₁, frequency offset Δω)<br>"
            "<b>K<sub>a</sub> = f<sub>b</sub>·k<sub>b</sub>·I</b> = backward exchange coupling matrix<br>"
            "<b>R1a, R2a</b> = longitudinal/transverse relaxation rates of water pool (s⁻¹)<br>"
            "<b>R1b, R2b</b> = relaxation rates of solute pool (s⁻¹)<br>"
            "<b>k<sub>b</sub></b> = solute→water exchange rate (s⁻¹);<br>"
            "<b>f<sub>b</sub></b> = solute proton fraction = M<sub>0b</sub>/M<sub>0a</sub><br>"
            "<b>Δω = ω<sub>rf</sub> − ω<sub>a</sub></b> = RF offset from water Larmor frequency<br><br>"
            "<b>Numerical solution:</b>  formal matrix-exponential solution for "
            "constant A; stepwise-constant approximation used for pulsed CEST "
            "Padé exponentiation.<br><br>"
            "<b>Analytical eigenspace solution:</b>  for off-resonant irradiation "
            "with ω<sub>eff</sub> ≫ 1/T₂ and t<sub>sat</sub> ≫ T₂, only the eigenvalue "
            "λ₁ = −R<sub>1ρ</sub> (smallest in modulus) contributes. "
            "Z decays mono-exponentially toward steady-state Z<sub>ss</sub> at rate R<sub>1ρ</sub>.<br>"
            "<b>P<sub>zeff</sub>, P<sub>z</sub></b> = projection factors linking initial/final "
            "magnetisation to the effective-field axis z<sub>eff</sub> (= cosθ for CEST)<br>"
            "<b>Z<sub>i</sub> = |M⃗<sub>i</sub>|/M₀</b> = initial Z-magnetisation"
        ),
        [
            ("Bloch F. (1946). Nuclear induction. Phys. Rev. 70, 460–474.",
             "https://doi.org/10.1103/PhysRev.70.460",
             "references/bloch/bloch.pdf"),
            ("McConnell H.M. (1958). Reaction rates by nuclear magnetic resonance. "
             "J. Chem. Phys. 28, 430–431.",
             "https://doi.org/10.1063/1.1744152",
             "references/bloch/Reaction Rates by Nuclear Magnetic Resonance.pdf"),
            ("Forsén S. & Hoffman R.A. (1963). Study of moderately rapid chemical exchange "
             "reactions by means of nuclear magnetic double resonance. "
             "J. Chem. Phys. 39, 2892–2901.",
             "https://doi.org/10.1063/1.1734121",
             "references/bloch/2892_1_online.pdf"),
            ("Woessner D.E., Zhang S., Merritt M.E., Sherry A.D. (2005). Numerical solution of "
             "the Bloch equations provides insights into the optimum design of PARACEST agents "
             "for MRI. Magn. Reson. Med. 53, 790–799.",
             "https://doi.org/10.1002/mrm.20408",
             "references/bloch/Magnetic Resonance in Med - 2005 - Woessner - Numerical solution of the Bloch equations provides insights into the optimum.pdf"),
            ("Sun P.Z. (2010). Simplified and scalable numerical solution for describing "
             "multi-pool chemical exchange saturation transfer (CEST) MRI contrast. "
             "J. Magn. Reson. 205, 235–241.",
             "https://doi.org/10.1016/j.jmr.2010.05.004",
             "references/bloch/Simplified and scalable numerical solution for describing "
             "multi-pool chemical exchange saturation transfer (CEST) MRI contrast.pdf"),
            ("Zaiss M. & Bachert P. (2012). Exchange-dependent relaxation in the rotating frame "
             "for slow and intermediate exchange — modeling off-resonant spin-lock and CEST. "
             "NMR Biomed. 26, 507–518.",
             "https://doi.org/10.1002/nbm.2887",
             "references/bloch/Zaiss Exchange dependent relaxation in the rotating frame for slow and intermediate exchange.pdf"),
            ("Zaiss M. & Bachert P. (2013). Chemical exchange saturation transfer (CEST) "
             "and MR Z-spectroscopy in vivo: a review of theoretical approaches and methods. "
             "Phys. Med. Biol. 58, R221–R269.",
             "https://doi.org/10.1088/0031-9155/58/22/R221",
             "references/bloch/Zaiss_2013_Phys._Med._Biol._58_R221.pdf"),
            ("Eng J., Ceckler T.L., Balaban R.S. (1991). Quantitative ¹H magnetization "
             "transfer imaging in vivo. Magn. Reson. Med. 17, 304–314.",
             "https://doi.org/10.1002/mrm.1910170206",
             "references/bloch/Quantitative 1H magnetization transfer imaging in vivo.pdf"),
            ("Abragam A. (1961). The Principles of Nuclear Magnetism. "
             "Oxford University Press, Oxford.",
             "",
             "references/bloch/Principles of Nuclear Magnetism.pdf"),
        ],
        {   # ── typeset equations (1)–(7) + Figure 5, from Zaiss 2013 ─────────
            "math": [
                {"text": "In the rotating frame of reference (<i>x, y, z</i>) defined by "
                         "the frequency ω<sub>rf</sub> of the oscillating field "
                         "<i>B</i><sub>1</sub>(<i>t</i>), the BM equations read"},
                {"img": "references/bloch/eq/eq1.png", "num": "(1)"},
                {"text": "with the six-dimensional magnetisation vector"},
                {"img": "references/bloch/eq/eq2.png", "num": "(2)"},
                {"text": "<i>A</i> is a block matrix"},
                {"img": "references/bloch/eq/eq3.png", "num": "(3)"},
                {"img": "references/bloch/fig5.png", "num": "", "figure": True},
                {"text": "The effective frame defined by the effective "
                         "field γ<i>B</i><sub>eff</sub> = ω<sub>eff</sub> = "
                         "(ω<sub>1</sub>, 0, Δω) in the rotating frame of reference "
                         "(<i>x, y, z</i>). <i>B</i><sub>eff</sub> is an eigenvector of "
                         "the smallest eigenvalue in modulus of the BM equation system, "
                         "with absolute value R<sub>1ρ</sub>. Thus in steady-state the "
                         "magnetisation vector of the system is dominated by the "
                         "contribution along z<sub>eff</sub>."},
                {"text": "consisting of 3 × 3 submatrices <i>K<sub>i</sub></i> = "
                         "<i>k<sub>i</sub></i> · <b>I</b> (see equations (6), (8) and "
                         "(9)), and <i>L<sub>i</sub></i>:"},
                {"img": "references/bloch/eq/eq4.png", "num": "(4)"},
                {"text": "where <i>i</i> = a, b. Finally, the constant vector "
                         "<i>C</i> is"},
                {"img": "references/bloch/eq/eq5.png", "num": "(5)"},
                {"text": "The quantity Δω = Δω<sub>a</sub> = ω<sub>rf</sub> − "
                         "ω<sub>a</sub> is the frequency offset relative to the Larmor "
                         "frequency ω<sub>a</sub> of pool a (for ¹H: "
                         "ω<sub>a</sub>/<i>B</i>₀ = γ = 267.5 rad μT⁻¹ s⁻¹). The offset "
                         "of pool b: Δω<sub>b</sub> = ω<sub>rf</sub> − ω<sub>b</sub> = "
                         "Δω − δ<sub>b</sub>ω<sub>a</sub>, is shifted by δ<sub>b</sub> "
                         "relative to the water proton resonance."},
                {"text": "The BM/BS equations pick out one relaxation pathway "
                         "explicitly. We therefore separate this process and define the "
                         "relaxation rates excluding this specific pathway by "
                         "<i>R</i>′ = 1/<i>T</i>₁′ and <i>R</i>₂′ = 1/<i>T</i>₂′ "
                         "(Neuhaus and Williamson 1989). In the context of the BM "
                         "equations the used relaxation rates already exclude exchange. "
                         "Therefore, <i>R</i><sub>1a</sub> = <i>R</i>′<sub>1a</sub> and "
                         "<i>R</i><sub>2a</sub> = <i>R</i>′<sub>2a</sub> and the same for "
                         "pool b. The matrix <i>K</i><sub>a</sub> = "
                         "<i>f</i><sub>b</sub><i>K</i><sub>b</sub> reads"},
                {"img": "references/bloch/eq/eq6.png", "num": "(6)"},
                {"text": "The proton fraction <i>f</i><sub>b</sub> is"},
                {"img": "references/bloch/eq/eq7.png", "num": "(7)"},
                {"text": "where [a] and [b] are the concentrations of pool a and b, "
                         "respectively, and <i>n</i><sub>a</sub> and "
                         "<i>n</i><sub>b</sub> are the numbers of protons per molecule "
                         "for the respective proton pools."},
                {"text": "BM equations generalised to <i>n</i> pools (water + solutes "
                         "s<sub>1</sub> … s<sub>n−1</sub>), the same banded structure "
                         "scales as"},
                {"img": "references/bloch/eq/eq9.png", "num": ""},
            ],
        },
    ),

    (
        "CEST Magnetic Resonance Fingerprinting",
        "#89dceb",
        [   # plain-text fallback (shown only if the typeset images are unavailable)
            r"Dot product:         DP(e,d) = <e,d> / ( ||e|| . ||d|| )",
            r"Euclidean distance:  ED(e,d) = (1/sqrt(N_t)) || Z_e - Z_d || ,   Z_e = e/M0e ,  Z_d = d/M0d",
            r"",
            r"Schedule (CRLB):     F = (1/sigma^2) J^T J ,    Var(theta_i) >= [F^-1]_ii",
            r"                     Lambda* = argmin_Lambda  (1/P) sum_i [F^-1(theta;Lambda)]_ii / theta_i^2",
        ],
        r"""\mathrm{DP}(\mathbf{e},\mathbf{d}) = \frac{\langle \mathbf{e},\mathbf{d}\rangle}{\lVert \mathbf{e}\rVert \cdot \lVert \mathbf{d}\rVert}
\qquad
\mathrm{ED}(\mathbf{e},\mathbf{d}) = \frac{1}{\sqrt{N_t}}\,\bigl\lVert \hat{\mathbf{Z}}_e - \hat{\mathbf{Z}}_d \bigr\rVert,
\quad \hat{\mathbf{Z}}_e = \frac{\mathbf{e}}{M_{0e}},\ \ \hat{\mathbf{Z}}_d = \frac{\mathbf{d}}{M_{0d}}

F_{ij}(\boldsymbol{\theta}) = \frac{1}{\sigma^{2}}\,\mathbf{J}^{\top}\mathbf{J},
\qquad \mathrm{Var}(\hat{\theta}_i) \ge \bigl[\mathbf{F}^{-1}(\boldsymbol{\theta})\bigr]_{ii},
\qquad \boldsymbol{\Lambda}^{*} = \underset{\boldsymbol{\Lambda}}{\arg\min}\,\frac{1}{P}\sum_i \frac{\bigl[\mathbf{F}^{-1}(\boldsymbol{\theta};\boldsymbol{\Lambda})\bigr]_{ii}}{\theta_i^{2}}""",
        (
            "CEST-MRF acquires a train of differently weighted images by varying the saturation "
            "power B₁ and saturation time t<sub>sat</sub> from one schedule iteration to the next, "
            "so that each tissue produces a distinctive signal trajectory (its <i>fingerprint</i>).<br><br>"
            "<b>Symbols.</b>  <b>e</b> = acquired (experimental) voxel trajectory;  "
            "<b>d</b> = a dictionary (simulated) trajectory;  ⟨·,·⟩ = inner product;  "
            "‖·‖ = L2 norm;  N<sub>t</sub> = number of schedule iterations;  "
            "M<sub>0e</sub>, M<sub>0d</sub> = unsaturated reference of the acquired / dictionary "
            "trajectory;  <b>θ</b> = (T<sub>1w</sub>, T<sub>2w</sub>, f<sub>s</sub>, k<sub>sw</sub>) "
            "the quantitative parameters;  <b>J</b> = ∂<b>s</b>/∂θ the trajectory Jacobian;  "
            "σ = noise standard deviation;  <b>F</b> = Fisher information matrix;  "
            "Λ = {B₁(t<sub>n</sub>), t<sub>sat</sub>(t<sub>n</sub>)} the saturation schedule;  "
            "P = number of parameters of interest.<br><br>"
            "<b>f<sub>s</sub></b> = solute proton fraction (mM)  "
            "<b>k<sub>sw</sub></b> = solute→water exchange rate (s⁻¹)."
        ),
        [
            ("Perlman O, Herz K, Zaiss M, Cohen O, Rosen MS, Farrar CT. CEST MR-Fingerprinting: "
             "practical considerations and insights for acquisition schedule design and improved "
             "reconstruction. Magn Reson Med. 2020;83(2):462–478.",
             "https://doi.org/10.1002/mrm.27937",
             "references/cestmrf/Perlman - CEST MR‐Fingerprinting  Practical considerations and insights for acquisition.pdf"),
            ("Cohen O, Huang S, McMahon MT, Rosen MS, Farrar CT. Rapid and quantitative chemical "
             "exchange saturation transfer (CEST) imaging with magnetic resonance fingerprinting "
             "(MRF). Magn Reson Med. 2018;80(6):2449–2463.",
             "https://doi.org/10.1002/mrm.27221",
             "references/cestmrf/Cohen - Rapid and quantitative chemical exchange saturation transfer  CEST  imaging with MRF.pdf"),
            ("Herz K, Mueller S, Perlman O, Zaiss M, et al. Pulseq-CEST: towards multi-site "
             "multi-vendor compatibility and reproducibility of CEST experiments using an "
             "open-source sequence standard. Magn Reson Med. 2021;86(4):1845–1858.",
             "https://doi.org/10.1002/mrm.28825",
             "references/cestmrf/Herz - Pulseq‐CEST  Towards multi‐site multi‐vendor compatibility and reproducibility of.pdf"),
            ("Perlman O, Ito H, Herz K, et al. Quantitative imaging of apoptosis following "
             "oncolytic virotherapy by magnetic resonance fingerprinting aided by deep learning. "
             "Nat Biomed Eng. 2022;6(5):648–657.",
             "https://doi.org/10.1038/s41551-021-00809-7",
             "references/cestmrf/Quantitative imaging of apoptosis following oncolytic virotherapy by magnetic resonance fingerprinting aided by deep learning.pdf"),
        ],
        {   # ── typeset equations: DP/ED matching + CRLB schedule design ──────
            "math": [
                {"text": "<b>Dictionary generation.</b>  The dictionary is built by forward-"
                         "simulating the expected signal trajectory for every parameter "
                         "combination on the grid. Each trajectory is obtained by propagating the "
                         "Bloch–McConnell equations with a Padé approximation of the matrix "
                         "exponential. Three exchanging compartments are represented, following "
                         "established multi-pool models: free water, the CEST solute pool, and a "
                         "semi-solid magnetisation-transfer pool, which together forms a 7×7  "
                         "matrix system. Because the parameter grid is large, the simulator is "
                         "written in C++, using the Eigen library for the linear-algebra operations "
                         "and OpenMP to distribute the independent parameter points."},
                {"text": "<b>Dictionary Matching.</b>Each acquired voxel trajectory <b>e</b> is "
                         "compared against every pre-simulated dictionary trajectory <b>d</b>. "
                         "Matching uses either the normalised dot product "
                         "(range 0–1, higher is better) or the Euclidean distance of the "
                         "M<sub>0</sub>-normalised trajectories (lower is better):"},
                {"img": "references/cestmrf/eq/dp.png", "num": "(1)"},
                {"img": "references/cestmrf/eq/ed.png", "num": "(2)"},
                {"text": "with the M<sub>0</sub>-normalised trajectories"},
                {"img": "references/cestmrf/eq/znorm.png", "num": "(3)"},
                {"text": "The dictionary entry that maximises DP (or minimises ED) assigns its "
                         "parameters (T<sub>1w</sub>, T<sub>2w</sub>, f<sub>s</sub>, k<sub>sw</sub>) "
                         "to the voxel."},
                {"text": "<b>Schedule design — Cramér–Rao lower bound.</b>  The saturation schedule "
                         "Λ = {B₁(t<sub>n</sub>), t<sub>sat</sub>(t<sub>n</sub>)} is chosen so the "
                         "acquisition is maximally sensitive to the parameters of interest. From the "
                         "trajectory Jacobian <b>J</b> = ∂<b>s</b>/∂θ and the noise level σ, the "
                         "Fisher information matrix is"},
                {"img": "references/cestmrf/eq/fisher.png", "num": "(4)"},
                {"text": "The CRLB bounds the variance of any unbiased estimator of θ from below by "
                         "the inverse Fisher information,"},
                {"img": "references/cestmrf/eq/crlb.png", "num": "(5)"},
                {"text": "which, expressed as a relative (normalised) error, becomes"},
                {"img": "references/cestmrf/eq/ncrb.png", "num": "(6)"},
                {"text": "The optimal schedule minimises the mean normalised CRB over the P "
                         "parameters of interest (e.g. f<sub>s</sub>, k<sub>sw</sub>) across the "
                         "dictionary:"},
                {"img": "references/cestmrf/eq/schedule.png", "num": "(7)"},
            ],
        },
    ),

    (
        "Pseudo-Voigt Line Fit",
        "#f38ba8",
        [   # plain-text fallback (shown only if the typeset images are unavailable)
            r"Pseudo-Voigt = weighted sum of Lorentzian (L) and Gaussian (G):",
            r"  V(dw)  ~=  alpha . L(dw)  +  (1 - alpha) . G(dw)",
            r"",
            r"  G(dw)  =  1/sqrt(2*pi*sigma) . exp( -(w1 - w)^2 / (2 sigma^2) )",
            r"  L(dw)  =  A / ( pi [ 1 + (w1 - w)/sigma ]^2 )",
        ],
        r"""V(\Delta\omega) \approx \alpha \times L(\Delta\omega) + (1-\alpha)\,G(\Delta\omega)

G(\Delta\omega) = \frac{1}{\sqrt{2\pi\sigma}}\, e^{-\frac{(\omega_1-\omega)^2}{2(\sigma)^2}}

L(\Delta\omega) = \frac{A}{\pi\left[1 + \dfrac{\omega_1-\omega}{\sigma}\right]^{2}}""",
        (
            "The true Voigt line-shape is a convolution of a Gaussian and a Lorentzian and is "
            "costly to evaluate, so each CEST/NOE peak in the Z-spectrum is approximated by a "
            "<b>pseudo-Voigt</b> profile — a simple weighted sum of the two. Fitting the mixing "
            "weight α together with the amplitude, centre and width per peak lets one form capture "
            "both the sharp core (Lorentzian) and the broader wings (Gaussian) of exchanging "
            "pools.<br><br>"
            "<b>Symbols.</b>  Δω = frequency-offset variable;  ω<sub>1</sub> = frequency offset "
            "from the water resonance;  ω = frequency offset (centre) of the CEST peak for the "
            "proton pool;  <b>α, 1−α ∈ [0,1]</b> = proportionality weights of the Lorentzian and "
            "Gaussian components;  <b>A</b> = peak amplitude;  <b>σ</b> = peak linewidth;  "
            "<b>G, L</b> = the Gaussian and Lorentzian component functions."
        ),
        [
            ("Zhang L, Zhao Y, Chen Y, Bie C, Liang Y, He X, Song X. Voxel-wise Optimization of "
             "Pseudo Voigt Profile (VOPVP) for Z-spectra fitting in chemical exchange saturation "
             "transfer (CEST) MRI. Quant Imaging Med Surg. 2019;9(10):1714–1730.",
             "https://doi.org/10.21037/qims.2019.10.01",
             "references/pseudovoigt/qims-09-10-1714.pdf"),
            ("Ida T, Ando M, Toraya H. Extended pseudo-Voigt function for approximating the Voigt "
             "profile. J. Appl. Crystallogr. 2000;33:1311–1316.  (figure: nt0146.pptx)",
             "https://doi.org/10.1107/S0021889800010219",
             "references/pseudovoigt/nt0146.pptx"),
            ("Petrakis L. Spectral line shapes: Gaussian and Lorentzian functions in magnetic "
             "resonance. J. Chem. Educ. 1967;44(8):432–436.",
             "https://doi.org/10.1021/ed044p432",
             "references/pseudovoigt/ed044p432.pdf"),
        ],
        {   # ── typeset equations: pseudo-Voigt profile (Zhang et al., VOPVP) ─
            "math": [
                {"text": "The true Voigt line-shape (a convolution of Gaussian and Lorentzian) is "
                         "costly to compute, so each CEST/NOE peak is approximated by a "
                         "<b>pseudo-Voigt</b> profile — a weighted sum of a Lorentzian L and a "
                         "Gaussian G:"},
                {"img": "references/pseudovoigt/eq/voigt.png", "num": "[2]"},
                {"text": "where α and 1−α are the proportionality weights of the Lorentzian and "
                         "Gaussian components, respectively. The Gaussian component is"},
                {"img": "references/pseudovoigt/eq/gauss.png", "num": "[3]"},
                {"text": "where ω<sub>1</sub> is the frequency offset from the water resonance and "
                         "ω is the frequency offset of the CEST peak for the proton pool. The "
                         "Lorentzian component is"},
                {"img": "references/pseudovoigt/eq/lorentz.png", "num": "[4]"},
                {"text": "where A, ω and σ are the amplitude, frequency offset and linewidth of the "
                         "CEST peak for the proton pool, respectively."},
            ],
        },
    ),

    (
        "Lorentzian Line Fit",
        "#fab387",
        [   # plain-text fallback (shown only if the typeset images are unavailable)
            r"Multi-pool Lorentzian Z-spectrum model:",
            r"  L(a_k, w_k^c, sigma_k) = 1 - I/I0 = sum_{k=1..K}  a_k / [ 1 + 4 ((w - w_k^c)/sigma_k)^2 ]",
            r"",
            r"Per-pool Lorentzian (with global shift):",
            r"  L_i(dw) = A_i / ( 1 + [ (dw - (ddelta_i + ddelta)) / (Gamma_i/2) ]^2 )",
        ],
        r"""L\!\left(a_{k},\omega_{k}^{c},\sigma_{k}\right) = 1 - \frac{I}{I_{0}} = \sum_{k=1}^{K}\frac{a_{k}}{1 + 4\!\left(\dfrac{\omega-\omega_{k}^{c}}{\sigma_{k}}\right)^{2}}

L_{i}(\Delta\omega) = \frac{A_{i}}{1 + \left[\dfrac{\Delta\omega-(\Delta\delta_{i}+\Delta\delta)}{\Gamma_{i}/2}\right]^{2}}

L(A,\Gamma,\Delta\omega) = \frac{A\,\Gamma^{2}/4}{\Gamma^{2}/4 + \Delta\omega^{2}}""",
        (
            "<b>Multi-pool Lorentzian fit.</b>  A Z-spectrum is decomposed into a sum of "
            "Lorentzian peaks — one for water, one for semi-solid MT, and one per CEST/NOE pool "
            "— so overlapping contributions can be separated and quantified.<br><br>"
            "<b>Symbols.</b>  I / I<sub>0</sub> = image intensity with / without pre-saturation;  "
            "K = number of Lorentzian components;  ω = saturation frequency;  "
            "a<sub>k</sub> (A<sub>i</sub>) = peak amplitude;  "
            "ω<sub>k</sub><sup>c</sup> (Δδ<sub>i</sub>) = chemical-shift offset of the k-th / i-th "
            "CEST proton pool;  σ<sub>k</sub>, Γ<sub>i</sub> = peak width / FWHM;  "
            "Δδ = common global (residual B<sub>0</sub>) offset;  "
            "DWS = direct water saturation;  PTR = proton-transfer ratio."
        ),
        [
            ("Wittsack HJ, Radke KL, et al. calf – Software for CEST Analysis with Lorentzian "
             "Fitting. J Med Syst. 2023;47(1):39.",
             "https://doi.org/10.1007/s10916-023-01931-6",
             "references/lorentzian/10916_2023_Article_1931.pdf"),
            ("Zaiss M, Schmitt B, Bachert P. Quantitative separation of CEST effect from "
             "magnetization transfer and spillover effects by Lorentzian-line-fit analysis of "
             "z-spectra. J Magn Reson. 2011;211(2):149–155.",
             "https://doi.org/10.1016/j.jmr.2011.05.001",
             "references/lorentzian/Quantitative separation of CEST effect from magnetization transfer and spillover effects by Lorentzian line fit analysis of z spectra.pdf"),
        ],
        {   # ── typeset equations: multi-pool Lorentzian Z-spectrum fit ───────
            "math": [
                {"text": "Any Z-spectrum can be modelled by a sum of several Lorentzian functions — "
                         "one per exchanging proton pool (plus water and semi-solid MT). In the "
                         "(1 − I/I<sub>0</sub>) domain the multi-pool model is"},
                {"img": "references/lorentzian/eq/multi.png", "num": "(1)"},
                {"text": "where I is the image intensity, I<sub>0</sub> the image intensity without "
                         "pre-saturation, K the number of Lorentzian components, ω the frequency, and "
                         "a<sub>k</sub>, ω<sub>k</sub><sup>c</sup> and σ<sub>k</sub> are the "
                         "amplitude, frequency offset and width of the k-th CEST proton pool."},
                {"text": "Equivalently, each pool i is an individual Lorentzian centred at its own "
                         "chemical shift Δδ<sub>i</sub> plus a common global shift Δδ (e.g. a "
                         "residual B<sub>0</sub> offset), with amplitude A<sub>i</sub> and FWHM "
                         "Γ<sub>i</sub>:"},
                {"img": "references/lorentzian/eq/single.png", "num": ""},
                {"text": "The Lorentzian line-shape itself follows from the steady-state "
                         "Bloch–McConnell solution: both the direct water saturation (DWS) and the "
                         "proton-transfer ratio (PTR) appear as Lorentzian lines of amplitude A and "
                         "FWHM Γ (Zaiss et al. 2011):"},
                {"img": "references/lorentzian/eq/zaiss.png", "num": ""},
            ],
        },
    ),

    (
        "Super-Lorentzian / Semi-Solid Line Fit",
        "#cba6f7",
        [   # plain-text fallback (shown only if the typeset images are unavailable)
            r"Base lineshapes:",
            r"  Lorentzian: g(2πΔ) = (T2/π) · 1/[1 + (2πΔ·T2)^2]",
            r"  Gaussian:   g(2πΔ) = (T2/sqrt(2π)) · exp(−(2πΔ·T2)^2/2)",
            r"",
            r"Super-Lorentzian (powder average over orientations θ, dipolar ∝ 3cos²θ−1):",
            r"  g(2πΔ) = ∫₀^{π/2} dθ sinθ √(2/π) · T2/|3cos²θ−1| · exp(−2(2πΔ·T2/|3cos²θ−1|)²)",
        ],
        r"""g(2\pi\Delta) = \frac{T_2}{\pi}\,\frac{1}{1+(2\pi\Delta\,T_2)^2}
\qquad
g(2\pi\Delta) = \frac{T_2}{\sqrt{2\pi}}\,e^{-\frac{(2\pi\Delta\,T_2)^2}{2}}

g(2\pi\Delta) = \int_{0}^{\pi/2} d\theta\,\sin\theta\;\sqrt{\frac{2}{\pi}}\;\frac{T_2}{|3\cos^2\theta-1|}\;\exp\!\left(-2\left(\frac{2\pi\Delta\,T_2}{|3\cos^2\theta-1|}\right)^{2}\right)""",
        (
            "<b>Symbols.</b>  g(2πΔ) = absorption lineshape at frequency offset Δ (Hz) — the "
            "saturation-rate weighting of the semi-solid pool;  T<sub>2</sub> = transverse "
            "relaxation time of the pool (µs — ~8–20 µs for the semi-solid pool, vs ms for "
            "liquid);  θ = orientation of the molecular symmetry axis relative to B<sub>0</sub>, "
            "averaged over the powder distribution;  Δω = 2πΔ = RF offset in rad·s⁻¹.<br><br>"
            "The magic-angle factor (3cos²θ − 1) vanishes at θ ≈ 54.7°, giving the "
            "super-Lorentzian its characteristic cusp; it is handled numerically by clamping the "
            "denominator away from zero."
        ),
        [
            ("Morrison C, Stanisz G, Henkelman RM. Modeling magnetization transfer for "
             "biological-like systems using a semi-solid pool with a super-Lorentzian lineshape "
             "and dipolar reservoir. J Magn Reson B. 1995;108(2):103–113.",
             "https://doi.org/10.1006/jmrb.1995.1012",
             "references/superlorentzian/henkelman.pdf"),
            ("Morrison C, Henkelman RM. A model for magnetization transfer in tissues. "
             "Magn Reson Med. 1995;33(4):475–482.",
             "https://doi.org/10.1002/mrm.1910330404",
             "references/superlorentzian/A Model for Magnetization Transfer in Tissues.pdf"),
            ("Henkelman RM, Huang X, Xiang QS, Stanisz GJ, Swanson SD, Bronskill MJ. Quantitative "
             "interpretation of magnetization transfer. Magn Reson Med. 1993;29(6):759–766.",
             "https://doi.org/10.1002/mrm.1910290607",
             "references/superlorentzian/Quantitative interpretation of magnetization transfer.pdf"),
            ("Glutamate-weighted CEST contrast after removal of magnetization transfer effect in "
             "human brain and rat brain with tumor. Mol Imaging Biol. 2020;22(3):724–734. "
             "(modified Lorentzian / Gaussian / super-Lorentzian MT fitting forms).",
             "https://doi.org/10.1007/s11307-019-01465-9",
             "references/superlorentzian/Glutamate Weighted CEST Contrast After Removal of Magnetization Transfer Effect in Human Brain and Rat Brain with Tumor.pdf"),
        ],
        {   # typeset lineshape derivation + figures (Morrison / Stanisz / Henkelman)
            "math": [
                {"text": "<b>Two-pool model.</b>  Magnetization transfer is modelled as exchange "
                         "(rate R) between a mobile <i>liquid</i> pool A (water) and a <i>semi-solid</i> "
                         "pool B (membranes, myelin), each with longitudinal relaxation R<sub>A/B</sub> "
                         "and RF saturation R<sub>rfA/B</sub>. The two pools differ mainly in the "
                         "lineshape of that RF saturation:"},
                {"img": "references/superlorentzian/fig_2pool.png", "figure": True},
                {"text": "<b>Base lineshapes.</b>  Rapid, isotropic motion averages the dipolar "
                         "coupling to zero, so the mobile pool has a narrow <b>Lorentzian</b> line; a "
                         "rigid, static system gives a broad <b>Gaussian</b>:"},
                {"img": "references/superlorentzian/eq/lorentzian.png", "num": "[4]"},
                {"img": "references/superlorentzian/eq/gaussian.png", "num": "[5]"},
                {"text": "<b>Where the super-Lorentzian comes from.</b>  In partially-ordered tissue "
                         "the residual dipolar coupling is <i>not</i> averaged to zero; it scales with "
                         "the orientation θ of the molecular symmetry axis to B<sub>0</sub> as"},
                {"img": "references/superlorentzian/eq/dipolar.png", "num": "[7]"},
                {"text": "Summing (powder-averaging) the elementary lineshape f over all orientations "
                         "θ therefore gives"},
                {"img": "references/superlorentzian/eq/sl_general.png", "num": "[8]"},
                {"text": "and, taking f to be Gaussian (as usually assumed), this becomes the "
                         "<b>super-Lorentzian</b> lineshape used for the semi-solid pool:"},
                {"img": "references/superlorentzian/eq/superlorentzian.png", "num": "[9]"},
                {"img": "references/superlorentzian/fig_lineshapes.png", "figure": True},
                {"text": "Absorption lineshapes g(2πΔ) on a log-frequency axis: (a) Lorentzian, "
                         "(b) super-Lorentzian, (c) Gaussian. The super-Lorentzian is far broader "
                         "than the Lorentzian, matching the semi-solid pool's saturation profile."},
                {"text": "<b>MT modelling by different lineshapes (fitting form).</b>  In practice "
                         "the MT effect dominates the Z-spectrum far off-resonance (beyond ±14 ppm "
                         "at 7 T, where CEST/NOE/direct-saturation are negligible), so those partial "
                         "Z-spectra are fitted with a single semi-solid MT component using one of "
                         "the three lineshapes below — each written with an explicit amplitude A and "
                         "the MT-pool offset w<sub>0</sub>:"},
                {"img": "references/superlorentzian/eq/lor_mod.png", "num": "[4]"},
                {"img": "references/superlorentzian/eq/gauss_mod.png", "num": "[5]"},
                {"img": "references/superlorentzian/eq/sl_mod.png", "num": "[6]"},
                {"text": "where w = frequency offset from the water resonance, w<sub>0</sub> = offset "
                         "of the MT pool, θ = dipolar-Hamiltonian angle, A = scaling factor, and "
                         "T<sub>2m</sub> = a lineshape time constant (which is <i>not</i> the true "
                         "T<sub>2</sub> of the MT pool, since the fit is to Z-spectrum data). The "
                         "Lorentzian falls off slowest, the super-Lorentzian is intermediate, and "
                         "the Gaussian falls off fastest — so the chosen lineshape sets how the MT "
                         "background is extrapolated under the CEST peaks."},
            ],
        },
    ),

    (
        "QUESP Fit",
        "#f9e2af",
        [],
        r"",
        (
            "<b>QUESP</b> (Quantification of Exchange rate using Saturation Power) sweeps the "
            "saturation amplitude B<sub>1</sub> at a fixed offset and fits the resulting CEST "
            "power series to recover the solute proton fraction <b>f<sub>s</sub></b> and the "
            "exchange rate <b>k<sub>sw</sub></b> at the same time. QUESP Analysis: "
            "a <i>regular</i> fit of MTR<sub>asym</sub>, and an <i>inverse</i> fit of the "
            "exchange-only metric MTR<sub>Rex</sub> whose reciprocal — the Ω-plot — is linear in "
            "1/ω<sub>1</sub><sup>2</sup> and returns both parameters independent of solute "
            "concentration."
        ),
        [
            ("Zaiss M., Angelovski G., Demetriou E., McMahon M.T., Golay X., Scheffler K. (2018). "
             "QUESP and QUEST revisited — fast and accurate quantitative CEST experiments. "
             "Magn. Reson. Med. 79, 1708–1721.",
             "https://doi.org/10.1002/mrm.26813",
             "references/quesp/QUESP and QUEST revisited fast and accurate quantitative CEST experiments.pdf"),
            ("McMahon M.T., Gilad A.A., Zhou J., Sun P.Z., Bulte J.W.M., van Zijl P.C.M. (2006). "
             "Quantifying exchange rates in chemical exchange saturation transfer agents using "
             "the saturation time and saturation power dependencies of the magnetization "
             "transfer effect on the MRI signal (QUEST and QUESP). Magn. Reson. Med. 55, 836–847.",
             "https://doi.org/10.1002/mrm.20818",
             "references/quesp/Quantifying exchange rates in chemical exchange saturation transfer agents.pdf"),
            ("Dixon W.T., Ren J., Lubag A.J.M., Ratnakar J., Vinogradov E., Hancu I., Lenkinski "
             "R.E., Sherry A.D. (2010). A concentration-independent method to measure exchange "
             "rates in PARACEST agents (the Ω-plot). Magn. Reson. Med. 63, 625–632.",
             "https://doi.org/10.1002/mrm.22242",
             "references/quesp/A Concentration Independent Method to Measure Exchange Rates in PARACEST Agents.pdf"),
            ("Woessner D.E., Zhang S., Merritt M.E., Sherry A.D. (2005). Numerical solution of "
             "the Bloch equations provides insights into the optimum design of PARACEST agents. "
             "Magn. Reson. Med. 53, 790–799.",
             "https://doi.org/10.1002/mrm.20408",
             "references/quesp/Numerical Solution of the Bloch Equations Provides Insights Into the Optimum Design of PARACEST Agents for MRI.pdf"),
        ],
        {   # QUESP & QUEST revisited (Zaiss et al.) — figure first, eqs [1]-[9]
            "math": [
                {"img": "references/quesp/fig/sequence.png", "figure": True},
                {"text": "<b>Figure 1.</b> The sequence diagram of a typical CEST experiment consists of three modules: a recovery module of duration t<sub>rec</sub>, a saturation module of duration t<sub>p</sub>, and an acquisition module of duration T<sub>A</sub>. (b) Magnetization course during continuous-wave irradiation of amplitude B<sub>1</sub>: if t<sub>rec</sub> ≪ 5 T<sub>1a</sub>, the Z-magnetization recovers to M<sub>i</sub> ≠ M<sub>0</sub>; if t<sub>p</sub> ≪ 5 T<sub>1a</sub>, the M<sub>sat</sub> can depend on M<sub>i</sub>. (c) Magnetization course for far off-resonant irradiation: the course during t<sub>p</sub> is also approximately governed by T<sub>1</sub> and therefore approaches M<sub>0</sub>. If t<sub>rec</sub> + t<sub>p</sub> ≪ 5 T<sub>1</sub>, this value can still be M<sub>offres</sub> < M<sub>0</sub>. Given that Z(Δω) = M<sub>sat</sub>(Δω)/M<sub>0</sub>, the Z-value notation of the z-magnetization is used in the rest of the text."},
                {"text": "We consider a two-pool system of the water pool (Pool a) with thermal magnetization M<sub>0a</sub>, and the CEST pool (Pool b) with thermal magnetization M<sub>0b</sub>, and the relative fraction f<sub>b</sub> = M<sub>0b</sub>/M<sub>0a</sub>. We assume that the initial magnetization at thermal equilibrium is M<sub>i</sub> = M<sub>0</sub>; thus, Z<sub>i</sub> = 1. It can be shown that then and only then the CEST effect can be described quantitatively by Equation [1]:"},
                (r"MTR_{asym}(\alpha(B_1), t_p) = \frac{R_{ex}^{lab}}{R_{1a} + R_{ex}^{lab}}\left(1 - e^{-(R_1 + R_{ex}^{lab})t_p}\right)", ""),
                (r"\qquad = \frac{f_b k_b\,\alpha}{R_{1a} + f_b k_b\,\alpha}\left(1 - e^{-(R_{1a} + f_b k_b\,\alpha)t_p}\right)", "[1]"),
                {"text": "For the so-called labeling efficiency α, different limits are published; we assume here large shifts between water and the CEST pool, then α reads as in Equation [2], where k<sub>b</sub> is the exchange rate, R<sub>2b</sub> is the transversal relaxation rate of the CEST pool, and ω<sub>1</sub> = γB<sub>1</sub> is the radiofrequency saturation amplitude:"},
                (r"\alpha(B_1) = \frac{\omega_1^2}{\omega_1^2 + k_b(k_b + R_{2b})}", "[2]"),
                {"text": "In the original QUEST/QUESP paper, the authors introduced MTR<sub>asym</sub>, given by Equation [3]:"},
                (r"MTR_{asym} = \frac{f_b k_b\,\alpha}{R_{1a} + f_b k_b}\left(1 - e^{-(R_{1a} + f_b k_b)t_p}\right)", "[3]"),
                {"text": "When comparing Equation [3] to Equation [1], two additional α factors appear that scale the product of the fractional concentration and the exchange rate f<sub>b</sub>k<sub>b</sub>."},
                {"text": "Using Equation [1] or [3], two experiments can be designed to quantify exchange rates from the water. A QUESP experiment can be understood as acquisition of MTR<sub>asym</sub>(B<sub>1</sub>) for varying B<sub>1</sub> at a fixed saturation duration t<sub>p</sub>, whereas a QUEST experiment can be understood as MTR<sub>asym</sub>(t<sub>p</sub>) for varying t<sub>p</sub> at a fixed saturation amplitude B<sub>1</sub>. Equation [3] can be used for arbitrary saturation times t<sub>p</sub>; the revised QUESP Equations [4] and [5], respectively, hold in steady state (i.e., t<sub>p</sub> → ∞):"},
                (r"MTR_{asym} = \frac{f_b k_b\,\alpha}{R_{1a} + f_b k_b}", "[4]"),
                (r"MTR_{asym} = \frac{f_b k_b\,\alpha}{R_{1a} + f_b k_b\,\alpha}", "[5]"),
                {"text": "This theory can be used in two ways to yield correct estimates of exchange rates. First, if the recovery time t<sub>rec</sub> (the delay before saturation, Fig. 1) is long enough that the initial magnetization is fully relaxed (Z<sub>i</sub> = M<sub>i</sub>/M<sub>0</sub> = 1), then arbitrary saturation times t<sub>p</sub> can be used and fitted by Equation [1]. Second, if the saturation time t<sub>p</sub> is long enough (> 3 T<sub>1a</sub>) that the saturation steady state is reached and is therefore independent of Z<sub>i</sub>, then exchange-rate quantification via QUESP experiments is possible using Equations [4] and [5]."},
                {"text": "<b>Ω-plot methods for steady-state QUESP</b>"},
                {"text": "In saturation steady state, using the inverse asymmetry MTR<sub>Rex</sub>, exchange rates can also be calculated using Equation [6], which eliminates spillover and semisolid magnetization transfer, and relates MTR<sub>Rex</sub> to k<sub>b</sub>, ω<sub>1</sub>, f<sub>b</sub>, and R<sub>1a</sub> as follows:"},
                (r"MTR_{Rex} = \frac{1}{Z_{lab}(B_1)} - \frac{1}{Z_{ref}(B_1)} = \frac{1}{R_{1a}}\,f_b k_b\,\frac{\omega_1^2}{\omega_1^2 + k_b^2}", "[6]"),
                {"text": "In addition, using 1/MTR<sub>Rex</sub>, the exchange rate can be obtained from the x-intercept of a plot of steady-state CEST intensity as a function of 1/ω<sub>1</sub><sup>2</sup>. The derived Equation [7] was given by Meissner et al. as follows:"},
                (r"y\!\left(\frac{1}{\omega_1^2}\right) = \frac{1}{\dfrac{1}{Z_{lab}} - \dfrac{1}{Z_{ref}}} = \frac{R_{1a}}{f_b k_b} + \frac{R_{1a} k_b}{f_b}\,\frac{1}{\omega_1^2}", "[7]"),
                {"text": "Equation [7] is referred to as the Ω-plot method and was originally introduced by Dixon et al. The original Ω-plot formula is given by Equation [8], a special case of Equation [7] for Z<sub>ref</sub> = 1:"},
                (r"y\!\left(\frac{1}{\omega_1^2}\right) = \frac{Z_{lab}}{1 - Z_{lab}} = \frac{1}{\dfrac{1}{Z_{lab}} - 1} = \frac{R_{1a}}{f_b k_b} + \frac{R_{1a} k_b}{f_b}\,\frac{1}{\omega_1^2}", "[8]"),
                {"text": "<b>Analytical solution for arbitrary initial magnetization M<sub>i</sub></b>"},
                {"text": "The theory described above can be extended to account for non-thermal-equilibrium initial magnetization Z<sub>i</sub>, resulting in Equation [9], which shows an explicit dependency on the initial magnetization before the saturation module, Z<sub>i</sub> = M<sub>i</sub>/M<sub>0</sub>:"},
                (r"MTR_{asym} = Z_{ref}(t_p, B_1) - Z_{lab}(t_p, B_1)", ""),
                (r"\qquad = \frac{f_b k_b\,\alpha}{R_{1a} + f_b k_b\,\alpha} + (Z_i - 1)\,e^{-R_{1a}t_p}", ""),
                (r"\qquad\quad -\, \left(Z_i - \frac{R_{1a}}{R_{1a} + f_b k_b\,\alpha}\right)e^{-(R_{1a} + f_b k_b\,\alpha)t_p}", "[9]"),
                {"text": "Therefore, for fast and accurate quantitative experiments, we suggest the following: (1) measure M<sub>0</sub> for a recovery time t<sub>rec</sub> equal to 5 T<sub>1a</sub> and use it to normalize all Z-spectra (alternatively, M<sub>far-offres</sub> at very long saturation time, 5 T<sub>1a</sub>, can serve as an M<sub>0</sub> estimate); (2) measure the initial magnetization right before saturation, Z<sub>i</sub> = M<sub>i</sub>/M<sub>0</sub> — obtained by running the sequence with the same recovery timing but removing the saturation block; (3) measure the Z-spectra with decreased recovery and saturation times as long as the SNR is sufficient; (4) measure the T<sub>1</sub> of the sample; and (5) fit the data using the full Bloch–McConnell equations, or Equation [9], with the measured R<sub>1a</sub>, M<sub>i</sub>, and M<sub>0</sub>."},
            ],
        },
    ),

    (
        "T1 / T2 / B1 Map",
        "#94e2d5",
        [   # plain-text fallback (shown only if the typeset images are unavailable)
            r"T1 Map:  S(TR) = M0 · ( 1 − exp(−TR/T1) )",
            r"T2 Map:  S(TE) = M0 · exp(−TE/T2)",
            r"",
            r"B1 Map (Double Angle Method):",
            r"  I2(r)/I1(r) = sin α2(r)·f2(T1,TR) / ( sin α1(r)·f1(T1,TR) )",
            r"  α(r) = arccos( | I2(r) / (2·I1(r)) | )   →   B1 [%] = α(r)/α_nominal · 100",
        ],
        r"""S(\mathrm{TR}) = M_0\left(1 - e^{-\mathrm{TR}/T_1}\right)
\qquad
S(\mathrm{TE}) = M_0\,e^{-\mathrm{TE}/T_2}

\dfrac{I_2(r)}{I_1(r)} = \dfrac{\sin\alpha_2(r)\,f_2(T_1,\mathrm{TR})}{\sin\alpha_1(r)\,f_1(T_1,\mathrm{TR})}
\qquad
\alpha(r) = \arccos\!\left(\left|\dfrac{I_2(r)}{2\,I_1(r)}\right|\right)""",
        (
            "<b>Symbols.</b>  S = image signal;  M<sub>0</sub> = equilibrium magnetisation;  "
            "TR = repetition time;  TE = echo time;  T<sub>1</sub>, T<sub>2</sub> = relaxation "
            "times;  I<sub>1</sub>, I<sub>2</sub> = the two double-angle images acquired at tip "
            "angles α<sub>1</sub> and α<sub>2</sub> = 2α<sub>1</sub>;  α(r) = actual per-voxel tip "
            "angle (a map of the B<sub>1</sub><sup>+</sup> field);  f<sub>1</sub>, f<sub>2</sub> = "
            "the T<sub>1</sub>/TR relaxation weightings of each acquisition.  "
            "B<sub>1</sub> [%] = α(r)/α<sub>nominal</sub> × 100 (100 % = perfect calibration)."
        ),
        [
            ("Cunningham CH, Pauly JM, Nayak KS. Saturated double-angle method for rapid B1+ "
             "mapping. Magn Reson Med. 2006;55(6):1326–1333.",
             "https://doi.org/10.1002/mrm.20896",
             "references/t1t2b1/Saturated double-angle method for rapid B1 mapping.pdf"),
            ("Sacolick LI, Wiesinger F, Hancu I, Vogel MW. B1 mapping by Bloch-Siegert shift. "
             "Magn Reson Med. 2010;63(5):1315–1322.",
             "https://doi.org/10.1002/mrm.22357",
             "references/t1t2b1/blochsiegert.pdf"),
            ("Yarnykh VL. Actual flip-angle imaging in the pulsed steady state: a method for rapid "
             "three-dimensional mapping of the transmitted radiofrequency field. Magn Reson Med. "
             "2007;57(1):192–200.",
             "https://doi.org/10.1002/mrm.21120",
             "references/t1t2b1/Actual flip-angle imaging in the pulsed steady state.pdf"),
            ("Insko EK, Bolinger L. Mapping of the radiofrequency field. J Magn Reson A. "
             "1993;103(1):82–85.",
             "https://doi.org/10.1006/jmra.1993.1020",
             "references/t1t2b1/Mapping of the RF field.pdf"),
            ("Haase A, Frahm J, Matthaei D, Hänicke W, Merboldt K-D. FLASH imaging — rapid NMR "
             "imaging using low flip-angle pulses. J Magn Reson. 1986;67(2):258–266.",
             "https://doi.org/10.1016/0022-2364(86)90433-6",
             "references/t1t2b1/FLASH imaging.pdf"),
            ("Meiboom S, Gill D. Modified spin-echo method for measuring nuclear relaxation times. "
             "Rev Sci Instrum. 1958;29(8):688–691.",
             "https://doi.org/10.1063/1.1716296",
             "references/t1t2b1/Modified SpinEcho Method for Measuring Nuclear Relaxation Times.pdf"),
        ],
        {   # typeset T1 / T2 recovery + double-angle B1 mapping
            "math": [
                {"text": "<b>T1 Map</b> — the signal recovers as the repetition time TR is varied "
                         "(variable-TR / saturation-recovery); a mono-exponential fit gives "
                         "T<sub>1</sub> and M<sub>0</sub>:"},
                {"img": "references/t1t2b1/eq/t1.png", "num": ""},
                {"text": "<b>T2 Map</b> — the signal decays with echo time TE (multi-echo); a "
                         "mono-exponential fit gives T<sub>2</sub>:"},
                {"img": "references/t1t2b1/eq/t2.png", "num": ""},
                {"text": "<b>B1 Map (Double Angle Method).</b>  Two images are acquired with tip "
                         "angles α<sub>1</sub> and α<sub>2</sub> = 2α<sub>1</sub>, all other "
                         "signal-affecting parameters kept constant. For each voxel their magnitude "
                         "ratio satisfies"},
                {"img": "references/t1t2b1/eq/b1_ratio.png", "num": ""},
                {"text": "If the T<sub>1</sub>/T<sub>2</sub> weighting is made equal for both "
                         "acquisitions (a long TR, or a magnetisation-reset / <i>saturated</i> DAM so "
                         "f<sub>1</sub> = f<sub>2</sub>), the actual per-voxel tip angle — i.e. the "
                         "B<sub>1</sub><sup>+</sup> map — follows directly from the ratio:"},
                {"img": "references/t1t2b1/eq/b1_alpha.png", "num": ""},
                {"text": "<b>B1 Map (Bloch-Siegert shift).</b>  An alternative to the double-angle "
                         "method. A strong RF pulse applied far off-resonance (kHz range) does not "
                         "excite the spins but shifts their precession frequency — the "
                         "<i>Bloch-Siegert shift</i> — producing an image <b>phase</b> that is "
                         "proportional to B<sub>1</sub>²:"},
                {"img": "references/t1t2b1/eq/bs_phase.png", "num": ""},
                {"text": "K<sub>BS</sub> is a pulse-specific constant (rad·G⁻²). Taking the phase "
                         "<i>difference</i> of two acquisitions with the RF pulse applied "
                         "symmetrically at +ω<sub>RF</sub> and −ω<sub>RF</sub> cancels unwanted "
                         "off-resonance phase, and B<sub>1</sub> follows directly:"},
                {"img": "references/t1t2b1/eq/bs_b1.png", "num": ""},
                {"text": "where φ<sub>BS</sub> = Bloch-Siegert phase shift, γ = gyromagnetic ratio, "
                         "B<sub>1</sub>(t) = RF envelope (peak B<sub>1,peak</sub>), ω<sub>RF</sub> = "
                         "off-resonance frequency, and T = pulse duration. Unlike the DAM the "
                         "Bloch-Siegert method is largely insensitive to T<sub>1</sub>, flip angle "
                         "and chemical shift, and needs no magnitude calibration."},
            ],
        },
    ),

    (
        "PLOF Fit",
        "#fe8019",
        [   # plain-text fallback (shown only if the typeset images are unavailable)
            r"Steady-state 2-peak PLOF (R1rho relaxation theory, Chen et al.):",
            r"  Z_ss     =  cos^2(theta) . R1 / R1rho",
            r"  R1rho    =  R_eff + R_back + R_peak1 + R_peak2",
            r"  R_eff    =  cos^2(theta) R1 + sin^2(theta) R2 ,   theta = atan(w1/Delta)",
            r"",
            r"  R_peakN  =  R_peakN_max . (w_peakN/2)^2 / [ (w_peakN/2)^2 + (Delta - Delta_peakN)^2 ]",
            r"  R_back   =  D0 + D1(Delta-2) + D2(Delta-2)^2 + D3(Delta-2)^3",
        ],
        r"""Z^{ss} = \frac{\cos^{2}\theta\, R_{1}}{R_{1\rho}}
\qquad R_{1\rho} = R_{\mathrm{eff}} + R_{\mathrm{back}} + R_{\mathrm{peak1}} + R_{\mathrm{peak2}}

R_{\mathrm{eff}} = \cos^{2}\theta\, R_{1} + \sin^{2}\theta\, R_{2}, \qquad \theta = \tan^{-1}(\omega_{1}/\Delta)

R_{\mathrm{peak}n} = R_{\mathrm{peak}n}^{\max}\,\frac{(w_{\mathrm{peak}n}/2)^{2}}{(w_{\mathrm{peak}n}/2)^{2}+(\Delta-\Delta_{\mathrm{peak}n})^{2}}

R_{\mathrm{back}} = D_{0} + D_{1}(\Delta-2) + D_{2}(\Delta-2)^{2} + D_{3}(\Delta-2)^{3}""",
        (
            "In the R<sub>1ρ</sub> (rotating-frame) relaxation framework — which is largely "
            "insensitive to the direct water-saturation spillover that biases MTR-based metrics — "
            "the steady-state water saturation signal, normalised by its unsaturated value "
            "S<sub>0</sub>, is Z<sup>ss</sup> = cos²θ·R<sub>1</sub>/R<sub>1ρ</sub>.<br><br>"
            "<b>Symbols.</b>  Δ = frequency offset (ppm);  ω<sub>1</sub> = saturation nutation rate;  "
            "θ = tan⁻¹(ω<sub>1</sub>/Δ) = tilt angle of the effective magnetisation about z;  "
            "R<sub>1</sub>, R<sub>2</sub> = water longitudinal / transverse relaxation rates (s⁻¹);  "
            "R<sub>1ρ</sub> = water relaxation rate under RF saturation;  "
            "R<sub>eff</sub> = rotating-frame water rate without solutes;  "
            "R<sub>back</sub> = background (direct saturation, semi-solid MT, aromatic protons, "
            "other metabolites);  R<sub>peak1</sub>, R<sub>peak2</sub> = the two targeted CEST peaks "
            "R<sub>peak</sub><sup>max</sup> = peak apparent relaxation rate;  "
            "w<sub>peak</sub> = peak FWHM;  Δ<sub>peak</sub> = peak chemical-shift offset;  "
            "D<sub>0</sub>–D<sub>3</sub> = zeroth- to third-order polynomial coefficients."
        ),
        [
            ("Chen L, et al. Creatine and phosphocreatine mapping of mouse skeletal muscle by a "
             "polynomial and Lorentzian line-shape fitting method. Magn Reson Med. "
             "2019;81(1):69–78.",
             "https://doi.org/10.1002/mrm.27514",
             "references/plof/Chen Creatine and phosphocreatine mapping of mouse skeletal muscle by a polynomial and.pdf"),
            ("Chen L, et al. High-resolution creatine mapping of mouse brain at 11.7 T using "
             "non-steady-state chemical exchange saturation transfer. NMR Biomed. 2019;32:e4168.",
             "https://doi.org/10.1002/nbm.4168",
             "references/plof/Chen High-resolution creatine mapping of mouse brain at 11 7 T using non-steady-state.pdf"),
            ("Chen L, et al. Investigation of the contribution of total creatine to the CEST "
             "Z-spectrum of brain using a knockout mouse model. NMR Biomed. 2017;30:e3834.",
             "https://doi.org/10.1002/nbm.3834",
             "references/plof/Chen Investigation of the contribution of total creatine to the CEST Zspectrum of brain.pdf"),
        ],
        {   # ── typeset equations: steady-state 2-peak PLOF (Chen et al.) ─────
            "math": [
                {"text": "<b>PLOF</b> (polynomial and Lorentzian line-shape fitting) works in the "
                         "R<sub>1ρ</sub> relaxation framework and is robust against the water "
                         "direct-saturation spillover. This two-peak variant extracts "
                         "two CEST signals simultaneously. "
                         "At steady state the normalised saturation signal for each offset is"},
                {"img": "references/plof/eq/zss.png", "num": "(1)"},
                {"text": "and the rotating-frame relaxation rate is decomposed into an "
                         "effective-water term, a smooth background, and two solute peaks:"},
                {"img": "references/plof/eq/r1rho.png", "num": "(2)"},
                {"img": "references/plof/eq/reff.png", "num": ""},
                {"text": "Each targeted resonance is modelled as a Lorentzian line-shape"},
                {"img": "references/plof/eq/rpeak1.png", "num": "(3)"},
                {"img": "references/plof/eq/rpeak2.png", "num": "(4)"},
                {"text": "and the background is a third-order polynomial in the frequency offset"},
                {"img": "references/plof/eq/rback.png", "num": "(5)"},
                {"text": "The Z-spectrum is fitted with Eqs. (1)–(2) using the two Lorentzians (3)–(4) "
                         "for R<sub>peak1</sub>, R<sub>peak2</sub> and the polynomial (5) for "
                         "R<sub>back</sub>; the fitted peak amplitudes gives two CEST parametric maps."},
            ],
        },
    ),

    (
        "DROF Fit",
        "#e0af68",
        [   # plain-text fallback (shown only if the typeset images are unavailable)
            r"Rotating-frame Z-spectrum (θ = atan(ω1/Δω)):",
            r"  Mz(t) = (M0 cos²θ − Mss) e^(−R1ρ t) + Mss ,   Mss = M0 R1w cos²θ / R1ρ",
            r"  Z(Δω) = Mz(tsat)/M0 = (cos²θ − R1w cos²θ/R1ρ) e^(−R1ρ tsat) + R1w cos²θ/R1ρ",
            r"",
            r"PLOF:  R1ρ = C0' + L_water + Σ Cn·Δω^n + L_amide + L_creatine",
            r"DROF:  R1ρ = C0' + L_water + L_rNOE + L_MT + L_amide + L_creatine",
        ],
        r"""M_z(t) = \left(M_0\cos^2\theta - M_{ss}\right)e^{-R_{1\rho}t} + M_{ss}
\qquad M_{ss} = \frac{M_0\,R_{1w}\cos^2\theta}{R_{1\rho}}

Z(\Delta\omega) = \frac{M_z(t_{\mathrm{sat}})}{M_0} = \left(\cos^2\theta - \frac{R_{1w}\cos^2\theta}{R_{1\rho}}\right)e^{-R_{1\rho}t_{\mathrm{sat}}} + \frac{R_{1w}\cos^2\theta}{R_{1\rho}}

R_{1\rho} = C_0' + L_{\mathrm{water}} + L_{\mathrm{rNOE}} + L_{\mathrm{MT}} + L_{\mathrm{amide}} + L_{\mathrm{creatine}}""",
        (
            "<b>DROF</b> (Double-step R<sub>1ρ</sub>-based Lorentzian Fitting) analyses the "
            "Z-spectrum in the rotating frame, where every pool — water direct saturation, MT, "
            "relayed-NOE and each CEST peak — is a Lorentzian in the relaxation rate "
            "R<sub>1ρ</sub>.<br><br>"
            "<b>Symbols.</b>  Z(Δω) = M<sub>z</sub>(t<sub>sat</sub>)/M<sub>0</sub> = normalised "
            "saturated signal;  θ = tan⁻¹(ω<sub>1</sub>/Δω) = tilt of the effective field off z;  "
            "ω<sub>1</sub> = γB<sub>1</sub> = saturation amplitude;  R<sub>1w</sub> = water "
            "longitudinal relaxation rate;  R<sub>1ρ</sub> = longitudinal relaxation rate in the "
            "rotating frame (under saturation);  M<sub>ss</sub> = steady-state magnetisation;  "
            "t<sub>sat</sub> = saturation duration;  L<sub>pool</sub> = a pool's Lorentzian;  "
            "C<sub>0</sub>′, C<sub>n</sub> = PLOF polynomial background coefficients."
        ),
        [
            ("Zhang H, Zeng S, Wang J, Cai P, Wang Z, Chen L, et al. Double-step R1ρ-based "
             "Lorentzian fitting (DROF): a new CEST analysis approach and its comparison with "
             "existing methods. NMR Biomed. 2025;38(4):e70082.",
             "https://doi.org/10.1002/nbm.70082",
             "references/drof/DROF.pdf"),
        ],
        {   # typeset R1ρ rotating-frame derivation + PLOF→DROF (Zhang et al. 2025)
            "math": [
                {"text": "<b>Rotating-frame Z-spectrum.</b>  Under off-resonant saturation the "
                         "effective field is tilted by θ = tan⁻¹(ω<sub>1</sub>/Δω) off the z-axis; "
                         "the water magnetisation relaxes bi-exponentially at the rotating-frame "
                         "rate R<sub>1ρ</sub> toward a steady state M<sub>ss</sub>:"},
                {"img": "references/drof/eq/mz.png", "num": "[8]"},
                {"img": "references/drof/eq/mss.png", "num": "[9]"},
                {"text": "so after a saturation pulse of duration t<sub>sat</sub> the measured "
                         "Z-spectrum is"},
                {"img": "references/drof/eq/zspec.png", "num": "[10]"},
                {"text": "<b>From PLOF to DROF.</b>  R<sub>1ρ</sub> is a sum of pool contributions. "
                         "The earlier <b>PLOF</b> method fits the CEST peaks as Lorentzians but the "
                         "MT background as an N-th-order polynomial:"},
                {"img": "references/drof/eq/plof.png", "num": "[12]"},
                {"text": "<b>DROF</b> instead replaces that polynomial with an explicit "
                         "<b>Lorentzian MT</b> term and adds an upfield <b>relayed-NOE</b> Lorentzian, "
                         "so every pool is a Lorentzian:"},
                {"img": "references/drof/eq/drof.png", "num": "[13]"},
                {"text": "A <b>double-step fitting</b> strategy improves low-concentration pools: "
                         "first a 3-pool fit (water DS, MT, rNOE) on the far offsets "
                         "(Δω ≤ 1 ppm and Δω ≥ 6 ppm); then a 5-pool fit over all offsets with the "
                         "first-step parameters fixed. Using the whole negative-offset range — not "
                         "just peak-free regions as in PLOF — gives a more robust MT lineshape and "
                         "rNOE quantification."},
            ],
        },
    ),

    (
        "Inverse Z-Spectroscopy",
        "#a6e3a1",
        [],
        r"",
        (
            "Under prolonged (continuous-wave) irradiation, chemical exchange shows up as an "
            "extra, exchange-driven <i>relaxation channel</i> in the rotating frame. Re-casting "
            "the measured Z-spectrum into its reciprocal (1/Z) turns this into a linear problem, "
            "so the individually small signals of amide / amine CEST pools, upfield rNOEs and the "
            "broad semi-solid MT background can be untangled and quantified one at a time. The "
            "steps below trace that formulation from the saturation-power (QUESP) description "
            "through to the final 1/Z fit."
        ),
        [
            ("Zaiss M. & Bachert P. (2013). Exchange-dependent relaxation in the rotating "
             "frame for slow and intermediate exchange — modeling off-resonant spin-lock "
             "and CEST. NMR Biomed. 26, 507–518.",
             "https://doi.org/10.1002/nbm.2887",
             "references/inversez/Zaiss Exchange dependent relaxation in the rotating frame for slow and intermediate exchange.pdf"),
            ("Zaiss M. et al. (2014). Inverse Z-spectrum analysis for spillover-, MT-, and "
             "T1-corrected steady-state pulsed CEST-MRI — application to pH-weighted MRI of "
             "acute stroke. NMR Biomed. 27, 240–252.",
             "https://doi.org/10.1002/nbm.3054",
             "references/inversez/Inverse Zspectrum analysis for spillover, MT and T1corrected steady state pulsed CESTMRI application to pHweighted MRI of acute stroke.pdf"),
            ("Zaiss M. & Bachert P. (2013). Chemical exchange saturation transfer (CEST) "
             "and MR Z-spectroscopy in vivo: a review of theoretical approaches and methods. "
             "Phys. Med. Biol. 58, R221–R269.",
             "https://doi.org/10.1088/0031-9155/58/22/R221",
             "references/inversez/Chemical exchange saturation transfer (CEST) and MR Z spectroscopy in vivo a review of theoretical approaches and methods.pdf"),
            ("CEST MRI quantification of transient ischemia using a combination method "
             "of 5-pool Lorentzian fitting and inverse Z-spectrum analysis. "
             "Quant. Imaging Med. Surg. 13(3), 1860–1873 (2023).",
             "https://doi.org/10.21037/qims-22-420",
             "references/inversez/Chemical exchange saturation transfer (CEST) magnetic resonance imaging (MRI) quantification of transient ischemia using a combination method of 5-pool Lorentzian fitting and inverse Z-spectrum analysis.pdf"),
        ],
        {   # ── typeset derivation (1)–(14): QUESP → 1/Z space (Zaiss et al.) ──
            "math": [
                {"text": "Labelling efficiency α measures how completely the solute protons "
                         "are saturated. It rises with the applied saturation amplitude "
                         "(ω<sub>1</sub> = γB<sub>1</sub>) and is eroded as the exchange rate "
                         "k<sub>sw</sub> grows:"},
                {"img": "references/inversez/eq/eq1.png", "num": "(1)"},
                {"text": "Fitting a set of MTR<sub>asym</sub> values recorded at different "
                         "ω<sub>1</sub> against the analytical Bloch–McConnell solution recovers "
                         "both the exchange rate k<sub>sw</sub> and the solute proton fraction "
                         "f<sub>s</sub>:"},
                {"img": "references/inversez/eq/eq2.png", "num": "(2)"},
                {"text": "Here t<sub>p</sub> is the length of the saturation pulse and "
                         "Z<sub>i</sub> is the starting Z-magnetisation, normalised to the "
                         "equilibrium value M<sub>0</sub>:"},
                {"img": "references/inversez/eq/eq3.png", "num": "(3)"},
                {"text": "with t<sub>rec</sub> the delay left for recovery between "
                         "acquisitions.<br><br>Once a voxel holds several overlapping "
                         "saturation-transfer pools together with upfield rNOEs and a diluting "
                         "MT background, the plain asymmetry metric MTR<sub>asym</sub> "
                         "(Z<sub>ref</sub> − Z<sub>lab</sub>) stops being a clean readout, and "
                         "the amplitudes of the separately fitted peaks become the quantity of "
                         "interest. Since saturation transfer under steady irradiation behaves "
                         "as an exchange-dependent relaxation process, the pool contributions "
                         "are most naturally combined in the reciprocal domain, where they add "
                         "linearly:"},
                {"img": "references/inversez/eq/eq4.png", "num": "(4)"},
                {"text": "At each offset Δω the signal relaxes mono-exponentially toward its "
                         "steady-state value Z<sub>ss</sub>, set by the rotating-frame rate "
                         "R<sub>1ρ</sub>:"},
                {"img": "references/inversez/eq/eq5.png", "num": "(5)"},
                {"text": "When the saturation lasts far longer than 1/R<sub>1ρ</sub> "
                         "(t<sub>sat</sub> ≫ T<sub>1ρ</sub>), the signal has effectively "
                         "settled onto Z<sub>ss</sub>:"},
                {"img": "references/inversez/eq/eq6.png", "num": "(6)"},
                {"text": "θ is the angle by which the effective field is tilted from the "
                         "z-axis in the rotating frame:"},
                {"img": "references/inversez/eq/eq7.png", "num": "(7)"},
                {"text": "For two pools the rotating-frame rate R<sub>1ρ</sub> splits cleanly "
                         "into two additive terms:"},
                {"img": "references/inversez/eq/eq8.png", "num": "(8)"},
                {"text": "R<sub>eff</sub> is the intrinsic relaxation of water along the "
                         "effective field, while R<sub>ex</sub> is the additional decay driven "
                         "by exchange:"},
                {"img": "references/inversez/eq/eq9.png", "num": "(9)"},
                {"text": "Rearranging isolates the exchange term:"},
                {"img": "references/inversez/eq/eq10.png", "num": "(10)"},
                {"text": "R<sub>2</sub>sin²θ describes the transverse relaxation of water and "
                         "R<sub>ex</sub> the exchange contribution; both are Lorentzian in "
                         "shape, which is precisely why Eq. (10) shares the additive form of "
                         "Eq. (4). The right-hand grouping "
                         "R<sub>1</sub>cos²θ (1/Z<sub>ss</sub> − 1) is what we call "
                         "<b>1/Z space</b>."},
                {"text": "Adding further pools — say two CEST pools alongside a semi-solid MT "
                         "pool — simply lengthens the sum:"},
                {"img": "references/inversez/eq/eq11.png", "num": "(11)"},
                {"text": "so, in the steady-state limit, the balance of Eq. (10) generalises "
                         "to"},
                {"img": "references/inversez/eq/eq12.png", "num": "(12)"},
                {"text": "and once the offset lies far from resonance (Δω ≫ ω<sub>1</sub>) it "
                         "collapses to"},
                {"img": "references/inversez/eq/eq13.png", "num": "(13)"},
                {"text": "After the peaks have been fitted in 1/Z, each is mapped back to its "
                         "Z-spectrum amplitude:"},
                {"img": "references/inversez/eq/eq14.png", "num": "(14)"},
                {"text": "Those recovered amplitudes are then passed to Eq. (2) to solve for "
                         "the pool fraction f<sub>b</sub> and its exchange rate k<sub>b</sub>."
                         "<br><br><b>Order of operations:</b> the broad semi-solid MT is "
                         "characterised first, directly in the Z-spectrum, from points well "
                         "off resonance (Super-Lorentzian), then projected into 1/Z and held "
                         "fixed while water and the sharper CEST pools are fitted across "
                         "roughly −2 to +5 ppm."},
            ],
        },
    ),

    (
        "AREX",
        "#74c7ec",
        [   # plain-text fallback (shown only if the typeset images are unavailable)
            r"With MTR = 1 − S_sat/S0 = 1 − Z:",
            r"  CESTR     = (S_ref − S_lab)/S0    = Z_ref − Z_lab",
            r"  CESTR^nr  = (S_ref − S_lab)/S_ref = (Z_ref − Z_lab)/Z_ref",
            r"  MTR_Rex   = (S_ref − S_lab)·S0/(S_ref·S_lab) = 1/Z_lab − 1/Z_ref",
            r"  AREX      = MTR_Rex / T1w = MTR_Rex · R1w",
        ],
        r"""\mathrm{CESTR} = \frac{S_{\mathrm{ref}}-S_{\mathrm{lab}}}{S_{0}} = Z_{\mathrm{ref}}-Z_{\mathrm{lab}}
\qquad
\mathrm{CESTR}^{nr} = \frac{S_{\mathrm{ref}}-S_{\mathrm{lab}}}{S_{\mathrm{ref}}} = \frac{Z_{\mathrm{ref}}-Z_{\mathrm{lab}}}{Z_{\mathrm{ref}}}

\mathrm{MTR}_{\mathrm{Rex}} = \frac{(S_{\mathrm{ref}}-S_{\mathrm{lab}})\,S_{0}}{S_{\mathrm{ref}}\,S_{\mathrm{lab}}} = \frac{1}{Z_{\mathrm{lab}}}-\frac{1}{Z_{\mathrm{ref}}}
\qquad
\mathrm{AREX} = \frac{\mathrm{MTR}_{\mathrm{Rex}}}{T_{1w}} = \frac{Z_{\mathrm{ref}}-Z_{\mathrm{lab}}}{Z_{\mathrm{ref}}\,Z_{\mathrm{lab}}}\cdot\frac{1}{T_{1w}}""",
        (
            "The <b>AREX</b> (Apparent Exchange-dependent Relaxation) family of metrics quantifies "
            "a CEST/NOE peak by comparing a <i>label</i> scan S<sub>lab</sub> (saturation at the "
            "solute offset) with a <i>reference</i> scan S<sub>ref</sub> (the spillover + MT "
            "background at the same offset), both normalised by the unsaturated signal S<sub>0</sub> "
            "— so Z = S<sub>sat</sub>/S<sub>0</sub> and MTR = 1 − Z.<br><br>"
            "Each metric removes more confounds: <b>CESTR</b> = Z<sub>ref</sub> − Z<sub>lab</sub> is "
            "the plain label–reference difference; <b>CESTRⁿʳ</b> divides by a reference scan to "
            "reduce spillover scaling; <b>MTR<sub>Rex</sub></b> works in inverse (1/Z) space so pool "
            "contributions add and direct water-saturation / MT cancel; and <b>AREX</b> multiplies by "
            "R<sub>1w</sub> = 1/T<sub>1w</sub> to also remove the water-T₁ weighting, giving a rate "
            "(s⁻¹) proportional to f<sub>s</sub>·k<sub>sw</sub>.<br><br>"
            "<b>Symbols.</b>  S<sub>lab</sub>, S<sub>ref</sub> = label / reference image intensity;  "
            "S<sub>0</sub> = unsaturated signal;  Z<sub>lab</sub>, Z<sub>ref</sub> = the corresponding "
            "normalised (S/S₀) intensities;  T<sub>1w</sub> (R<sub>1w</sub> = 1/T<sub>1w</sub>) = "
            "water longitudinal relaxation time (rate)."
        ),
        [
            ("Heo HY, Lee DH, Zhang Y, Zhao X, Jiang S, Chen M, Zhou J. Insight into the quantitative "
             "metrics of chemical exchange saturation transfer (CEST) imaging. Magn Reson Med. "
             "2017;77(5):1853–1865.",
             "https://doi.org/10.1002/mrm.26264",
             "references/arex/nihms780471.pdf"),
            ("Zaiss M, Xu J, Goerke S, Khan IS, Singer RJ, Gore JC, Gochberg DF, Bachert P. Inverse "
             "Z-spectrum analysis for spillover-, MT-, and T1-corrected steady-state pulsed CEST-MRI "
             "— application to pH-weighted MRI of acute stroke. NMR Biomed. 2014;27(3):240–252.",
             "https://doi.org/10.1002/nbm.3054",
             "references/arex/nihms700214.pdf"),
            ("Zaiss M, Bachert P. Exchange-dependent relaxation in the rotating frame for slow and "
             "intermediate exchange — modeling off-resonant spin-lock and CEST. NMR Biomed. "
             "2013;26(5):507–518.",
             "https://doi.org/10.1002/nbm.2887",
             "references/arex/Zaiss Exchange dependent relaxation in the rotating frame for slow and intermediate exchange.pdf"),
        ],
        {   # typeset metric chain (Heo et al. 2017): CESTR → CESTRⁿʳ → MTR_Rex → AREX
            "math": [
                {"text": "With MTR = 1 − S<sub>sat</sub>/S<sub>0</sub> = 1 − Z, the standard "
                         "label–reference CEST metric (CESTR) is"},
                {"img": "references/arex/eq/cestr.png", "num": "[2]"},
                {"text": "Normalising by a reference scan rather than S<sub>0</sub> gives the "
                         "reference-normalised form"},
                {"img": "references/arex/eq/cestrnr.png", "num": "[3]"},
                {"text": "In inverse (1/Z) space the metric becomes spillover- and MT-cancelling — "
                         "the intrinsic inverse metric MTR<sub>Rex</sub>:"},
                {"img": "references/arex/eq/mtrrex.png", "num": "[4]"},
                {"text": "and dividing by the water T₁ (multiplying by R<sub>1w</sub>) removes the "
                         "longitudinal-relaxation weighting, yielding AREX — a relaxation rate "
                         "proportional to f<sub>s</sub>·k<sub>sw</sub>:"},
                {"img": "references/arex/eq/arex.png", "num": "[5]"},
            ],
        },
    ),

    (
        "Gaussian Line Fit",
        "#b4befe",
        [],
        r"",
        (
            "Simple 3-parameter Gaussian used for fast voxelwise fitting of "
            "individual CEST peaks in the Z-spectrum or (1−Z) domain.<br><br>"
            "The Gaussian decays more rapidly than a Lorentzian at large offsets "
            "(sub-Gaussian tails), making it suitable for narrow exchange peaks "
            "where the far off-resonance contribution is negligible.<br><br>"
            "For broader or asymmetric peaks the Pseudo-Voigt (Gaussian+Lorentzian "
            "mixture) is preferred.<br><br>"
            "<b>A</b> = amplitude (peak height in Z-attenuation or 1−Z units)<br>"
            "<b>FWHM</b> = full-width at half-maximum (ppm)<br>"
            "<b>Δω₀</b> = peak centre chemical shift offset (ppm)<br>"
            "<b>σ</b> = Gaussian standard deviation (ppm); σ = FWHM/2.355<br><br>"
            "In OCEAN, G(Δω) is implemented as "
            "<code>A · exp(−4·ln2·(Δω−Δω₀)²/FWHM²)</code> "
            "for numerical stability."
        ),
        [
            ("Petrakis L. (1967). Spectral line shapes: Gaussian and Lorentzian "
             "functions in magnetic resonance. J. Chem. Educ. 44(8), 432–436.",
             "https://doi.org/10.1021/ed044p432",
             "references/gaussian/Spectral Line Shapes.pdf"),
        ],
        {   # typeset equations (match the other line-fit boxes)
            "math": [
                {"text": "A simple three-parameter "
                         "Gaussian peak — amplitude A, centre Δω<sub>0</sub> and "
                         "full-width-at-half-maximum:"},
                {"img": "references/gaussian/eq/fwhm.png", "num": "",
                 "tex": r"G(\Delta\omega) = A\,\exp\!\left("
                        r"-\frac{4\ln 2\,(\Delta\omega-\Delta\omega_0)^2}"
                        r"{\mathrm{FWHM}^2}\right)"},
                {"text": "<b>Equivalent σ-parameterisation.</b>"},
                {"img": "references/gaussian/eq/sigma.png", "num": "",
                 "tex": r"G(\Delta\omega) = A\,\exp\!\left("
                        r"-\frac{(\Delta\omega-\Delta\omega_0)^2}{2\sigma^2}\right)"},
                {"text": "with the width conversion"},
                {"img": "references/gaussian/eq/srel.png", "num": "",
                 "tex": r"\sigma = \frac{\mathrm{FWHM}}{2\sqrt{2\ln 2}}"
                        r" \approx \frac{\mathrm{FWHM}}{2.355}"},
                {"text": "The peak value is A at Δω = Δω<sub>0</sub>, and G → 0 as "
                         "|Δω − Δω<sub>0</sub>| → ∞ — i.e. compact (sub-Gaussian) tails "
                         "compared with a Lorentzian."},
            ],
        },
    ),

    # ── WASABI (Water Shift And B1) ────────────────────────────────────────
    (
        "WASABI  (B₀ & B₁ mapping)",
        "#6c5ce7",
        [
            r"── Bloch solution for a short off-resonant block pulse ──",
            r"",
            r"  Mz(tp) = M0·cos α(tp) = M0·[ 1 − 2·sin²θ · sin²(ωeff·tp / 2) ]        [1]",
            r"",
            r"    tan θ = γB₁ / Δω ,    ω_eff = √( (γB₁)² + (Δω)² )",
            r"",
            r"── Normalised Z-spectrum   Z(Δω) = Mz(Δω) / M0 ──",
            r"",
            r"  Z(Δω) = 1 − 2·sin²( tan⁻¹(γB₁ / Δω) ) · sin²( √((γB₁)² + (Δω)²) · tp / 2 )        [2]",
            r"",
            r"    Δω = ωrf − ω0   (RF offset from water Larmor ω0)",
            r"    γ / 2π = 42.578 MHz / T",
            r"",
            r"── WASABI fit model  (baseline c, depth d, B₀ shift δω) ──",
            r"",
            r"  Z(Δω) = c − d·sin²( tan⁻¹( γB₁ / (Δω − δω) ) ) · sin²( √((γB₁)² + (Δω − δω)²) · tp / 2 )   [3]",
            r"",
            r"  Four free parameters:  c, d, B₁, δω",
            r"    c, d → amplitude modulation (offset-independent)",
            r"    B₁   → Z(Δω)                     → relative B₁ map",
            r"    δω   → symmetry axis (B₀ shift)  → ΔB₀ map",
        ],
        # LaTeX
        r"""M_z(t_p) = M_0\cos\alpha(t_p)
        = M_0\left[\,1 - 2\sin^2\theta\,\sin^2\!\left(\tfrac{\omega_{\mathrm{eff}}t_p}{2}\right)\right],
\quad \tan\theta=\frac{\gamma B_1}{\Delta\omega},\;
\omega_{\mathrm{eff}}=\sqrt{(\gamma B_1)^2+(\Delta\omega)^2}

Z(\Delta\omega)=1-2\sin^2\!\left(\tan^{-1}\tfrac{\gamma B_1}{\Delta\omega}\right)
        \sin^2\!\left(\tfrac{t_p}{2}\sqrt{(\gamma B_1)^2+(\Delta\omega)^2}\right)

Z(\Delta\omega)=c-d\,\sin^2\!\left(\tan^{-1}\frac{\gamma B_1}{\Delta\omega-\delta\omega}\right)
        \sin^2\!\left(\frac{t_p}{2}\sqrt{(\gamma B_1)^2+(\Delta\omega-\delta\omega)^2}\right)""",
        (
            "<b>Mathematical Description and Fit Model</b><br><br>"
            "The Bloch equations for short "
            "off-resonant irradiation are solved by neglecting T₁ and T₂ relaxation. "
            "Defining <b>α(t)</b> as the angle between the z-axis and <b>M⃗(t)</b> "
            "(Fig. 1) yields the z-magnetisation after a block pulse of duration "
            "<b>t<sub>p</sub></b>:<br><br>"
            "&nbsp;&nbsp;<b>M<sub>z</sub>(t<sub>p</sub>) = M₀·cos α(t<sub>p</sub>) "
            "= M₀·[1 − 2 sin²θ · sin²(ω<sub>eff</sub>·t<sub>p</sub>/2)]</b>&nbsp;&nbsp;[1]"
            "<br><br>"
            "with <b>tan θ = γB₁/Δω</b> and "
            "<b>ω<sub>eff</sub> = √((γB₁)² + (Δω)²)</b>. Writing "
            "<b>Z(Δω) = M<sub>z</sub>(Δω)/M₀</b> gives Eq. [2].<br><br>"
            "The quantity <b>Δω = ω<sub>rf</sub> − ω₀</b> is the radiofrequency offset "
            "relative to the Larmor frequency ω₀ (for ¹H: ω₀/B₀ = γ = 2π·42.578 MHz/T)."
            "<br><br>"
            "To account for the initial magnetisation and relaxation during the pulse, "
            "the fit parameters <b>c</b> and <b>d</b> are introduced. B₀ inhomogeneity "
            "causes an additional per-voxel frequency shift <b>δω</b>, so the actual "
            "offset in a voxel is <b>Δω − δω</b>, leading to the fit model Eq. [3].<br><br>"
            "With Δω and the pulse duration t<sub>p</sub> as input, this is a model with "
            "<b>four free parameters: c, d, B₁, and δω</b>. While c and d describe solely "
            "the amplitude modulation (independent of the frequency offset), the parameter "
            "<b>B₁ changes the periodicity</b> and <b>δω the symmetry axis</b> of the "
            "function. By sampling Z(Δω) for several frequency offsets around the water "
            "Larmor frequency, both the <b>water-frequency shift δω</b> (→ B₀) and the "
            "<b>B₁ amplitude</b> of the pulse can be determined simultaneously. Passing the "
            "adjusted system resonance frequency converts δω into the absolute water "
            "frequency ω₀, and therefore the actual field strength B₀."
        ),
        [
            ("Schuenke P, Windschuh J, Roeloffs V, Ladd ME, Bachert P, Zaiss M. "
             "Simultaneous mapping of water shift and B1 (WASABI) — Application to "
             "field-inhomogeneity correction of CEST MRI data. Magn Reson Med. "
             "2017 Feb;77(2):571–580. doi:10.1002/mrm.26133. Epub 2016 Feb 9. "
             "PMID: 26857219.",
             "https://doi.org/10.1002/mrm.26133",
             "references/WASABI/WASABI.pdf"),
        ],
        # extra: figure + open-as-PDF / link buttons
        {
            # Publication-quality typeset equations (pre-rendered with LaTeX →
            # Computer-Modern, matching the Bloch–McConnell box exactly).
            "math": [
                {"img": "references/WASABI/eq/eq1.png",  "num": "[1]"},
                {"img": "references/WASABI/eq/with.png", "num": ""},
                {"img": "references/WASABI/eq/eq2.png",  "num": "[2]"},
                {"img": "references/WASABI/eq/fit.png",  "num": ""},
                {"img": "references/WASABI/eq/eq3.png",  "num": "[3]"},
            ],
            "figure": "references/WASABI/wasabi_fig1.png",
            "caption_above": True,
            "figure_caption": (
                "Precession of the magnetisation around the effective field "
                "B_eff.  The rotation angle φ = ω_eff·t_p depends on the pulse duration "
                "t_p, the amplitude B₁ of the excitation field, and the frequency "
                "offset Δω."),
        },
    ),

    # ── WASSR (Water Saturation Shift Referencing) ─────────────────────────
    (
        "WASSR  (B₀ mapping)",
        "#a6e3a1",
        [   # plain-text fallback (shown only if the typeset image is unavailable)
            r"WASSR B0 mapping — Maximum-Symmetry Center Frequency (MSCF):",
            r"  MSCF = argmin_C  < ( f(x_i) − f~(2C − x_i) )^2 >   for  x(1) ≤ 2C − x_i ≤ x(N)",
        ],
        r"""\mathrm{MSCF} = \underset{C}{\operatorname{arg\,min}}\left\langle\bigl(f(x_i)-\tilde{f}(2C-x_i)\bigr)^{2}\right\rangle_{x_{(1)}\le 2C-x_i\le x_{(N)}}""",
        (
            "<b>WASSR (Water Saturation Shift Referencing)</b> maps the per-voxel B<sub>0</sub> "
            "offset from a densely-sampled, low-power Z-spectrum. The direct water-saturation dip "
            "is intrinsically symmetric about the true water frequency, and that symmetry survives "
            "field inhomogeneity — so the local water frequency in each voxel is found as the axis "
            "of symmetry of its measured Z-spectrum.<br><br>"
            "<b>Symbols.</b>  f(x<sub>i</sub>) = measured signal at the sampled WASSR offset "
            "x<sub>i</sub> (N offsets, i = 1…N);  C = estimated centre (water) frequency;  "
            "f̃(2C − x<sub>i</sub>) = the spectrum reflected about C and cubic-spline interpolated "
            "back onto the sampled grid;  ⟨·⟩ = mean over the points whose reflected offset "
            "2C − x<sub>i</sub> stays within the acquired range x<sub>(1)</sub>…x<sub>(N)</sub>.  "
            "The recovered offset (C − water reference) corrects the CEST frequency axis per voxel."
        ),
        [
            ("Kim M, Gillen J, Landman BA, Zhou J, van Zijl PCM. Water saturation shift "
             "referencing (WASSR) for chemical exchange saturation transfer (CEST) experiments. "
             "Magn Reson Med. 2009;61(6):1441–1450.",
             "https://doi.org/10.1002/mrm.21873",
             "references/wassr/nihms196109.pdf"),
        ],
        {   # typeset equation + explanation
            "math": [
                {"text": "<b>Symmetry-based B<sub>0</sub> estimation.</b>  The observed spectrum is "
                         "mirrored about a trial centre C and compared with itself; the "
                         "<i>maximum-symmetry centre frequency</i> (MSCF) is the C that minimises "
                         "the mean squared difference between the measured curve and its reflection:"},
                {"img": "references/wassr/eq/mscf.png", "num": "[2]"},
                {"text": "The reflected curve f̃(2C − x<sub>i</sub>) is obtained by cubic-spline "
                         "interpolation, and the average runs only over offsets whose mirror image "
                         "falls inside the sampled range. The minimisation is solved with a "
                         "Nelder–Mead simplex search, initialised at the median frequency of the "
                         "points near half the dip's depth. Because it relies on symmetry, WASSR "
                         "needs finely-spaced offsets on both sides of the water dip."},
            ],
        },
    ),
]


# ── Interactive concept map (draggable node graph) ──────────────────────────────
from PyQt6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsItem,
                             QGraphicsPathItem)
from PyQt6.QtGui import QPainterPath, QPen, QBrush, QPolygonF, QPainter
from PyQt6.QtCore import QRectF, QPointF

# Default node positions (scene units) laid out like a CEST concept map.
# The line-shape / fit boxes are packed adjacent (centres one node-width apart)
# into a single horizontal strip so they all fit in the window.
_MAP_POS = {
    "WASSR  (B₀ mapping)":                     (260,  40),
    "WASABI  (B₀ & B₁ mapping)":               (500,  40),
    "Denoising":                               (760, 120),
    "QUESP Fit":                               ( 40, 300),
    "CEST MRI":                                (340, 300),
    "Bloch–McConnell Equations":               (650, 300),
    "CEST Magnetic Resonance Fingerprinting":  (950, 300),
    "T1 / T2 / B1 Map":                        (150, 560),
    # packed line-shape fit strip (touching, one node width apart), reordered,
    # sitting under the "line shape fit" banner (added by the view).
    "Gaussian Line Fit":                       (560, 620),
    "Lorentzian Line Fit":                     (760, 620),
    "Pseudo-Voigt Line Fit":                   (960, 620),
    "Super-Lorentzian / Semi-Solid Line Fit":  (1160, 620),
    "PLOF Fit":                                (1360, 620),
    "DROF Fit":                                (1560, 620),
    "Inverse Z-Spectroscopy":                  (330, 810),
    "AREX":                                    (600, 810),
}
# Core concepts drawn as ellipses; everything else as rounded rectangles.
_MAP_ELLIPSE = {
    "CEST MRI", "Bloch–McConnell Equations",
    "CEST Magnetic Resonance Fingerprinting",
}
# Each edge is (a, b) — single arrow a→b — or (a, b, True) — double-headed.
_MAP_EDGES = [
    ("CEST MRI", "Bloch–McConnell Equations", True),
    ("Bloch–McConnell Equations", "CEST Magnetic Resonance Fingerprinting", True),
    ("CEST MRI", "WASSR  (B₀ mapping)"),
    ("CEST MRI", "WASABI  (B₀ & B₁ mapping)"),
    ("CEST MRI", "Denoising"),
    ("CEST MRI", "QUESP Fit"),
    ("CEST MRI", "T1 / T2 / B1 Map"),
    ("CEST MRI", "Inverse Z-Spectroscopy"),
    ("Inverse Z-Spectroscopy", "AREX", True),
    # CEST MRI → line-shape-fit banner is added in the view (banner is not a section)
]
_LINE_SHAPE_BANNER = "Quantitative methods for fitting line shape"
_NODE_W = 176


def _rect_edge_point(cx, cy, tx, ty, half_w, half_h):
    """Point on the axis-aligned box (cx,cy,±half) in the direction of (tx,ty)."""
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return QPointF(cx, cy)
    sx = half_w / abs(dx) if dx != 0 else 1e9
    sy = half_h / abs(dy) if dy != 0 else 1e9
    s = min(sx, sy)
    return QPointF(cx + dx * s, cy + dy * s)


class _MapNode(QGraphicsItem):
    """Draggable, clickable concept-map node.  Click opens the equation dialog;
    drag repositions it (edges follow).  Positions reset each session."""

    def __init__(self, section, ellipse, opener, width=None):
        super().__init__()
        self._section = section
        self._ellipse = ellipse
        self._opener = opener            # None → non-clickable (e.g. a banner)
        self._w = float(width) if width is not None else float(_NODE_W)
        self._edges = []
        self._hover = False
        self._press = None
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        if opener is not None:
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setZValue(2)

    def half_h(self):
        return 48.0 if self._ellipse else 33.0

    def half_w(self):
        return self._w / 2.0

    def boundingRect(self):
        return QRectF(-self._w / 2 - 3, -self.half_h() - 3, self._w + 6, 2 * self.half_h() + 6)

    def add_edge(self, e):
        self._edges.append(e)

    def paint(self, p, opt, widget=None):
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        fill = QColor("#2a86a6") if self._hover else QColor("#1c6b87")
        p.setBrush(QBrush(fill))
        p.setPen(QPen(QColor("#0c4457"), 2))
        r = QRectF(-self._w / 2, -self.half_h(), self._w, 2 * self.half_h())
        if self._ellipse:
            p.drawEllipse(r)
        else:
            p.drawRoundedRect(r, 12, 12)
        p.setPen(QColor("#ffffff"))
        f = QFont(); f.setPointSize(11); f.setBold(True); p.setFont(f)
        p.drawText(r.adjusted(10, 4, -10, -4),
                   int(Qt.AlignmentFlag.AlignCenter) | int(Qt.TextFlag.TextWordWrap),
                   self._section[0])

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for e in self._edges:
                e.update_path()
        return super().itemChange(change, value)

    def hoverEnterEvent(self, e):
        self._hover = True; self.update(); super().hoverEnterEvent(e)

    def hoverLeaveEvent(self, e):
        self._hover = False; self.update(); super().hoverLeaveEvent(e)

    def mousePressEvent(self, e):
        self._press = e.scenePos(); super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        moved = (self._press is not None
                 and (e.scenePos() - self._press).manhattanLength() > 6)
        super().mouseReleaseEvent(e)
        if not moved and self._opener is not None:
            self._opener(self._section)


class _MapEdge(QGraphicsPathItem):
    """A line + arrowhead between two nodes that tracks their positions."""

    def __init__(self, src, dst, bidir=False):
        super().__init__()
        self._src = src
        self._dst = dst
        self._bidir = bidir
        self.setZValue(1)
        self.setPen(QPen(QColor("#5a8a99"), 2))
        self.setBrush(QBrush(QColor("#5a8a99")))
        src.add_edge(self)
        dst.add_edge(self)
        self.update_path()

    @staticmethod
    def _arrowhead(path, frm, to):
        """Add a filled arrowhead at `to`, pointing from `frm` → `to`."""
        import math
        dx, dy = to.x() - frm.x(), to.y() - frm.y()
        d = math.hypot(dx, dy) or 1.0
        ux, uy = dx / d, dy / d
        ah, w = 11.0, 5.5
        base = QPointF(to.x() - ah * ux, to.y() - ah * uy)
        left = QPointF(base.x() - w * uy, base.y() + w * ux)
        right = QPointF(base.x() + w * uy, base.y() - w * ux)
        path.addPolygon(QPolygonF([to, left, right]))
        path.closeSubpath()

    def update_path(self):
        a = self._src.scenePos(); b = self._dst.scenePos()
        pa = _rect_edge_point(a.x(), a.y(), b.x(), b.y(), self._src.half_w() + 4, self._src.half_h() + 4)
        pb = _rect_edge_point(b.x(), b.y(), a.x(), a.y(), self._dst.half_w() + 4, self._dst.half_h() + 4)
        path = QPainterPath(pa); path.lineTo(pb)
        self._arrowhead(path, pa, pb)            # arrow at the destination
        if self._bidir:
            self._arrowhead(path, pb, pa)        # and back at the source
        self.setPath(path)


class _ConceptMapView(QGraphicsView):
    """Scrollable canvas of draggable concept-map nodes + edges."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setBackgroundBrush(QBrush(QColor(_BG)))
        self.setStyleSheet(f"QGraphicsView {{ border: none; background: {_BG}; }}")
        self._dialogs = []
        self._nodes = {}
        _fb = 0
        for section in _SECTIONS:
            title = section[0]
            if title in _MAP_POS:
                x, y = _MAP_POS[title]
            else:                                   # any unlisted box → spare column
                x, y = 1560, 700 + _fb * 120; _fb += 1
            node = _MapNode(section, title in _MAP_ELLIPSE, self._open)
            node.setPos(x, y)
            self._scene.addItem(node)
            self._nodes[title] = node
        for e in _MAP_EDGES:
            a, b = e[0], e[1]
            bidir = len(e) > 2 and bool(e[2])
            if a in self._nodes and b in self._nodes:
                self._scene.addItem(_MapEdge(self._nodes[a], self._nodes[b], bidir))
        # Wide "line shape fit" banner grouping the reordered fit strip below it;
        # non-clickable, and the CEST MRI arrow points to it.
        banner = _MapNode((_LINE_SHAPE_BANNER,), False, None, width=1180)
        banner.setPos(1060, 520)
        self._scene.addItem(banner)
        self._nodes[_LINE_SHAPE_BANNER] = banner
        if "CEST MRI" in self._nodes:
            self._scene.addItem(_MapEdge(self._nodes["CEST MRI"], banner))
        # Forward arrows from the banner to each line-shape fit box (one way).
        for _fit in ("Gaussian Line Fit", "Lorentzian Line Fit", "Pseudo-Voigt Line Fit",
                     "Super-Lorentzian / Semi-Solid Line Fit", "PLOF Fit", "DROF Fit"):
            if _fit in self._nodes:
                self._scene.addItem(_MapEdge(banner, self._nodes[_fit]))
        self._scene.setSceneRect(
            self._scene.itemsBoundingRect().adjusted(-60, -60, 60, 60))

    def _open(self, section):
        s = section
        ex = s[6] if len(s) > 6 else {}
        dlg = _EqDialog(s[0], s[2], s[3], s[4], s[5], s[1], parent=self, extra=ex)
        dlg.show(); dlg.raise_(); dlg.activateWindow()
        self._dialogs.append(dlg)


# ── Main tab ──────────────────────────────────────────────────────────────────

class EquationsTab(QWidget):
    """Interactive concept-map of equations & references."""

    def __init__(self):
        super().__init__()
        _apply_eq_palette()                    # match chrome to the OS theme
        self.setStyleSheet(f"background: {_BG};")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Page header
        hdr = QLabel("  Equations  &  References")
        hdr.setStyleSheet(
            f"color: {_HDR_CLR}; font-size: 18px; font-weight: bold;"
            f"background: {_DLG_CARD}; padding: 12px 20px;"
            f"border-bottom: 2px solid {_SEP_CLR};"
        )
        outer.addWidget(hdr)

        # Interactive concept map — draggable nodes (ellipses = core concepts,
        # rounded rects = fits), connected by arrows.  Resets each session.
        self._map_view = _ConceptMapView(self)
        outer.addWidget(self._map_view, stretch=1)
