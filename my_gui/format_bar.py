"""
format_bar.py  —  Reusable matplotlib-title formatting toolbar
==============================================================
Creates a compact row of  Bold / Italic / Superscript / Subscript  buttons.

• Superscript / Subscript  use Unicode characters (⁰¹²… / ₀₁₂…) — zero LaTeX,
  zero crash risk, displays correctly in matplotlib with any font.
• Bold / Italic  still use LaTeX mathtext ($\\mathbf{…}$ / $\\mathit{…}$) but are
  only applied on a button click, not during free typing.  A 400 ms debounce
  on the title field ensures that partial LaTeX strings never reach matplotlib
  while the user is mid-typing.

Usage
-----
    from my_gui.format_bar import add_title_format_bar, connect_title_debounced

    # Wire the title field (debounced, crash-safe):
    connect_title_debounced(self.edit_map_title, self._refresh_display)

    # Add the format toolbar row below the title field:
    add_title_format_bar(self.edit_map_title, parent_layout)
"""
from __future__ import annotations

from functools import lru_cache

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QLineEdit
from PyQt6.QtCore import QTimer, Qt

# ── Unicode conversion maps ───────────────────────────────────────────────────
_SUB_MAP = str.maketrans(
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-=()",
    "₀₁₂₃₄₅₆₇₈₉ₐbcdₑfgₕᵢⱼₖₗₘₙₒₚqᵣₛₜᵤᵥwₓyz"
    "ₐBCDₑFGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥWₓYZ₊₋₌₍₎"
)

_SUP_MAP = str.maketrans(
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-=()",
    "⁰¹²³⁴⁵⁶⁷⁸⁹ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖᑫʳˢᵗᵘᵛʷˣʸᶻ"
    "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁᵛᵂˣʸᶻ⁺⁻⁼⁽⁾"
)


def _to_sub(text: str) -> str:
    return text.translate(_SUB_MAP)


def _to_sup(text: str) -> str:
    return text.translate(_SUP_MAP)


# ── Shared button stylesheet ──────────────────────────────────────────────────
@lru_cache(maxsize=512)
def safe_mathtext(title: str) -> str:
    """Return a title string that is SAFE to hand to matplotlib.

    matplotlib parses ``$…$`` mathtext lazily, at ``draw()`` time — so an
    invalid span (e.g. an unclosed ``$\\mathbf{``) raises deep inside the Agg
    renderer and aborts the whole process.  We validate every math span
    eagerly here; if the ``$`` are unbalanced or any span fails to parse we
    strip the LaTeX markup and return plain text, so a half-typed or malformed
    title degrades to a readable label instead of crashing the app.
    """
    if not title or '$' not in title:
        return title
    import re
    parts = re.split(r'(?<!\\)\$', title)        # split on unescaped $
    ok = (len(parts) % 2 == 1)                    # balanced $ required
    if ok:
        try:
            from matplotlib import mathtext
            parser = mathtext.MathTextParser('agg')
            for i in range(1, len(parts), 2):     # odd segments are math
                parser.parse('$' + parts[i] + '$')
        except Exception:
            ok = False
    if ok:
        return title
    # Fallback: strip LaTeX markup, keep the readable text.
    p = re.sub(r'\\math[a-zA-Z]+\s*\{', '', title)   # \mathbf{ / \mathit{ → ''
    p = re.sub(r'\\[a-zA-Z]+', '', p)                 # other commands → ''
    p = p.replace('$', '').replace('{', '').replace('}', '')
    return p.strip()


_BTN_SS = (
    "QPushButton {"
    "  border: 1px solid #555;"
    "  border-radius: 3px;"
    "  padding: 1px 5px;"
    "  font-size: 11px;"
    "  min-width: 24px;"
    "}"
    "QPushButton:hover   { background: #2a3050; border-color: #7986cb; }"
    "QPushButton:pressed { background: #1a1a2e; }"
)


# ── Public API ────────────────────────────────────────────────────────────────

def connect_title_debounced(edit: QLineEdit, callback, delay_ms: int = 400):
    """Connect edit.textChanged to callback via a 400 ms debounce timer.

    The callback fires only after the user has stopped typing for *delay_ms*
    milliseconds.  This means partial / invalid LaTeX strings during live
    editing never reach matplotlib, eliminating the ParseFatalException crash.

    The timer is stored on the widget as ``_title_debounce_timer`` so it stays
    alive for the widget's lifetime.
    """
    timer = QTimer()
    timer.setSingleShot(True)
    timer.setInterval(delay_ms)
    timer.timeout.connect(callback)
    edit.textChanged.connect(lambda _: timer.start(delay_ms))
    edit._title_debounce_timer = timer   # keep alive
    return timer


def add_title_format_bar(edit: QLineEdit, parent_layout, target_row=None,
                         default_getter=None) -> None:
    """
    Append a compact Bold / Italic / Superscript / Subscript toolbar row
    to *parent_layout*, wired to the given *edit* widget.

    Sub/Superscript use Unicode characters — no LaTeX, no crash.
    Bold/Italic use LaTeX mathtext and are safe because they're only
    applied on button click, not during free typing.
    """

    # ── Core helpers ──────────────────────────────────────────────────────────
    # Track last known selection so NoFocus buttons can still read it
    _sel: dict = {'start': 0, 'end': 0, 'text': ''}

    def _save_sel():
        t = edit.selectedText()
        if t:
            _sel['start'] = edit.selectionStart()
            _sel['end']   = edit.selectionStart() + len(t)
            _sel['text']  = t

    edit.selectionChanged.connect(_save_sel)

    def _wrap(prefix: str, suffix: str) -> None:
        """Wrap selected text with LaTeX markers.
        Works even after button click clears focus, using saved selection."""
        sel  = edit.selectedText() or _sel['text']
        text = edit.text()
        if not sel:
            # Nothing selected — do nothing, just refocus
            edit.setFocus()
            return
        start = edit.selectionStart() if edit.selectedText() else _sel['start']
        end   = start + len(sel)
        new_text = text[:start] + prefix + sel + suffix + text[end:]
        edit.setText(new_text)
        edit.setSelection(start, len(prefix) + len(sel) + len(suffix))
        _sel['text'] = ''   # consume
        edit.setFocus()

    def _unicode_replace(converter) -> None:
        """Replace selected text with Unicode sub/superscript equivalent.
        Works even after button click clears focus, using saved selection."""
        sel  = edit.selectedText() or _sel['text']
        text = edit.text()
        if not sel:
            # No selection.  If the box is empty, pull the current (default)
            # title into it so the user can select a character — e.g. the "1"
            # in "T1 map (ms)" — and apply sub/superscript, without retyping the
            # whole title.  The first click loads it; then select + click again.
            if not text and default_getter is not None:
                try:
                    dt = (default_getter() or "").strip()
                except Exception:
                    dt = ""
                if dt:
                    edit.setText(dt)
            edit.setFocus()
            return
        start    = edit.selectionStart() if edit.selectedText() else _sel['start']
        end      = start + len(sel)
        replaced = converter(sel)
        new_text = text[:start] + replaced + text[end:]
        edit.setText(new_text)
        edit.setSelection(start, len(replaced))
        _sel['text'] = ''   # consume
        edit.setFocus()

    # ── Layout ───────────────────────────────────────────────────────────────
    # If a target_row is supplied, append the format buttons to it (e.g. onto
    # the plot bar's fonts line); otherwise build a dedicated row.
    if target_row is not None:
        row = target_row
    else:
        row = QHBoxLayout()
        row.setSpacing(4)
        row.setContentsMargins(0, 1, 0, 2)

    # Bold / Italic are handled as whole-title toggles on the plot bar (they
    # style the DEFAULT title too), so only the per-character sub/superscript
    # helpers live here.

    # Superscript  (Unicode — always safe)
    btn_sup = QPushButton("x²")
    btn_sup.setFixedSize(30, 22)
    btn_sup.setToolTip("Superscript")
    btn_sup.setStyleSheet(_BTN_SS)
    btn_sup.clicked.connect(lambda: _unicode_replace(_to_sup))

    # Subscript  (Unicode — always safe)
    btn_sub = QPushButton("x₂")
    btn_sub.setFixedSize(30, 22)
    btn_sub.setToolTip("Subscript")
    btn_sub.setStyleSheet(_BTN_SS)
    btn_sub.clicked.connect(lambda: _unicode_replace(_to_sub))

    for btn in (btn_sup, btn_sub):
        # NoFocus: button clicks never steal focus from QLineEdit,
        # so selectedText() stays intact when the handler runs.
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        row.addWidget(btn)

    if target_row is None:
        row.addStretch()
        parent_layout.addLayout(row)
