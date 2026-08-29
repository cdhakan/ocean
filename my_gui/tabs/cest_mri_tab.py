"""
cest_mri_tab.py  —  CEST MRI grouped parent tab
=================================================
Groups the CEST-MRI analysis steps under one top-level tab with an inner
QTabWidget (mirrors the CEST MRF grouped tab):

    Load CEST Datas     — CEST / WASSR Z-spectroscopy loading & analysis
    Inverse Z Analysis  — 1/Z (AREX / inverse Z-spectrum) analysis
    QUESP               — QUESP fₛ / kₛw quantification

All inner tabs are exposed as public attributes (`zspec_tab`,
`inv_zspec_tab`, `quesp_tab`) so app.py can wire cross-tab callbacks exactly
as before.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel
from PyQt6.QtCore import Qt

from my_gui.tabs.zspec_tab      import ZSpecTab
from my_gui.tabs.inv_zspec_tab  import InvZSpecTab
from my_gui.tabs.quesp_tab      import QUESPTab
from my_gui.tabs.synth_cest_tab import SynthCestTab


class CestMriTab(QWidget):
    """Parent tab hosting the three CEST-MRI sub-tabs in a nested QTabWidget."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Create inner tabs ────────────────────────────────────────────────
        self.zspec_tab     = ZSpecTab()
        self.inv_zspec_tab = InvZSpecTab()
        self.quesp_tab     = QUESPTab()
        self.synth_cest_tab = SynthCestTab()

        # Let "Synthetic CEST MRI" push its Z-stack into "Quantitative Z Analysis".
        self._wire_synth_push()

        # ── Inner tab widget ─────────────────────────────────────────────────
        self.sub_tabs = QTabWidget()
        self.sub_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.sub_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #555;
                border-radius: 4px;
            }
            QTabBar::tab {
                padding: 5px 18px;
                font-size: 12px;
                min-width: 130px;
                background: #2d2d2d;
                color: #ccc;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #1565c0;
                color: white;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background: #3a3a3a;
                color: white;
            }
        """)

        self.sub_tabs.addTab(self.zspec_tab,      "Quantitative Z Analysis")
        self.sub_tabs.addTab(self.inv_zspec_tab,  "Inverse Z Analysis")
        self.sub_tabs.addTab(self.quesp_tab,      "QUESP")
        self.sub_tabs.addTab(self.synth_cest_tab, "Synthetic CEST MRI")

        # ── Banner ────────────────────────────────────────────────────────────
        banner = QLabel(
            "CEST-MRI Pipeline  →  Load CEST / WASSR Datas  →  Motion Correction  →  "
            "Denoising  →  B₀ / B₁ correction  →  CEST Analysis  →  "
            "Inverse Z Analysis (if needed)  →  QUESP (if needed, to quantify fₛ and kₛw)"
        )
        banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        banner.setWordWrap(False)
        banner.setStyleSheet(
            "QLabel {"
            "  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "    stop:0 #0d47a1, stop:0.5 #1565c0, stop:1 #00695c);"
            "  color: white;"
            "  font-size: 12px;"
            "  font-weight: bold;"
            "  padding: 5px 10px;"
            "}"
        )

        # ── Layout ────────────────────────────────────────────────────────────
        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)
        vl.addWidget(banner)
        vl.addWidget(self.sub_tabs, stretch=1)

    # ── Synthetic-phantom → Quantitative Z Analysis push ─────────────────────

    def _wire_synth_push(self):
        """Connect the Synthetic CEST MRI tab so its Z-stack lands in the
        Quantitative Z Analysis tab (and that tab is brought to front)."""
        from my_gui.tabs.zspec_tab import _finish_cest_load

        def _push(img_all, offsets):
            _finish_cest_load(self.zspec_tab, img_all, offsets, "Synthetic phantom")
            self.sub_tabs.setCurrentWidget(self.zspec_tab)

        self.synth_cest_tab.set_push_callback(_push)

    # ── Convenience pass-through for ROI manager wiring ──────────────────────

    def connect_roi_manager(self, roi_manager):
        """Forward ROI manager to any inner tab that supports it."""
        for tab in (self.zspec_tab, self.inv_zspec_tab, self.quesp_tab):
            if hasattr(tab, 'connect_roi_manager'):
                tab.connect_roi_manager(roi_manager)
