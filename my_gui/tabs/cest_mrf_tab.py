"""
cest_mrf_tab.py  —  CEST MRF grouped parent tab
=================================================
Groups three core MRF workflow steps under one top-level tab with
an inner QTabWidget:

    Configuration        — pool parameters (water, CEST, MT)
    Sequence / Simulation — MRF pulse schedule editor + dictionary dialog
    MRF Viewer            — quantitative map viewer

The dictionary generation / matching UI (DictTab) is no longer a dedicated
sub-tab; it lives in a resizable dialog launched from the prominent
"Generate Dictionary Simulation & Matching" button in the Sequence /
Simulation toolbar.

All inner tabs are still instantiated here and exposed as public attributes
(`config_tab`, `seq_tab`, `dict_tab`, `results_tab`) so that app.py can wire
cross-tab callbacks exactly as before.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel
from PyQt6.QtCore import Qt

from my_gui.tabs.config_tab   import ConfigTab
from my_gui.tabs.sequence_tab import SequenceTab
from my_gui.tabs.dict_tab     import DictTab
from my_gui.tabs.results_tab  import ResultsTab


class CestMrfTab(QWidget):
    """
    Parent tab that hosts four MRF sub-tabs in a nested QTabWidget.

    Parameters
    ----------
    on_generate_clicked : callable
        Forwarded to DictTab as its generate-button callback.
        (In practice this is MainWindow._on_generate.)
    """

    def __init__(self, on_generate_clicked=None, parent=None):
        super().__init__(parent)

        # ── Create inner tabs ────────────────────────────────────────────────
        self.config_tab  = ConfigTab()
        self.seq_tab     = SequenceTab()
        self.dict_tab    = DictTab(on_generate_clicked=on_generate_clicked)
        self.results_tab = ResultsTab()

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

        self.sub_tabs.addTab(self.config_tab,  "Configuration")
        self.sub_tabs.addTab(self.seq_tab,     "Sequence / Simulation")
        self.sub_tabs.addTab(self.results_tab, "MRF Viewer")

        # Wire the DictTab into the Sequence tab toolbar button
        # (dict_tab is NOT added as a sub-tab — it opens as a dialog instead)
        self.seq_tab.set_dict_tab(self.dict_tab)

        # ── Banner ────────────────────────────────────────────────────────────
        banner = QLabel(
            "CEST MRF Pipeline  →  Configure Parameters  →  "
            "Load / Build Sequence  →  "
            "Generate CEST-MRF Dictionary Simulation & Matching  →  "
            "View CEST-MRF Parametric Maps"
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

    # ── Convenience pass-through for ROI manager wiring ──────────────────────

    def connect_roi_manager(self, roi_manager):
        """Forward ROI manager to any inner tab that supports it."""
        for tab in (self.config_tab, self.seq_tab,
                    self.dict_tab, self.results_tab):
            if hasattr(tab, 'connect_roi_manager'):
                tab.connect_roi_manager(roi_manager)

    # ── Jump to a specific sub-tab from outside (e.g. after generation done) ──

    def show_results(self):
        """Switch the inner tab widget to the MRF Viewer sub-tab (index 2)."""
        self.sub_tabs.setCurrentIndex(2)

    def show_simulation(self):
        """Open the Dictionary Simulation & Matching dialog."""
        # Navigate to Sequence / Simulation tab first so the button is visible,
        # then open the dialog.
        self.sub_tabs.setCurrentIndex(1)
        self.seq_tab._open_sim_dialog()
