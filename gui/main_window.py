"""
gui/main_window.py
---------------------
Top-level window: a simple sidebar to switch between the three required
pages (Dashboard, Interactive Map, Analytics), and the glue code that
calls into data/scenario_generator.py and ai/pipeline.py when the user
clicks "Generate Scenario" / "Run AI" on the Dashboard.

Kept deliberately simple (no threading, no extra abstraction layers) --
the full pipeline runs in well under a second on the Cambridge network,
so a synchronous call with a wait-cursor is enough for a course project.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QStackedWidget,
    QMessageBox, QApplication, QLabel,
)

from gui.dashboard import DashboardPage
from gui.map_widget import MapPage
from gui.analytics import AnalyticsPage
from gui.route_risk_demo import RouteRiskDemoPage

from data.network_loader import load_cambridge_graph
from data.scenario_generator import generate_scenario
from ai.pipeline import run_full_pipeline


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Emergency Medical Dispatch Decision Support System")
        self.resize(1200, 800)

        self.graph = None       # loaded lazily on first use
        self.scenario = None    # current Scenario from data.scenario_generator
        self.result = None      # current PipelineResult from ai.pipeline

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        # --- sidebar navigation ------------------------------------------------
        sidebar = QWidget()
        sidebar.setObjectName("appSidebar")
        sidebar.setFixedWidth(180)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(14, 20, 14, 16)
        sidebar_layout.setSpacing(8)
        sidebar_layout.setAlignment(Qt.AlignTop)

        brand = QLabel("AIDISPATCH")
        brand.setObjectName("brandLabel")
        brand_subtitle = QLabel("EMERGENCY ROUTE LAB")
        brand_subtitle.setObjectName("brandSubtitle")
        sidebar_layout.addWidget(brand)
        sidebar_layout.addWidget(brand_subtitle)
        sidebar_layout.addSpacing(18)

        self.dashboard_button = QPushButton("Dashboard")
        self.map_button = QPushButton("Interactive Map")
        self.analytics_button = QPushButton("Analytics")
        self.course_demo_button = QPushButton("Course Demo")
        self.nav_buttons = (
            self.dashboard_button, self.map_button, self.analytics_button,
            self.course_demo_button,
        )
        for btn in self.nav_buttons:
            btn.setCheckable(True)
            btn.setMinimumHeight(40)
            sidebar_layout.addWidget(btn)
        sidebar_layout.addStretch()
        author = QLabel("Altynbek Adil\nAI & ML")
        author.setObjectName("authorLabel")
        sidebar_layout.addWidget(author)

        self.dashboard_button.setChecked(True)
        self.dashboard_button.clicked.connect(lambda: self._switch_page(0))
        self.map_button.clicked.connect(lambda: self._switch_page(1))
        self.analytics_button.clicked.connect(lambda: self._switch_page(2))
        self.course_demo_button.clicked.connect(lambda: self._switch_page(3))

        root_layout.addWidget(sidebar)
        self.setStyleSheet("""
            QWidget#appSidebar { background: #14212B; }
            QLabel#brandLabel { color: #F4F7F8; font-size: 16px; font-weight: 800; letter-spacing: 1px; }
            QLabel#brandSubtitle { color: #91A4B0; font-size: 8px; font-weight: 700; }
            QLabel#authorLabel { color: #C7D2D8; font-size: 10px; padding-top: 10px; border-top: 1px solid #31434F; }
            QWidget#appSidebar QPushButton { background: transparent; color: #D3DDE2; border: 0; border-radius: 7px; text-align: left; padding: 10px 11px; font-size: 11px; }
            QWidget#appSidebar QPushButton:hover { background: #223542; }
            QWidget#appSidebar QPushButton:checked { background: #167C70; color: white; font-weight: 700; }
        """)

        # --- pages ---------------------------------------------------------
        self.pages = QStackedWidget()
        self.dashboard_page = DashboardPage()
        self.map_page = MapPage()
        self.analytics_page = AnalyticsPage()
        self.course_demo_page = RouteRiskDemoPage()
        self.pages.addWidget(self.dashboard_page)
        self.pages.addWidget(self.map_page)
        self.pages.addWidget(self.analytics_page)
        self.pages.addWidget(self.course_demo_page)
        root_layout.addWidget(self.pages)

        self.dashboard_page.generate_scenario_requested.connect(self.on_generate_scenario)
        self.dashboard_page.run_ai_requested.connect(self.on_run_ai)

    # ------------------------------------------------------------------
    def _switch_page(self, index: int):
        self.pages.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)

    # ------------------------------------------------------------------
    def _ensure_graph_loaded(self):
        if self.graph is None:
            self.dashboard_page.set_status("Loading Cambridge road network...")
            QApplication.processEvents()
            self.graph = load_cambridge_graph()

    # ------------------------------------------------------------------
    def on_generate_scenario(self, n_calls: int):
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self._ensure_graph_loaded()
            self.scenario = generate_scenario(self.graph, n_calls=n_calls)
            self.result = None  # invalidate any previous AI results

            self.dashboard_page.set_scenario_stats(
                n_calls=len(self.scenario.calls),
                n_ambulances=len(self.scenario.ambulances),
                n_hospitals=len(self.scenario.hospitals),
            )
            self.map_page.update_map(self.scenario, ["Unknown"] * len(self.scenario.calls), [], self.graph)
        except Exception as exc:
            QMessageBox.critical(self, "Scenario generation failed", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    # ------------------------------------------------------------------
    def on_run_ai(self):
        if self.scenario is None:
            QMessageBox.warning(self, "No scenario", "Generate a scenario first.")
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.dashboard_page.set_status("Running AI pipeline (triage, assignment, routing, hospitals)...")
            QApplication.processEvents()

            self.result = run_full_pipeline(graph=self.graph, scenario=self.scenario)
            # scenario is unchanged by the pipeline; kept as-is for clarity.

            self.map_page.update_map(
                self.scenario, self.result.predicted_priorities, self.result.routes, self.graph,
            )
            self.analytics_page.update_results(self.result)
            self.dashboard_page.set_status("AI run complete. See the Interactive Map and Analytics pages.")
            self._switch_page(1)
        except Exception as exc:
            QMessageBox.critical(self, "AI run failed", str(exc))
        finally:
            QApplication.restoreOverrideCursor()
