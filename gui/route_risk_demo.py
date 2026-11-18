"""Interactive GUI for the integrated A* + Bayesian + MDP course demo."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QFrame, QGridLayout,
)

from ai.route_demo import (
    PRIOR_BLOCKED, SENSOR_FALSE_POSITIVE_RATE, SENSOR_SENSITIVITY,
    RISK_EDGE, run_route_demo,
)


class MetricCard(QFrame):
    def __init__(self, label: str, value: str = "—", detail: str = ""):
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 13, 16, 13)
        title = QLabel(label.upper())
        title.setObjectName("metricLabel")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        self.detail_label = QLabel(detail)
        self.detail_label.setObjectName("metricDetail")
        self.detail_label.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(self.value_label)
        layout.addWidget(self.detail_label)

    def set_value(self, value: str, detail: str = ""):
        self.value_label.setText(value)
        self.detail_label.setText(detail)


class RouteCanvas(QWidget):
    """Compact, native Qt drawing of the toy road graph and selected path."""

    POINTS = {
        "AMB": (0.10, 0.51), "JCT": (0.48, 0.51), "CALL": (0.89, 0.51),
        "A": (0.28, 0.83), "B": (0.68, 0.83),
    }
    EDGES = (("AMB", "JCT"), ("JCT", "CALL"), ("AMB", "A"),
             ("A", "B"), ("B", "CALL"))

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(245)
        self.route = []
        self.alert = False

    def set_scenario(self, route, alert):
        self.route = list(route)
        self.alert = bool(alert)
        self.update()

    def paintEvent(self, event):
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#101923"))
        w, h = self.width(), self.height()
        margin_x, margin_y = 42, 30

        def xy(node):
            rx, ry = self.POINTS[node]
            return (margin_x + rx * (w - margin_x * 2),
                    margin_y + ry * (h - margin_y * 2))

        route_edges = {frozenset((u, v)) for u, v in zip(self.route, self.route[1:])}
        risk = frozenset(RISK_EDGE)
        for u, v in self.EDGES:
            x1, y1 = xy(u)
            x2, y2 = xy(v)
            is_route = frozenset((u, v)) in route_edges
            is_risk = frozenset((u, v)) == risk
            if is_route:
                color, width = QColor("#45D6B0"), 6
            elif is_risk and self.alert:
                color, width = QColor("#FF6B6B"), 4
            elif is_risk:
                color, width = QColor("#F3B84B"), 4
            else:
                color, width = QColor("#455665"), 3
            pen = QPen(color, width, Qt.DashLine if is_risk and self.alert and not is_route else Qt.SolidLine)
            p.setPen(pen)
            p.drawLine(int(x1), int(y1), int(x2), int(y2))
            p.setPen(QColor("#9BAAB5"))
            p.setFont(QFont("Arial", 9))
            length = "100 m" if frozenset((u, v)) in {
                frozenset(("AMB", "JCT")), frozenset(("JCT", "CALL"))
            } else ("120 m" if frozenset((u, v)) == frozenset(("AMB", "A")) else "150 m")
            p.drawText(int((x1+x2)/2 - 20), int((y1+y2)/2 - 8), length)

        for node, (nx, ny) in self.POINTS.items():
            x, y = xy(node)
            fill = QColor("#45D6B0") if node in self.route else QColor("#263744")
            if node == "CALL":
                fill = QColor("#E9F0F4")
            p.setPen(QPen(QColor("#E9F0F4"), 2))
            p.setBrush(fill)
            p.drawEllipse(int(x-15), int(y-15), 30, 30)
            p.setPen(QColor("#101923") if node == "CALL" else QColor("#F5F8FA"))
            p.setFont(QFont("Arial", 8, QFont.Bold))
            p.drawText(int(x-28), int(y+32), 56, 18, Qt.AlignCenter, node)
        p.end()


class RouteRiskDemoPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("courseDemoPage")
        self._build_ui()
        self._apply_style()
        self.run_scenario()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(26, 22, 26, 22)
        outer.setSpacing(15)

        heading_row = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel("Route risk decision lab")
        title.setObjectName("pageTitle")
        subtitle = QLabel("A* search + Bayesian inference + a one-step MDP")
        subtitle.setObjectName("pageSubtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        heading_row.addLayout(titles)
        heading_row.addStretch()
        heading_row.addWidget(QLabel("SENSOR SCENARIO"))
        self.signal_combo = QComboBox()
        self.signal_combo.addItem("No alert", False)
        self.signal_combo.addItem("Road alert", True)
        self.signal_combo.setMinimumWidth(140)
        heading_row.addWidget(self.signal_combo)
        self.run_button = QPushButton("Run decision")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self.run_scenario)
        heading_row.addWidget(self.run_button)
        outer.addLayout(heading_row)

        cards = QGridLayout()
        cards.setHorizontalSpacing(12)
        self.risk_card = MetricCard("Posterior road risk")
        self.action_card = MetricCard("MDP action")
        self.distance_card = MetricCard("Selected route")
        cards.addWidget(self.risk_card, 0, 0)
        cards.addWidget(self.action_card, 0, 1)
        cards.addWidget(self.distance_card, 0, 2)
        outer.addLayout(cards)

        self.canvas = RouteCanvas()
        outer.addWidget(self.canvas)
        legend = QLabel("Mint: selected route    Amber: monitored road    Red dashed: alerted segment avoided")
        legend.setObjectName("legend")
        outer.addWidget(legend)

        lower = QHBoxLayout()
        lower.setSpacing(14)
        bayes = QFrame()
        bayes.setObjectName("infoCard")
        bayes_layout = QVBoxLayout(bayes)
        bayes_title = QLabel("1  BAYESIAN UPDATE")
        bayes_title.setObjectName("sectionTitle")
        self.bayes_text = QLabel()
        self.bayes_text.setWordWrap(True)
        self.bayes_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bayes_layout.addWidget(bayes_title)
        bayes_layout.addWidget(self.bayes_text)
        mdp = QFrame()
        mdp.setObjectName("infoCard")
        mdp_layout = QVBoxLayout(mdp)
        mdp_title = QLabel("2  MDP ACTION VALUES")
        mdp_title.setObjectName("sectionTitle")
        self.mdp_text = QLabel()
        self.mdp_text.setWordWrap(True)
        self.mdp_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        mdp_layout.addWidget(mdp_title)
        mdp_layout.addWidget(self.mdp_text)
        lower.addWidget(bayes, 1)
        lower.addWidget(mdp, 1)
        outer.addLayout(lower)

        note = QLabel("Teaching scenario: probabilities, rewards and road lengths are illustrative; this is not a live dispatch or safety system.")
        note.setObjectName("disclaimer")
        note.setWordWrap(True)
        outer.addWidget(note)

    def run_scenario(self):
        result = run_route_demo(bool(self.signal_combo.currentData()))
        posterior = result.bayes.posterior_blocked
        alert_word = "alert" if result.sensor_alert else "no alert"
        self.risk_card.set_value(f"{posterior:.1%}", f"After sensor: {alert_word}")
        self.action_card.set_value(result.mdp.action.upper(), "Chosen by belief-weighted Bellman value")
        self.distance_card.set_value(f"{result.selected_route.distance_m:.0f} m", " → ".join(result.selected_route.path))
        self.canvas.set_scenario(result.selected_route.path, result.sensor_alert)
        self.bayes_text.setText(
            f"Prior P(blocked) = {PRIOR_BLOCKED:.0%}.  "
            f"Sensitivity = {SENSOR_SENSITIVITY:.0%}; false-positive rate = "
            f"{SENSOR_FALSE_POSITIVE_RATE:.0%}.\n\n"
            f"P(blocked | {alert_word}) = {posterior:.1%}. Bayes updates the prior "
            "using how likely this sensor observation is in each road state."
        )
        values = result.mdp.belief_action_values
        self.mdp_text.setText(
            f"Continue: {values['continue']:.2f}\n"
            f"Reroute: {values['reroute']:.2f}\n\n"
            f"The MDP chooses the larger expected value. A* then computes the route "
            f"for that action. The initial shortest route is "
            f"{result.initial_route.distance_m:.0f} m."
        )

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget#courseDemoPage { background: #F4F7F8; color: #14212B; }
            QLabel#pageTitle { font-size: 24px; font-weight: 700; color: #14212B; }
            QLabel#pageSubtitle { font-size: 12px; color: #5C6E79; }
            QLabel#metricLabel { color: #61727D; font-size: 9px; font-weight: 700; }
            QLabel#metricValue { color: #14212B; font-size: 22px; font-weight: 700; }
            QLabel#metricDetail { color: #61727D; font-size: 10px; }
            QFrame#metricCard, QFrame#infoCard { background: white; border: 1px solid #DDE5E9; border-radius: 11px; }
            QLabel#sectionTitle { color: #167C70; font-size: 11px; font-weight: 700; }
            QLabel#legend, QLabel#disclaimer { color: #61727D; font-size: 10px; }
            QPushButton#primaryButton { background: #167C70; color: white; border: 0; border-radius: 7px; padding: 9px 15px; font-weight: 700; }
            QPushButton#primaryButton:hover { background: #116A60; }
            QComboBox { background: white; padding: 7px; border: 1px solid #CDD8DE; border-radius: 6px; }
        """)
