"""Build the report and two editable-before-signing statement PDFs."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parent
INK = colors.HexColor("#14243A")
BLUE = colors.HexColor("#2364AA")
PALE = colors.HexColor("#EAF2F8")


def styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(name="ReportTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=23,
                            leading=28, textColor=INK, alignment=TA_CENTER, spaceAfter=18))
    base.add(ParagraphStyle(name="Section", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=14,
                            leading=18, textColor=BLUE, spaceBefore=11, spaceAfter=6, keepWithNext=True))
    base.add(ParagraphStyle(name="BodySmall", parent=base["BodyText"], fontName="Helvetica", fontSize=9.3,
                            leading=13, spaceAfter=6))
    base.add(ParagraphStyle(name="Statement", parent=base["BodyText"], fontName="Helvetica", fontSize=11,
                            leading=16, spaceAfter=10))
    return base


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D7E1EC"))
    canvas.line(0.65 * inch, 0.55 * inch, 7.85 * inch, 0.55 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(0.65 * inch, 0.38 * inch, "Mini Emergency Exit Planner | Educational simulation")
    canvas.drawRightString(7.85 * inch, 0.38 * inch, str(doc.page))
    canvas.restoreState()


def report():
    s = styles()
    story = [Paragraph("Mini Emergency Exit Planner", s["ReportTitle"]),
             Paragraph("Technical Report | Course Topic 05", ParagraphStyle("Sub", parent=s["BodyText"], alignment=TA_CENTER, textColor=BLUE)),
             Spacer(1, 8), Paragraph("Author: [Enter student name] &nbsp;&nbsp;&nbsp; Date: 19 November 2026", ParagraphStyle("Meta", parent=s["BodyText"], alignment=TA_CENTER)), Spacer(1, 14)]
    sections = [
        ("1. Problem and objective", "A shortest route can become unsafe when a corridor is blocked. This classroom simulation connects search, Bayesian inference, and a one-step Markov decision model: A* plans to Exit A; a smoke reading updates blockage belief; the MDP compares proceeding and detouring; a detour closes the monitored cell before A* plans to Exit B.<br/><br/><b>Safety:</b> map, sensor, and reward values are hypothetical. This is not a life-safety product and has no real sensor feeds, verified building plans, fire model, or human movement dynamics."),
        ("2. Environment and assumptions", "The 6 x 6 baseline uses start (2, 0), Exit A (2, 5), Exit B (5, 5), walls at (1, 1), (3, 1), and (4, 3), and monitored cell (2, 3). Moves are four-directional at unit cost. A 10 x 10 stress map provides a second benchmark. Coordinates are zero-indexed (row, column).<br/><br/>Prior blockage probability is 0.25. Smoke rates are P(Alert|Blocked)=0.85 and P(Alert|Clear)=0.15. Optional heat rates are 0.80 and 0.20, assumed conditionally independent given corridor state. Rewards are +100 for proceed-clear, -100 for proceed-blocked, and +40 for detour."),
        ("3. Architecture and algorithms", "A* and BFS share the same map and four neighbors. A* orders nodes by f(n)=g(n)+h(n), with Manhattan distance h. This heuristic is admissible and consistent with four-way unit-cost movement. The Bayesian module calculates the observation likelihood, evidence, and posterior. The MDP compares one-step expected utilities. If DETOUR wins, the pipeline adds the monitored corridor to the blocked cells and searches again for Exit B. An unreachable path is returned as None and shown as an error in the app."),
        ("4. Verified calculations", "Bayes' rule: P(B|E)=P(E|B)P(B)/[P(E|B)P(B)+P(E|not B)P(not B)].<br/><br/><b>Clear:</b> evidence = 0.15*0.25 + 0.85*0.75 = 0.675; posterior = 0.0375/0.675 = 0.05556 (5.56%).<br/><b>Alert:</b> evidence = 0.85*0.25 + 0.15*0.75 = 0.325; posterior = 0.2125/0.325 = 0.65385 (65.38%).<br/><br/>At default rewards EU(Proceed)=100-200p and EU(Detour)=40. Equality gives p*=0.30. Clear produces utilities 88.89 vs 40 (PROCEED); alert produces -30.77 vs 40 (DETOUR). Exact ties select DETOUR."),
        ("5. Testing and benchmark", "Nine tests cover the two sensor presets, noisy evidence crossing the boundary, sequential heat evidence, probability extremes, tie behavior, invalid probabilities, unreachable destinations, and equal A*/BFS path costs on both maps. They are pytest-compatible and also run with Python unittest."),
    ]
    for title, body in sections:
        story += [Paragraph(title, s["Section"]), Paragraph(body, s["BodySmall"])]
    data = [["Map", "Search", "Path cost", "Expanded", "Runtime (ms)"],
            ["6 x 6 baseline", "A*", "5", "6", "0.0153"],
            ["6 x 6 baseline", "BFS", "5", "22", "0.0405"],
            ["10 x 10 stress", "A*", "9", "10", "0.0211"],
            ["10 x 10 stress", "BFS", "9", "37", "0.0631"]]
    table = Table(data, colWidths=[1.7*inch, .72*inch, .82*inch, 1.0*inch, 1.15*inch], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), INK), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                               ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 8.5),
                               ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#CBD5E1")),
                               ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, PALE]),
                               ("ALIGN", (1,1), (-1,-1), "CENTER"), ("TOPPADDING", (0,0), (-1,-1), 6),
                               ("BOTTOMPADDING", (0,0), (-1,-1), 6)]))
    story += [table, Paragraph("One illustrative local run on Python 3.12.14. Sub-millisecond timings vary by host and are noisy; equal path costs and expanded-node counts are more useful here.", s["BodySmall"]),
              Paragraph("6. Interface and deployment", s["Section"]), Paragraph("The Streamlit app provides three presets, two grid sizes, baseline wall editing, live probabilities and risk settings, optional heat evidence, route playback, math traces, a benchmark table, and a policy-boundary chart. The Dockerfile serves port 8501. GitHub Actions installs test dependencies and runs pytest. Hosting is documented but not provisioned.", s["BodySmall"]),
              Paragraph("7. Limitations and future work", s["Section"]), Paragraph("The model uses one-step rewards rather than transitions over time, a binary corridor state, known sensor rates, a static map, and a single agent. Future classroom extensions could add a small transition model, several monitored corridors, sensor dependence, dynamic hazards, multiple agents, or multi-floor graphs. Real use would require validated plans and qualified safety engineering.", s["BodySmall"]),
              Paragraph("8. Defense takeaway", s["Section"]), Paragraph("Search proposes a route, Bayes updates belief from evidence, the MDP selects an action at a derived boundary, and search recomputes an alternate route after a detour.", s["BodySmall"])]
    SimpleDocTemplate(str(ROOT / "technical_report.pdf"), pagesize=letter, rightMargin=.65*inch, leftMargin=.65*inch,
                      topMargin=.62*inch, bottomMargin=.75*inch, title="Mini Emergency Exit Planner - Technical Report").build(story, onFirstPage=footer, onLaterPages=footer)


def statement_pdf(path, title, paragraphs):
    s = styles()
    story = [Paragraph(title, s["ReportTitle"]), Spacer(1, 18)]
    story += [Paragraph(p, s["Statement"]) for p in paragraphs]
    SimpleDocTemplate(str(path), pagesize=letter, rightMargin=.85*inch, leftMargin=.85*inch,
                      topMargin=.9*inch, bottomMargin=.8*inch, title=title).build(story, onFirstPage=footer, onLaterPages=footer)


def presentation_pdf():
    """Assemble the verified slide previews into a landscape handout PDF."""
    build = ROOT.parent / ".build"
    output = ROOT / "presentation.pdf"
    page_size = (960, 540)
    pdf = canvas.Canvas(str(output), pagesize=page_size, pageCompression=1)
    pdf.setTitle("Mini Emergency Exit Planner - Presentation")
    for index in range(1, 8):
        image = build / f"slide-{index:02d}.png"
        pdf.drawImage(ImageReader(str(image)), 0, 0, width=page_size[0], height=page_size[1])
        pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    report()
    if all((ROOT.parent / ".build" / f"slide-{index:02d}.png").exists() for index in range(1, 8)):
        presentation_pdf()
    statement_pdf(ROOT / "ai_usage_declaration.pdf", "AI Usage Declaration", [
        "AI assistance was used to draft the project architecture, application code, tests, technical explanations, and documentation.",
        "The project author must review each module, run the test suite, verify the calculations, and be able to explain submitted code. Edit this declaration to match the actual work and applicable course policy before submission.",
    ])
    statement_pdf(ROOT / "team_contribution_statement.pdf", "Individual Contribution Statement", [
        "Author: [Enter student name]",
        "I completed the project as an individual submission and reviewed the algorithms, application, tests, and written materials. AI tools assisted with planning, code drafting, and documentation; I checked and understood the final work before submission.",
        "Edit this statement to reflect the actual contributions and your course's AI-use rules. Do not sign it until it is accurate.",
    ])
