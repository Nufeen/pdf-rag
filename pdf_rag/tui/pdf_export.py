from datetime import datetime
from pathlib import Path

from fpdf import FPDF


def export_to_pdf(question: str, answer: str, output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"pedro_{ts}.pdf"

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", style="B", size=12)
    pdf.multi_cell(0, 8, f"Q: {question}")
    pdf.ln(4)

    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, answer)

    pdf.output(str(path))
    return path
