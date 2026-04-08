from datetime import datetime
from pathlib import Path

from fpdf import FPDF

FONTS_DIR = Path(__file__).parent / "fonts"


def export_to_pdf(question: str, answer: str, output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"pedro_{ts}.pdf"

    pdf = FPDF()
    pdf.add_page()

    pdf.add_font("dejavu-sans", style="", fname=str(FONTS_DIR / "DejaVuSans.ttf"))
    pdf.add_font("dejavu-sans", style="b", fname=str(FONTS_DIR / "DejaVuSans-Bold.ttf"))

    pdf.set_font("dejavu-sans", style="B", size=12)
    pdf.multi_cell(0, 8, f"Q: {question}")
    pdf.ln(4)

    pdf.set_font("dejavu-sans", size=11)
    pdf.multi_cell(0, 6, answer)

    pdf.output(str(path))
    return path
