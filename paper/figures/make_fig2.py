"""Render Figure 2 (task schematic) from its hand-drawn SVG master.

Same toolchain as make_fig1.py: svglib + reportlab + pymupdf.
Usage: python make_fig2.py
"""

import os

import fitz
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg

HERE = os.path.dirname(__file__)
SVG = os.path.join(HERE, "fig2_task.svg")
PDF = os.path.join(HERE, "fig2_task.pdf")
PNG = os.path.join(HERE, "fig2_task.png")


def main():
    renderPDF.drawToFile(svg2rlg(SVG), PDF)
    doc = fitz.open(PDF)
    doc[0].get_pixmap(matrix=fitz.Matrix(2.2, 2.2)).save(PNG)
    print("wrote", PDF, "and", PNG)


if __name__ == "__main__":
    main()
