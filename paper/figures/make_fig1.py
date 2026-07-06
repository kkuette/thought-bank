"""Render Figure 1 (architecture schematic) from its hand-drawn SVG master.

The SVG is the editable source; this script converts it to the PDF used by
the paper and a PNG preview. Pure-Python toolchain (svglib + reportlab +
pymupdf), no system cairo needed. Note: keep the SVG to ASCII-ish text —
unicode sub/superscripts have no glyphs in the default PDF fonts.

Usage: python make_fig1.py
"""

import os

import fitz
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg

HERE = os.path.dirname(__file__)
SVG = os.path.join(HERE, "fig1_architecture.svg")
PDF = os.path.join(HERE, "fig1_architecture.pdf")
PNG = os.path.join(HERE, "fig1_architecture.png")


def main():
    renderPDF.drawToFile(svg2rlg(SVG), PDF)
    doc = fitz.open(PDF)
    doc[0].get_pixmap(matrix=fitz.Matrix(2.2, 2.2)).save(PNG)
    print("wrote", PDF, "and", PNG)


if __name__ == "__main__":
    main()
