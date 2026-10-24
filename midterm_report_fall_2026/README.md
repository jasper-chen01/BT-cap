# BT-CAP Midterm Report (Fall 2026)

This directory contains the complete IEEE-style LaTeX source and compiled PDF for the BT-CAP midterm report.

## Files

- `main.tex` - report source
- `references.bib` - BibTeX bibliography
- `figures/portal_card.png` - portal evidence figure used by the report
- `main.pdf` - PDF compiled from `main.tex`
- `Midterm_Report_Fall_2026.pdf` - submission-ready copy of the compiled PDF

The system-architecture figure is authored directly in `main.tex` with TikZ, so no separate image is required for it.

## Build

A TeX distribution with `IEEEtran`, `natbib`, `TikZ`, and standard graphics packages is required. From this directory, run:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The generated `main.pdf` should contain four pages, two figures, one performance table, and four bibliography entries.
