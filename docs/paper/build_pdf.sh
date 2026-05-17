#!/usr/bin/env bash
# Build termplanmt_v3.pdf with bibliography (pdflatex + bibtex).
set -euo pipefail
cd "$(dirname "$0")"
main=termplanmt_v3
pdflatex -interaction=nonstopmode "$main.tex"
bibtex "$main"
pdflatex -interaction=nonstopmode "$main.tex"
pdflatex -interaction=nonstopmode "$main.tex"
echo "Wrote $(pwd)/${main}.pdf"
