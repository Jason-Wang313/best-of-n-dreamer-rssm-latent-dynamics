$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
}
finally {
  Pop-Location
}
