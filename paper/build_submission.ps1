param(
  [string]$DesktopCopy = "$env:USERPROFILE\OneDrive\Desktop\best of n dreamer rssm latent dynamics-v2.pdf"
)

$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  if ($DesktopCopy) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "main.pdf") -Destination $DesktopCopy -Force
    Write-Host "Desktop PDF: $DesktopCopy"
  }
}
finally {
  Pop-Location
}
