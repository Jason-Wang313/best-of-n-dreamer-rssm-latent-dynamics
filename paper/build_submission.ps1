param(
  [string]$FinalCopy = "$PSScriptRoot\final\best of n dreamer rssm latent dynamics-v3.pdf",
  [string]$DesktopCopy = ""
)

$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  if ($FinalCopy) {
    $finalDir = Split-Path -Parent $FinalCopy
    if ($finalDir) {
      New-Item -ItemType Directory -Path $finalDir -Force | Out-Null
    }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "main.pdf") -Destination $FinalCopy -Force
    Write-Host "Final PDF: $FinalCopy"
  }
  if ($DesktopCopy) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "main.pdf") -Destination $DesktopCopy -Force
    Write-Host "Desktop PDF: $DesktopCopy"
  }
}
finally {
  Pop-Location
}
