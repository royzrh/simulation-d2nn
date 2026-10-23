$ErrorActionPreference='Stop'
$Host.UI.RawUI.WindowTitle='D2NN imaging - 1550 nm S6 / S7 scans (queued)'
Set-Location -LiteralPath 'H:\PROJECT\study\simulation_d2nn\imaging_reproduction'
Write-Host '1550 nm architecture scans: all input distances remain positive.' -ForegroundColor Cyan
Write-Host 'Waiting for the existing main and 10000-image workflows, then executing one experiment at a time.'
& 'C:\Users\dell\miniforge3\envs\d2nn\python.exe' -B -u .\imaging\workflows\run_suite.py --stage adapted1550 2>&1 | Tee-Object -FilePath '.\misc\logs\suite1550.log' -Append
if ($LASTEXITCODE -ne 0) { Write-Host '1550 nm scan queue stopped; inspect results/suite1550_state.json.' -ForegroundColor Red }
