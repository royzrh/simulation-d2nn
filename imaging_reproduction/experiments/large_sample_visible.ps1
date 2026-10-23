$ErrorActionPreference='Stop'
$Host.UI.RawUI.WindowTitle='D2NN imaging - 10000 training images (queued)'
Set-Location -LiteralPath 'H:\PROJECT\study\simulation_d2nn\imaging_reproduction'
Write-Host 'Separate larger-sample reproduction, supplement p10. Original held-out set and original dataset stay unchanged.' -ForegroundColor Cyan
& 'C:\Users\dell\miniforge3\envs\d2nn\python.exe' -B -u .\imaging\workflows\run_large_sample.py 2>&1 | Tee-Object -FilePath '.\misc\logs\large_sample_queue.log'
