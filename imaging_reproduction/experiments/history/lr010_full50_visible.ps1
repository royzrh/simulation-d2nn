$ErrorActionPreference='Stop'
$Host.UI.RawUI.WindowTitle='D2NN imaging - LR 0.01 full comparison (queued)'
Set-Location -LiteralPath 'H:\PROJECT\study\simulation_d2nn\imaging_reproduction'
try {
 Write-Host 'Waiting for uniform-initialization comparison; this avoids adding a third GPU training process.' -ForegroundColor Cyan
 $deadline=(Get-Date).AddHours(12)
 while(-not (Test-Path -LiteralPath '.\misc\uniform_full50_status.json')) {
  if((Get-Date) -gt $deadline){throw 'Wait timed out; no training started'}
  Start-Sleep -Seconds 10
 }
 $status=Get-Content -LiteralPath '.\misc\uniform_full50_status.json' -Raw | ConvertFrom-Json
 if($status.exit_code -ne 0){throw 'Prior comparison failed; inspect before continuing'}
 $Host.UI.RawUI.WindowTitle='D2NN imaging - LR 0.01, init 0.1 (epochs 11-50)'
 & 'C:\Users\dell\miniforge3\envs\d2nn\python.exe' -B -u .\train.py --name lr010_init01_full50 --epochs 50 --oversample 2 --init-std 0.1 --input-encoding amplitude_gray --lr 0.01 --seed 42 --continue-from '.\results\lr010_init01_pilot10\last_checkpoint.pt' 2>&1 | Tee-Object -FilePath '.\misc\logs\train_lr010_init01_full50.log'
 if($LASTEXITCODE -ne 0){throw "Continuation failed: $LASTEXITCODE"}
 @{exit_code=0;finished_at=(Get-Date).ToString('o');test_used=$false} | ConvertTo-Json | Set-Content '.\misc\lr010_full50_status.json'
 Write-Host '50 total epochs complete; validation comparison ready.' -ForegroundColor Green
} catch {
 @{exit_code=1;error=$_.ToString()} | ConvertTo-Json | Set-Content '.\misc\lr010_full50_status.json'
 Write-Host $_ -ForegroundColor Red
}
