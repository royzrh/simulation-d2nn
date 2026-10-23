$ErrorActionPreference='Stop'
$Host.UI.RawUI.WindowTitle='D2NN imaging - uniform initialization 50 epochs (queued)'
Set-Location -LiteralPath 'H:\PROJECT\study\simulation_d2nn\imaging_reproduction'
$python='C:\Users\dell\miniforge3\envs\d2nn\python.exe'
try {
 $check=Get-Content -LiteralPath '.\misc\continuation_check_status.json' -Raw | ConvertFrom-Json
 if($check.exit_code -ne 0){throw 'Continuation verification did not pass'}
 Write-Host 'Waiting for validation parameter checks. This terminal will display training when it starts.' -ForegroundColor Cyan
 $deadline=(Get-Date).AddHours(12)
 while(-not (Test-Path -LiteralPath '.\misc\optimizer_range_status.json')) {
  if((Get-Date) -gt $deadline){throw 'Parameter-check wait timed out; training was not started'}
  Start-Sleep -Seconds 10
 }
 $status=Get-Content -LiteralPath '.\misc\optimizer_range_status.json' -Raw | ConvertFrom-Json
 if($status.exit_code -ne 0){throw 'Parameter checks failed'}
 foreach($name in @('lr010_init01_pilot10','lr001_init07_pilot10')) {
  $complete=Get-Content -LiteralPath ('.\results\'+$name+'\training_complete.json') -Raw | ConvertFrom-Json
  if($complete.epochs -ne 10){throw "Incomplete prerequisite: $name"}
 }
 $Host.UI.RawUI.WindowTitle='D2NN imaging - uniform initialization (epochs 11-50)'
 Write-Host 'Continuing 10 -> 50 total epochs in NEW uniform_init_full50. Original run and images are preserved.' -ForegroundColor Yellow
 & $python -B -u .\train.py --name uniform_init_full50 --epochs 50 --oversample 2 --init-std 0 --input-encoding amplitude_gray --lr 0.005 --seed 42 --continue-from '.\results\uniform_init_pilot10\last_checkpoint.pt' 2>&1 | Tee-Object -FilePath '.\misc\logs\train_uniform_init_full50.log'
 if($LASTEXITCODE -ne 0){throw "Uniform continuation failed: $LASTEXITCODE"}
 @{exit_code=0;finished_at=(Get-Date).ToString('o');test_used=$false} | ConvertTo-Json | Set-Content '.\misc\uniform_full50_status.json'
 Write-Host '50 total epochs complete. Compare validation before any test evaluation.' -ForegroundColor Green
} catch {
 @{exit_code=1;error=$_.ToString()} | ConvertTo-Json | Set-Content '.\misc\uniform_full50_status.json'
 Write-Host $_ -ForegroundColor Red
}
