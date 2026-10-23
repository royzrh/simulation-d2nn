$ErrorActionPreference='Stop'
$Host.UI.RawUI.WindowTitle='D2NN imaging - uniform initialization comparison'
Set-Location -LiteralPath 'H:\PROJECT\study\simulation_d2nn\imaging_reproduction'
Write-Host 'Validation-only initialization comparison: uniform phase, 10 epochs' -ForegroundColor Cyan
Write-Host 'Same 1500 training / 200 validation images. Test set is not evaluated.'
& 'C:\Users\dell\miniforge3\envs\d2nn\python.exe' -B -u .\train.py --name uniform_init_pilot10 --epochs 10 --init-std 0 --lr 0.005 --seed 42 2>&1 | Tee-Object -FilePath '.\misc\logs\train_uniform_init_pilot10.log'
$jobExit=$LASTEXITCODE
@{exit_code=$jobExit;finished_at=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath '.\misc\uniform_init_pilot10_status.json' -Encoding utf8
if($jobExit -ne 0){Write-Host "Training failed with exit code $jobExit" -ForegroundColor Red}
else{Write-Host 'Initialization comparison completed.' -ForegroundColor Green}
