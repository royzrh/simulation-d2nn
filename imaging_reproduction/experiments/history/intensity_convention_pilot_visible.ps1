$ErrorActionPreference='Stop'
$Host.UI.RawUI.WindowTitle='D2NN imaging - grayscale intensity convention comparison'
Set-Location -LiteralPath 'H:\PROJECT\study\simulation_d2nn\imaging_reproduction'
Write-Host 'Validation-only gray encoding comparison: input amplitude sqrt(gray), target intensity gray' -ForegroundColor Cyan
Write-Host 'Same geometry, Adam/MSE, 1500 train / 200 validation. Test images are not evaluated.'
& 'C:\Users\dell\miniforge3\envs\d2nn\python.exe' -B -u .\train.py --name intensity_gray_pilot10 --epochs 10 --init-std 0.7 --input-encoding intensity_gray --lr 0.005 --seed 42 2>&1 | Tee-Object -FilePath '.\misc\logs\train_intensity_gray_pilot10.log'
$jobExit=$LASTEXITCODE
@{exit_code=$jobExit;finished_at=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath '.\misc\intensity_gray_pilot10_status.json' -Encoding utf8
if($jobExit -ne 0){Write-Host "Training failed with exit code $jobExit" -ForegroundColor Red}
else{Write-Host 'Convention comparison completed.' -ForegroundColor Green}
