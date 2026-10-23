$ErrorActionPreference='Stop'
$Host.UI.RawUI.WindowTitle='D2NN imaging - numerical sampling control OS1'
Set-Location -LiteralPath 'H:\PROJECT\study\simulation_d2nn\imaging_reproduction'
Write-Host 'Sampling control: 50 epochs at OS=1; same images, geometry and intensity MSE' -ForegroundColor Cyan
Write-Host 'The trained masks will also be checked on finer grids to detect numerical artifacts.'
& 'C:\Users\dell\miniforge3\envs\d2nn\python.exe' -B -u .\train.py --name baseline_os1_diagnostic --epochs 50 --oversample 1 --init-std 0.7 --input-encoding amplitude_gray --lr 0.005 --seed 42 2>&1 | Tee-Object -FilePath '.\misc\logs\train_baseline_os1_diagnostic.log'
$jobExit=$LASTEXITCODE
if($jobExit -eq 0){
 Write-Host 'Training complete; evaluating unchanged masks on OS=1/2/3/4.' -ForegroundColor Yellow
 & 'C:\Users\dell\miniforge3\envs\d2nn\python.exe' -B -u .\imaging\evaluation\evaluate.py --checkpoint '.\results\baseline_os1_diagnostic\best_model.pt' --convergence 2>&1 | Tee-Object -FilePath '.\misc\logs\evaluate_baseline_os1_diagnostic.log'
 $jobExit=$LASTEXITCODE
}
@{exit_code=$jobExit;finished_at=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath '.\misc\baseline_os1_diagnostic_status.json' -Encoding utf8
if($jobExit -ne 0){Write-Host "Sampling control failed with exit code $jobExit" -ForegroundColor Red}
else{Write-Host 'Sampling control completed.' -ForegroundColor Green}
