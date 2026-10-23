$ErrorActionPreference='Stop'
$Host.UI.RawUI.WindowTitle='D2NN imaging - validation parameter range checks'
Set-Location -LiteralPath 'H:\PROJECT\study\simulation_d2nn\imaging_reproduction'
Write-Host 'Validation-only optimizer/initialization checks. Original images and splits remain fixed.' -ForegroundColor Cyan
if(Get-Process -Id 28608 -ErrorAction SilentlyContinue){
 Write-Host 'Waiting for sampling-control training PID 28608 before using additional GPU time.'
 Wait-Process -Id 28608 -ErrorAction SilentlyContinue
}
$jobs=@(
 @{name='lr010_init01_pilot10';lr='0.01';init='0.1'},
 @{name='lr001_init07_pilot10';lr='0.001';init='0.7'}
)
foreach($job in $jobs){
 $Host.UI.RawUI.WindowTitle='D2NN imaging - '+$job.name
 Write-Host "`nSTART $($job.name): 10 epochs, same 1500 train / 200 validation images" -ForegroundColor Yellow
 $jobArgs=@('-B','-u','.\train.py','--name',$job.name,'--epochs','10','--oversample','2','--init-std',$job.init,'--input-encoding','amplitude_gray','--lr',$job.lr,'--seed','42')
 & 'C:\Users\dell\miniforge3\envs\d2nn\python.exe' @jobArgs 2>&1 | Tee-Object -FilePath ('.\misc\logs\train_'+$job.name+'.log')
 if($LASTEXITCODE -ne 0){throw "Validation run $($job.name) failed: $LASTEXITCODE"}
}
@{exit_code=0;finished_at=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath '.\misc\optimizer_range_status.json' -Encoding utf8
Write-Host 'Validation range checks complete. No test-set evaluation was performed.' -ForegroundColor Green
