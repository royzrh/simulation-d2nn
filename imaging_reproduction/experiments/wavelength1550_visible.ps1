$ErrorActionPreference='Stop'
$Host.UI.RawUI.WindowTitle='D2NN imaging - 1550 nm / fine grid OS16 / 50 epochs'
Set-Location -LiteralPath 'H:\PROJECT\study\simulation_d2nn\imaging_reproduction'
$researchPython='C:\Users\dell\miniforge3\envs\d2nn\python.exe'
$researchRun='.\results\wavelength1550_paper_geometry_os16'
$researchLog='.\misc\logs\wavelength1550_paper_geometry_os16.log'
try {
    @{exit_code=$null;status='training';wavelength_nm=1550} | ConvertTo-Json | Set-Content '.\misc\wavelength1550_status.json'
    Write-Host '1550 nm; five phase masks; pitch 300 um; input/gap/output = 4/4/7 mm.' -ForegroundColor Cyan
    Write-Host 'Paper dimensions retained as a controlled numerical comparison. Hardware geometry remains unconfirmed.'
    Write-Host 'OS16; padding 16/15; effective batch4 accumulated from microbatch1. Layer activations recomputed during backward.' -ForegroundColor Cyan
    & $researchPython -B -u .\imaging\validation\check_physics.py 2>&1 | Tee-Object -FilePath $researchLog -Append
    if ($LASTEXITCODE -ne 0) { throw 'Physics/gradient-accumulation verification failed; training will not start.' }
    $researchArguments=@('-B','-u','.\train.py','--name','wavelength1550_paper_geometry_os16','--wavelength-nm','1550','--pitch-um','300','--layers','5','--input-mm','4','--gap-mm','4','--output-mm','7','--epochs','50','--batch-size','4','--micro-batch-size','1','--activation-checkpointing','--oversample','16','--pad-factor','1.0666666666666667','--lr','0.005','--init-std','0.7','--seed','42','--input-encoding','amplitude_gray')
    if (Test-Path -LiteralPath "$researchRun\last_checkpoint.pt") {
        $researchArguments+=@('--resume',"$researchRun\last_checkpoint.pt")
    }
    if (-not (Test-Path -LiteralPath "$researchRun\training_complete.json")) {
        & $researchPython @researchArguments 2>&1 | Tee-Object -FilePath $researchLog -Append
        if ($LASTEXITCODE -ne 0) { throw '1550 nm training failed; preserve checkpoints and inspect log.' }
    }
    @{exit_code=$null;status='evaluating';wavelength_nm=1550} | ConvertTo-Json | Set-Content '.\misc\wavelength1550_status.json'
    & $researchPython -B -u .\imaging\evaluation\evaluate.py --checkpoint "$researchRun\best_model.pt" --convergence 2>&1 | Tee-Object -FilePath $researchLog -Append
    if ($LASTEXITCODE -ne 0) { throw '1550 nm evaluation failed.' }
    & $researchPython -B -u .\imaging\evaluation\figures.py --checkpoint "$researchRun\best_model.pt" 2>&1 | Tee-Object -FilePath $researchLog -Append
    if ($LASTEXITCODE -ne 0) { throw '1550 nm figure export failed.' }
    & $researchPython -B -u .\imaging\evaluation\report.py 2>&1 | Tee-Object -FilePath $researchLog -Append
    if ($LASTEXITCODE -ne 0) { throw 'Report generation failed.' }
    @{exit_code=0;status='training_and_evaluation_completed';wavelength_nm=1550;hardware_validation_complete=$false} | ConvertTo-Json | Set-Content '.\misc\wavelength1550_status.json'
    Write-Host '1550 nm numerical run completed. Images and measurements are in the experiment directory.' -ForegroundColor Green
} catch {
    @{exit_code=1;status='failed';error=$_.Exception.Message;wavelength_nm=1550} | ConvertTo-Json | Set-Content '.\misc\wavelength1550_status.json'
    Write-Host $_ -ForegroundColor Red
}
