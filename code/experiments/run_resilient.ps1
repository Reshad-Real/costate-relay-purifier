# Runs the remaining simulation stages with automatic retry (stages e2 and e4 resume from saved progress).
param([string]$Stages = "e2 e4 e5 e7 e8 e9 e1b", [int]$Workers = 8, [int]$MaxAttempts = 6)
$py = "C:\Users\Asus ROG Strix G16\AppData\Local\Python\bin\python.exe"
$root = "E:\FYDP\Revision"
$env:AQ_WORKERS = "$Workers"
Remove-Item Env:QUICK -ErrorAction SilentlyContinue
for ($a = 1; $a -le $MaxAttempts; $a++) {
    Add-Content "$root\results_run.log" ("{0} === attempt {1} (workers={2})" -f (Get-Date -Format HH:mm:ss), $a, $Workers)
    $p = Start-Process -FilePath $py -ArgumentList "run_all.py $Stages" -WorkingDirectory "$root\code\experiments" `
         -RedirectStandardOutput "$root\results_run_attempt$a.log" -RedirectStandardError "$root\results_run_attempt$a.err" `
         -WindowStyle Hidden -PassThru -Wait
    Get-Content "$root\results_run_attempt$a.log" | Add-Content "$root\results_run.log"
    if ($p.ExitCode -eq 0) { Add-Content "$root\results_run.log" "ALL STAGES COMPLETE"; break }
    Add-Content "$root\results_run.log" ("{0} attempt {1} failed (exit {2}); retrying in 60 s" -f (Get-Date -Format HH:mm:ss), $a, $p.ExitCode)
    Start-Sleep 60
}
