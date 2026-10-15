# Stops stray Python servers left running from other projects so the transcriber
# has enough RAM. Run in PowerShell:  .\free_memory.ps1
# It only touches Python processes whose command line matches the patterns below.

$patterns = 'piper\.http_server|uvicorn app\.main|serve\.py|app_from_notebook\.py|run_notebook\.py'

$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match $patterns }

if (-not $procs) {
    Write-Host "No stray Python servers found."
} else {
    $procs | ForEach-Object {
        Write-Host ("Stopping PID {0}: {1}" -f $_.ProcessId, $_.CommandLine.Substring(0, [math]::Min(80, $_.CommandLine.Length)))
    }
    Stop-Process -Id ($procs.ProcessId) -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

$os = Get-CimInstance Win32_OperatingSystem
Write-Host ("Free RAM now: {0:N0} MB of {1:N0} MB" -f ($os.FreePhysicalMemory / 1KB), ($os.TotalVisibleMemorySize / 1KB))
Write-Host ("Free disk C: {0:N0} MB" -f ((Get-PSDrive C).Free / 1MB))
Write-Host "Now start the transcriber with:  python run_notebook.py   (opens on http://127.0.0.1:8002)"
