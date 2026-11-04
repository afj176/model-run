param(
    [string]$ModelsDir,
    [string[]]$ExtraPaths = @(),
    [int]$Port = 8766,
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
$arguments = @("-m", "model_run.server", "--port", "$Port")
if ($ModelsDir) { $arguments += @("--models-dir", ('"' + $ModelsDir + '"')) }
foreach ($config in $ExtraPaths) { $arguments += @("--extra-paths", ('"' + $config + '"')) }
$logDirectory = Join-Path $PSScriptRoot ".tmp"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) { throw "Port $Port is already in use. Existing processes were left unchanged." }
$process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDirectory "app-server.stdout.log") -RedirectStandardError (Join-Path $logDirectory "app-server.stderr.log")
$ready = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    if ($process.HasExited) { throw "Model Run exited. Check .tmp/app-server.stderr.log." }
    try {
        $response = Invoke-WebRequest "http://127.0.0.1:$Port/api/session" -TimeoutSec 1 -NoProxy
        if ($response.StatusCode -eq 200) { $ready = $true; break }
    } catch { Start-Sleep -Milliseconds 200 }
}
if (-not $ready) { throw "Server did not become reachable. PID $($process.Id); check the local logs." }
Write-Output "Model Run is ready at http://127.0.0.1:$Port/ (PID $($process.Id))"
