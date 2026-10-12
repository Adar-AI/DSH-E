# 启动后端服务。
#
# 用法：
#   .\scripts\run.ps1                 # 真实模型模式（默认，读 .env）
#   .\scripts\run.ps1 -Mode mock      # 离线模式：不调用任何模型，先把后端逻辑跑通
#   .\scripts\run.ps1 -Reload         # 开发时热重载
#   .\scripts\run.ps1 -Python 'x:\path\to\python.exe'

[CmdletBinding()]
param(
    [ValidateSet('auto', 'mock')][string]$Mode = 'auto',
    [int]$Port = 0,
    [string]$HostName = '',
    [switch]$Reload,
    [string]$Python = ''
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'resolve_python.ps1')

$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $python = Get-ProjectPython -Override $Python
    if (-not (Test-ProjectDependencies -Python $python)) { exit 1 }

    $env:LLM_MODE = $Mode

    $arguments = @('-m', 'uvicorn', 'app.main:app', '--log-level', 'info')
    if ($HostName) { $arguments += @('--host', $HostName) }
    if ($Port -gt 0) { $arguments += @('--port', "$Port") }
    if ($Reload) { $arguments += '--reload' }

    $hostText = if ($HostName) { $HostName } else { '127.0.0.1（默认，见 .env）' }
    $portText = if ($Port -gt 0) { $Port } else { '8000（默认，见 .env）' }
    Write-Host "解释器：$python" -ForegroundColor Cyan
    Write-Host "模型模式：$Mode$(if ($Mode -eq 'mock') { '（离线，不调用任何真实模型）' })" -ForegroundColor Cyan
    Write-Host "访问地址：http://${hostText}:${portText}" -ForegroundColor Green
    Write-Host ''

    & $python @arguments
}
finally {
    Pop-Location
}
