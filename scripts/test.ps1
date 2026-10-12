# 顺序执行两套测试：Node 文档/仓库校验 + Python 单元测试。
#
# 用法：.\scripts\test.ps1 [-Python 'x:\path\to\python.exe']
# 说明：之所以不用 npm script 串起来，是因为受限沙箱下 Node 无法带管道派生
#       子进程（spawn EPERM），PowerShell 直接调用两个解释器最稳。

[CmdletBinding()]
param([string]$Python = '')

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'resolve_python.ps1')

$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    Write-Host '=== 1/2 仓库文档与完整性校验（Node）===' -ForegroundColor Cyan
    npm test
    if ($LASTEXITCODE -ne 0) { throw "npm test 失败（退出码 $LASTEXITCODE）" }

    $pythonPath = Get-ProjectPython -Override $Python
    if (-not (Test-ProjectDependencies -Python $pythonPath)) { exit 1 }

    Write-Host "=== 2/2 Python 单元测试（$pythonPath）===" -ForegroundColor Cyan
    & $pythonPath -m pytest tests
    if ($LASTEXITCODE -ne 0) { throw "pytest 失败（退出码 $LASTEXITCODE）" }

    Write-Host '全部测试通过 ✔' -ForegroundColor Green
}
finally {
    Pop-Location
}
