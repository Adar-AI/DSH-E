# 解析项目使用的 Python 解释器（被 run.ps1 / test.ps1 / smoke_backend.ps1 复用）。
#
# 优先级：调用方传入 > 环境变量 PROJECT_PYTHON > 项目约定的 venv > 本机 Python 安装 > PATH
# 找到之后会做一次依赖自检，缺依赖时给出明确提示。

function Get-ProjectPython {
    [CmdletBinding()]
    param([string]$Override = '')

    $candidates = @()
    if ($Override) { $candidates += $Override }
    if ($env:PROJECT_PYTHON) { $candidates += $env:PROJECT_PYTHON }
    $candidates += 'C:\Users\19181\PyCharmMiscProject\.venv\Scripts\python.exe'
    $candidates += 'E:\PyCharmMiscProject\python.exe'

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) { return $candidate }
    }

    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }

    throw '找不到可用的 Python 解释器：请设置 $env:PROJECT_PYTHON 指向 python.exe'
}

function Test-ProjectDependencies {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Python)

    $probe = 'import fastapi, uvicorn, langgraph, sqlite_vec, sentence_transformers, pypdfium2, PIL; print("ok")'
    $output = & $Python -c $probe 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "解释器 $Python 缺少依赖：" -ForegroundColor Yellow
        Write-Host ($output | Out-String) -ForegroundColor DarkGray
        Write-Host "请先安装：& '$Python' -m pip install -r requirements.txt" -ForegroundColor Yellow
        return $false
    }
    return $true
}
