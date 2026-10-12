# 后端全链路冒烟测试：不需要前端、也不需要真实模型（配 -Mode mock 即可全离线验证）。
#
# 覆盖：健康检查 → 静态页 → 文字对话(SSE) → 图片对话(SSE) → 未命中兜底 →
#       工单定稿 → 列表筛选 → 状态流转(含非法流转) → CSV 导出 → 会话删除后工单留存
#
# 用法：
#   .\scripts\smoke_backend.ps1                                  # 测已在运行的服务
#   .\scripts\smoke_backend.ps1 -BaseUrl http://127.0.0.1:8000
#   .\scripts\smoke_backend.ps1 -Ingest                          # 知识库为空时先入库演示资料

[CmdletBinding()]
param(
    [string]$BaseUrl = 'http://127.0.0.1:8000',
    [int]$TimeoutSec = 300,
    [switch]$Ingest,
    [string]$Python = ''
)

$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot 'resolve_python.ps1')

$script:Passed = 0
$script:Failed = 0
$root = Split-Path -Parent $PSScriptRoot

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

function Test-Step {
    param([string]$Name, [scriptblock]$Body)
    try {
        & $Body
        Write-Host "  [PASS] $Name" -ForegroundColor Green
        $script:Passed++
    }
    catch {
        Write-Host "  [FAIL] $Name -> $($_.Exception.Message)" -ForegroundColor Red
        $script:Failed++
    }
}

function ConvertFrom-Sse {
    param([string]$Raw)
    $events = @()
    foreach ($block in ($Raw -split "`n`n")) {
        if (-not $block.Trim()) { continue }
        $name = 'message'
        $data = '{}'
        foreach ($line in ($block -split "`n")) {
            $line = $line.TrimEnd("`r")
            if ($line.StartsWith('event:')) { $name = $line.Substring(6).Trim() }
            elseif ($line.StartsWith('data:')) { $data = $line.Substring(5).Trim() }
        }
        try { $parsed = $data | ConvertFrom-Json } catch { $parsed = $null }
        $events += [pscustomobject]@{ Name = $name; Data = $parsed }
    }
    return $events
}

function Get-SseEvent {
    param($Events, [string]$Name)
    return ($Events | Where-Object { $_.Name -eq $Name } | Select-Object -First 1)
}

function Invoke-Chat {
    param([hashtable]$Form)
    $response = Invoke-WebRequest -Uri "$BaseUrl/api/chat" -Method Post -Form $Form -TimeoutSec $TimeoutSec
    return (ConvertFrom-Sse $response.Content)
}

Write-Host "冒烟测试目标：$BaseUrl" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------- 健康检查

$health = $null
Test-Step '健康检查 /api/health' {
    $script:health = Invoke-RestMethod -Uri "$BaseUrl/api/health" -TimeoutSec 30
    Assert-True ($script:health.status -eq 'ok') "status 不是 ok"
    Assert-True ($script:health.chat.channels.Count -ge 1) '没有对话通道'
}
$health = $script:health

if ($health) {
    Write-Host ("    知识库：{0} 份文档 / {1} 个分块 / {2} 条向量，检索模式 {3}" -f `
        $health.knowledge.documents, $health.knowledge.chunks, $health.knowledge.vectors, $health.retrieval.mode) -ForegroundColor DarkGray
    Write-Host ("    通道：{0}；视觉：{1}" -f `
        (($health.chat.channels | ForEach-Object { "$($_.name)/$($_.model)" }) -join ', '), $health.vision.model) -ForegroundColor DarkGray
}

# ------------------------------------------------- 知识库为空时先入库演示资料

if ($health -and $health.knowledge.documents -eq 0) {
    Write-Host "  知识库为空，先入库演示资料…" -ForegroundColor Yellow
    $pythonPath = Get-ProjectPython -Override $Python
    & $pythonPath (Join-Path $root 'scripts\ingest_samples.py') 2>&1 | Select-Object -Last 4 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    $health = Invoke-RestMethod -Uri "$BaseUrl/api/health" -TimeoutSec 30
}
elseif ($Ingest) {
    $pythonPath = Get-ProjectPython -Override $Python
    & $pythonPath (Join-Path $root 'scripts\ingest_samples.py') 2>&1 | Select-Object -Last 4 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
}

# ---------------------------------------------------------------- 静态页

Test-Step '静态首页可访问' {
    $page = Invoke-WebRequest -Uri "$BaseUrl/" -TimeoutSec 30
    Assert-True ($page.StatusCode -eq 200) "首页返回 $($page.StatusCode)"
    Assert-True ($page.Content -match '智能客服') '首页内容不含项目名'
}

# ---------------------------------------------------------------- 文字对话

$conversationId = $null
$textEvents = $null
Test-Step '文字对话：SSE 事件序列完整' {
    $script:textEvents = Invoke-Chat @{ text = '我的打印机报错E05，硒鼓刚换过，怎么处理？' }
    $names = $script:textEvents | ForEach-Object { $_.Name }
    Assert-True ($names -contains 'meta') "缺少 meta 事件（实际：$($names -join ',')）"
    Assert-True ($names -contains 'retrieval') '缺少 retrieval 事件'
    Assert-True ($names -contains 'token') '缺少 token 流式事件'
    Assert-True ($names -contains 'done') '缺少 done 事件'

    $meta = Get-SseEvent $script:textEvents 'meta'
    $done = Get-SseEvent $script:textEvents 'done'
    $script:conversationId = $meta.Data.conversation_id
    Assert-True ($script:conversationId -like 'conv-*') "会话 ID 异常：$($script:conversationId)"
    Assert-True ($done.Data.answer.Length -gt 10) '回答为空'
    $streamed = ($script:textEvents | Where-Object { $_.Name -eq 'token' } | ForEach-Object { $_.Data.text }) -join ''
    Assert-True ($streamed -eq $done.Data.answer) 'token 拼接结果与最终回答不一致'
}

Test-Step '文字对话：检索命中并给出依据' {
    $retrieval = Get-SseEvent $script:textEvents 'retrieval'
    Assert-True ($retrieval.Data.hit -eq $true) "未命中知识库（最高分 $($retrieval.Data.best_score)）"
    Assert-True ($retrieval.Data.chunks.Count -ge 1) '没有返回支撑片段'
    $done = Get-SseEvent $script:textEvents 'done'
    Assert-True ($done.Data.citations.Count -ge 1) '回答没有引用来源'
    Assert-True ($done.Data.ticket_draft.error_info -match 'E05') "工单草稿没抓到报错码：$($done.Data.ticket_draft.error_info)"
}

Test-Step '会话与消息已落库' {
    $messages = Invoke-RestMethod -Uri "$BaseUrl/api/conversations/$($script:conversationId)/messages" -TimeoutSec 30
    Assert-True ($messages.items.Count -eq 2) "消息条数应为 2，实际 $($messages.items.Count)"
    Assert-True ($messages.items[0].role -eq 'user') '第一条不是用户消息'
    Assert-True ($messages.items[1].role -eq 'assistant') '第二条不是客服消息'
}

# ---------------------------------------------------------------- 图片对话

$pngBase64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=='
$imagePath = Join-Path $root 'data\test_tmp\smoke\smoke.png'
New-Item -ItemType Directory -Force (Split-Path -Parent $imagePath) | Out-Null
[System.IO.File]::WriteAllBytes($imagePath, [Convert]::FromBase64String($pngBase64))

Test-Step '图片对话：上传成功且推送 vision 事件' {
    $events = Invoke-Chat @{ text = '屏幕上有个报错提示，帮我看下'; image = Get-Item $imagePath }
    $names = $events | ForEach-Object { $_.Name }
    Assert-True ($names -contains 'vision') "缺少 vision 事件（实际：$($names -join ',')）"
    $vision = Get-SseEvent $events 'vision'
    Assert-True ($null -ne $vision.Data.ok) 'vision 事件缺少 ok 字段'
    if ($vision.Data.ok) {
        Assert-True ($vision.Data.summary.Length -gt 0) 'vision 识别成功但描述为空'
    }
    Assert-True ($names -contains 'done') '图片对话没有正常结束'
}

Test-Step '图片已落盘到 uploads 目录' {
    $uploads = Get-ChildItem (Join-Path $root 'data\uploads') -Filter 'img-*' -ErrorAction SilentlyContinue
    Assert-True ($uploads.Count -ge 1) 'data/uploads 下没有保存上传的图片'
}

# ---------------------------------------------------------------- 兜底链路

Test-Step '未命中知识库：走兜底话术且不编造' {
    $events = Invoke-Chat @{ text = '帮我订一张去上海的机票，顺便推荐酒店' }
    $retrieval = Get-SseEvent $events 'retrieval'
    $done = Get-SseEvent $events 'done'
    Assert-True ($retrieval.Data.hit -eq $false) "这个问题不该命中知识库（最高分 $($retrieval.Data.best_score)）"
    Assert-True ($done.Data.trace.fallback_used -eq $true) 'fallback_used 应为 true'
    Assert-True ($done.Data.citations.Count -eq 0) '兜底回答不应带引用'
    Assert-True ($done.Data.answer -match '未收录') "兜底话术缺少「未收录」表述：$($done.Data.answer)"
}

# ---------------------------------------------------------------- 工单定稿

$ticket = $null
Test-Step '结束会话：定稿生成工单' {
    $script:ticket = Invoke-RestMethod -Uri "$BaseUrl/api/conversations/$($script:conversationId)/close" -Method Post -TimeoutSec $TimeoutSec
    Assert-True ($script:ticket.ticket_no -match '^T-\d{8}-\d{4}$') "工单号格式异常：$($script:ticket.ticket_no)"
    Assert-True ($script:ticket.status -eq '待处理') "初始状态应为待处理，实际 $($script:ticket.status)"
    Assert-True ($script:ticket.extraction.source -in @('llm', 'draft')) "抽取来源异常：$($script:ticket.extraction.source)"
    Assert-True ($script:ticket.customer_request.Length -gt 0) '客户诉求为空'
    Assert-True ($script:ticket.events.Count -ge 1) '没有生成流转事件'
}
$ticket = $script:ticket

if ($ticket) {
    Write-Host ("    工单 {0}｜来源 {1}｜填充 {2}｜报错 {3}｜型号 {4}" -f `
        $ticket.ticket_no, $ticket.extraction.source, ($ticket.extraction.filled -join '/'), `
        $ticket.error_info, $ticket.product_model) -ForegroundColor DarkGray
}

Test-Step '重复定稿同一会话不会重复建单' {
    $again = Invoke-RestMethod -Uri "$BaseUrl/api/conversations/$($script:conversationId)/close" -Method Post -TimeoutSec $TimeoutSec
    Assert-True ($again.id -eq $script:ticket.id) '重复定稿应返回同一张工单'
}

Test-Step '工单列表与关键字筛选' {
    $list = Invoke-RestMethod -Uri "$BaseUrl/api/tickets" -TimeoutSec 30
    Assert-True ($list.total -ge 1) '工单列表为空'
    Assert-True ($list.status_options.Count -eq 4) '状态选项不完整'
    $filtered = Invoke-RestMethod -Uri "$BaseUrl/api/tickets?q=$([uri]::EscapeDataString('E05'))" -TimeoutSec 30
    Assert-True ($filtered.total -ge 1) '按报错码 E05 搜索不到工单'
}

# ---------------------------------------------------------------- 状态流转

Test-Step '状态流转：待处理 → 处理中 → 已解决' {
    $working = Invoke-RestMethod -Uri "$BaseUrl/api/tickets/$($script:ticket.id)" -Method Patch `
        -ContentType 'application/json' -Body '{"to_status":"处理中","note":"客服已接单"}' -TimeoutSec 30
    Assert-True ($working.status -eq '处理中') "状态未变更为处理中：$($working.status)"

    $resolved = Invoke-RestMethod -Uri "$BaseUrl/api/tickets/$($script:ticket.id)" -Method Patch `
        -ContentType 'application/json' -Body '{"to_status":"已解决"}' -TimeoutSec 30
    Assert-True ($resolved.status -eq '已解决') "状态未变更为已解决：$($resolved.status)"
    Assert-True ($resolved.events.Count -eq 3) "事件数应为 3，实际 $($resolved.events.Count)"
}

Test-Step '非法流转被拒绝（已解决 → 待处理）' {
    $failed = $false
    try {
        Invoke-RestMethod -Uri "$BaseUrl/api/tickets/$($script:ticket.id)" -Method Patch `
            -ContentType 'application/json' -Body '{"to_status":"待处理"}' -TimeoutSec 30 | Out-Null
    }
    catch { $failed = $true }
    Assert-True $failed '非法流转居然成功了'
}

# ---------------------------------------------------------------- 导出与留存

Test-Step 'CSV 导出（带 BOM 与中文表头）' {
    $response = Invoke-WebRequest -Uri "$BaseUrl/api/tickets/export.csv" -TimeoutSec 30
    $contentType = [string]$response.Headers['Content-Type']
    Assert-True ($contentType -match 'text/csv') "Content-Type 不是 text/csv：$contentType"
    $body = [string]$response.Content
    Assert-True ($body.StartsWith([char]0xFEFF)) 'CSV 缺少 UTF-8 BOM'
    $firstLine = ($body -split "`r?`n")[0]
    Assert-True ($firstLine -match '工单号') "表头异常：$firstLine"
    Assert-True ($body -match [regex]::Escape($script:ticket.ticket_no)) '导出内容里没有该工单'
}

Test-Step '删除会话后工单仍然留存' {
    Invoke-RestMethod -Uri "$BaseUrl/api/conversations/$($script:conversationId)" -Method Delete -TimeoutSec 30 | Out-Null
    $detail = Invoke-RestMethod -Uri "$BaseUrl/api/tickets/$($script:ticket.id)" -TimeoutSec 30
    Assert-True ($detail.ticket_no -eq $script:ticket.ticket_no) '会话删除后工单丢失'
}

# ---------------------------------------------------------------- 汇总

Write-Host ""
Write-Host ("冒烟测试结果：通过 {0} 项，失败 {1} 项" -f $script:Passed, $script:Failed) -ForegroundColor $(if ($script:Failed -eq 0) { 'Green' } else { 'Red' })
if ($script:Failed -gt 0) { exit 1 }
exit 0
