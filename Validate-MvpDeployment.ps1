[CmdletBinding()]
param(
    [Parameter()]
    [string]$ProjectPath = (Get-Location).Path,

    [Parameter()]
    [string]$ApiBaseUrl = "http://localhost:8080",

    [Parameter()]
    [string]$UiBaseUrl = "http://localhost:3000",

    [Parameter()]
    [string]$CharacterId = "elena_voss",

    [Parameter()]
    [int]$StartupTimeoutSeconds = 120,

    [Parameter()]
    [int]$RequestTimeoutSeconds = 180,

    [Parameter()]
    [switch]$BuildAndStart,

    [Parameter()]
    [switch]$SkipUi,

    [Parameter()]
    [switch]$SkipExecution,

    [Parameter()]
    [bool]$CollectLogsOnFailure = $true,

    [Parameter()]
    [string]$ReportPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Net.Http

$script:Checks = [System.Collections.Generic.List[object]]::new()
$script:HttpClient = [System.Net.Http.HttpClient]::new()
$script:HttpClient.Timeout = [TimeSpan]::FromSeconds($RequestTimeoutSeconds)
$script:SessionId = $null
$script:SelectedCharacterId = $null
$script:WorkerBackends = @{}

function Add-Check {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][ValidateSet("PASS", "WARN", "FAIL")][string]$Status,
        [Parameter(Mandatory = $true)][string]$Detail
    )

    $record = [pscustomobject]@{
        Name   = $Name
        Status = $Status
        Detail = $Detail
    }
    $script:Checks.Add($record)

    switch ($Status) {
        "PASS" { Write-Host "[PASS] $Name - $Detail" -ForegroundColor Green }
        "WARN" { Write-Host "[WARN] $Name - $Detail" -ForegroundColor Yellow }
        "FAIL" { Write-Host "[FAIL] $Name - $Detail" -ForegroundColor Red }
    }
}

function Invoke-NativeCapture {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter()][string[]]$Arguments = @()
    )

    $lines = @(& $FilePath @Arguments 2>&1 | ForEach-Object { $_.ToString() })
    return [pscustomobject]@{
        ExitCode = $LASTEXITCODE
        Output   = ($lines -join [Environment]::NewLine).Trim()
        Lines    = $lines
    }
}

function Invoke-Http {
    param(
        [Parameter(Mandatory = $true)][string]$Method,
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter()][object]$Body = $null
    )

    $request = $null
    $response = $null
    try {
        $request = [System.Net.Http.HttpRequestMessage]::new(
            [System.Net.Http.HttpMethod]::new($Method.ToUpperInvariant()),
            $Uri
        )

        if ($null -ne $Body) {
            $json = $Body | ConvertTo-Json -Depth 30 -Compress
            $request.Content = [System.Net.Http.StringContent]::new(
                $json,
                [System.Text.Encoding]::UTF8,
                "application/json"
            )
        }

        $response = $script:HttpClient.SendAsync($request).GetAwaiter().GetResult()
        $raw = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        $parsed = $null
        if (-not [string]::IsNullOrWhiteSpace($raw)) {
            try { $parsed = $raw | ConvertFrom-Json } catch { $parsed = $null }
        }

        return [pscustomobject]@{
            StatusCode = [int]$response.StatusCode
            IsSuccess  = [bool]$response.IsSuccessStatusCode
            Body       = $parsed
            Raw        = $raw
        }
    }
    finally {
        if ($null -ne $response) { $response.Dispose() }
        if ($null -ne $request) { $request.Dispose() }
    }
}

function Wait-ForEndpoint {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $lastError = "No response received."

    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $response = Invoke-Http -Method GET -Uri $Uri
            if ($response.StatusCode -lt 500) { return $response }
            $lastError = "HTTP $($response.StatusCode): $($response.Raw)"
        }
        catch {
            $lastError = $_.Exception.Message
        }
        Start-Sleep -Milliseconds 500
    }

    throw "Endpoint did not become ready: $Uri. Last error: $lastError"
}

function Assert-Success {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][object]$Response
    )

    if (-not $Response.IsSuccess) {
        throw "$Name returned HTTP $($Response.StatusCode): $($Response.Raw)"
    }
}

function Get-PropertyValue {
    param(
        [Parameter()][object]$Object,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter()]$Default = $null
    )

    if ($null -eq $Object) { return $Default }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $Default }
    return $property.Value
}

function Test-DockerPrerequisites {
    if ($null -eq (Get-Command docker -ErrorAction SilentlyContinue)) {
        Add-Check -Name "Docker CLI" -Status FAIL -Detail "docker is not installed or is not on PATH."
        throw "Docker CLI is unavailable."
    }
    Add-Check -Name "Docker CLI" -Status PASS -Detail "docker command is available."

    $dockerInfo = Invoke-NativeCapture -FilePath "docker" -Arguments @("info", "--format", "{{.ServerVersion}}")
    if ($dockerInfo.ExitCode -ne 0) {
        Add-Check -Name "Docker daemon" -Status FAIL -Detail $dockerInfo.Output
        throw "Docker daemon is unavailable."
    }
    Add-Check -Name "Docker daemon" -Status PASS -Detail "Docker engine $($dockerInfo.Output) is reachable."

    $composeVersion = Invoke-NativeCapture -FilePath "docker" -Arguments @("compose", "version")
    if ($composeVersion.ExitCode -ne 0) {
        Add-Check -Name "Docker Compose" -Status FAIL -Detail $composeVersion.Output
        throw "Docker Compose v2 is unavailable."
    }
    Add-Check -Name "Docker Compose" -Status PASS -Detail $composeVersion.Output
}

function Test-ComposeDefinition {
    $composeFile = Join-Path $ProjectPath "docker-compose.yml"
    if (-not (Test-Path -LiteralPath $composeFile -PathType Leaf)) {
        Add-Check -Name "Compose file" -Status FAIL -Detail "Missing $composeFile"
        throw "docker-compose.yml was not found."
    }
    Add-Check -Name "Compose file" -Status PASS -Detail $composeFile

    $config = Invoke-NativeCapture -FilePath "docker" -Arguments @("compose", "config", "--quiet")
    if ($config.ExitCode -ne 0) {
        Add-Check -Name "Compose configuration" -Status FAIL -Detail $config.Output
        throw "docker compose config failed."
    }
    Add-Check -Name "Compose configuration" -Status PASS -Detail "docker-compose.yml is syntactically valid."

    $serviceResult = Invoke-NativeCapture -FilePath "docker" -Arguments @("compose", "config", "--services")
    if ($serviceResult.ExitCode -ne 0) { throw "Unable to enumerate Compose services." }

    $services = @($serviceResult.Lines | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    $expected = @("ui", "orchestrator", "memory", "left-model", "right-model", "executive-model")
    $missing = @($expected | Where-Object { $_ -notin $services })
    if ($missing.Count -gt 0) {
        Add-Check -Name "Compose services" -Status FAIL -Detail "Missing: $($missing -join ', ')"
        throw "Compose service definition is incomplete."
    }
    Add-Check -Name "Compose services" -Status PASS -Detail "All required services are defined."
}

function Start-MvpIfRequested {
    if (-not $BuildAndStart) { return }

    Write-Host "`nBuilding and starting the MVP..." -ForegroundColor Cyan
    $up = Invoke-NativeCapture -FilePath "docker" -Arguments @("compose", "up", "-d", "--build")
    if ($up.ExitCode -ne 0) {
        Add-Check -Name "Compose deployment" -Status FAIL -Detail $up.Output
        throw "docker compose up failed."
    }
    Add-Check -Name "Compose deployment" -Status PASS -Detail "Stack built and started successfully."
}

function Test-RunningServices {
    $runningResult = Invoke-NativeCapture -FilePath "docker" -Arguments @("compose", "ps", "--services", "--status", "running")
    if ($runningResult.ExitCode -ne 0) { throw "Unable to query running services." }

    $running = @($runningResult.Lines | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    $expected = @("ui", "orchestrator", "memory", "left-model", "right-model", "executive-model")
    $missing = @($expected | Where-Object { $_ -notin $running })

    if ($missing.Count -gt 0) {
        $hint = if ($BuildAndStart) {
            "Required containers failed to remain running."
        }
        else {
            "Start them with docker compose up -d --build or rerun with -BuildAndStart."
        }
        Add-Check -Name "Running services" -Status FAIL -Detail "Not running: $($missing -join ', '). $hint"
        throw "Required containers are not running."
    }
    Add-Check -Name "Running services" -Status PASS -Detail "All six MVP services are running."
}

function Get-WorkerBackends {
    foreach ($service in @("left-model", "right-model", "executive-model")) {
        $result = Invoke-NativeCapture -FilePath "docker" -Arguments @(
            "compose", "exec", "-T", $service, "sh", "-lc", 'printf "%s" "$WORKER_BACKEND"'
        )
        $backend = if ($result.ExitCode -eq 0 -and -not [string]::IsNullOrWhiteSpace($result.Output)) {
            $result.Output.Trim()
        }
        else {
            "unknown"
        }
        $script:WorkerBackends[$service] = $backend
    }

    $detail = ($script:WorkerBackends.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ", "
    if (@($script:WorkerBackends.Values | Where-Object { $_ -eq "unknown" }).Count -gt 0) {
        Add-Check -Name "Worker backends" -Status WARN -Detail $detail
    }
    else {
        Add-Check -Name "Worker backends" -Status PASS -Detail $detail
    }
}

function Test-ServiceHealth {
    $health = Wait-ForEndpoint -Uri "$ApiBaseUrl/health" -TimeoutSeconds $StartupTimeoutSeconds
    Assert-Success -Name "Orchestrator health" -Response $health

    $status = [string](Get-PropertyValue -Object $health.Body -Name "status" -Default "")
    if ($status -ne "ok") {
        Add-Check -Name "Orchestrator health" -Status FAIL -Detail "Status=$status. Body: $($health.Raw)"
        throw "Orchestrator dependencies are degraded."
    }
    Add-Check -Name "Orchestrator health" -Status PASS -Detail "Orchestrator and all four internal dependencies report healthy."

    if ($SkipUi) { return }

    $ui = Wait-ForEndpoint -Uri "$UiBaseUrl/" -TimeoutSeconds $StartupTimeoutSeconds
    Assert-Success -Name "UI" -Response $ui
    Add-Check -Name "UI" -Status PASS -Detail "$UiBaseUrl is reachable."

    $proxy = Invoke-Http -Method GET -Uri "$UiBaseUrl/api/health"
    Assert-Success -Name "UI API proxy" -Response $proxy
    $proxyStatus = [string](Get-PropertyValue -Object $proxy.Body -Name "status" -Default "")
    if ($proxyStatus -ne "ok") {
        Add-Check -Name "UI API proxy" -Status FAIL -Detail "nginx proxy returned status=$proxyStatus."
        throw "UI API proxy failed."
    }
    Add-Check -Name "UI API proxy" -Status PASS -Detail "nginx /api proxy reaches the orchestrator."
}

function Get-CharactersAndSelectOne {
    $response = Invoke-Http -Method GET -Uri "$ApiBaseUrl/characters"
    Assert-Success -Name "Character discovery" -Response $response

    $characters = @($response.Body)
    if ($characters.Count -lt 1) {
        Add-Check -Name "Character discovery" -Status FAIL -Detail "No character primers were loaded."
        throw "No characters are available."
    }

    $ids = @($characters | ForEach-Object { [string](Get-PropertyValue -Object $_ -Name "id" -Default "") })
    if ($CharacterId -in $ids) {
        $script:SelectedCharacterId = $CharacterId
    }
    else {
        $script:SelectedCharacterId = $ids[0]
        Add-Check -Name "Requested character" -Status WARN -Detail "'$CharacterId' was not found; using '$($script:SelectedCharacterId)'."
    }

    Add-Check -Name "Character discovery" -Status PASS -Detail "Loaded $($characters.Count) character(s): $($ids -join ', ')."

    $state = Invoke-Http -Method GET -Uri "$ApiBaseUrl/characters/$($script:SelectedCharacterId)/state"
    Assert-Success -Name "Character state" -Response $state
    if ($null -eq (Get-PropertyValue -Object $state.Body -Name "character")) {
        Add-Check -Name "Character state" -Status FAIL -Detail "State payload does not contain a character document."
        throw "Character state payload is incomplete."
    }
    Add-Check -Name "Character state" -Status PASS -Detail "Persistent state is readable for '$($script:SelectedCharacterId)'."
}

function Test-ExecutionLifecycle {
    if ($SkipExecution) {
        Add-Check -Name "Execution lifecycle" -Status WARN -Detail "Skipped by -SkipExecution."
        return
    }

    $sessionResponse = Invoke-Http -Method POST -Uri "$ApiBaseUrl/sessions" -Body @{
        character_id = $script:SelectedCharacterId
    }
    Assert-Success -Name "Session creation" -Response $sessionResponse

    $sid = [string](Get-PropertyValue -Object $sessionResponse.Body -Name "id" -Default "")
    if ([string]::IsNullOrWhiteSpace($sid)) { throw "Session creation returned no session ID." }
    $script:SessionId = $sid
    Add-Check -Name "Session creation" -Status PASS -Detail "Created $sid."

    # Birthplace is a deterministic bootstrap topic in the current MVP. The run-specific
    # wording is kept semantically equivalent while avoiding dependence on prior session IDs.
    $first = Invoke-Http -Method POST -Uri "$ApiBaseUrl/sessions/$sid/chat" -Body @{
        message = "Where were you born?"
    }
    Assert-Success -Name "First cognitive turn" -Response $first

    $message1 = [string](Get-PropertyValue -Object $first.Body -Name "message" -Default "")
    $cognition = Get-PropertyValue -Object $first.Body -Name "cognition"
    if ([string]::IsNullOrWhiteSpace($message1) -or $null -eq $cognition) {
        Add-Check -Name "First cognitive turn" -Status FAIL -Detail "Missing speech or cognition payload. Body: $($first.Raw)"
        throw "Cognitive turn contract failed."
    }
    foreach ($role in @("left", "right", "executive")) {
        if ($null -eq (Get-PropertyValue -Object $cognition -Name $role)) {
            throw "Cognitive turn omitted '$role' output."
        }
    }
    Add-Check -Name "First cognitive turn" -Status PASS -Detail "Left, Right, and Executive all produced output."

    $firstInteraction = Get-PropertyValue -Object $first.Body -Name "interaction"
    $topic = [string](Get-PropertyValue -Object $firstInteraction -Name "topic" -Default "")
    if ($topic -ne "self.birthplace") {
        Add-Check -Name "Topic resolution" -Status FAIL -Detail "Expected self.birthplace; got '$topic'."
        throw "Bootstrap topic resolver failed."
    }
    Add-Check -Name "Topic resolution" -Status PASS -Detail "Birthplace question resolved to self.birthplace."

    $second = Invoke-Http -Method POST -Uri "$ApiBaseUrl/sessions/$sid/chat" -Body @{
        message = "What's your hometown again?"
    }
    Assert-Success -Name "Repeated cognitive turn" -Response $second

    $interaction2 = Get-PropertyValue -Object $second.Body -Name "interaction"
    $type2 = [string](Get-PropertyValue -Object $interaction2 -Name "interaction_type" -Default "")
    $timesAsked = [int](Get-PropertyValue -Object $interaction2 -Name "times_asked" -Default 0)
    $related = @(Get-PropertyValue -Object $interaction2 -Name "related_event_ids" -Default @())
    if ($type2 -ne "repeated_question" -or $timesAsked -lt 2 -or $related.Count -lt 1) {
        Add-Check -Name "Repeated-question continuity" -Status FAIL -Detail "Expected repeated_question with prior history. Body: $($second.Raw)"
        throw "Repeated-question continuity failed."
    }
    Add-Check -Name "Repeated-question continuity" -Status PASS -Detail "Repeat was recognized and linked to prior history."

    $eventsResponse = Invoke-Http -Method GET -Uri "$ApiBaseUrl/sessions/$sid/events"
    Assert-Success -Name "Event persistence" -Response $eventsResponse
    $events = @($eventsResponse.Body)
    $userCount = @($events | Where-Object { (Get-PropertyValue -Object $_ -Name "event_type") -eq "user_message" }).Count
    $characterCount = @($events | Where-Object { (Get-PropertyValue -Object $_ -Name "event_type") -eq "character_message" }).Count
    if ($userCount -lt 2 -or $characterCount -lt 2) {
        Add-Check -Name "Event persistence" -Status FAIL -Detail "Expected at least 2 user and 2 character events; found user=$userCount character=$characterCount."
        throw "Raw event persistence failed."
    }
    Add-Check -Name "Event persistence" -Status PASS -Detail "Raw event stream contains both validation turns."

    $reflection = Invoke-Http -Method POST -Uri "$ApiBaseUrl/sessions/$sid/reflect" -Body @{}
    Assert-Success -Name "Reflection" -Response $reflection
    $summary1 = [string](Get-PropertyValue -Object $reflection.Body -Name "summary" -Default "")
    if ([string]::IsNullOrWhiteSpace($summary1)) {
        Add-Check -Name "Reflection" -Status FAIL -Detail "Reflection returned no summary."
        throw "Reflection failed."
    }
    Add-Check -Name "Reflection" -Status PASS -Detail $summary1

    $reflection2 = Invoke-Http -Method POST -Uri "$ApiBaseUrl/sessions/$sid/reflect" -Body @{}
    Assert-Success -Name "Reflection idempotency" -Response $reflection2
    $summary2 = [string](Get-PropertyValue -Object $reflection2.Body -Name "summary" -Default "")
    if ($summary2 -ne $summary1) {
        Add-Check -Name "Reflection idempotency" -Status FAIL -Detail "Sequential reflection changed without a new conversational event."
        throw "Reflection idempotency failed."
    }

    $postReflectionEvents = Invoke-Http -Method GET -Uri "$ApiBaseUrl/sessions/$sid/events"
    Assert-Success -Name "Persisted reflection" -Response $postReflectionEvents
    $reflectionCount = @(@($postReflectionEvents.Body) | Where-Object {
        (Get-PropertyValue -Object $_ -Name "event_type") -eq "reflection"
    }).Count
    if ($reflectionCount -ne 1) {
        Add-Check -Name "Reflection idempotency" -Status FAIL -Detail "Expected exactly one persisted reflection; found $reflectionCount."
        throw "Sequential reflection created duplicate persisted reflections."
    }
    Add-Check -Name "Reflection idempotency" -Status PASS -Detail "Repeated reflection reused the committed reflection."

    $close = Invoke-Http -Method POST -Uri "$ApiBaseUrl/sessions/$sid/close" -Body @{}
    Assert-Success -Name "Session close" -Response $close
    $closedSession = Get-PropertyValue -Object $close.Body -Name "session"
    $closedStatus = [string](Get-PropertyValue -Object $closedSession -Name "status" -Default "")
    if ($closedStatus -ne "closed") {
        Add-Check -Name "Session close" -Status FAIL -Detail "Expected closed status; got '$closedStatus'."
        throw "Session close failed."
    }
    Add-Check -Name "Session close" -Status PASS -Detail "Session closed after reflection."

    $afterClose = Invoke-Http -Method POST -Uri "$ApiBaseUrl/sessions/$sid/chat" -Body @{
        message = "This request should be rejected because the interaction is closed."
    }
    if ($afterClose.StatusCode -ne 409) {
        Add-Check -Name "Closed-session guard" -Status FAIL -Detail "Expected HTTP 409; got HTTP $($afterClose.StatusCode)."
        throw "Closed-session guard failed."
    }
    Add-Check -Name "Closed-session guard" -Status PASS -Detail "Chat after close is rejected with HTTP 409."
}

function Write-ValidationReport {
    $passCount = @($script:Checks | Where-Object { $_.Status -eq "PASS" }).Count
    $warnCount = @($script:Checks | Where-Object { $_.Status -eq "WARN" }).Count
    $failCount = @($script:Checks | Where-Object { $_.Status -eq "FAIL" }).Count

    $report = [pscustomobject]@{
        timestamp_utc         = [DateTime]::UtcNow.ToString("o")
        project_path          = $ProjectPath
        api_base_url          = $ApiBaseUrl
        ui_base_url           = $UiBaseUrl
        selected_character_id = $script:SelectedCharacterId
        validation_session_id = $script:SessionId
        worker_backends       = $script:WorkerBackends
        summary               = [pscustomobject]@{
            pass = $passCount
            warn = $warnCount
            fail = $failCount
        }
        checks                = @($script:Checks)
    }

    if (-not [string]::IsNullOrWhiteSpace($ReportPath)) {
        $resolved = $ReportPath
        if (-not [System.IO.Path]::IsPathRooted($resolved)) { $resolved = Join-Path $ProjectPath $resolved }
        $parent = Split-Path -Parent $resolved
        if ($parent -and -not (Test-Path -LiteralPath $parent)) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }
        $report | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $resolved -Encoding UTF8
        Write-Host "Validation report: $resolved" -ForegroundColor Cyan
    }

    Write-Host "`nValidation summary: PASS=$passCount WARN=$warnCount FAIL=$failCount" -ForegroundColor Cyan
}

$initialLocation = Get-Location
try {
    $ProjectPath = (Resolve-Path -LiteralPath $ProjectPath).Path
    Push-Location -LiteralPath $ProjectPath

    Write-Host "Cognitive Character Runtime MVP deployment validator" -ForegroundColor Cyan
    Write-Host "Project: $ProjectPath"
    Write-Host "API:     $ApiBaseUrl"
    if (-not $SkipUi) { Write-Host "UI:      $UiBaseUrl" }
    Write-Host ""

    Test-DockerPrerequisites
    Test-ComposeDefinition
    Start-MvpIfRequested
    Test-RunningServices
    Get-WorkerBackends
    Test-ServiceHealth
    Get-CharactersAndSelectOne
    Test-ExecutionLifecycle
}
catch {
    if (@($script:Checks | Where-Object { $_.Status -eq "FAIL" }).Count -eq 0) {
        Add-Check -Name "Unhandled validation error" -Status FAIL -Detail $_.Exception.Message
    }
    else {
        Write-Host "`nValidation stopped: $($_.Exception.Message)" -ForegroundColor Red
    }
}
finally {
    try {
        Write-ValidationReport
        $failCount = @($script:Checks | Where-Object { $_.Status -eq "FAIL" }).Count
        if ($failCount -gt 0 -and $CollectLogsOnFailure) {
            Write-Host "`n--- docker compose ps ---" -ForegroundColor Yellow
            & docker compose ps 2>&1 | ForEach-Object { Write-Host $_ }
            Write-Host "`n--- recent docker compose logs ---" -ForegroundColor Yellow
            & docker compose logs --tail 120 2>&1 | ForEach-Object { Write-Host $_ }
        }
    }
    finally {
        $script:HttpClient.Dispose()
        Pop-Location -ErrorAction SilentlyContinue
        Set-Location -LiteralPath $initialLocation -ErrorAction SilentlyContinue
    }
}

$finalFailures = @($script:Checks | Where-Object { $_.Status -eq "FAIL" }).Count
if ($finalFailures -gt 0) { exit 1 }
exit 0
