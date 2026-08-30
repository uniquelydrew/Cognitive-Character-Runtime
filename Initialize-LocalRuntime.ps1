[CmdletBinding()]
param(
    [string]$ProjectPath = $PSScriptRoot,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$resolvedProject = (Resolve-Path -LiteralPath $ProjectPath).Path
$envPath = Join-Path $resolvedProject '.env'

if ((Test-Path -LiteralPath $envPath) -and -not $Force) {
    throw "Refusing to overwrite existing $envPath. Use -Force only after preserving its secret."
}

$bytes = New-Object byte[] 48
[System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
$token = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
$content = @"
# Generated for this local runtime. Keep this file private and out of source control.
API_AUTH_TOKEN=$token
OLLAMA_MODEL=llama3.2:3b
"@
[System.IO.File]::WriteAllText($envPath, $content, [System.Text.UTF8Encoding]::new($false))

Write-Host "Created private local runtime configuration at $envPath." -ForegroundColor Green
Write-Host "Start with: docker compose up --build -d" -ForegroundColor Cyan
