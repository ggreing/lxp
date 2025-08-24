param (
    [string]$BaseUrl = "http://localhost:8000",
    [string]$UserId = "u1",
    [string]$FilePath = "README.md",
    [string]$Prompt = "What is this repo about?",
    [string]$SubFunction = "qa",
    [switch]$NoStream,
    [switch]$VerboseCurl
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Ensure-Curl {
    $curlPath = (Get-Command curl.exe -ErrorAction SilentlyContinue).Path
    if (-not $curlPath) {
        throw "curl.exe not found in PATH. Please ensure curl.exe is available on Windows 10+ or install it."
    }
}

function Invoke-CurlJson {
    param(
        [Parameter(Mandatory=$true)][string]$Method,
        [Parameter(Mandatory=$true)][string]$Url,
        [string]$BodyJson
    )
    $args = @("-L","-s","-X",$Method,$Url)
    if ($BodyJson) {
        # IMPORTANT: wrap the payload in quotes so PowerShell doesn't split it on spaces
        $args += @("-H","Content-Type: application/json","-d",$BodyJson)
    }
    if ($VerboseCurl) { Write-Host "curl.exe $($args -join ' ')" -ForegroundColor DarkGray }
    $raw = & curl.exe @args
    if (-not $raw) { return $null }
    try {
        return $raw | ConvertFrom-Json
    } catch {
        Write-Warning "Non-JSON response from ${Url}:`n$raw"
        return $null
    }
}

function Invoke-CurlMultipart {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][string]$FilePath
    )
    $args = @("-L","-s","-F","file=@$FilePath",$Url)
    if ($VerboseCurl) { Write-Host "curl.exe $($args -join ' ')" -ForegroundColor DarkGray }
    $raw = & curl.exe @args
    if (-not $raw) { return $null }
    try {
        return $raw | ConvertFrom-Json
    } catch {
        Write-Warning "Non-JSON response from ${Url}:`n$raw"
        return $null
    }
}

try {
    Ensure-Curl

    if (-not (Test-Path -LiteralPath $FilePath)) {
        throw "File not found: $FilePath (set -FilePath to a valid file to upload)"
    }

    # Normalize to absolute Windows path for curl -F
    $FilePath = (Resolve-Path -LiteralPath $FilePath).Path

    Write-Host "1) Creating document collection..." -ForegroundColor Cyan
    $vsResp = Invoke-CurlJson -Method "POST" -Url "$BaseUrl/coach/documents"
    $vsId = $vsResp.id
    if (-not $vsId) { throw "Failed to create vectorstore. Response:`n$($vsResp | ConvertTo-Json -Depth 6)" }
    Write-Host "   -> vectorstore_id: $vsId" -ForegroundColor Green

    Write-Host "2) Uploading file & metadata..." -ForegroundColor Cyan
    $uploadUrl = "$BaseUrl/files/upload?user_id=$UserId&vectorstore_id=$vsId"
    $upResp = Invoke-CurlMultipart -Url $uploadUrl -FilePath $FilePath
    if (-not $upResp) { Write-Warning "Upload response was empty or not JSON." }
    else { Write-Host "   -> upload ok" -ForegroundColor Green }

    Write-Host "3) Indexing (Qdrant upsert)..." -ForegroundColor Cyan
    $idxResp = Invoke-CurlJson -Method "POST" -Url "$BaseUrl/coach/documents/$vsId/index"
    if (-not $idxResp) { Write-Warning "Index response was empty or not JSON." }
    else { Write-Host "   -> index requested" -ForegroundColor Green }

    Write-Host "4) Creating Assist job (+RAG)..." -ForegroundColor Cyan
    $body = (@{
        user_id       = $UserId
        prompt        = $Prompt
        vectorstore_id= $vsId
        sub_function  = $SubFunction
    } | ConvertTo-Json -Compress)

    # Escape quotes for curl.exe -d "<json>" (wrap later, so keep backslashes)
    $escapedBody = $body.Replace('"','\"')
    $assistUrl = "$BaseUrl/coach/recommendations"
    $jobResp = Invoke-CurlJson -Method "POST" -Url $assistUrl -BodyJson $escapedBody
    $jobId = $jobResp.job_id
    if (-not $jobId) {
        $dump = if ($jobResp) { $jobResp | ConvertTo-Json -Depth 8 } else { "(empty or non-JSON)" }
        throw "Failed to create job. Response:`n$dump"
    }
    Write-Host "   -> job_id: $jobId" -ForegroundColor Green

    if (-not $NoStream) {
        Write-Host "5) Streaming SSE for job $jobId (Ctrl+C to stop)..." -ForegroundColor Cyan
        & curl.exe -L -N "$BaseUrl/events/jobs/$jobId"
    } else {
        Write-Host "Skipping SSE stream due to -NoStream switch." -ForegroundColor Yellow
    }
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}
