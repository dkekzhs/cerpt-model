param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingPid,
    [Parameter(Mandatory = $true)]
    [string]$ProjectPath
)

$outputDir = Join-Path $ProjectPath 'artifacts\cerpt-korean-stage1-resumed'
$logPath = Join-Path $outputDir 'final_upload.log'

while (Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 60
}

$historyPath = Join-Path $outputDir 'training_history.json'
if (-not (Test-Path -LiteralPath $historyPath)) {
    Add-Content -LiteralPath $logPath -Value 'Training ended without training_history.json.'
    exit 2
}

$history = Get-Content -LiteralPath $historyPath -Raw | ConvertFrom-Json
$lastEpoch = [int]$history[-1].epoch
if ($lastEpoch -lt 30) {
    Add-Content -LiteralPath $logPath -Value "Training ended before epoch 30: $lastEpoch"
    exit 3
}

Set-Location -LiteralPath $ProjectPath
hf upload qweqwqw113/cerpt-korean-stage1 artifacts/cerpt-korean-stage1-resumed . --type model --exclude 'latest/**' --exclude 'tokenizer/**' --exclude '*.log' --exclude '*.error.log' --commit-message 'Upload final 30-epoch Korean Stage 1 CERPT checkpoint' *>&1 | Tee-Object -FilePath $logPath
hf upload qweqwqw113/cerpt-korean-stage1 docs/MODEL_CARD_KOREAN_STAGE1.md README.md --type model --commit-message 'Update final Korean Stage 1 model card' *>&1 | Tee-Object -FilePath $logPath -Append
