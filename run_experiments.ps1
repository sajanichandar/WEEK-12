$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Create it first with: py -3.14 -m venv .venv"
}

$results = Join-Path $PSScriptRoot "experiment-results"
New-Item -ItemType Directory -Force -Path $results | Out-Null

$runs = @(
    @{ Name = "baseline-10-epochs"; Epochs = 10; SequenceLength = 100; HiddenSize = 128 },
    @{ Name = "baseline"; Epochs = 2; SequenceLength = 100; HiddenSize = 128 },
    @{ Name = "shorter-sequence"; Epochs = 2; SequenceLength = 60; HiddenSize = 128 },
    @{ Name = "fewer-units"; Epochs = 2; SequenceLength = 100; HiddenSize = 64 },
    @{ Name = "more-epochs"; Epochs = 4; SequenceLength = 100; HiddenSize = 128 }
)

foreach ($run in $runs) {
    $log = Join-Path $results "$($run.Name).txt"
    Write-Host "`n=== $($run.Name): epochs=$($run.Epochs), seq-length=$($run.SequenceLength), hidden-size=$($run.HiddenSize) ==="
    & $python (Join-Path $PSScriptRoot "train.py") `
        --epochs $run.Epochs `
        --seq-length $run.SequenceLength `
        --hidden-size $run.HiddenSize `
        --seed "ROMEO:" `
        --generate-length 160 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment '$($run.Name)' failed with exit code $LASTEXITCODE. See $log"
    }
}

Write-Host "`nExperiment logs saved under: $results"
