param(
    [ValidateRange(1, 100000)]
    [int]$Reps = 100,
    [ValidateSet('ttest', 'l1lr')]
    [string]$Screening = 'ttest',
    [ValidateSet('none', 'zscore', 'rank_int')]
    [string]$Transform = 'none',
    [string]$RscriptPath = 'D:/R-4.4.1/bin/Rscript.exe'
)

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $RscriptPath)) {
    throw "Rscript was not found: $RscriptPath"
}

& $RscriptPath (Join-Path $PSScriptRoot 'run_prop_experiment.R') `
    "--reps=$Reps" "--screening=$Screening" "--transform=$Transform"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
