# Windows equivalent of run_all.sh. Run from the repository root in PowerShell:  .\run_all.ps1
# Regenerates every derived table and figure from the archived data. No API calls.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force data\derived, outputs\figures | Out-Null
# Native-command warnings on stderr must not stop the run (PowerShell 5.1 turns redirected stderr into errors);
# failure is judged by the exit code only.
function Run($cmd) { Write-Host ">> $cmd"; $ErrorActionPreference = "Continue"; Invoke-Expression $cmd; if ($LASTEXITCODE -ne 0) { throw "Step failed: $cmd" } }

# Locate Rscript: on Windows R is usually not on PATH. Order: RSCRIPT env var, PATH, R_HOME, newest R under Program Files.
$Rscript = $env:RSCRIPT
if (-not $Rscript) { $Rscript = (Get-Command Rscript -ErrorAction SilentlyContinue).Source }
if (-not $Rscript -and $env:R_HOME) { $Rscript = Join-Path $env:R_HOME "bin\Rscript.exe" }
if (-not $Rscript) {
  $candidates = @("$env:ProgramFiles\R", "${env:ProgramFiles(x86)}\R", "$env:LOCALAPPDATA\Programs\R") |
    Where-Object { $_ -and (Test-Path $_) } |
    ForEach-Object { Get-ChildItem $_ -Directory -Filter "R-*" } |
    Sort-Object { [version]($_.Name -replace '^R-', '' -replace '[^0-9.].*$', '') } -Descending
  foreach ($c in $candidates) { $p = Join-Path $c.FullName "bin\Rscript.exe"; if (Test-Path $p) { $Rscript = $p; break } }
}
if (-not $Rscript -or -not (Test-Path $Rscript)) {
  throw "Rscript.exe not found. Install R (https://cran.r-project.org) or set `$env:RSCRIPT to the full path of Rscript.exe, e.g. `$env:RSCRIPT = 'C:\Program Files\R\R-4.5.1\bin\Rscript.exe'"
}
Write-Host "Using Rscript at $Rscript"
function RunR($script) { Write-Host ">> Rscript $script"; $ErrorActionPreference = "Continue"; & $Rscript $script; if ($LASTEXITCODE -ne 0) { throw "Step failed: Rscript $script" } }

Write-Host "== Stage 3 (offline part): judge validation from archived judge-vs-human comparisons"
Run "python 03_llm_judge/build_judge_validation_table.py"
Run "python 03_llm_judge/inter_rater_reliability.py | Out-File -Encoding utf8 data/derived/inter_rater_reliability_report.txt"

Write-Host "== Stage 5a: statement- and summary-level score tables"
Run "python 05_analysis/build_score_tables.py"

Write-Host "== Stage 4: retrieval and agent behaviour (from raw generation logs)"
Run "python 04_retrieval_traces/parse_tool_traces.py"
Run "python 04_retrieval_traces/recheck_traces.py"
Run "python 04_retrieval_traces/true_retrieval_recall.py"
Run "python 04_retrieval_traces/retrieval_summary_stats.py"

Write-Host "== Stage 5b: failure tags and bootstrap statistics (tables)"
Run "python 05_analysis/build_failure_reasons.py --results-dir data/judge_outputs/by_model --output-dir data/derived"
RunR "05_analysis/analysis.R"
RunR "05_analysis/reframed_metrics.R"

Write-Host "== Stage 5c: all figures"
RunR "05_analysis/figures.R"

Write-Host "== Stage 7: file manifest"
Run "python make_manifest.py"

Write-Host "== Verify against the paper"
Run "python verify_results.py"
