param(
    [int]$TotalEnvSteps = 100000,
    [string]$OutputRoot = "artifacts/service_paper",
    [int]$EvalEpisodes = 8
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path "."
$logDir = Join-Path $root $OutputRoot
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir "full_run.log"
Start-Transcript -Path $logPath -Append | Out-Null

try {

    Write-Host "== Best stochastic checkpoint evaluation =="
    python -m ServiceComputing.scripts.evaluate_best_checkpoints `
      --root $OutputRoot `
      --policy_mode stochastic `
      --eval_episodes 50

    Write-Host "== Summarize and plot =="
    python -m ServiceComputing.scripts.summarize_service_results --root $OutputRoot
    python -m ServiceComputing.scripts.plot_paper_results --root $OutputRoot --smooth_window 10
}
finally {
    Stop-Transcript | Out-Null
}
