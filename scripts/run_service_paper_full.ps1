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
    Write-Host "== Core补跑: medium/medium seeds 7,42 MAPPO vs SLG-SAGE =="
    python -m ServiceComputing.scripts.run_paper_experiments `
      --suite core `
      --difficulty medium `
      --scale medium `
      --algos mappo,slg_sage `
      --seeds 7,42 `
      --total_env_steps $TotalEnvSteps `
      --policy_mode stochastic `
      --eval_episodes $EvalEpisodes `
      --output_root $OutputRoot

    Write-Host "== Scale sensitivity: medium small/medium/large core algorithms =="
    python -m ServiceComputing.scripts.run_paper_experiments `
      --suite scale `
      --difficulty medium `
      --scales small,medium,large `
      --algos random,greedy,mappo,slg_sage,qmix,coma,happo `
      --seeds 1,7,42 `
      --total_env_steps $TotalEnvSteps `
      --policy_mode stochastic `
      --eval_episodes $EvalEpisodes `
      --output_root $OutputRoot

    Write-Host "== Hard robustness: hard small/medium/large core algorithms =="
    python -m ServiceComputing.scripts.run_paper_experiments `
      --suite hard `
      --difficulty hard `
      --scales small,medium,large `
      --algos random,greedy,mappo,slg_sage,qmix,coma,happo `
      --seeds 1,7,42 `
      --total_env_steps $TotalEnvSteps `
      --policy_mode stochastic `
      --eval_episodes $EvalEpisodes `
      --output_root $OutputRoot

    Write-Host "== Full supplement: medium/medium remaining algorithms =="
    python -m ServiceComputing.scripts.run_paper_experiments `
      --suite full `
      --difficulty medium `
      --scale medium `
      --algos wqmix,madqn,qtran,maddpg,masac,matd3 `
      --seeds 1,7,42 `
      --total_env_steps $TotalEnvSteps `
      --policy_mode stochastic `
      --eval_episodes $EvalEpisodes `
      --output_root $OutputRoot

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
