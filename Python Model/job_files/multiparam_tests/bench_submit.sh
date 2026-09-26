#!/bin/bash
# Submit the benchmark grid: one inference run per (system, parameter) config that
# bench_configs.py produced. Each cell frees exactly one scaling group.
#
# Job names are "<tag>_<system>_<param>" so bench_metrics.py can find the logs.
set -u
TAG="${1:-bench}"
ROOT=/projects/anth4580/Bayesian
SB="$ROOT/Bayesian Inference/run_inference_segment_gpu_blanca.sh"
OUT="$ROOT/job_files/${TAG}_jobids.txt"
module load slurm/blanca >/dev/null 2>&1
: > "$OUT"
n=0
for cfg in "$ROOT/Results/Chain Scaling Tests/Chain "*" - $TAG "*"/solver_params.json"; do
  [ -r "$cfg" ] || continue
  d=$(dirname "$cfg"); b=$(basename "$d")
  sys=$(echo "$b" | sed -E "s/^Chain (.+) - $TAG .+$/\1/")
  par=$(echo "$b" | sed -E "s/^Chain .+ - $TAG (.+)$/\1/")
  rm -rf "$d/checkpoint"          # a stale checkpoint would resume a different prior
  jid=$(sbatch --parsable --job-name="${TAG}_${sys}_${par}" "$SB" "$cfg" 2>&1)
  if [[ "$jid" =~ ^[0-9]+$ ]]; then
    echo "$sys $par $jid" >> "$OUT"; n=$((n+1))
    printf "  %-10s %-5s -> %s\n" "$sys" "$par" "$jid"
  else
    printf "  %-10s %-5s -> SUBMIT FAILED: %s\n" "$sys" "$par" "$jid"
  fi
done
echo "submitted $n cells; ids in $OUT"
