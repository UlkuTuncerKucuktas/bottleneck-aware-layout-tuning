#!/bin/bash
#SBATCH -p kolyoz-cuda
#SBATCH -A ukucuktas
#SBATCH -J layout-tune
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue

# One or more single-node experiments:
#   sbatch slurm/single.sbatch bottleneck_budget
#   sbatch slurm/single.sbatch all
# Resubmitting the same command after a timeout continues from results.jsonl.

set -euo pipefail

RUNDIR=${LAYOUT_SCRATCH:-/arf/scratch/$USER/layout-tuning}
mkdir -p "$RUNDIR" logs
rsync -a --delete "$SLURM_SUBMIT_DIR/layout_tuning" "$RUNDIR/"
cd "$RUNDIR"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST at $(date)"
lfs --version
lfs df . | grep -E 'MDT|filesystem_summary'
echo "OSTs: $(lfs df . | grep -c 'OST:')"
echo "counters readable: $(ls /proc/fs/lustre/llite/*/stats 2>/dev/null | wc -l)"

PYTHONPATH="$RUNDIR" python3 -m layout_tuning.run "$@"

echo "finished $(date)"
