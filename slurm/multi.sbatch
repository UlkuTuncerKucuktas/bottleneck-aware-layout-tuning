#!/bin/bash
#SBATCH -p kolyoz-cuda
#SBATCH -A ukucuktas
#SBATCH -J layout-tune-multi
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue

# Node count comes from the command line, so one script covers both:
#   sbatch -N 2 slurm/multi.sbatch neighbour_cost
#   sbatch -N 4 slurm/multi.sbatch mds_scaling      (repeat at -N 2, 4, 8)

set -euo pipefail

RUNDIR=${LAYOUT_SCRATCH:-/arf/scratch/$USER/layout-tuning}
mkdir -p "$RUNDIR" logs
rsync -a --delete "$SLURM_SUBMIT_DIR/layout_tuning" "$RUNDIR/"
cd "$RUNDIR"
rm -f barrier_*        # stale rendezvous markers from an interrupted run

echo "job $SLURM_JOB_ID on $SLURM_NODELIST ($SLURM_NNODES nodes) at $(date)"

srun env PYTHONPATH="$RUNDIR" python3 -m layout_tuning.run "$@"

echo "finished $(date)"
