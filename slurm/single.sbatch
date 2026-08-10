#!/bin/bash
#SBATCH -p kolyoz-cuda
#SBATCH -A ukucuktas
#SBATCH -J layout-tune
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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
#
# kolyoz-cuda requires one GPU per 16 cores, so the defaults here are 16 cores
# and one GPU. stripe_grid sweeps up to 32 reader threads and wants twice that,
# which the command line can override:
#   sbatch --cpus-per-task=32 --gres=gpu:2 slurm/single.sbatch stripe_grid

# No experiment here uses a GPU. The suite measures filesystem behaviour and the
# node acts as a Lustre client, so the GPU sits idle for the whole run. The
# request below exists only because kolyoz-cuda refuses jobs without one, at a
# ratio of one GPU per 16 cores.
#
# On a CPU-only partition, override both on the command line -- neither can be
# changed from inside the job, since the allocation is already made by then:
#   sbatch -p <cpu-partition> --gres=NONE slurm/single.sbatch <experiment>

set -euo pipefail

RUNDIR=${LAYOUT_SCRATCH:-/arf/scratch/$USER/layout-tuning}
mkdir -p "$RUNDIR" logs
rsync -a "$SLURM_SUBMIT_DIR/layout_tuning" "$RUNDIR/"
cd "$RUNDIR"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST at $(date)"
lfs --version
lfs df . | grep -E 'MDT|filesystem_summary'
echo "OSTs: $(lfs df . | grep -c 'OST:')"
echo "counters readable: $(ls /proc/fs/lustre/llite/*/stats 2>/dev/null | wc -l)"
echo "cores allocated: ${SLURM_CPUS_PER_TASK:-unknown}"

PYTHONPATH="$RUNDIR" python3 -m layout_tuning.run "$@"

echo "finished $(date)"
