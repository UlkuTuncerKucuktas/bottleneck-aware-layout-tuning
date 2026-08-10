#!/bin/bash
# Run a one-off diagnostic on a compute node:
#
#   sbatch --chdir=$LAYOUT_SCRATCH -p $LAYOUT_PARTITION -A $LAYOUT_ACCOUNT \
#       slurm/probe.sbatch slurm/evict_probe.py
#
# The directive block below matters more than it looks. A partition that
# enforces a GPU-per-core ratio computes it PER NODE, so it needs -N and
# --ntasks to work out what a job is asking for; an `sbatch --wrap` with only
# --cpus-per-task and --gres is rejected on a request that is arithmetically
# fine ("node basina her 20 cekirdek icin 1 GPU talep edilmelidir"). Reusing the
# same shape as single.sbatch is what makes a probe submittable at all.
#SBATCH -p kolyoz-cuda
#SBATCH -A ukucuktas
#SBATCH -J layout-probe
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

SCRIPT="${1:?usage: sbatch slurm/probe.sbatch <script.py>}"

RUNDIR=${LAYOUT_SCRATCH:-/arf/scratch/$USER/layout-tuning}
mkdir -p "$RUNDIR" logs
rsync -a "$SLURM_SUBMIT_DIR/layout_tuning" "$RUNDIR/"
cd "$RUNDIR"

echo "job $SLURM_JOB_ID on $SLURM_NODELIST, $SLURM_CPUS_PER_TASK cores"
echo "started $(date)"
echo

PYTHONPATH="$RUNDIR" python3 "$SLURM_SUBMIT_DIR/$SCRIPT"

echo
echo "finished $(date)"
