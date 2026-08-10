#!/bin/bash
# Run a one-off diagnostic on a compute node:
#
#   source slurm/barbun.env
#   ./slurm/submit_probe.sh slurm/evict_probe.py
#
# Use that wrapper rather than sbatch directly. The directives below are only a
# default for the cluster this was written on: #SBATCH lines are comments the
# scheduler parses before any shell runs, so they cannot read LAYOUT_CORES, and
# a hardcoded core count is rejected by any partition with a different quantum
# (16 on kolyoz-cuda, 20 on barbun-cuda). submit_probe.sh passes the geometry on
# the command line, where it overrides these.
#
# The directive block below matters more than it looks. A partition that
# enforces a GPU-per-core ratio computes it PER NODE, so it needs -N and
# --ntasks to work out what a job is asking for; an `sbatch --wrap` with only
# --cpus-per-task and --gres is rejected on a request that is arithmetically
# fine ("node basina her 20 cekirdek icin 1 GPU talep edilmelidir"). Reusing the
# same shape as single.sbatch is what makes a probe submittable at all.
# Partition, account, core count and GPU request are deliberately absent: they
# are site-specific, a directive cannot read a variable, and a stale default
# here would silently send a probe to the wrong cluster or violate its core
# quantum. submit_probe.sh supplies all four. A bare `sbatch slurm/probe.sbatch`
# fails on the missing partition, which is the right kind of failure.
#SBATCH -J layout-probe
#SBATCH -N 1
#SBATCH --ntasks=1
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
