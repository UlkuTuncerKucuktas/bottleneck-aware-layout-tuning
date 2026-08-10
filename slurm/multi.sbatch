#!/bin/bash
#SBATCH -p kolyoz-cuda
#SBATCH -A ukucuktas
#SBATCH -J layout-tune-multi
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue

# Node count comes from the command line, so one script covers both:
#   sbatch -N 4 slurm/multi.sbatch mds_scaling      (repeat at -N 2, 4, 8)
#   sbatch --exclusive -N 2 slurm/multi.sbatch neighbour_cost
#
# --exclusive is NOT set here, because it is only right for one of the two
# experiments. mds_scaling reads single-threaded and measures contention on the
# metadata server, which is shared filesystem-wide regardless of who else is on
# the node -- so exclusivity buys almost nothing and makes an 8-node job wait for
# 8 nodes to drain at once, which on a busy partition can mean never.
# neighbour_cost measures OST bandwidth interference between jobs, where a
# co-tenant would confound the result, so pass --exclusive on its command line.

# No experiment here uses a GPU. The suite measures filesystem behaviour and the
# node acts as a Lustre client, so the GPU sits idle for the whole run. The
# request below exists only because kolyoz-cuda refuses jobs without one, at a
# ratio of one GPU per 16 cores.
#
# On a CPU-only partition, override both on the command line -- neither can be
# changed from inside the job, since the allocation is already made by then:
#   sbatch -p <cpu-partition> --gres=NONE slurm/multi.sbatch <experiment>

# 30 minutes, not hours. The 2-node mds_scaling run took 90 seconds and the
# 8-node one does the same work per client, so a long request buys nothing and
# costs everything: the backfill scheduler can slot a short job into a gap
# between larger reservations, while a two-hour job waits for a two-hour gap.
set -euo pipefail

RUNDIR=${LAYOUT_SCRATCH:-/arf/scratch/$USER/layout-tuning}
mkdir -p "$RUNDIR" logs
# No --delete: a concurrent job may be importing from this tree right now, and
# removing files under a live interpreter fails in ways that look like unrelated
# import errors. slurm/reset.sh removes the whole copied tree between campaigns.
rsync -a "$SLURM_SUBMIT_DIR/layout_tuning" "$RUNDIR/"
cd "$RUNDIR"
# Only this job's own markers, never "barrier_*": markers are named with the job
# id, and a blanket sweep run at startup deletes the LIVE markers of a concurrent
# job already waiting at its barrier -- which then blocks until its timeout and
# records a failure. Markers from dead jobs are harmless; they can never match a
# live job id.
rm -f "barrier_${SLURM_JOB_ID}"_*

echo "job $SLURM_JOB_ID on $SLURM_NODELIST ($SLURM_NNODES nodes) at $(date)"

srun env PYTHONPATH="$RUNDIR" python3 -m layout_tuning.run "$@"

echo "finished $(date)"
