#!/bin/bash
# Submit the whole campaign as a chain: each job starts only after the previous
# one ends. Returns immediately -- nothing here waits.
#
#   ./slurm/run_campaign.sh            submit everything
#   ./slurm/run_campaign.sh --dry-run  print the sbatch calls without submitting
#
# Sequential rather than parallel on purpose. Every experiment reads and writes
# the same 24 flash OSTs, so two throughput jobs running at once measure each
# other's interference instead of their own layouts. neighbour_cost is the one
# exception and it creates its interference itself.
set -e
cd "$(dirname "$0")/.."

DRY=""
[ "$1" = "--dry-run" ] && DRY="echo"

mkdir -p logs

# The ledger skips cells it already holds, which is what makes resume work but
# also means rows written by superseded code are never redone. Report them and
# let the caller decide rather than deleting anyone's results here.
SCRATCH="${LAYOUT_SCRATCH:-/arf/scratch/$USER/layout-tuning}"
if [ -f "$SCRATCH/results.jsonl" ]; then
    stale=$(grep -cE '"cell": "(stripe_grid|bottleneck_budget/(minimal|wide))/' \
            "$SCRATCH/results.jsonl" || true)
    if [ "${stale:-0}" -gt 0 ]; then
        echo "warning: $SCRATCH/results.jsonl holds $stale rows from superseded code."
        echo "         Those cells will be SKIPPED, not remeasured. To redo them:"
        echo
        echo "  cd $SCRATCH && cp results.jsonl results.jsonl.bak"
        echo "  grep -vE '\"cell\": \"(stripe_grid|bottleneck_budget/(minimal|wide))/' \\"
        echo "      results.jsonl > results.keep && mv results.keep results.jsonl"
        echo
        read -r -p "submit anyway? [y/N] " reply
        [ "$reply" = "y" ] || exit 1
    fi
fi

# afterany, not afterok: a timed-out job still leaves finished cells in the
# ledger, and the next experiment is independent of whether it completed.
previous=""
submit() {
    local geometry="$1"; shift
    local dependency=""
    [ -n "$previous" ] && dependency="--dependency=afterany:$previous"

    if [ -n "$DRY" ]; then
        echo "sbatch $dependency $geometry $*"
        previous="<jobid>"
        return
    fi

    local jobid
    jobid=$(sbatch --parsable $dependency $geometry "$@")
    echo "$jobid   $*"
    previous="$jobid"
}

# 32 cores where the sweep needs them, 16 otherwise. The partition wants one GPU
# per 16 cores, so the two go together.
WIDE="--cpus-per-task=32 --gres=gpu:2 slurm/single.sbatch"
NARROW="slurm/single.sbatch"

echo "jobid       experiment"
submit "$WIDE"   stripe_grid
submit "$WIDE"   cores_vs_throughput
submit "$NARROW" dom_cutoff dom_footprint flush_cost tail_latency
submit "$NARROW" repeated_epochs mixed_classes write_path pool_selection
submit "$NARROW" bottleneck_budget

# Metadata scaling is measured against client count, so each width is its own
# job and they must not overlap.
# neighbour_cost first: it needs only 2 nodes and it needs them exclusively, so
# queuing it behind the 8-node job would make a cheap measurement hostage to the
# most expensive one in the campaign.
submit "--exclusive -N 2 slurm/multi.sbatch" neighbour_cost
submit "-N 2 slurm/multi.sbatch" mds_scaling
submit "-N 4 slurm/multi.sbatch" mds_scaling
submit "-N 8 slurm/multi.sbatch" mds_scaling

echo
echo "chain submitted. watch with:  squeue -u \$USER"
echo "logs appear in logs/ as each job starts"
