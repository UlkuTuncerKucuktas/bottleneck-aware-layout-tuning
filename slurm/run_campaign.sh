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

# Where the jobs go. The scripts carry kolyoz-cuda defaults in their #SBATCH
# directives; set these to run the same campaign anywhere else, since a
# partition name, an account and a GPU ratio are all site-specific:
#
#   LAYOUT_PARTITION=barbun LAYOUT_ACCOUNT=<acct> LAYOUT_GRES= ./slurm/run_campaign.sh
#
# LAYOUT_GRES set but EMPTY means "request no GPU", which is what a CPU
# partition needs; leaving it unset keeps the GPU request the cuda partitions
# demand. No experiment here uses a GPU either way.
# Some sites refuse jobs whose working directory is outside the scratch
# filesystem ("Lutfen islerinizi /arf/scratch/ dizini altinda calistiriniz").
# The job's workdir is the submit directory unless --chdir says otherwise, and
# this repository normally lives in $HOME -- so point the workdir at the scratch
# run directory, which is under /arf/scratch by construction. SLURM_SUBMIT_DIR
# still refers to where sbatch was invoked, so the scripts continue to find the
# module tree they copy from.
COMMON="--chdir=$SCRATCH"
[ -n "${LAYOUT_PARTITION-}" ] && COMMON="$COMMON -p $LAYOUT_PARTITION"
[ -n "${LAYOUT_ACCOUNT-}" ]   && COMMON="$COMMON -A $LAYOUT_ACCOUNT"

# --output is relative to the workdir and Slurm opens it before the job runs,
# so the directory has to exist at submit time.
[ -z "$DRY" ] && mkdir -p "$SCRATCH/logs"

if [ "${LAYOUT_GRES-gpu:1}" = "" ]; then
    # --gres=NONE, not an omitted flag. The sbatch scripts carry their own
    # "#SBATCH --gres=gpu:1", which stays in force unless the command line
    # overrides it; leaving the flag out would silently keep requesting a GPU
    # on a partition that has none to give.
    GRES_NARROW="--gres=NONE"
    GRES_WIDE="--gres=NONE"
else
    GRES_NARROW="--gres=${LAYOUT_GRES-gpu:1}"
    # One GPU per 16 cores where the ratio is enforced, so 32 cores needs two.
    GRES_WIDE="--gres=${LAYOUT_GRES_WIDE-gpu:2}"
fi

# 32 cores where the sweep needs them, 16 otherwise.
WIDE="$COMMON --cpus-per-task=32 $GRES_WIDE slurm/single.sbatch"
NARROW="$COMMON $GRES_NARROW slurm/single.sbatch"
MULTI="$COMMON $GRES_NARROW slurm/multi.sbatch"

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
submit "--exclusive -N 2 $MULTI" neighbour_cost
submit "-N 2 $MULTI" mds_scaling
submit "-N 4 $MULTI" mds_scaling
submit "-N 8 $MULTI" mds_scaling

echo
echo "chain submitted. watch with:  squeue -u \$USER"
echo "logs:  $SCRATCH/logs/"
