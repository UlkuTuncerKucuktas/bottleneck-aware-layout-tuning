#!/bin/bash
# Submit a one-off diagnostic with the site's geometry applied:
#
#   source slurm/barbun.env
#   ./slurm/submit_probe.sh slurm/evict_probe.py
#
# #SBATCH directives cannot read variables -- they are comments the scheduler
# parses before any shell runs -- so a script whose header hardcodes a core
# count is wrong everywhere except the cluster it was written on. The geometry
# therefore goes on the command line here, exactly as run_campaign.sh does it,
# and the header in probe.sbatch is only a default for a bare sbatch.
set -euo pipefail

SCRIPT="${1:?usage: ./slurm/submit_probe.sh <script.py>}"
[ -f "$SCRIPT" ] || { echo "no such script: $SCRIPT"; exit 1; }

SCRATCH="${LAYOUT_SCRATCH:-/arf/scratch/$USER/layout-tuning}"
CORES="${LAYOUT_CORES-16}"

GEOMETRY="--chdir=$SCRATCH --cpus-per-task=$CORES"
[ -n "${LAYOUT_PARTITION-}" ] && GEOMETRY="$GEOMETRY -p $LAYOUT_PARTITION"
[ -n "${LAYOUT_ACCOUNT-}" ]   && GEOMETRY="$GEOMETRY -A $LAYOUT_ACCOUNT"

# A partition that enforces a GPU-per-core ratio wants the count to match the
# cores requested, so the probe's single-GPU default only fits the narrow count.
if [ "${LAYOUT_GRES-gpu:1}" = "" ]; then
    GEOMETRY="$GEOMETRY --gres=NONE"
else
    GEOMETRY="$GEOMETRY --gres=${LAYOUT_GRES-gpu:1}"
fi

mkdir -p "$SCRATCH/logs"

echo "scratch:   $SCRATCH"
echo "cores:     $CORES   gres: ${LAYOUT_GRES-gpu:1}"
echo "sbatch $GEOMETRY slurm/probe.sbatch $SCRIPT"
jobid=$(sbatch --parsable $GEOMETRY slurm/probe.sbatch "$SCRIPT")
echo
echo "$jobid   $SCRIPT"
echo "log:  $SCRATCH/logs/layout-probe-$jobid.out"
