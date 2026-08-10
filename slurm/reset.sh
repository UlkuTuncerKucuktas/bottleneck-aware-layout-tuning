#!/bin/bash
# Start a campaign from scratch: archive the ledger, clear leftover working
# directories, leave the run ready for ./slurm/run_campaign.sh.
#
#   ./slurm/reset.sh --dry-run   list what would move and what would go
#   ./slurm/reset.sh             do it
#
# Results are MOVED into archive/<timestamp>/, never deleted, so a reset that
# turns out to be a mistake is recoverable. Only the transient measurement
# directories the suite creates are removed.
set -e
cd "$(dirname "$0")/.."

DRY=""
[ "$1" = "--dry-run" ] && DRY="yes"

SRCDIR="$PWD"
RUNDIR="${LAYOUT_SCRATCH:-/arf/scratch/$USER/layout-tuning}"
[ -d "$RUNDIR" ] || { echo "nothing to reset: $RUNDIR does not exist"; exit 0; }

# A reset while jobs are queued would archive a ledger those jobs then append to,
# leaving one campaign split across two files.
if command -v squeue >/dev/null 2>&1; then
    live=$(squeue -u "$USER" -h -o '%i' 2>/dev/null | wc -l)
    if [ "${live:-0}" -gt 0 ]; then
        echo "$live job(s) still queued or running. Cancel them first:"
        squeue -u "$USER" -o '  %.10i %.9T %j'
        echo
        echo "  scancel -u $USER      # or scancel the specific ids"
        echo
        echo "afterany dependencies treat a cancelled job as finished, so cancel"
        echo "the WHOLE chain at once -- cancelling only the running job releases"
        echo "the next one."
        exit 1
    fi
fi

cd "$RUNDIR"
stamp=$(date +%Y%m%d-%H%M%S)

ledgers=$(ls results*.jsonl run*.log 2>/dev/null || true)
if [ -n "$ledgers" ]; then
    rows=$(cat results*.jsonl 2>/dev/null | wc -l || echo 0)
    echo "archive -> archive/$stamp/  ($rows result rows)"
    for f in $ledgers; do echo "    $f"; done
    if [ -z "$DRY" ]; then
        mkdir -p "archive/$stamp"
        for f in $ledgers; do mv "$f" "archive/$stamp/"; done
    fi
else
    echo "no ledger to archive"
fi

# Transient per-cell directories. Each cell removes its own on success, so
# anything here is debris from an interrupted run.
#
# The prefixes are read out of the source rather than listed here. A hand-written
# list silently stops matching when an experiment is added or a path renamed, and
# the script then reports a clean scratch while GiB of debris sits in it.
# Only paths built with an interpolated name, i.e. "{SCRATCH}/e4_{...}" -- a bare
# literal in a docstring is an illustration, not a directory the suite creates,
# and deleting on a generic prefix like "thing*" would reach a user's own data.
prefixes=$(grep -rhoE '\{SCRATCH\}/[A-Za-z0-9_]+_\{' "$SRCDIR/layout_tuning" \
           --include='*.py' \
           | sed 's|{SCRATCH}/||' | sed 's/_{$//' | sort -u)
prefixes="$prefixes preflight"
[ -n "$prefixes" ] || { echo "could not read scratch prefixes from $SRCDIR/layout_tuning"; exit 1; }
echo "prefixes found in source: $(echo $prefixes | tr '\n' ' ')"

globs=""
for prefix in $prefixes; do globs="$globs ${prefix}*"; done
leftovers=$(ls -d $globs 2>/dev/null || true)
if [ -n "$leftovers" ]; then
    count=$(echo "$leftovers" | wc -w)
    echo "remove $count leftover working path(s):"
    echo "$leftovers" | tr ' ' '\n' | sed 's/^/    /' | head -12
    [ "$count" -gt 12 ] && echo "    ... and $((count - 12)) more"
    [ -z "$DRY" ] && rm -rf $leftovers
else
    echo "no leftover working directories"
fi

echo
if [ -n "$DRY" ]; then
    echo "dry run: nothing changed"
else
    echo "reset done. archived copies are in $RUNDIR/archive/$stamp/"
    echo "next:  ./slurm/run_campaign.sh"
fi
