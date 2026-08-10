#!/bin/bash
# Why is data-on-MDT not inlining? Answers in seconds, on a login node, no job.
#
#   ./slurm/dom_probe.sh
#
# An earlier run on this filesystem measured DoM reads at 39 us/file; the suite
# now measures 1656 us, with the cost moved from open to read. That is the
# signature of the data not being inlined. This prints the four facts that
# distinguish the candidate causes, so the next cluster run is not spent guessing.
set -u
command -v lfs >/dev/null || {
    echo "no lfs on PATH: run this on a Lustre client, not a laptop"
    exit 1
}
DIR="${LAYOUT_SCRATCH:-/arf/scratch/$USER/layout-tuning}/domprobe"

echo "=== 1. what the MDT will grant ==="
# The requested -E component is capped by this. If it reads ~114688 rather than
# 1048576, a request of 1M yields a ~112 KiB component -- which is exactly where
# the earlier run's cliff sat, and it would explain every flat sweep since.
lctl get_param -n mdt.*.dom_stripesize 2>/dev/null || echo "  (not readable as a user)"
lctl get_param -n llite.*.inode_cache 2>/dev/null | head -1

echo
echo "=== 2. what a file actually gets ==="
rm -rf "$DIR"; mkdir -p "$DIR"
lfs setstripe -E 1M -L mdt -E -1 -c 1 -S 1M "$DIR"
echo "-- directory layout as set:"
lfs getstripe -d "$DIR" | grep -E 'lcme|stripe|pattern' | head -8
dd if=/dev/urandom of="$DIR/f" bs=64K count=1 status=none
echo "-- the file's granted layout:"
lfs getstripe "$DIR/f" | grep -E 'lcme_id|lcme_extent|pattern|stripe_count' | head -10

echo
echo "=== 3. does the file have OST objects? ==="
# A properly inlined DoM file has NO object on any OST for its first component.
# If objects appear covering byte 0, the data went to an OST and DoM did nothing.
lfs getstripe "$DIR/f" | grep -E 'l_ost_idx|obdidx|objid' | head -5 \
    || echo "  no OST objects listed -- consistent with data on the MDT"

echo
echo "=== 4. is the component the size we asked for? ==="
lfs getstripe "$DIR/f" | awk '/lcme_extent.e_end/ {print "  component ends at", $2, "bytes =", $2/1024, "KiB"}'

rm -rf "$DIR"
echo
echo "Read line 4 against the 1048576 bytes requested. If it is smaller, the cap"
echo "is the answer: every DoM sweep has been placed relative to the wrong boundary."
