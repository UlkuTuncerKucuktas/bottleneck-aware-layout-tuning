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

echo "=== 1. the STORAGE boundary: where the data lives ==="
# lod.*.dom_stripesize is the per-MDT maximum for a DoM component. A larger
# request is silently truncated to it, so what was asked for is not necessarily
# what a file got. Default 1 MiB, 64 KiB aligned, 1 GiB ceiling.
lctl get_param -n lod.*.dom_stripesize 2>/dev/null \
    || lctl get_param -n mdt.*.dom_stripesize 2>/dev/null \
    || echo "  (server-side parameter, not readable from a client)"

echo
echo "=== 1b. the LATENCY boundary: how much arrives with the open ==="
# A separate mechanism. The MDT ships file data inside the open reply when it
# fits the reply buffer, so attributes, lock and data arrive in one RPC. Past
# that the file is still on the MDT but needs a second RPC to read. The limit
# is reply-buffer space, NOT the component size -- which is why a component
# granted at 1 MiB can still show a read cliff around 100-130 KiB.
# The leaf is mdc_dom_min_repsize, not dom_min_repsize: the mdc_ prefix is part
# of the parameter name, not just the namespace.
lctl get_param -n mdc.*.mdc_dom_min_repsize 2>/dev/null \
    || lctl get_param -n mdc.*.dom_min_repsize 2>/dev/null \
    || echo "  (no dom repsize parameter on this client)"
echo "-- (this is the MINIMUM reply size the client asks for, not the ceiling;"
echo "    the cliff is where the payload stops fitting the reply buffer)"

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
# A DoM file's first component holds its data on the MDT, so no OST object
# should exist for it. The second component is declared but uninstantiated
# until the file grows past the boundary, and an uninstantiated component has
# no objects either -- so an empty result here is the expected, correct answer.
# Capture first: a pipeline's status is the LAST command's, and `head` exits 0
# on empty input, so `grep ... | head || echo` never reaches the fallback.
objects=$(lfs getstripe "$DIR/f" | grep -E 'l_ost_idx|obdidx|objid' | head -5)
if [ -n "$objects" ]; then
    echo "$objects"
    echo "  ^ the file HAS OST objects, so its data did not stay on the MDT"
else
    echo "  no OST objects: the data is on the MDT, which is what DoM should do"
fi

echo
echo "=== 4. is the component the size we asked for? ==="
lfs getstripe "$DIR/f" | awk '/lcme_extent.e_end/ {print "  component ends at", $2, "bytes =", $2/1024, "KiB"}'

rm -rf "$DIR"
echo
echo "Line 4 is the STORAGE boundary: below it a file's data sits on the MDT."
echo "A grant smaller than requested means the MDT truncated it, and every DoM"
echo "sweep would be placed against the wrong size."
echo
echo "The LATENCY boundary is separate and is not printed by any of the above:"
echo "it is where the payload stops fitting the open reply buffer, which on this"
echo "filesystem measured near 112-128 KiB with a fully granted 1 MiB component."
echo "dom_cutoff locates it empirically, which is the only way to see it."
