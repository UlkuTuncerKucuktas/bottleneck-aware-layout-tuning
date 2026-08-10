"""Which eviction strategy lets data-on-MDT be measured cold?

DoM's benefit is that the open reply carries the file's data, so a working
measurement shows an expensive open and a nearly free read. If the client
already holds the file's read lock, the server sends no data with the open and
the client fetches it separately: a cheap open and an expensive read. The two
are easy to tell apart, which is what this probe does.

Three strategies, on identical freshly written files:

    none      read straight after writing, no eviction at all
    flush     the write-flush ioctl on an O_WRONLY fd (no read lock taken)
    fadvise   open O_RDONLY and drop pages (takes the read lock)

Run it on a compute node, not a login node:

    sbatch --chdir=$LAYOUT_SCRATCH -p <partition> -A <account> \
        --cpus-per-task=<n> --wrap "python3 $PWD/slurm/evict_probe.py"

or directly if you have an interactive allocation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layout_tuning.layout import (SCRATCH, OST_PLAIN, dom_layout, evict_by_flush,
                                  fresh_dir, drop_client_locks)
from layout_tuning.io import write_files
from layout_tuning.probes import profile_reads

FILE_KIB = 64
COUNT = 200


def fadvise_evict(paths):
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def measure(layout, label, evictor):
    directory = fresh_dir(f"{SCRATCH}/evictprobe_{label}", layout)
    paths, _ = write_files(directory, COUNT, FILE_KIB << 10, flush=True)
    if evictor:
        evictor(paths)
    stats = profile_reads(paths)
    return stats["open_us_p50"], stats["read_us_p50"], stats["total_us_p50"]


def main():
    print(f"{COUNT} files of {FILE_KIB} KiB, medians in microseconds per file")
    print(f"lru_size writable: {drop_client_locks()}")
    print()
    print(f"{'layout':>5} {'eviction':>9} {'open':>9} {'read':>9} {'total':>9}   verdict")

    results = {}
    for layout, lname in ((dom_layout(), "DoM"), (OST_PLAIN, "OST")):
        for ename, evictor in (("none", None),
                               ("flush", evict_by_flush),
                               ("fadvise", fadvise_evict)):
            open_us, read_us, total_us = measure(layout, f"{lname}_{ename}", evictor)
            results[(lname, ename)] = (open_us, read_us, total_us)
            # Inlining shows as the open carrying the cost and the read being
            # nearly free. The ratio is what separates the cases; absolute
            # numbers move with load.
            if lname == "DoM":
                verdict = "INLINED" if read_us < open_us else "not inlined"
            else:
                verdict = ""
            print(f"{lname:>5} {ename:>9} {open_us:>9.1f} {read_us:>9.1f} "
                  f"{total_us:>9.1f}   {verdict}")

    print()
    for ename in ("none", "flush", "fadvise"):
        dom = results[("DoM", ename)][2]
        ost = results[("OST", ename)][2]
        print(f"  {ename:>7}: DoM is {ost / dom:.2f}x OST")
    print()
    print("A strategy is usable when DoM's read is well under its open AND the")
    print("OST arm is slow enough to show it was measured cold. An OST arm as")
    print("fast as its DoM counterpart means neither was evicted.")


if __name__ == "__main__":
    main()
