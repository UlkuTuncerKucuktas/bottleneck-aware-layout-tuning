"""What has to happen between writing a file and reading it, for a cold read?

DoM's benefit is that the open reply carries the file's data, so a working
measurement shows an expensive open and a nearly free read. If the client
already holds the file's read lock the server sends nothing with the open and
the client fetches separately: a cheap open and an expensive read. The ratio
between the two phases tells the cases apart.

Two independent factors, crossed:

    write flush   the data-version ioctl on the write descriptor, or not.
                  Releasing the write lock is what pushes data to the server
                  and drops the client's cached pages -- Lustre caches under
                  lock, so surrendering the lock surrenders the cache.

    after write   nothing / reopen the file O_WRONLY and flush again /
                  reopen it O_RDONLY and drop pages. Both reopens exist to
                  answer whether ANY reopen before the measured read defeats
                  inlining, or only one that takes a read lock.

Crossing them matters because the two were conflated: an arm labelled "no
eviction" that still flushes at write time cannot show what the write flush
contributes, and every arm having it makes the control useless for that
question.

Run it on a compute node:

    source slurm/barbun.env
    ./slurm/submit_probe.sh slurm/evict_probe.py
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
    if not hasattr(os, "posix_fadvise"):
        raise RuntimeError("posix_fadvise is Linux-only")
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def measure(layout, label, evictor, write_flush):
    directory = fresh_dir(f"{SCRATCH}/evictprobe_{label}", layout)
    paths, _ = write_files(directory, COUNT, FILE_KIB << 10, flush=write_flush)
    if evictor:
        evictor(paths)
    stats = profile_reads(paths)
    # first_read_us is the phase that distinguishes the cases: it is the read
    # that either finds the data already in hand from the open reply or has to
    # issue an RPC for it. rest_read_us covers the remaining chunks of a file
    # larger than one buffer, which for these 64 KiB files is nothing.
    return stats["open_us_p50"], stats["first_read_us_p50"], stats["total_us_p50"]


def main():
    print(f"{COUNT} files of {FILE_KIB} KiB, medians in microseconds per file")
    print(f"lru_size writable: {drop_client_locks()}")
    print()
    print(f"{'layout':>5} {'wflush':>6} {'after':>9} {'open':>9} {'read':>9} "
          f"{'total':>9}   verdict")

    results = {}
    for layout, lname in ((dom_layout(), "DoM"), (OST_PLAIN, "OST")):
      for write_flush in (True, False):
        for ename, evictor in (("none", None),
                               ("reopen-w", evict_by_flush),
                               ("reopen-r", fadvise_evict)):
            # One strategy failing must not cost the rows already measured: the
            # comparison is the point, and a partial table still shows it.
            wf = "yes" if write_flush else "no"
            try:
                open_us, read_us, total_us = measure(
                    layout, f"{lname}_{wf}_{ename}", evictor, write_flush)
            except (OSError, RuntimeError) as exc:
                print(f"{lname:>5} {wf:>6} {ename:>9} {'-':>9} {'-':>9} "
                      f"{'-':>9}   unavailable: {exc}")
                continue
            results[(lname, write_flush, ename)] = (open_us, read_us, total_us)
            # Inlining shows as the open carrying the cost and the read being
            # nearly free. The ratio is what separates the cases; absolute
            # numbers move with load.
            if lname == "DoM":
                # Inlining is not merely "read faster than open" -- on a warm
                # cache both are microseconds and the ratio means nothing. The
                # signature is a read that is a small FRACTION of the open,
                # because the payload arrived with the reply and the read is a
                # memory copy. The validated run on this filesystem showed
                # 668 us open against 39 us read, a ratio near 0.06; the broken
                # configuration showed 332 against 1656, a ratio of 5.
                ratio = read_us / open_us if open_us else float("inf")
                if ratio < 0.25:
                    verdict = f"INLINED (read/open {ratio:.2f})"
                elif ratio > 1.0:
                    verdict = f"NOT inlined (read/open {ratio:.2f})"
                else:
                    verdict = f"unclear (read/open {ratio:.2f}), likely warm"
            else:
                verdict = ""
            print(f"{lname:>5} {wf:>6} {ename:>9} {open_us:>9.1f} "
                  f"{read_us:>9.1f} {total_us:>9.1f}   {verdict}")

    print()
    for write_flush in (True, False):
        wf = "yes" if write_flush else "no"
        for ename in ("none", "reopen-w", "reopen-r"):
            key_d = ("DoM", write_flush, ename)
            key_o = ("OST", write_flush, ename)
            if key_d not in results or key_o not in results:
                print(f"  wflush={wf:<3} {ename:>9}: incomplete, no comparison")
                continue
            dom = results[key_d][2]
            ost = results[key_o][2]
            print(f"  wflush={wf:<3} {ename:>9}: DoM is {ost / dom:.2f}x OST")
    print()
    print("A strategy is usable when DoM's read is well under its open AND the")
    print("OST arm is slow enough to show it was measured cold. An OST arm as")
    print("fast as its DoM counterpart means neither was evicted.")


if __name__ == "__main__":
    main()
