"""Whether a wider layout buys anything once compute dominates."""

import os, time, shutil

from ..layout import SCRATCH, OST_PLAIN, OST_WIDE, dom_layout, fresh_dir, evict
from ..io import (write_files, read_many, read_ranges, mdt_used_kib,
                  osts_per_file, stat_rate, burn_cpu)
from ..probes import profile_reads
from ..runner import experiment, measured, median_by, rows, log, REPEATS


@experiment("bottleneck_budget")
def bottleneck_budget():
    """The premise: when compute dominates, a wider layout buys nothing.

    Two file sizes, because stripe width can only pay when a file spans more
    than one stripe. At 256 KiB against a 1 MiB stripe every file sits on a
    single OST whatever the count says, so widening is pure overhead. At 16 MiB
    a file spans 16 stripes and wide striping has real parallelism to offer,
    which is the regime where the budget argument has to hold on its merits.
    """
    batches = 24
    shapes = [("small", 256 << 10, 64), ("large", 16 << 20, 4)]
    arms = [("minimal", "-c 1 -S 1M"), ("moderate", "-c 8 -S 1M"), ("wide", OST_WIDE)]

    for shape, batch_size, batch_files in shapes:
        for compute_ms in [0, 5, 25, 100]:
            for name, layout in arms:
                for repeat in range(REPEATS):
                    cell = f"bottleneck_budget/{shape}/{name}/{compute_ms}/{repeat}"

                    def body(shape=shape, name=name, layout=layout, compute_ms=compute_ms,
                             batch_size=batch_size, batch_files=batch_files):
                        directory = fresh_dir(f"{SCRATCH}/budget_{shape}_{name}_{compute_ms}",
                                              layout)
                        paths, _ = write_files(directory, batch_files * batches, batch_size,
                                               flush=False)
                        evict(paths)
                        io_seconds, compute_seconds = 0.0, 0.0
                        bytes_read = 0
                        started = time.perf_counter()
                        for b in range(batches):
                            batch = paths[b * batch_files:(b + 1) * batch_files]
                            elapsed, got = read_many(batch, threads=8)
                            io_seconds += elapsed
                            bytes_read += got
                            t = time.perf_counter()
                            burn_cpu(compute_ms)
                            compute_seconds += time.perf_counter() - t
                        epoch = time.perf_counter() - started
                        per_file = osts_per_file(paths[0])
                        row = {"shape": shape, "layout": name, "compute_ms": compute_ms,
                               "files": len(paths), "file_kib": batch_size >> 10,
                               "epoch_s": epoch, "io_s": io_seconds,
                               "compute_s": compute_seconds,
                               "io_share": io_seconds / epoch,
                               "read_mib_per_s": bytes_read / io_seconds / (1 << 20),
                               "osts_per_file": per_file,
                               "ost_objects": len(paths) * per_file}
                        shutil.rmtree(directory, ignore_errors=True)
                        return row

                    measured(cell, body)

            seen = rows(f"bottleneck_budget/{shape}/")
            def at(name, key):
                return median_by(seen, key, layout=name, compute_ms=compute_ms)
            best = min(arms, key=lambda a: at(a[0], "epoch_s"))[0]
            log(f"{shape:>5} compute {compute_ms:>4}ms  epoch  " + "  ".join(
                f"{n} {at(n, 'epoch_s'):.2f}s" for n, _ in arms)
                + f"   fastest={best}")
            log(f"{'':>5} {'':>14}  I/O    " + "  ".join(
                f"{n} {at(n, 'io_s'):.2f}s" for n, _ in arms)
                + "   objects " + "/".join(f"{at(n, 'ost_objects'):.0f}" for n, _ in arms))


@experiment("mixed_classes")
def mixed_classes():
    """One dataset, two file classes: does per-class layout beat any single choice?"""
    small_count, large_count = 8000, 40
    small_bytes, large_bytes = 32 << 10, 64 << 20
    # Four uniform plans against one per-class plan. The uniform plans are the
    # honest competition: each is the best single choice for one of the two
    # classes, so beating all of them is what justifies deciding per class.
    plans = {
        "all_ost_plain":    (OST_PLAIN, OST_PLAIN),
        "all_ost_moderate": ("-c 8 -S 1M", "-c 8 -S 1M"),
        "all_ost_wide":     (OST_WIDE, OST_WIDE),
        "all_dom":          (dom_layout(), dom_layout()),
        "per_class":        (dom_layout(), "-c 8 -S 1M"),
    }
    for name, (small_layout, large_layout) in plans.items():
        for repeat in range(REPEATS):
            cell = f"mixed_classes/{name}/{repeat}"

            def body(name=name, small_layout=small_layout, large_layout=large_layout):
                small_dir = fresh_dir(f"{SCRATCH}/e3_{name}_small", small_layout)
                large_dir = fresh_dir(f"{SCRATCH}/e3_{name}_large", large_layout)
                small, small_w = write_files(small_dir, small_count, small_bytes, profile=True)
                large, large_w = write_files(large_dir, large_count, large_bytes)
                evict(small + large)
                small_time, _ = read_many(small, threads=8)
                large_time, _ = read_many(large, threads=8)
                row = {"plan": name, "files": len(small) + len(large),
                       "small_osts": osts_per_file(small[0]),
                       "large_osts": osts_per_file(large[0]),
                       "total_s": small_time + large_time,
                       "small_s": small_time, "large_s": large_time,
                       "small_files_per_s": len(small) / small_time,
                       "large_mib_per_s": large_count * (large_bytes >> 20) / large_time,
                       "ost_objects": (small_count * osts_per_file(small[0])
                                       + large_count * osts_per_file(large[0])),
                       "small_write_creates_per_s": small_w["creates_per_s"]}
                shutil.rmtree(small_dir, ignore_errors=True)
                shutil.rmtree(large_dir, ignore_errors=True)
                return row

            measured(cell, body)
        seen = rows("mixed_classes/")
        log(f"{name:>17}: {median_by(seen,'total_s',plan=name):5.1f}s total "
            f"({median_by(seen,'small_s',plan=name):.1f}s small + "
            f"{median_by(seen,'large_s',plan=name):.1f}s large), "
            f"{median_by(seen,'ost_objects',plan=name):>7.0f} OST objects")
