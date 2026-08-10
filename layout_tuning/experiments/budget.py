"""Whether a wider layout buys anything once compute dominates."""

import os, time, shutil

from ..layout import SCRATCH, OST_PLAIN, OST_WIDE, dom_layout, fresh_dir, evict
from ..io import (write_files, read_many, read_ranges, mdt_used_kib,
                  osts_per_file, stat_rate, burn_cpu)
from ..probes import profile_reads
from ..runner import experiment, measured, median_by, rows, log, REPEATS


@experiment("bottleneck_budget")
def bottleneck_budget():
    """The premise: when compute dominates, a wider layout buys nothing."""
    batch_files, batch_size, batches = 64, 256 << 10, 24
    for compute_ms in [0, 5, 25, 100]:
        for name, layout in [("minimal", OST_PLAIN), ("wide", OST_WIDE)]:
            for repeat in range(REPEATS):
                cell = f"bottleneck_budget/{name}/{compute_ms}/{repeat}"

                def body(name=name, layout=layout, compute_ms=compute_ms):
                    directory = fresh_dir(f"{SCRATCH}/e5_{name}_{compute_ms}", layout)
                    paths, _ = write_files(directory, batch_files * batches, batch_size,
                                           flush=False)
                    evict(paths)
                    io_seconds, compute_seconds = 0.0, 0.0
                    started = time.perf_counter()
                    for b in range(batches):
                        batch = paths[b * batch_files:(b + 1) * batch_files]
                        elapsed, _ = read_many(batch, threads=8)
                        io_seconds += elapsed
                        t = time.perf_counter()
                        burn_cpu(compute_ms)
                        compute_seconds += time.perf_counter() - t
                    epoch = time.perf_counter() - started
                    row = {"layout": name, "compute_ms": compute_ms, "files": len(paths),
                           "epoch_s": epoch, "io_s": io_seconds,
                           "compute_s": compute_seconds,
                           "io_share": io_seconds / epoch,
                           "ost_objects": len(paths) * osts_per_file(paths[0])}
                    shutil.rmtree(directory, ignore_errors=True)
                    return row

                measured(cell, body)
        seen = rows("bottleneck_budget/")
        wide = median_by(seen, "epoch_s", layout="wide", compute_ms=compute_ms)
        minimal = median_by(seen, "epoch_s", layout="minimal", compute_ms=compute_ms)
        log(f"compute {compute_ms:>4}ms: epoch wide {wide:6.1f}s vs minimal "
                   f"{minimal:6.1f}s ({minimal/wide:.2f}x), OST objects "
                   f"{median_by(seen,'ost_objects',layout='wide',compute_ms=compute_ms):.0f}"
                   f" vs {median_by(seen,'ost_objects',layout='minimal',compute_ms=compute_ms):.0f}")


@experiment("mixed_classes")
def mixed_classes():
    """One dataset, two file classes: does per-class layout beat any single choice?"""
    small_count, large_count = 8000, 40
    small_bytes, large_bytes = 32 << 10, 64 << 20
    plans = {
        "all_ost_plain": (OST_PLAIN, OST_PLAIN),
        "all_ost_wide":  (OST_WIDE, OST_WIDE),
        "all_dom":       (dom_layout(), dom_layout()),
        "per_class":     (dom_layout(), OST_WIDE),
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
        log(f"{name:>14}: {median_by(seen,'total_s',plan=name):.1f}s, "
                   f"{median_by(seen,'ost_objects',plan=name):.0f} OST objects")
