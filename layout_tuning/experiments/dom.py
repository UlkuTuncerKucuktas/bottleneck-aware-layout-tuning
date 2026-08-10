"""Data-on-MDT: where inlining stops, what it costs, and how it behaves."""

import os, time, shutil

from ..layout import SCRATCH, OST_PLAIN, OST_WIDE, dom_layout, fresh_dir, evict
from ..io import (write_files, read_many, read_ranges, mdt_used_kib,
                  osts_per_file, stat_rate, burn_cpu)
from ..probes import profile_reads
from ..runner import experiment, measured, median_by, rows, log, REPEATS


@experiment("dom_cutoff")
def dom_cutoff():
    """Where exactly does inlining stop, and does it move with the -E component?"""
    sizes_kib = [96, 112, 116, 120, 124, 128, 132, 160]
    for component in ["64K", "256K", "1M"]:
        for size_kib in sizes_kib:
            for repeat in range(REPEATS):
                cell = f"dom_cutoff/{component}/{size_kib}/{repeat}"

                def body(component=component, size_kib=size_kib):
                    directory = fresh_dir(f"{SCRATCH}/e1_{component}_{size_kib}",
                                          dom_layout(component))
                    paths, wstats = write_files(directory, 300, size_kib << 10, profile=True)
                    evict(paths)
                    row = {"component": component, "size_kib": size_kib,
                           "files": len(paths), **wstats, **profile_reads(paths)}
                    shutil.rmtree(directory, ignore_errors=True)
                    return row

                measured(cell, body)
        seen = rows(f"dom_cutoff/{component}/")
        line = [f"{median_by(seen, 'total_us_p50', size_kib=s):.0f}" for s in sizes_kib]
        log(f"-E {component:>4} p50 us/file: " + " ".join(f"{v:>6}" for v in line))
    log("sizes(KiB):          " + " ".join(f"{s:>6}" for s in sizes_kib))


@experiment("dom_footprint")
def dom_footprint():
    """What DoM costs on the MDT, and what it does to metadata op rates."""
    for size_kib in [4, 16, 64, 112]:
        for arm, layout in [("ost", OST_PLAIN), ("dom", dom_layout())]:
            cell = f"dom_footprint/{arm}/{size_kib}"

            def body(arm=arm, layout=layout, size_kib=size_kib):
                directory = fresh_dir(f"{SCRATCH}/e2_{arm}_{size_kib}", layout)
                before = mdt_used_kib()
                paths, wstats = write_files(directory, 20_000, size_kib << 10, profile=True)
                after = mdt_used_kib()
                evict(paths)
                row = {"arm": arm, "size_kib": size_kib, "files": len(paths),
                       "mdt_kib_per_file": (after - before) / len(paths),
                       "mdt_kib_total": after - before,
                       "stats_per_s": stat_rate(paths),
                       "osts_per_file": osts_per_file(paths[0]), **wstats}
                shutil.rmtree(directory, ignore_errors=True)
                return row

            measured(cell, body)
        seen = rows("dom_footprint/")
        log(f"{size_kib:>4}K  " + "  ".join(
            f"{a}: {median_by(seen,'mdt_kib_per_file',arm=a,size_kib=size_kib):.2f} KiB/file MDT, "
            f"{median_by(seen,'creates_per_s',arm=a,size_kib=size_kib):.0f} creates/s"
            for a in ["ost", "dom"]))


@experiment("flush_cost")
def flush_cost():
    """DoM's speedup needs a per-file flush. How many reads pay it back?"""
    for size_kib in [4, 64, 112]:
        for arm, layout, flush in [("ost", OST_PLAIN, False),
                                   ("dom_noflush", dom_layout(), False),
                                   ("dom_flush", dom_layout(), True)]:
            cell = f"flush_cost/{arm}/{size_kib}"

            def body(arm=arm, layout=layout, flush=flush, size_kib=size_kib):
                directory = fresh_dir(f"{SCRATCH}/e7_{arm}_{size_kib}", layout)
                paths, wstats = write_files(directory, 4000, size_kib << 10,
                                            flush=flush, profile=True)
                evict(paths)
                row = {"arm": arm, "size_kib": size_kib, "files": len(paths),
                       **wstats, **profile_reads(paths)}
                shutil.rmtree(directory, ignore_errors=True)
                return row

            measured(cell, body)

        seen = rows(f"flush_cost/")
        def pick(arm, key):
            return median_by(seen, key, arm=arm, size_kib=size_kib)
        extra_write = pick("dom_flush", "w_flush_us_mean")
        saved = pick("ost", "total_us_mean") - pick("dom_flush", "total_us_mean")
        log(f"{size_kib:>4}K: flush {extra_write:.0f} us/file at write, "
                   f"saves {saved:.0f} us/file per read, breakeven "
                   f"{extra_write/saved if saved > 0 else float('inf'):.2f} reads")


@experiment("tail_latency")
def tail_latency():
    """A dataloader stalls on its slowest reads, not its median one."""
    for size_kib in [4, 64, 112]:
        for arm, layout in [("ost", OST_PLAIN), ("dom", dom_layout())]:
            cell = f"tail_latency/{arm}/{size_kib}"

            def body(arm=arm, layout=layout, size_kib=size_kib):
                directory = fresh_dir(f"{SCRATCH}/e8_{arm}_{size_kib}", layout)
                paths, _ = write_files(directory, 4000, size_kib << 10)
                evict(paths)
                row = {"arm": arm, "size_kib": size_kib, "files": len(paths),
                       **profile_reads(paths)}
                shutil.rmtree(directory, ignore_errors=True)
                return row

            measured(cell, body)
        seen = rows("tail_latency/")
        for arm in ["ost", "dom"]:
            log(f"{size_kib:>4}K {arm:>4}: open p50/p99 "
                       f"{median_by(seen,'open_us_p50',arm=arm,size_kib=size_kib):.0f}/"
                       f"{median_by(seen,'open_us_p99',arm=arm,size_kib=size_kib):.0f}  "
                       f"total p50/p95/p99 "
                       f"{median_by(seen,'total_us_p50',arm=arm,size_kib=size_kib):.0f}/"
                       f"{median_by(seen,'total_us_p95',arm=arm,size_kib=size_kib):.0f}/"
                       f"{median_by(seen,'total_us_p99',arm=arm,size_kib=size_kib):.0f} us")


@experiment("repeated_epochs")
def repeated_epochs():
    """Training reads a dataset many times. Does the DoM win survive re-reads?"""
    for arm, layout in [("ost", OST_PLAIN), ("dom", dom_layout())]:
        for epoch in range(4):
            cell = f"repeated_epochs/{arm}/{epoch}"

            def body(arm=arm, layout=layout, epoch=epoch):
                directory = f"{SCRATCH}/e9_{arm}"
                if epoch == 0:
                    fresh_dir(directory, layout)
                    paths, _ = write_files(directory, 6000, 64 << 10)
                    evict(paths)
                else:
                    paths = sorted(os.path.join(directory, n)
                                   for n in os.listdir(directory))
                elapsed, _ = read_many(paths, threads=8)
                return {"arm": arm, "epoch": epoch, "files": len(paths),
                        "read_s": elapsed, "files_per_s": len(paths) / elapsed}

            measured(cell, body)
        shutil.rmtree(f"{SCRATCH}/e9_{arm}", ignore_errors=True)
    seen = rows("repeated_epochs/")
    for arm in ["ost", "dom"]:
        rates = [median_by(seen, "files_per_s", arm=arm, epoch=e) for e in range(4)]
        log(f"{arm:>4}: " + " ".join(f"{v:.0f}" for v in rates) + " files/s per epoch")
