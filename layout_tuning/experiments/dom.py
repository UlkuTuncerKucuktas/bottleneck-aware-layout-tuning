"""Data-on-MDT: where inlining stops, what it costs, and how it behaves."""

import os, time, shutil

from ..layout import (SCRATCH, OST_PLAIN, OST_WIDE, dom_layout, fresh_dir,
                       working_dir, evict)
from ..io import (write_files, read_many, mdt_used_kib, dom_component_kib,
                  osts_per_file, stat_rate, burn_cpu)
from ..probes import profile_reads
from ..runner import experiment, measured, median_by, rows, log, REPEATS


@experiment("dom_cutoff")
def dom_cutoff():
    """Where exactly does inlining stop, and does it move with the -E component?"""
    # Wide enough to bracket the boundary wherever it lands. An earlier run with
    # -E 1M cliffed between 112 and 128 KiB, which is far below the 1 MiB asked
    # for -- so the boundary is set by what the MDT grants, not by the request,
    # and a grid placed by the requested size can miss it entirely. Each row now
    # records the granted component so the cliff can be located against it.
    sizes_kib = [32, 64, 96, 112, 128, 144, 192, 256, 384]
    for component in ["64K", "256K", "1M"]:
        for size_kib in sizes_kib:
            for repeat in range(REPEATS):
                cell = f"dom_cutoff/{component}/{size_kib}/{repeat}"

                def body(component=component, size_kib=size_kib):
                    with working_dir(f"{SCRATCH}/e1_{component}_{size_kib}",
                                          dom_layout(component)) as directory:
                        paths, wstats = write_files(directory, 300, size_kib << 10, profile=True)
                        evict(paths)
                        row = {"component": component, "size_kib": size_kib,
                               "granted_kib": dom_component_kib(paths[0]),
                               "files": len(paths), **wstats, **profile_reads(paths)}
                    return row

                measured(cell, body)
        seen = rows(f"dom_cutoff/{component}/")
        line = [f"{median_by(seen, 'total_us_p50', size_kib=s):.0f}" for s in sizes_kib]
        granted = median_by(seen, "granted_kib", size_kib=sizes_kib[0])
        log(f"-E {component:>4} (granted {granted:.0f}K) p50 us/file: "
            + " ".join(f"{v:>6}" for v in line))
    log("sizes(KiB):          " + " ".join(f"{s:>6}" for s in sizes_kib))


@experiment("dom_footprint")
def dom_footprint():
    """What DoM costs on the MDT, and what it does to metadata op rates."""
    for size_kib in [4, 16, 64, 112]:
        for arm, layout in [("ost", OST_PLAIN), ("dom", dom_layout())]:
            cell = f"dom_footprint/{arm}/{size_kib}"

            def body(arm=arm, layout=layout, size_kib=size_kib):
                with working_dir(f"{SCRATCH}/e2_{arm}_{size_kib}", layout) as directory:
                    before = mdt_used_kib()
                    paths, wstats = write_files(directory, 20_000, size_kib << 10, profile=True)
                    after = mdt_used_kib()
                    evict(paths)
                    row = {"arm": arm, "size_kib": size_kib, "files": len(paths),
                           "mdt_kib_per_file": (after - before) / len(paths),
                           "mdt_kib_total": after - before,
                           "stats_per_s": stat_rate(paths),
                           "osts_per_file": osts_per_file(paths[0]), **wstats}
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
        # The OST arm is flushed too. The ioctl forces data out to the servers,
        # so an unflushed arm is read from the client's own cache and is not a
        # comparable baseline: in an earlier run the same OST files measured
        # 338.9 us unflushed against 2763.8 us flushed. Comparing a flushed DoM
        # arm against an unflushed OST arm charges DoM for the flush and credits
        # OST with a warm cache, which is what produced a negative saving.
        for arm, layout, flush in [("ost", OST_PLAIN, True),
                                   ("dom_noflush", dom_layout(), False),
                                   ("dom_flush", dom_layout(), True)]:
            cell = f"flush_cost/{arm}/{size_kib}"

            def body(arm=arm, layout=layout, flush=flush, size_kib=size_kib):
                with working_dir(f"{SCRATCH}/e7_{arm}_{size_kib}", layout) as directory:
                    paths, wstats = write_files(directory, 4000, size_kib << 10,
                                                flush=flush, profile=True)
                    evict(paths)
                    row = {"arm": arm, "size_kib": size_kib, "files": len(paths),
                           "granted_kib": dom_component_kib(paths[0]),
                           **wstats, **profile_reads(paths)}
                return row

            measured(cell, body)

        seen = rows(f"flush_cost/")
        def pick(arm, key):
            return median_by(seen, key, arm=arm, size_kib=size_kib)
        extra_write = pick("dom_flush", "w_flush_us_mean")
        # against DoM without the flush: the question is what the flush buys on
        # the same layout, not what DoM buys over OST (tail_latency answers that).
        saved = pick("dom_noflush", "total_us_mean") - pick("dom_flush", "total_us_mean")
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
                with working_dir(f"{SCRATCH}/e8_{arm}_{size_kib}", layout) as directory:
                    paths, _ = write_files(directory, 4000, size_kib << 10)
                    evict(paths)
                    row = {"arm": arm, "size_kib": size_kib, "files": len(paths),
                           "granted_kib": dom_component_kib(paths[0]),
                           **profile_reads(paths)}
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
        # Files deliberately survive between epochs -- re-reading the same set is
        # the measurement -- so cleanup waits for the last one and runs even if
        # an epoch failed.
        shutil.rmtree(f"{SCRATCH}/e9_{arm}", ignore_errors=True)
    seen = rows("repeated_epochs/")
    for arm in ["ost", "dom"]:
        rates = [median_by(seen, "files_per_s", arm=arm, epoch=e) for e in range(4)]
        log(f"{arm:>4}: " + " ".join(f"{v:.0f}" for v in rates) + " files/s per epoch")
