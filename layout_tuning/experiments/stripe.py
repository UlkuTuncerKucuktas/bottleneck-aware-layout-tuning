"""Stripe geometry against concurrency, for reads and for writes."""

import os, time, shutil

from ..layout import SCRATCH, OST_PLAIN, OST_WIDE, dom_layout, fresh_dir, evict
from ..io import (write_files, read_many, read_ranges, mdt_used_kib,
                  osts_per_file, stat_rate, burn_cpu)
from ..probes import profile_reads
from ..runner import experiment, measured, median_by, rows, log, REPEATS


@experiment("stripe_grid")
def stripe_grid():
    """Stripe count against reader threads, for whole-file and shared-file reads."""
    # A file only benefits from N stripes if it spans N of them. The whole-file
    # sizes bracket the 1 MiB stripe size deliberately: 0.25 MiB fits in a single
    # stripe however many are requested, so its row should stay flat, and 4 MiB
    # spans four. Wide striping gets its fair test in the shared arm, which reads
    # one cell-sized file spanning every level in the sweep, rather than from a
    # whole-file size large enough to starve the 32-thread cells of files.
    bytes_per_cell = 256 << 20
    thread_levels = [1, 2, 4, 8, 16, 32]
    stripe_levels = [1, 4, 8, 16, -1]
    for pattern in ["fpw", "shared"]:
        for size_mib in ([0.25, 4] if pattern == "fpw" else [bytes_per_cell >> 20]):
            for stripe_count in stripe_levels:
                for threads in thread_levels:
                    for repeat in range(REPEATS):
                        cell = f"stripe_grid/{pattern}/{size_mib}/{stripe_count}/{threads}/{repeat}"

                        def body(pattern=pattern, size_mib=size_mib,
                                 stripe_count=stripe_count, threads=threads):
                            directory = fresh_dir(
                                f"{SCRATCH}/e4_{pattern}_{stripe_count}_{threads}_{size_mib}",
                                f"-c {stripe_count} -S 1M")
                            size_bytes = int(size_mib * (1 << 20))
                            if pattern == "fpw":
                                paths, wstats = write_files(
                                    directory, bytes_per_cell // size_bytes, size_bytes,
                                    flush=False)
                                evict(paths)
                                elapsed, total = read_many(paths, threads)
                            else:
                                paths, wstats = write_files(directory, 1, bytes_per_cell,
                                                            flush=False)
                                evict(paths)
                                elapsed, total = read_ranges(paths[0], bytes_per_cell, threads)
                            row = {"pattern": pattern, "size_mib": size_mib,
                                   "files_per_thread": len(paths) / threads,
                                   "stripe_count": stripe_count, "threads": threads,
                                   "files": len(paths), "read_s": elapsed,
                                   "mib_per_s": total / elapsed / (1 << 20),
                                   "osts_per_file": osts_per_file(paths[0]),
                                   "write_mib_per_s": (len(paths) * size_bytes / (1 << 20)
                                                       / wstats["write_wall_s"])}
                            shutil.rmtree(directory, ignore_errors=True)
                            return row

                        measured(cell, body)
            seen = rows(f"stripe_grid/{pattern}/{size_mib}/")
            log(f"{pattern} {size_mib} MiB, read MiB/s by stripes x threads")
            for stripe_count in stripe_levels:
                line = [median_by(seen, "mib_per_s", stripe_count=stripe_count, threads=t)
                        for t in thread_levels]
                label = "all" if stripe_count < 0 else str(stripe_count)
                log(f"   {label:>4}: " + " ".join(f"{v:>7.0f}" for v in line))
            gains = []
            for threads in thread_levels:
                one = median_by(seen, "mib_per_s", stripe_count=1, threads=threads)
                widest = median_by(seen, "mib_per_s", stripe_count=-1, threads=threads)
                gains.append(widest / one if one else float("nan"))
            log("   gain widest/1 stripe: " + " ".join(f"{g:>6.2f}x" for g in gains))
            granted = " ".join(
                f"{'all' if c < 0 else c}->{median_by(seen, 'osts_per_file', stripe_count=c):.0f}"
                for c in stripe_levels)
            log(f"   stripes requested->granted: {granted}")


@experiment("write_path")
def write_path():
    """Everything else measures reads. Checkpoints are writes."""
    # Checkpoint writes are the case wide striping is meant for, so the shapes
    # bracket it: many small files (one stripe each, whatever is requested) and
    # one large file that spans the whole sweep.
    bytes_per_cell = 256 << 20
    for label, size_mib, count in [("many_small", 0.25, bytes_per_cell // (256 << 10)),
                                   ("one_large", 256.0, 1)]:
        for stripe_count in [1, 4, 16, -1]:
            for repeat in range(REPEATS):
                cell = f"write_path/{label}/{stripe_count}/{repeat}"

                def body(label=label, size_mib=size_mib, count=count,
                         stripe_count=stripe_count):
                    directory = fresh_dir(f"{SCRATCH}/e11_{label}_{stripe_count}",
                                          f"-c {stripe_count} -S 1M")
                    paths, wstats = write_files(directory, count,
                                                int(size_mib * (1 << 20)),
                                                flush=False, profile=True)
                    row = {"shape": label, "stripe_count": stripe_count,
                           "files": len(paths),
                           "mib_per_s": count * size_mib / wstats["write_wall_s"],
                           "osts_per_file": osts_per_file(paths[0]), **wstats}
                    shutil.rmtree(directory, ignore_errors=True)
                    return row

                measured(cell, body)
        seen = rows(f"write_path/{label}/")
        levels = [1, 4, 16, -1]
        line = [f"{median_by(seen,'mib_per_s',stripe_count=c):.0f}" for c in levels]
        held = [f"{median_by(seen,'osts_per_file',stripe_count=c):.0f}" for c in levels]
        log(f"{label:>11} write MiB/s at 1/4/16/all stripes: "
            + " ".join(f"{v:>6}" for v in line)
            + "   objects per file: " + "/".join(held))
