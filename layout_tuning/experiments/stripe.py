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
    bytes_per_cell = 256 << 20
    for pattern in ["fpw", "shared"]:
        for size_mib in ([0.25, 4] if pattern == "fpw" else [4]):
            for stripe_count in [1, 2, 4, 8, -1]:
                for threads in [1, 2, 4, 8, 16, 32]:
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
            for stripe_count in [1, 2, 4, 8, -1]:
                line = [median_by(seen, "mib_per_s", stripe_count=stripe_count, threads=t)
                        for t in [1, 2, 4, 8, 16, 32]]
                label = "all" if stripe_count < 0 else str(stripe_count)
                log(f"   {label:>4}: " + " ".join(f"{v:>7.0f}" for v in line))
            gains = []
            for threads in [1, 2, 4, 8, 16, 32]:
                one = median_by(seen, "mib_per_s", stripe_count=1, threads=threads)
                widest = median_by(seen, "mib_per_s", stripe_count=-1, threads=threads)
                gains.append(widest / one if one else float("nan"))
            log("   gain widest/1 stripe: " + " ".join(f"{g:>6.2f}x" for g in gains))


@experiment("write_path")
def write_path():
    """Everything else measures reads. Checkpoints are writes."""
    bytes_per_cell = 256 << 20
    for label, size_mib, count in [("many_small", 0.25, bytes_per_cell // (256 << 10)),
                                   ("one_large", 256.0, 1)]:
        for stripe_count in [1, 4, -1]:
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
        line = [f"{median_by(seen,'mib_per_s',stripe_count=c):.0f}" for c in [1, 4, -1]]
        held = [f"{median_by(seen,'osts_per_file',stripe_count=c):.0f}" for c in [1, 4, -1]]
        log(f"{label:>11} write MiB/s at 1/4/all stripes: "
                   + " ".join(f"{v:>6}" for v in line) + "   OSTs held: " + "/".join(held))
