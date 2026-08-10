"""Stripe geometry against concurrency, for reads and for writes."""

import os, time, shutil

from ..layout import SCRATCH, OST_PLAIN, OST_WIDE, dom_layout, fresh_dir, evict
from ..io import (write_files, read_many, read_ranges, ranged_reader, read_until,
                  mdt_used_kib,
                  osts_per_file, stat_rate, burn_cpu, restricted_to_cores,
                  allocated_cores, pool_names, pool_members, inherited_pool,
                  target_inventory)
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
    # A single flash OST already serves ~2.9 GiB/s to one thread, so a few
    # hundred MiB is read in tens of milliseconds and the measurement captures
    # thread startup rather than steady-state throughput. Cells therefore read
    # repeatedly until the timed region reaches MIN_SECONDS, with the cache
    # dropped before every pass.
    bytes_per_cell = 2 << 30
    min_seconds = 2.0
    # 2 GiB of 0.25 MiB files would be 8192 files per cell, which measures
    # metadata rate more than stripe behaviour. 1 GiB is the compromise: 4096
    # files is still a dataloader-shaped set, and each pass is long enough that
    # per-pass RPC latency does not dominate it.
    cell_bytes = {0.25: 1 << 30, 4: bytes_per_cell}
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
                                volume = cell_bytes[size_mib]
                                paths, wstats = write_files(
                                    directory, volume // size_bytes, size_bytes,
                                    flush=False)
                                elapsed, total, passes = read_until(
                                    paths, threads, min_seconds)
                            else:
                                paths, wstats = write_files(directory, 1, bytes_per_cell,
                                                            flush=False)
                                elapsed, total, passes = read_until(
                                    paths, threads, min_seconds,
                                    reader=ranged_reader(paths[0], bytes_per_cell))
                            row = {"pattern": pattern, "size_mib": size_mib,
                                   "files_per_thread": len(paths) / threads,
                                   "read_s": elapsed, "passes": passes,
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


@experiment("cores_vs_throughput")
def cores_vs_throughput():
    """Read throughput against cores available, for three stripe widths.

    Reader threads that block in read() hold no core, so thread count and core
    count are separate resources: `stripe_grid` sweeps threads on a fixed
    allocation, and this sweeps the allocation itself. A job whose cores are
    consumed by computation cannot drive many concurrent reads, so the width
    its layout can actually exploit is bounded by the cores it has spare.

    Threads are set to twice the core count, which is enough to keep the
    allocation busy without measuring a thread-count effect instead.
    """
    bytes_per_cell = 2 << 30
    min_seconds = 2.0
    file_size = 4 << 20
    core_levels = [c for c in [1, 2, 4, 8, 16] if c <= allocated_cores()]
    stripe_levels = [1, 8, -1]

    for stripe_count in stripe_levels:
        for cores in core_levels:
            for repeat in range(REPEATS):
                cell = f"cores_vs_throughput/{stripe_count}/{cores}/{repeat}"

                def body(stripe_count=stripe_count, cores=cores):
                    directory = fresh_dir(f"{SCRATCH}/cores_{stripe_count}_{cores}",
                                          f"-c {stripe_count} -S 1M")
                    paths, _ = write_files(directory, bytes_per_cell // file_size,
                                           file_size, flush=False)
                    with restricted_to_cores(cores):
                        elapsed, total, passes = read_until(paths, cores * 2, min_seconds)
                    row = {"stripe_count": stripe_count, "cores": cores,
                           "threads": cores * 2, "files": len(paths),
                           "read_s": elapsed, "passes": passes,
                           "mib_per_s": total / elapsed / (1 << 20),
                           "osts_per_file": osts_per_file(paths[0])}
                    shutil.rmtree(directory, ignore_errors=True)
                    return row

                measured(cell, body)

    seen = rows("cores_vs_throughput/")
    log("read MiB/s by stripe width x cores available")
    log("   cores: " + " ".join(f"{c:>7}" for c in core_levels))
    for stripe_count in stripe_levels:
        line = [median_by(seen, "mib_per_s", stripe_count=stripe_count, cores=c)
                for c in core_levels]
        label = "all" if stripe_count < 0 else str(stripe_count)
        log(f"   {label:>4}: " + " ".join(f"{v:>7.0f}" for v in line))

    # The number the advisor needs: at each core count, does extra width pay?
    log("   gain of widest over 1 stripe, per core count:")
    gains = []
    for cores in core_levels:
        one = median_by(seen, "mib_per_s", stripe_count=1, cores=cores)
        widest = median_by(seen, "mib_per_s", stripe_count=-1, cores=cores)
        gains.append(widest / one if one else float("nan"))
    log("         " + " ".join(f"{g:>6.2f}x" for g in gains))
    granted = " ".join(
        f"{'all' if c < 0 else c}->{median_by(seen, 'osts_per_file', stripe_count=c):.0f}"
        for c in stripe_levels)
    log(f"   stripes requested->granted: {granted}")


@experiment("pool_selection")
def pool_selection():
    """What the OST pool a file lands in is worth, at matched stripe widths.

    Pool selection is the fourth layout attribute. Where a filesystem has a
    flash tier and a capacity tier, the same stripe width means very different
    throughput, and the cheaper tier may already be fast enough for a workload
    that is not I/O-bound. Pools are discovered rather than assumed, so this is
    a no-op where none exist.
    """
    inventory = target_inventory()
    active = sum(1 for entry in inventory if entry["active"])
    log(f"{len(inventory)} OSTs visible, {active} active")
    log(f"scratch inherits pool: '{inherited_pool(SCRATCH) or 'none'}'")

    pools = pool_names()
    if not pools:
        log("no OST pools on this filesystem, nothing to compare")
        return
    for pool in pools:
        log(f"pool {pool}: {len(pool_members(pool))} members")

    # Each pool at matched widths, so the comparison isolates the medium.
    # Both read and write, because a capacity tier usually costs more on writes.
    arms = [(pool, f"-p {pool}") for pool in pools]
    file_size, count = 4 << 20, 512
    min_seconds = 2.0
    widths = [1, 8]

    for pool, pool_flag in arms:
        for stripe_count in widths:
            for repeat in range(REPEATS):
                cell = f"pool_selection/{pool}/{stripe_count}/{repeat}"

                def body(pool=pool, pool_flag=pool_flag, stripe_count=stripe_count):
                    directory = fresh_dir(f"{SCRATCH}/pool_{pool}_{stripe_count}",
                                          f"-c {stripe_count} -S 1M {pool_flag}")
                    paths, wstats = write_files(directory, count, file_size, flush=False)
                    elapsed, total, passes = read_until(paths, 8, min_seconds)
                    row = {"pool": pool, "stripe_count": stripe_count,
                           "files": len(paths), "read_s": elapsed, "passes": passes,
                           "read_mib_per_s": total / elapsed / (1 << 20),
                           "write_mib_per_s": (count * file_size / (1 << 20)
                                               / wstats["write_wall_s"]),
                           "osts_per_file": osts_per_file(paths[0]),
                           "granted_pool": inherited_pool(paths[0])}
                    shutil.rmtree(directory, ignore_errors=True)
                    return row

                measured(cell, body)

    seen = rows("pool_selection/")
    log("read / write MiB/s by pool and stripe width")
    for pool, _ in arms:
        for stripe_count in widths:
            log(f"   {pool:>12} -c {stripe_count:<2} "
                f"read {median_by(seen, 'read_mib_per_s', pool=pool, stripe_count=stripe_count):7.0f}  "
                f"write {median_by(seen, 'write_mib_per_s', pool=pool, stripe_count=stripe_count):7.0f}  "
                f"objects {median_by(seen, 'osts_per_file', pool=pool, stripe_count=stripe_count):.0f}")

    # The advisor's question: is a narrow layout on the faster tier better than a
    # wide one on the slower tier? Which pool is faster is measured, not assumed
    # from the order `lfs pool_list` happens to return, because reading that
    # order as fast-first silently inverts the comparison.
    if len(arms) >= 2:
        by_speed = sorted(
            (median_by(seen, "read_mib_per_s", pool=pool, stripe_count=1), pool)
            for pool, _ in arms)
        slow_rate, slow = by_speed[0]
        fast_rate, fast = by_speed[-1]
        log(f"faster tier at -c 1: {fast} ({fast_rate:.0f} MiB/s) "
            f"vs {slow} ({slow_rate:.0f} MiB/s)")
        for stripe_count in widths:
            a = median_by(seen, "read_mib_per_s", pool=fast, stripe_count=stripe_count)
            b = median_by(seen, "read_mib_per_s", pool=slow, stripe_count=stripe_count)
            if a and b:
                log(f"   -c {stripe_count}: {fast} is {a / b:.2f}x {slow} on reads")
        narrow_fast = median_by(seen, "read_mib_per_s", pool=fast, stripe_count=1)
        wide_slow = median_by(seen, "read_mib_per_s", pool=slow, stripe_count=widths[-1])
        if narrow_fast and wide_slow:
            log(f"   1 stripe on {fast} vs {widths[-1]} on {slow}: "
                f"{narrow_fast / wide_slow:.2f}x the throughput for "
                f"1/{widths[-1]} the objects")
