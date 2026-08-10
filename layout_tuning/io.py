"""Reading and writing files, and querying what the filesystem says about them."""

import os, time
import threading
from concurrent.futures import ThreadPoolExecutor

from .layout import CHUNK, SCRATCH, capture, flush_dom_lock, run
from .probes import distribution

def write_files(directory, count, size_bytes, flush=True, profile=False,
                durable=False):
    """Create `count` files, optionally timing each write phase separately.

    Returns (paths, stats). stats always carries creates_per_s; with
    profile=True it also carries the per-phase distributions, so the cost of
    the DoM flush can be separated from the cost of the write itself.

    durable=True fsyncs each file before its timer stops. Without it a write
    returns once the data is in the client's page cache, so for any volume that
    fits in memory the number is a memcpy rate -- identical whatever the layout,
    because no layout has been exercised yet. Anything drawing a conclusion
    about write behaviour needs durable=True; callers that only need files on
    disk to read back should leave it off and not pay the cost.

    Both are recorded: write_wall_s stops before the fsync, write_durable_s
    after it. The gap is how much a client can absorb before the layout starts
    to matter, which is itself the answer to a checkpoint-sizing question.
    """
    payload = os.urandom(min(size_bytes, CHUNK))
    paths = []
    sync_seconds = 0.0
    phases = {"create_us": [], "write_us": [], "flush_us": [],
              "fsync_us": [], "close_us": []}
    started = time.perf_counter()
    for i in range(count):
        path = os.path.join(directory, f"f{i:06d}")
        t0 = time.perf_counter()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        t1 = time.perf_counter()
        written = 0
        while written < size_bytes:
            n = min(CHUNK, size_bytes - written)
            # os.write may accept fewer bytes than offered; counting the request
            # rather than the result would leave files short of size_bytes and
            # quietly change what is being measured.
            written += os.write(fd, payload[:n])
        t2 = time.perf_counter()
        if flush:
            flush_dom_lock(fd)
        t3 = time.perf_counter()
        if durable:
            os.fsync(fd)
        t_sync = time.perf_counter()
        os.close(fd)
        t4 = time.perf_counter()
        sync_seconds += t_sync - t3
        paths.append(path)
        if profile:
            phases["create_us"].append((t1 - t0) * 1e6)
            phases["write_us"].append((t2 - t1) * 1e6)
            phases["flush_us"].append((t3 - t2) * 1e6)
            phases["close_us"].append((t4 - t_sync) * 1e6)
            phases["fsync_us"].append((t_sync - t3) * 1e6)
    elapsed = time.perf_counter() - started
    run("sync")

    stats = {"creates_per_s": count / elapsed,
             "write_wall_s": elapsed - sync_seconds,
             "write_durable_s": elapsed,
             "write_durable": durable}
    if profile:
        for name, values in phases.items():
            for stat_name, stat_value in distribution(values).items():
                stats[f"w_{name}_{stat_name}"] = stat_value
    return paths, stats


def read_whole(path):
    total = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            total += len(chunk)
    return total



def prewarm(pool, threads):
    """Force every worker thread to exist before the caller starts timing.

    ThreadPoolExecutor spawns lazily, so a timer started around `pool.map` also
    measures thread creation. At a few hundred MiB per pass that startup cost is
    a large share of the measurement, and it grows with the thread count -- which
    is exactly the axis being swept, so it masquerades as a concurrency effect.
    A barrier makes all workers exist first: each task blocks until every worker
    has arrived, so the pool is fully populated when the barrier releases.
    """
    barrier = threading.Barrier(threads + 1)
    for _ in range(threads):
        pool.submit(barrier.wait)
    # A timeout rather than an unbounded wait: a pool that cannot start every
    # worker should fail the cell, not hold the allocation until the wall clock.
    barrier.wait(timeout=60)


def read_many(paths, threads, pool=None):
    if pool is not None:
        started = time.perf_counter()
        total = sum(pool.map(read_whole, paths))
        return time.perf_counter() - started, total
    with ThreadPoolExecutor(max_workers=threads) as own:
        prewarm(own, threads)
        started = time.perf_counter()
        total = sum(own.map(read_whole, paths))
        return time.perf_counter() - started, total


def read_until(paths, threads, min_seconds, reader=None):
    """Read `paths` until the timed region reaches `min_seconds`, cold throughout.

    Each pass takes a DIFFERENT slice of the file set, so no byte is read twice
    and every pass is genuinely cold. Re-reading the same files instead would
    warm the servers' own caches -- evict() clears this client, and nothing
    more -- and the bias would not be uniform: a faster arm needs more passes to
    fill the same seconds, so it collects a larger share of warm reads and looks
    faster still. That inflates exactly the ratios the sweep exists to measure.

    When the set is exhausted before the floor is reached, the read stops rather
    than wrapping around. A short cell is visible in `passes` and in the elapsed
    time; a wrapped one would silently be part warm.

    The thread pool is built once and reused, since creating it costs time that
    scales with the thread count and would otherwise masquerade as a
    concurrency effect.
    """
    from .layout import evict

    reader = reader or read_many
    total_bytes, elapsed, passes = 0, 0.0, 0
    offset = 0
    # Enough files per pass to keep every thread busy, and at least one.
    per_pass = max(threads, len(paths) // 8) or 1
    with ThreadPoolExecutor(max_workers=threads) as pool:
        prewarm(pool, threads)
        while elapsed < min_seconds and offset < len(paths) and passes < 50:
            batch = paths[offset:offset + per_pass]
            offset += per_pass
            evict(batch)
            took, got = reader(batch, threads, pool=pool)
            elapsed += took
            total_bytes += got
            passes += 1
    return elapsed, total_bytes, passes


def ranged_reader(path, size_bytes):
    """A read_until-compatible reader for disjoint ranges of one file.

    Returned as a callable that accepts an existing pool, so the shared-file arm
    gets the same one-pool-per-cell treatment as the whole-file arm rather than
    paying thread creation on every pass.
    """
    def read(paths, threads, pool=None):
        fd = os.open(path, os.O_RDONLY)
        span = size_bytes // threads
        bounds = [(k * span, size_bytes if k == threads - 1 else (k + 1) * span)
                  for k in range(threads)]

        def worker(bound):
            start, end = bound
            got, offset = 0, start
            while offset < end:
                chunk = os.pread(fd, min(CHUNK, end - offset), offset)
                if not chunk:
                    break
                offset += len(chunk)
                got += len(chunk)
            return got

        try:
            if pool is not None:
                started = time.perf_counter()
                total = sum(pool.map(worker, bounds))
                return time.perf_counter() - started, total
            with ThreadPoolExecutor(max_workers=threads) as own:
                prewarm(own, threads)
                started = time.perf_counter()
                total = sum(own.map(worker, bounds))
                return time.perf_counter() - started, total
        finally:
            os.close(fd)

    return read


def dom_component_kib(path):
    """Size of the file's data-on-MDT component in KiB, or 0 if it has none.

    The server caps this: `lfs setstripe -E 1M -L mdt` may be granted a much
    smaller component depending on the MDT's dom_stripesize, so what was asked
    for is not what a file gets. Every DoM measurement therefore has to record
    the granted size -- without it, a sweep cannot be placed relative to the
    boundary it is trying to find, and a flat result cannot be told apart from
    a boundary that sits outside the swept range.
    """
    text = capture(f"lfs getstripe {path}")
    granted = 0
    for block in text.split("lcme_id")[1:]:
        if "mdt" not in block.lower():
            continue
        for line in block.splitlines():
            if "lcme_extent.e_end" in line:
                token = line.split(":")[-1].strip()
                if token.isdigit():
                    granted = int(token) // 1024
    return granted


def mdt_used_kib():
    """Space used across *every* metadata target, not just the first.

    A filesystem usually has several MDTs and a directory's files can land on
    any of them, so reading one line undercounts by roughly the MDT count.
    """
    used = [int(line.split()[2]) for line in capture(f"lfs df {SCRATCH}").splitlines()
            if "[MDT:" in line]
    if not used:
        raise RuntimeError("no MDT line in lfs df")
    return sum(used)


def target_counts():
    text = capture(f"lfs df {SCRATCH}")
    return (sum(1 for line in text.splitlines() if "[MDT:" in line),
            sum(1 for line in text.splitlines() if "[OST:" in line))


def ost_count():
    return target_counts()[1]


def osts_per_file(path):
    """How many OST objects the file actually holds.

    Not `lfs getstripe -c`: on a composite layout that prints the whole layout,
    and harvesting integers from it picks up extent offsets. The largest token
    in a DoM file's output is 1048576 -- the component boundary in bytes -- so a
    max() over those integers reports a small inlined file as occupying a
    million objects, which then propagates into every footprint figure.

    `--stripe-count` on each instantiated component is the object count, and a
    DoM component reports 0 because its data is on the MDT. Summing across
    components therefore gives what the file costs on the OSTs, which is the
    quantity the footprint argument needs.
    """
    text = capture(f"lfs getstripe --stripe-count {path}")
    counts = []
    for token in text.split():
        if token.lstrip("-").isdigit():
            value = int(token)
            # -1 means "all available" on an uninstantiated component: no
            # objects exist yet, so it contributes nothing to the footprint.
            if value > 0:
                counts.append(value)
    return sum(counts)


def stat_rate(paths):
    """Metadata ops per second, as an MDS-load proxy."""
    started = time.perf_counter()
    for path in paths:
        os.stat(path)
    return len(paths) / (time.perf_counter() - started)


def pool_names():
    """Pools defined on this filesystem, as `lfs pool_list` reports them.

    Returns [] when the filesystem has no pools, which is the common case and
    not an error: the pool experiment skips itself rather than inventing names.
    """
    text = capture(f"lfs pool_list {SCRATCH} 2>/dev/null")
    names = []
    for line in text.splitlines()[1:]:
        token = line.strip()
        if token and not token.endswith(":"):
            names.append(token.split(".")[-1])
    return names


def pool_members(pool):
    """OSTs in `pool`, which may be given short ('flash') or qualified.

    `lfs pool_list` wants the qualified name even though `lfs setstripe -p`
    accepts the short one, so a short name is resolved against the filesystem
    first. Without that this returns nothing and the run logs every pool as
    having zero members.
    """
    if "." not in pool:
        qualified = [name for name in _qualified_pool_names()
                     if name.split(".")[-1] == pool]
        pool = qualified[0] if qualified else pool
    text = capture(f"lfs pool_list {pool} 2>/dev/null")
    return [line.strip() for line in text.splitlines()[1:] if line.strip()]


def _qualified_pool_names():
    text = capture(f"lfs pool_list {SCRATCH} 2>/dev/null")
    return [line.strip() for line in text.splitlines()[1:]
            if line.strip() and not line.strip().endswith(":")]


def inherited_pool(path):
    """The pool a new file in `path` would land in, or '' if none."""
    text = capture(f"lfs getstripe --pool {path} 2>/dev/null").strip()
    return text.split()[-1] if text else ""


def target_inventory():
    """Every OST with its index and whether it is active, from `lfs osts`."""
    entries = []
    for line in capture(f"lfs osts {SCRATCH}").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].rstrip(":").isdigit():
            entries.append({"index": int(parts[0].rstrip(":")),
                            "uuid": parts[1],
                            "active": "INACTIVE" not in line.upper()})
    return entries


class restricted_to_cores:
    """Run a block as if the job had been allocated only `count` cores.

    Reader threads that block in read() hold no core while waiting, so thread
    count and core count are different resources. Pinning the process lets one
    job measure what a smaller allocation could have achieved, which is the
    quantity a layout has to be sized against.
    """

    def __init__(self, count):
        self.count = count
        self.original = None

    def __enter__(self):
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError(
                "os.sched_setaffinity is unavailable, so core count cannot be "
                "restricted; this measurement requires Linux")
        self.original = os.sched_getaffinity(0)
        chosen = sorted(self.original)[:self.count]
        if len(chosen) < self.count:
            raise RuntimeError(
                f"asked for {self.count} cores but only {len(self.original)} are allocated")
        os.sched_setaffinity(0, set(chosen))
        return self

    def __exit__(self, *exc):
        os.sched_setaffinity(0, self.original)
        return False


def allocated_cores():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def burn_cpu(milliseconds):
    """Busy work standing in for an application's compute phase."""
    deadline = time.perf_counter() + milliseconds / 1000.0
    x = 0.0
    while time.perf_counter() < deadline:
        for _ in range(2000):
            x = x * 1.000001 + 1.0
    return x
