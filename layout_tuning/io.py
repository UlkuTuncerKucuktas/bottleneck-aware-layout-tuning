"""Reading and writing files, and querying what the filesystem says about them."""

import os, time
from concurrent.futures import ThreadPoolExecutor

from .layout import CHUNK, SCRATCH, capture, flush_dom_lock, run
from .probes import distribution

def write_files(directory, count, size_bytes, flush=True, profile=False):
    """Create `count` files, optionally timing each write phase separately.

    Returns (paths, stats). stats always carries creates_per_s; with
    profile=True it also carries the per-phase distributions, so the cost of
    the DoM flush can be separated from the cost of the write itself.
    """
    payload = os.urandom(min(size_bytes, CHUNK))
    paths = []
    phases = {"create_us": [], "write_us": [], "flush_us": [], "close_us": []}
    started = time.perf_counter()
    for i in range(count):
        path = os.path.join(directory, f"f{i:06d}")
        t0 = time.perf_counter()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        t1 = time.perf_counter()
        written = 0
        while written < size_bytes:
            n = min(CHUNK, size_bytes - written)
            os.write(fd, payload[:n])
            written += n
        t2 = time.perf_counter()
        if flush:
            flush_dom_lock(fd)
        t3 = time.perf_counter()
        os.close(fd)
        t4 = time.perf_counter()
        paths.append(path)
        if profile:
            phases["create_us"].append((t1 - t0) * 1e6)
            phases["write_us"].append((t2 - t1) * 1e6)
            phases["flush_us"].append((t3 - t2) * 1e6)
            phases["close_us"].append((t4 - t3) * 1e6)
    elapsed = time.perf_counter() - started
    run("sync")

    stats = {"creates_per_s": count / elapsed, "write_wall_s": elapsed}
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



def read_many(paths, threads):
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        total = sum(pool.map(read_whole, paths))
    return time.perf_counter() - started, total


def read_ranges(path, size_bytes, threads):
    """All threads read disjoint ranges of one file (the checkpoint shape)."""
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
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as pool:
            total = sum(pool.map(worker, bounds))
        elapsed = time.perf_counter() - started
    finally:
        os.close(fd)
    return elapsed, total


def mdt_used_kib():
    for line in capture(f"lfs df {SCRATCH}").splitlines():
        if "[MDT:" in line:
            return int(line.split()[2])
    raise RuntimeError("no MDT line in lfs df")


def ost_count():
    return sum(1 for line in capture(f"lfs df {SCRATCH}").splitlines() if "[OST:" in line)


def osts_per_file(path):
    return int(capture(f"lfs getstripe -c {path}").strip().splitlines()[-1])


def stat_rate(paths):
    """Metadata ops per second, as an MDS-load proxy."""
    started = time.perf_counter()
    for path in paths:
        os.stat(path)
    return len(paths) / (time.perf_counter() - started)


def burn_cpu(milliseconds):
    """Busy work standing in for an application's compute phase."""
    deadline = time.perf_counter() + milliseconds / 1000.0
    x = 0.0
    while time.perf_counter() < deadline:
        for _ in range(2000):
            x = x * 1.000001 + 1.0
    return x
