"""Reading Lustre's own client counters, and timing a read phase by phase."""

import os, time, glob

from .layout import CHUNK

# ---------------------------------------------------------------- client counters
# Lustre exports per-client operation and RPC counts under /proc/fs/lustre.
# Diffing them across a measured phase says which calls the time went into,
# rather than only how long the phase took.

COUNTER_GLOBS = [
    ("llite", "/proc/fs/lustre/llite/*/stats"),     # VFS ops the application made
    ("mdc",   "/proc/fs/lustre/mdc/*/stats"),       # RPCs to the metadata server
    ("osc",   "/proc/fs/lustre/osc/*/stats"),       # RPCs to the object servers
]
INTERESTING = {
    "open", "close", "getattr", "setattr", "readdir", "statfs", "inode_permission",
    "read_bytes", "write_bytes", "ost_read", "ost_write", "ost_setattr",
    "mds_getattr", "mds_getattr_lock", "mds_close", "mds_readpage", "mds_statfs",
    "ldlm_ibits_enqueue", "ldlm_extent_enqueue", "ldlm_cancel", "req_waittime",
}


def read_counters():
    """Current value of every counter we care about, flattened to one dict."""
    import glob
    out = {}
    for family, pattern in COUNTER_GLOBS:
        for path in glob.glob(pattern):
            try:
                text = open(path).read()
            except OSError:
                continue
            for line in text.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0] in INTERESTING:
                    key = f"{family}.{parts[0]}"
                    out[key] = out.get(key, 0) + int(parts[1])
    return out


def counter_delta(before, after, per=1):
    """What changed, normalised per file (or per whatever `per` counts)."""
    delta = {}
    for key in set(before) | set(after):
        change = after.get(key, 0) - before.get(key, 0)
        if change:
            delta[f"n_{key}_per_unit"] = change / per
    return delta


def lock_unused_count():
    """Client-side DLM locks held but not in use, from their own experiment 4."""
    import glob
    total = 0
    for path in glob.glob("/proc/fs/lustre/ldlm/namespaces/*/lock_unused_count"):
        try:
            total += int(open(path).read().strip())
        except (OSError, ValueError):
            pass
    return total


# ---------------------------------------------------------------- phase timing
def read_file_phases(path, dirfd=None, with_stat=False):
    """Time each syscall in a whole-file read separately.

    with_stat is off by default and must stay off in any DoM measurement:
    a stat() before open takes MDS_INODELOCK_DOM in read mode and suppresses
    the inlined-data path, so it changes the thing being measured. Turn it on
    only in a cell that is explicitly about lookup cost.
    """
    phases = {}
    if with_stat:
        t = time.perf_counter()
        os.stat(path) if dirfd is None else os.stat(path, dir_fd=dirfd)
        phases["stat_us"] = (time.perf_counter() - t) * 1e6

    t0 = time.perf_counter()
    fd = os.open(path, os.O_RDONLY) if dirfd is None else os.open(path, os.O_RDONLY, dir_fd=dirfd)
    t1 = time.perf_counter()

    first = os.read(fd, CHUNK)
    t2 = time.perf_counter()

    total = len(first)
    while True:
        chunk = os.read(fd, CHUNK)
        if not chunk:
            break
        total += len(chunk)
    t3 = time.perf_counter()

    os.close(fd)
    t4 = time.perf_counter()

    phases.update({
        "open_us": (t1 - t0) * 1e6,
        "first_read_us": (t2 - t1) * 1e6,
        "rest_read_us": (t3 - t2) * 1e6,
        "close_us": (t4 - t3) * 1e6,
        "total_us": (t4 - t0) * 1e6,
        "bytes": total,
    })
    return phases


def distribution(values):
    """Percentiles, not just a mean: a dataloader waits on its slow reads."""
    if not values:
        return {}
    ordered = sorted(values)
    def at(p):
        return ordered[min(len(ordered) - 1, int(len(ordered) * p))]
    return {"mean": sum(ordered) / len(ordered), "p50": at(0.50), "p90": at(0.90),
            "p95": at(0.95), "p99": at(0.99), "max": ordered[-1], "min": ordered[0],
            "n": len(ordered)}


def profile_reads(paths, with_stat=False):
    """Read every file once, single threaded, keeping every phase separately."""
    dirfd = os.open(os.path.dirname(paths[0]), os.O_RDONLY)
    collected = {}
    try:
        for path in paths:
            phases = read_file_phases(os.path.basename(path), dirfd=dirfd, with_stat=with_stat)
            for name, value in phases.items():
                if name != "bytes":
                    collected.setdefault(name, []).append(value)
    finally:
        os.close(dirfd)

    summary = {}
    for name, values in collected.items():
        for stat_name, stat_value in distribution(values).items():
            summary[f"{name}_{stat_name}"] = stat_value
    # what share of the time each phase accounts for
    means = {n: sum(v) / len(v) for n, v in collected.items() if n != "total_us"}
    whole = sum(means.values())
    for name, value in means.items():
        summary[f"share_{name}"] = value / whole if whole else 0.0
    return summary


