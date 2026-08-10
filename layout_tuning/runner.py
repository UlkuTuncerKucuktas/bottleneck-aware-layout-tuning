"""Running experiments: the registry, the per-cell wrapper, and the entry point.

Each experiment is a function registered by name. A cell is one measured
configuration; finished cells are recorded so an interrupted run resumes.
"""

import os, sys, time, statistics, traceback

from .ledger import Ledger
from .probes import read_counters, counter_delta, lock_unused_count

REPEATS = int(os.environ.get("LAYOUT_REPEATS", "3"))

_registry = {}
_state = {"ledger": None}


def experiment(name, nodes=1):
    """Register a function as a named experiment. nodes>1 means it needs srun."""
    def wrap(fn):
        _registry[name] = {"fn": fn, "nodes": nodes, "doc": (fn.__doc__ or "").strip()}
        return fn
    return wrap


def registry():
    return dict(_registry)


def ledger():
    if _state["ledger"] is None:
        raise RuntimeError("no ledger: call configure() or run through main()")
    return _state["ledger"]


def log(message):
    ledger().log(message)


def configure(suffix=""):
    _state["ledger"] = Ledger(path=f"results{suffix}.jsonl", log_path=f"run{suffix}.log")
    return _state["ledger"]


def measured(cell, body):
    """Run one cell unless already recorded, wrapping it in counter deltas.

    `body` returns the row of measurements. The counter delta around it says
    which client operations the time went into.

    A cell that raises is recorded as failed and the run continues. One bad
    configuration should not cost the other hundred cells in the job, and a
    recorded failure is not retried on resume unless the row is deleted -- so a
    systematic failure does not loop, while a transient one can be redone by
    removing its line from the ledger.
    """
    book = ledger()
    if book.done(cell):
        return
    book.log(f"start {cell}")
    started = time.perf_counter()
    before = read_counters()
    locks_before = lock_unused_count()
    try:
        row = body()
    except Exception as exc:
        elapsed = time.perf_counter() - started
        book.append(cell, {"failed": True, "error": f"{type(exc).__name__}: {exc}",
                           "cell_wall_s": elapsed})
        book.log(f"FAILED {cell} after {elapsed:.1f}s: {type(exc).__name__}: {exc}")
        return
    row.update(counter_delta(before, read_counters(), per=row.get("files", 1)))
    row["lock_unused_delta"] = lock_unused_count() - locks_before
    row["cell_wall_s"] = time.perf_counter() - started
    book.append(cell, row)
    book.log(f"done  {cell} in {row['cell_wall_s']:.1f}s")


def median_by(rows, key, **match):
    """Median of `key` over rows matching every keyword, ignoring failed cells.

    A failed cell carries no measurements, so including it would either raise or
    silently shift a median depending on the key.
    """
    values = [r[key] for r in rows
              if not r.get("failed") and key in r
              and all(r.get(k) == v for k, v in match.items())]
    return statistics.median(values) if values else float("nan")


def rows(prefix):
    return ledger().rows(prefix)


def _preflight():
    """Fail in seconds rather than hours if the environment cannot measure.

    Every check here corresponds to something that produced wrong numbers or a
    late crash in an earlier run: a layout that cannot be applied, a cache that
    cannot be dropped, a DoM flush the kernel rejects.
    """
    import shutil as _shutil

    from .layout import SCRATCH, OST_PLAIN, dom_layout, working_dir, evict
    from .io import write_files, osts_per_file

    if _shutil.which("lfs") is None:
        log("preflight skipped: no lfs on PATH, so this is not a Lustre client")
        return

    with working_dir(f"{SCRATCH}/preflight", OST_PLAIN) as directory:
        paths, _ = write_files(directory, 2, 1 << 20, flush=False)
        width = osts_per_file(paths[0])
        assert width >= 1, f"lfs getstripe reported {width} objects for a plain file"
        evict(paths)

    with working_dir(f"{SCRATCH}/preflight_dom", dom_layout()) as directory:
        paths, _ = write_files(directory, 2, 8 << 10, flush=True)
        log(f"preflight ok: plain file on {width} OST(s), DoM layout applied and flushed")


def main(argv=None):
    from . import experiments  # registers everything on import

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "list"):
        print("usage: python -m layout_tuning.run <experiment> [more ...]   (or 'all')\n")
        for name, entry in sorted(_registry.items()):
            scale = "multi-node" if entry["nodes"] > 1 else "single node"
            first_line = entry["doc"].splitlines()[0] if entry["doc"] else ""
            print(f"  {name:<20} {scale:<12} {first_line}")
        return 0

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    nodes = int(os.environ.get("SLURM_NTASKS", "1"))

    names = sorted(n for n, e in _registry.items() if e["nodes"] == 1) \
        if argv == ["all"] else argv
    unknown = [n for n in names if n not in _registry]
    if unknown:
        print(f"unknown experiment(s): {unknown}; try 'list'", file=sys.stderr)
        return 2

    multi = any(_registry[n]["nodes"] > 1 for n in names)
    configure(f"_rank{rank}" if multi else "")
    log(f"=== {' '.join(names)} | job {os.environ.get('SLURM_JOB_ID', 'local')} "
        f"| node {os.environ.get('SLURMD_NODENAME', 'local')} | rank {rank}/{nodes} ===")
    try:
        from .io import target_counts, pool_names, pool_members, inherited_pool
        from .layout import SCRATCH
        mdts, osts = target_counts()
        log(f"filesystem has {mdts} MDTs and {osts} OSTs")
        pools = pool_names()
        if pools:
            sizes = ", ".join(f"{name}={len(pool_members(name))}" for name in pools)
            log(f"pools: {sizes}")
        # Which pool the measurements land in bounds what -c -1 can grant and
        # what medium the OST arms actually used, so it belongs in every log.
        current = inherited_pool(SCRATCH)
        log(f"measuring in pool: '{current or 'filesystem default'}'")
    except Exception as exc:
        log(f"could not read target counts: {exc}")

    if not read_counters():
        log("note: /proc/fs/lustre counters are unreadable here, so rows carry "
            "phase timings but no per-call attribution")

    _preflight()

    failures = []
    for name in names:
        entry = _registry[name]
        log(f"===== {name} =====")
        started = time.perf_counter()
        try:
            entry["fn"](rank=rank, nodes=nodes) if entry["nodes"] > 1 else entry["fn"]()
        except Exception as exc:
            # Summary code outside a cell can fail on its own; the cells it was
            # summarising are already in the ledger, and the next experiment in
            # this job is independent of this one.
            failures.append(name)
            log(f"!!!!! {name} aborted: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        log(f"----- {name} took {(time.perf_counter() - started) / 60:.1f} min")

    failed_cells = sum(1 for r in ledger().rows("") if r.get("failed"))
    if failed_cells:
        log(f"{failed_cells} cell(s) recorded as failed; "
            "grep FAILED in the log, or 'failed' in the ledger")
    if failures:
        log(f"experiments that aborted: {', '.join(failures)}")
        return 1
    log("all requested work complete")
    return 0
