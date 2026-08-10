"""Running experiments: the registry, the per-cell wrapper, and the entry point.

Each experiment is a function registered by name. A cell is one measured
configuration; finished cells are recorded so an interrupted run resumes.
"""

import os, sys, time, statistics

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
    """
    book = ledger()
    if book.done(cell):
        return
    book.log(f"start {cell}")
    started = time.perf_counter()
    before = read_counters()
    locks_before = lock_unused_count()
    row = body()
    row.update(counter_delta(before, read_counters(), per=row.get("files", 1)))
    row["lock_unused_delta"] = lock_unused_count() - locks_before
    row["cell_wall_s"] = time.perf_counter() - started
    book.append(cell, row)
    book.log(f"done  {cell} in {row['cell_wall_s']:.1f}s")


def median_by(rows, key, **match):
    values = [r[key] for r in rows if key in r and all(r.get(k) == v for k, v in match.items())]
    return statistics.median(values) if values else float("nan")


def rows(prefix):
    return ledger().rows(prefix)


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

    for name in names:
        entry = _registry[name]
        log(f"===== {name} =====")
        started = time.perf_counter()
        entry["fn"](rank=rank, nodes=nodes) if entry["nodes"] > 1 else entry["fn"]()
        log(f"----- {name} took {(time.perf_counter() - started) / 60:.1f} min")

    log("all requested work complete")
    return 0
