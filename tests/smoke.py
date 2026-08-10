"""Run every experiment against a local directory, on real code paths.

    python tests/smoke.py

Only the calls that genuinely need Lustre are replaced: `lfs setstripe`,
`lfs df`, `lfs getstripe`, and the DoM lock flush ioctl. Everything else runs
for real, including eviction, so a missing import or a typo in a rarely-taken
branch fails here instead of on the cluster.

Sizes are shrunk to a few files per cell, so the numbers mean nothing. The
point is that every line executes.
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = tempfile.mkdtemp(prefix="layout-tuning-smoke-")
os.environ["LAYOUT_SCRATCH"] = os.path.join(ROOT, "scratch")
os.environ["LAYOUT_REPEATS"] = "1"

from layout_tuning import io as io_module          # noqa: E402
from layout_tuning import layout as layout_module  # noqa: E402
from layout_tuning.runner import main, registry    # noqa: E402

REAL_WRITE = io_module.write_files


def patch(pairs):
    """Point every experiment module at the replacement, not just its source."""
    from layout_tuning import experiments
    modules = [layout_module, io_module] + [
        getattr(experiments, name) for name in ("dom", "stripe", "budget", "multinode")]
    for name, value in pairs:
        for module in modules:
            if hasattr(module, name):
                setattr(module, name, value)


def make_dir_ignoring_layout(path, layout):
    """Stand in for `lfs setstripe`: create the directory, ignore the layout."""
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return path


def shell(cmd):
    """`sync` is real; anything starting with lfs needs Lustre and is skipped."""
    if cmd.strip() == "sync" or cmd.startswith("lfs"):
        return
    raise AssertionError(f"unexpected shell command in a smoke run: {cmd}")


def small_write(directory, count, size_bytes, flush=True, profile=False):
    return REAL_WRITE(directory, min(count, 6), min(size_bytes, 8192), flush, profile)


def run():
    patch([
        ("fresh_dir", make_dir_ignoring_layout),
        ("run", shell),
        ("flush_dom_lock", lambda fd: None),
        ("mdt_used_kib", lambda: 1000),
        ("osts_per_file", lambda path: 4),
        ("write_files", small_write),
    ])

    # Core pinning is Linux-only, like eviction. Off Linux the real call raises
    # a specific error and the smoke run tolerates that one message only.
    if not hasattr(os, "sched_setaffinity"):
        print("note: no sched_setaffinity here, running cores_vs_throughput unpinned")
        import contextlib
        from layout_tuning import io as io_mod
        patch([("restricted_to_cores", lambda count: contextlib.nullcontext()),
               ("allocated_cores", lambda: 4)])
        io_mod.allocated_cores = lambda: 4

    # evict() is deliberately NOT replaced: it is a real code path, it has
    # broken before, and stubbing it is what hid that. Off Linux it raises a
    # specific error, which is the only failure tolerated here.
    if not hasattr(os, "posix_fadvise"):
        print("note: no posix_fadvise on this platform, tolerating evict()'s refusal")
        original = layout_module.evict

        def tolerated(paths):
            try:
                original(paths)
            except RuntimeError as exc:
                if "posix_fadvise" not in str(exc):
                    raise
        patch([("evict", tolerated)])

    os.chdir(ROOT)
    single = sorted(n for n, e in registry().items() if e["nodes"] == 1)
    main(single)

    os.environ["SLURM_PROCID"], os.environ["SLURM_NTASKS"] = "0", "1"
    for name in sorted(n for n, e in registry().items() if e["nodes"] > 1):
        main([name])

    # re-running must add nothing: that is what makes an interrupted job resumable
    before = sum(1 for _ in open("results.jsonl"))
    main(single)
    after = sum(1 for _ in open("results.jsonl"))
    assert before == after, f"resume added {after - before} duplicate cells"

    print(f"\n{before} single-node cells, resume added none")
    print(f"workspace: {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
