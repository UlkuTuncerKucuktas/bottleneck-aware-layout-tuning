# Bottleneck-Aware, Resource-Minimal Layout Tuning

Measurement suite for *Bottleneck-Aware, Resource-Minimal Layout Tuning for
Shared Parallel Filesystems*.

A layout decision — Data-on-MDT residency, stripe count, stripe size, OST pool —
is usually made once for a whole namespace and left alone. These experiments
measure both sides of that decision: the speed a layout delivers, and the share
of a shared filesystem it occupies to deliver it. The aim is to find the
smallest layout that still meets an application's demand, rather than the
fastest one available.

## Running an experiment

```bash
python -m layout_tuning.run list                  # every experiment, one line each
python -m layout_tuning.run bottleneck_budget     # one
python -m layout_tuning.run all                   # every single-node experiment
```

On a cluster:

```bash
sbatch slurm/single.sbatch bottleneck_budget
sbatch -N 4 slurm/multi.sbatch mds_scaling
```

Or submit every experiment as a dependency chain and leave it:

```bash
./slurm/run_campaign.sh
```

`stripe_grid` sweeps up to 32 reader threads, so it wants a bigger allocation
than the default 16 cores:

```bash
sbatch --cpus-per-task=32 --gres=gpu:2 slurm/single.sbatch stripe_grid
```

See `docs/experiments.md` for the partition's submit-filter rules.

Results append to `results.jsonl` and a timestamped log to `run.log`. Both are
resumable: **re-running the same command skips cells already recorded**, so a
job that hits its time limit continues where it stopped rather than starting
over. Multi-node runs write one ledger per rank.

## The experiments

| name | question |
| --- | --- |
| `dom_cutoff` | At what file size does Data-on-MDT stop inlining, and does it move with the `-E` component? |
| `dom_footprint` | What does DoM consume on the metadata target, and what does it do to metadata op rates? |
| `flush_cost` | DoM's read speedup needs a per-file flush at write time. How many reads pay it back? |
| `tail_latency` | A dataloader stalls on its slowest reads. What do p95 and p99 look like, not just the mean? |
| `repeated_epochs` | Training reads a dataset many times. Does the advantage survive re-reads? |
| `stripe_grid` | Stripe count against reader threads, for whole-file and shared-file access. |
| `write_path` | Checkpoints are writes. Where does wide striping earn its OSTs rather than merely hold them? |
| `mixed_classes` | One dataset, two file classes. Does per-class layout beat any single uniform choice? |
| `bottleneck_budget` | When compute dominates, does a wider layout buy anything at all? |
| `mds_scaling` | DoM reads are almost entirely metadata work, and metadata has one server. What happens as clients are added? |
| `neighbour_cost` | What does an over-provisioned neighbour cost the job scheduled beside it? |

`docs/experiments.md` says what each one measures and why it is in the set.

## What a result row contains

Each row is one measured configuration, and records more than a single number:

- **phase timings** — `open`, first `read`, remaining `read`s, `close`, each with
  p50/p90/p95/p99/max, plus a `share_*` field for the fraction of time it took.
  Separating the first read from the rest is what isolates the round trip.
- **write phases** — `create`, `write`, `flush`, `close` separately, so the cost
  of the DoM flush is distinguishable from the cost of writing.
- **client counter deltas** — `n_mdc.*`, `n_osc.*`, `n_llite.*` per file, read
  from `/proc/fs/lustre`. These say which operations the time went into, turning
  "faster" into a claim about mechanism.
- **footprint** — OST objects held, metadata-target bytes consumed, DLM locks.

## Layout

```
layout_tuning/
  layout.py         layout definitions and the lfs commands that apply them
  io.py             reading and writing files, and filesystem facts
  probes.py         Lustre client counters, phase timing, distributions
  ledger.py         append-only results with resume
  runner.py         registry, per-cell wrapper, entry point
  experiments/
    dom.py          Data-on-MDT residency
    stripe.py       stripe geometry and concurrency
    budget.py       bottleneck-derived budgets
    multinode.py    experiments needing several clients
analysis/load.py    load ledgers into a DataFrame
slurm/              submit scripts
docs/               what each experiment measures, and cluster notes
```

## Adding an experiment

Write a function in the module its subject belongs to, decorate it, and it
appears in `list` automatically:

```python
@experiment("example_probe")
def example_probe():
    """One line saying what question this answers."""
    for pool in ["flash", "capacity"]:
        cell = f"example_probe/{pool}"

        def body(pool=pool):
            directory = fresh_dir(f"{SCRATCH}/pool_{pool}", f"-p {pool} -c 1")
            paths, _ = write_files(directory, 4000, 64 << 10)
            evict(paths)
            row = {"pool": pool, "files": len(paths), **profile_reads(paths)}
            shutil.rmtree(directory, ignore_errors=True)
            return row

        measured(cell, body)
```

Three rules make it behave like the rest:

1. Give every cell a unique key. `measured()` skips keys already in the ledger,
   which is what makes a run resumable — a key that varies between runs breaks it.
2. Bind loop variables as default arguments (`def body(pool=pool)`). Python
   closures capture the variable, not its value, so without this every cell
   measures the last one.
3. Return `files` in the row. Counter deltas are normalised per file.

## Checks

```bash
tests/run_all.sh
```

Three things, none of which need Lustre. `tests/smoke.py` runs every experiment
against a temporary directory, replacing only the calls that genuinely require
Lustre — `lfs setstripe`, `lfs df`, `lfs getstripe`, and the DoM flush ioctl —
so a missing import or a typo in a rarely-taken branch fails locally rather than
on the cluster. It also asserts that a second run adds no cells, which is the
resume guarantee. `tests/readme_example.py` extracts the snippet above out of
this file and executes it, so the documentation cannot drift from working code.

Cache eviction is deliberately *not* replaced in the smoke run: it is a real
code path, it has broken before, and stubbing it is what hid the breakage.

## Requirements

Python 3.8+ and a Lustre client. `pandas` for the analysis helpers only.
Nothing needs root: every experiment uses `lfs setstripe` on directories the
user owns. Counter reading degrades gracefully — if `/proc/fs/lustre` is not
readable, timings still work and the per-call attribution is simply absent.
