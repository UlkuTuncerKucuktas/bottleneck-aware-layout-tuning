# What each experiment measures, and why it is in the set

Every experiment here exists to support a specific claim in the paper. The
table below is the mapping, so an experiment that stops defending a claim can
be dropped, and a claim with nothing behind it is visible.

| claim | evidence needed | experiment |
| --- | --- | --- |
| C1 footprint | MDT bytes per file (DoM) | `dom_footprint` capacity sweep: 20k files per size, one delta over the whole batch |
| C1 footprint | OST objects per file | `dom_footprint` records it alongside MDT bytes |
| C1 footprint | MDS load | `dom_footprint` measures create/stat ops per second, DoM vs OST |
| C1 footprint | client RPC concurrency | `stripe_grid` sweeps threads jointly with stripe count |
| C2 bottleneck budget | extra I/O gives no end-to-end gain when compute-bound | `bottleneck_budget` synthetic loop: read batch, then spin CPU; vary compute per batch |
| C2 bottleneck budget | saturation knee (marginal gain -> 0) | `stripe_grid` reports marginal MiB/s per added stripe/thread |
| C3 DoM residency | cutoff location | `dom_cutoff` fine sweep 112-128K in 4K steps |
| C3 DoM residency | cutoff vs layout component | `dom_cutoff` repeats at -E 64K / 256K / 1M |
| C4 joint tuning | stripe x threads interaction | `stripe_grid` (same grid, folded in) |
| C5 multi-tenant | neighbour damage from over-provisioning | `neighbour_cost` two nodes: victim + aggressor at wide vs minimal layout |
| motivation | real file-size distribution | not measured; cited from the Lustre survey instead |
| evaluation | mixed file classes in one dataset | `mixed_classes` small+large dataset, uniform vs per-class layout |
| evaluation | write path | `write_path` write path: create rate and MiB/s vs stripe count |
| C1 footprint | MDS as a shared bottleneck | `mds_scaling` multi-client: DoM vs OST as client count rises |
| C2 bottleneck budget | write-side amortisation of the flush | `write_path` + `flush_cost` breakeven read count |
| evaluation | tail latency, not just mean | `tail_latency` p50/p95/p99 per arm |
| evaluation | benefit across repeated epochs | `repeated_epochs` four consecutive epochs |

## Notes that cost real time to learn

**A `stat()` before `open()` suppresses DoM inlining.** It takes the DoM inode
lock in read mode, so the data no longer arrives with the open reply. This is
why `read_file_phases` has `with_stat=False` by default: an innocent-looking
lookup changes the thing being measured.

**DoM needs an explicit flush to show any benefit.** Without releasing the
write lock, a reader gets the same path as OST and the measured speedup is
1.0x. An early version of this work concluded DoM does nothing for exactly
this reason. `write_files(..., flush=True)` handles it; `flush_cost` measures
what it costs.

**Read at least ten thousand files before trusting a metadata-target delta.**
`lfs df` reports whole-filesystem usage, so with a few hundred files the
delta is smaller than other users' activity and comes back as noise, or
negative.

**Below the stripe size, stripe count does nothing.** A file smaller than one
stripe occupies a single OST whatever the count says. `stripe_grid` sweeps
sizes on both sides of the stripe size so this shows up as a flat row rather
than a puzzle.

**`-c -1` is not necessarily every OST.** On TRUBA's `/arf` (48 OSTs) a file
created with `-c -1` came back from `lfs getstripe -c` with **24** objects, not
48. Take the per-file stripe count from `lfs getstripe`, never from the
filesystem's OST count: they are different numbers and the second one flatters
the footprint figure. `stripe_grid` now records `osts_per_file` in every row so
the requested and the granted width can be compared.

**On TRUBA, scratch is a flash pool, so every OST arm is flash.** `/arf` has
two pools, `lustre1.disk` and `lustre1.flash`, of 24 targets each, and
`/arf/scratch` inherits `flash`. Three things follow. `-c -1` grants 24 objects
rather than 48 because "all" means all-in-pool, which is correct behaviour and
not a cap. A file's footprint denominator is the 24 OSTs of its pool, so a
"fraction of the filesystem" claim has to name the pool. And every DoM-vs-OST
comparison measured DoM against flash NVMe, which is the harder test: the
speedup is a lower bound on what the disk pool would give. The runner logs the
inherited pool on every run so a result set can never be read as
pool-agnostic.

**Never measure on a login node.** Its client-side cache and lock state carry
whatever every other logged-in user is doing.

## Cluster notes (TRUBA)

The `kolyoz-cuda` partition has three submit-filter rules, each reported only
as a rejection at submit time:

- every job must request at least one GPU,
- the core count must be a multiple of 16,
- and there must be **one GPU per 16 cores**.

So 16 cores go with `--gres=gpu:1` and 32 cores with `--gres=gpu:2`. The GPUs sit
idle — these are I/O measurements — but they are the price of admission. The
submit scripts default to 16 cores and one GPU, which is enough for every
experiment except `stripe_grid`, whose thread sweep reaches 32:

    sbatch --cpus-per-task=32 --gres=gpu:2 slurm/single.sbatch stripe_grid

Run everything on Lustre under `/arf/scratch`, never on node-local storage, and
never on a login node.

## Measuring throughput on flash

A single flash OST on TRUBA serves roughly 2.9 GiB/s to one thread, so a
few hundred MiB is read in tens of milliseconds. Two things follow, and the
first run of `stripe_grid` fell into both.

Thread-pool creation must sit outside the timed region. `ThreadPoolExecutor`
spawns workers lazily, so timing around `pool.map` also times thread creation --
a cost that grows with the thread count, which is the axis being swept. It shows
up as throughput falling at high concurrency and is easily mistaken for a real
client-side ceiling. `read_many` and `ranged_reader` now populate the pool with
a barrier before starting the clock.

Re-reading does not by itself dilute a fixed per-pass cost. Averaging N passes
that each pay `c` of startup gives `N*w / (N*(c + w/r))`, which is exactly the
single-pass rate. `read_until` therefore builds one pool per cell and reuses it
across passes, so the floor lengthens one timed region rather than summing short
ones. Rows carry `read_s` and `passes` so a cell that never reached the floor is
visible.

## pool_selection writes to a shared tier

`pool_selection` is the only experiment that writes outside the pool scratch
inherits. With the defaults (512 files of 4 MiB, two widths, three repeats) each
pool arm writes 6 x 2 GiB = 12 GiB, so a two-pool filesystem sees 24 GiB in
total and half of it lands on the capacity tier other users' jobs sit on. It is
deleted as each cell finishes, but the write bandwidth is real while it runs.
Lower it with LAYOUT_REPEATS=1 for a first look.

## Running the whole campaign

    ./slurm/run_campaign.sh --dry-run    # see the sbatch calls
    ./slurm/run_campaign.sh              # submit the chain, then walk away

Each job carries `--dependency=afterany` on the one before it, so they run one
at a time and the script returns as soon as everything is queued. `afterany`
rather than `afterok` because a job that hits its time limit still leaves
finished cells in the ledger, and the next experiment does not depend on its
predecessor completing.

## Multi-node runs share the filesystem, so names must carry the run

Every rank of every job writes into the same scratch directory, so a path built
from only the arm and the rank collides between a 2-node and a 4-node run of the
same experiment. Scratch directories therefore include the node count, and the
rendezvous barrier includes the job id as well: a marker file left behind by a
killed run would otherwise satisfy a later barrier immediately, letting some
ranks read while others are still writing. That produces a plausible-looking
number with none of the concurrency the experiment exists to create, which is
worse than a crash. The barrier also gives up after ten minutes rather than
holding the allocation until the wall clock ends.
