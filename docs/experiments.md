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
