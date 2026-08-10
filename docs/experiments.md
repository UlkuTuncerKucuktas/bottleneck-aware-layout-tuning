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

    ./slurm/reset.sh --dry-run           # what a fresh start would archive
    ./slurm/reset.sh                     # archive the ledger, clear debris
    ./slurm/run_campaign.sh --dry-run    # see the sbatch calls
    ./slurm/run_campaign.sh              # submit the chain, then walk away

Nothing in the chain should be pulled out and run alongside it. Every experiment
reads and writes the same 24 flash OSTs, and that includes the ones that look
cheap: `write_path` and `pool_selection` move GiB at a time, and `tail_latency`
reports p95 and p99, which is the metric a concurrent neighbour distorts first.
`neighbour_cost` is the sole exception, and it creates its own interference.

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

## What happens when a cell fails

A cell that raises is recorded with `failed: true` and its error, and the run
continues to the next one -- a twelve-hour job should not be lost to one bad
configuration. Recorded failures are not retried on resume, so a systematic
failure cannot loop; to redo one, delete its line from the ledger. An experiment
that aborts outside a cell (in its summary code, say) also lets the ones after it
run, and the job exits non-zero so a failure is visible without reading logs.

`analysis.load` drops failed rows by default and prints how many it dropped;
pass `include_failed=True` to inspect them. The in-log medians skip them too.

Working directories clean themselves up through `working_dir`, so a failed cell
does not leave GiB behind or, worse, a directory whose layout a later cell might
inherit.

Before any measurement, `_preflight` writes a plain file and a DoM file, reads
back the granted stripe width, drops the cache and flushes a DoM lock. Each check
corresponds to something that has produced wrong numbers or a late crash here, so
a broken environment fails in seconds rather than hours.

## What the MDT grants is not what you asked for

`lfs setstripe -E 1M -L mdt` requests a 1 MiB data-on-MDT component, but the
server caps it at the MDT's `dom_stripesize`, so files can get far less. An
earlier run requesting 1M cliffed between 112 and 128 KiB, which means the
boundary was set by the grant and not by the request.

Every DoM row therefore records `granted_kib`, read back from `lfs getstripe`.
Without it a flat sweep is ambiguous: it cannot be told apart from a boundary
sitting outside the swept range, and results from two runs cannot be compared at
all if the effective component differed between them.

## Flush comparisons need both arms flushed

The data-version ioctl forces data out to the servers, so an unflushed arm is
read back from the client's own cache. In an earlier run the same OST files
measured 338.9 us unflushed against 2763.8 us flushed -- an 8x difference from
the ioctl alone. `flush_cost` therefore flushes its OST arm too, and measures
what the flush buys within the DoM layout (flushed vs unflushed DoM) rather than
across layouts, which is what `tail_latency` is for.

## Exclusivity is per-experiment, not a default

`multi.sbatch` does not set `--exclusive`. The two multi-node experiments need
different things from a node, and the default was blocking the whole campaign.

`mds_scaling` reads single-threaded on each client and measures contention on the
metadata server, which is shared filesystem-wide whoever else is on the node.
Exclusivity therefore buys almost nothing while forcing an 8-node job to wait for
eight nodes to drain simultaneously -- on a partition with 50 of 55 nodes
allocated, that wait is unbounded, and Slurm gives such a job no start estimate
at all. Without it the job needs eight nodes with 16 free cores and a free GPU
each, which partially-used nodes can satisfy.

`neighbour_cost` is the opposite: it measures OST bandwidth interference between
jobs, so an uninvited co-tenant would land in the middle of the measurement.
Pass `--exclusive` on its command line, as `run_campaign.sh` does.

## Moving to another cluster

No experiment uses a GPU. The suite measures filesystem behaviour and the node
acts as a Lustre client, so a `--gres=gpu:1` request is a scheduling tax paid
only because `kolyoz-cuda` refuses jobs without one. On a CPU partition, drop it:

    sbatch -p <cpu-partition> --gres=NONE -A <account> slurm/single.sbatch <exp>

Two things must be checked before any result from a new cluster is compared with
an existing one.

**Does it mount the same filesystem?** `df /arf` and `lfs df /arf` on the new
login node. If `/arf` is absent, the results measure a different filesystem and
belong in a separate ledger, not appended to this one.

**Is the hardware the same?** Read throughput depends on the client's cores,
memory bandwidth and network. A number measured on one node type cannot be put
in the same table as one from another. Every row records `node` and `partition`
so a mixed ledger can at least be separated afterwards, but the cleaner course
is one ledger per cluster.

Walltime is a scheduling lever, not a safety margin. Multi-node cells take
minutes, so `multi.sbatch` asks for 30 of them: the backfill scheduler can slot a
short job into a gap between larger reservations, while a two-hour request waits
for a two-hour gap. On a busy partition that difference decides whether an
8-node job ever starts.

## Concurrent jobs share a scratch directory

Every job runs in the same `$LAYOUT_SCRATCH`, so anything written with a fixed
name is shared whether or not that was intended. Three things follow, and each
was a real defect before it was a rule.

Each job writes `results_<jobid>.jsonl`, and under `srun` each rank appends its
own suffix: `results_<jobid>_rank<n>.jsonl`. The job id has to lead -- a name
built from the rank alone collides the moment two multi-node jobs run at once,
since both have a rank 0 in the same directory. Appends are line-atomic so a
shared file is never corrupted, but resume would then depend on which jobs
happened to run, and one experiment's rerun could not be separated from
another's. Resume and analysis both read `results*.jsonl`, so a job still skips
cells any earlier job completed.

`multi.sbatch` removes only `barrier_$SLURM_JOB_ID_*`. A blanket `rm -f barrier_*`
at job start deletes the live markers of a concurrent job already waiting at its
rendezvous, which then blocks to its timeout and records a failure.

`rsync` runs without `--delete`, because a concurrent job may be importing from
that tree; removing modules under a live interpreter surfaces as unrelated import
errors. The cost is that a module deleted from the repo would survive in the copy
indefinitely, so `slurm/reset.sh` removes `$RUNDIR/layout_tuning` outright -- it
refuses to run while any job is queued, which is what makes that safe.

## A shared filesystem is not a shared machine

The TRUBA clusters mount the same `/arf` -- same MDS and OSS addresses, same
`lustre1.disk` and `lustre1.flash` pools -- so a run from a different login node
lands in the same scratch directory and resumes off the same ledger.

That is a hazard, not a convenience. Resume skips cells another job recorded,
which across clusters means the gaps in one curve get filled with points from
different client hardware. Throughput is a client-side quantity: cores, memory
bandwidth and the network adapter all differ between `kolyoz` and `barbun`, and
a scaling curve assembled from both looks clean while meaning nothing.

Use one scratch directory per cluster:

    export LAYOUT_SCRATCH=/arf/scratch/$USER/layout-tuning-barbun

The runner also warns at startup when an experiment it is about to run already
has rows from another partition. Filesystem-level results (what DoM costs on the
MDT, which pool is faster) are portable across clusters; throughput and latency
numbers are not.

## Why eviction drops locks, not just pages

An adversarial audit traced the loss of this suite's DoM effect to `evict()`.
Dropping pages needs an open descriptor, so a page-only evictor opens every file
immediately before the measured read. That open takes the DoM ibits lock and
leaves it cached; the measured open then finds the lock held, so the server sends
no inlined data, while the pages it would have used are gone. The result is a
cheap open and an expensive read -- DoM behaving exactly like an OST file.

The earlier standalone script that measured DoM at 3.9x had no eviction at all,
and its read function said why: "Opening a file more than once engages Lustre's
open cache and bypasses the DoM path." Adding cache control reintroduced the very
pre-open that rule forbids, and the docstring claimed the opposite.

Locks are now released through `ldlm.namespaces.*.lru_size=clear`, which opens
nothing. The page-drop path remains only as a fallback, and the runner says at
startup which mode it got -- a fallback run produces DoM numbers that read as a
null result rather than as a failure.

## Three parsing and arithmetic defects the same audit found

`osts_per_file` harvested integers from `lfs getstripe -c`, which prints the whole
layout for a composite file. The largest token in a DoM file's output is 1048576 --
the component boundary in bytes -- so every DoM arm reported a million objects per
file. It now sums `--stripe-count` across components.

`distribution` indexed percentiles with `int(N*p)`, one rank too high and
saturating: p99 was literally the maximum for every N <= 100, p95 for N <= 20. It
now uses nearest-rank and omits a percentile the sample cannot support.

The counter delta wrapped the whole cell, including `working_dir`'s cleanup, so
per-file RPC attribution was charged with a create+unlink cycle the measured phase
never performed. Cells that read through `profile_reads` now close the window
where the reads end; rows say which window they got.

## Five design defects an adversarial audit found

**The premise experiment could not fail.** Its loop reads a batch, waits, then
computes, so epoch = io + compute by construction; adding the same compute to
every arm leaves the ranking untouched, whatever the compute level. A real
loader prefetches, so the two overlap and the epoch approaches
max(io, compute) -- and past the crossover the arms become indistinguishable,
which is the claim the experiment exists to test. Both bounds are now reported.
A finite prefetch queue sits between them, so the pair brackets reality rather
than either being the truth.

**A missing arm could win.** median_by returns nan for an arm with no usable
rows, nan compares false against everything, and min() then keeps whichever arm
came first -- "minimal", the answer the paper wants. The summary now refuses to
name a winner unless every arm produced rows, and says which arm was missing.

**Re-reads were not cold.** Reaching the duration floor by reading the same
files repeatedly warms the servers' caches; evict() clears this client and
nothing else. The bias was not uniform either: a faster arm needs more passes,
collects a larger share of warm reads, and looks faster still -- inflating the
very ratios being measured. Each pass now takes a slice no pass has touched,
and the floor came down to 0.5 s because cold data is finite. Precision comes
from REPEATS, whose cells write fresh files.

**Writes were timed to memory.** A write returns once the data is in the
client's page cache, so any volume that fits in memory measured a memcpy rate
identical across layouts. write_path now fsyncs before stopping the timer, and
reports the cached rate alongside: the gap is how much of a checkpoint burst
lands before the layout matters at all.

**Arms that could not differ, and what they measure instead.** With a 1 MiB
stripe a 0.25 MiB file has its data on one OST at every width, and a 4 MiB file
on at most four -- so those arms hold placement fixed. They are kept, because
they still lose 39 % and 25 % as the requested width rises: the client fetches
and locks a layout entry per configured target on every open, whether or not it
holds anything. That is the cost of asking for width you cannot use, and it is
the regime small-file workloads live in. A 64 MiB arm was added alongside so
that placement can vary too, and the two mechanisms can be told apart.

## Running the campaign on a different cluster

The `#SBATCH` directives carry `kolyoz-cuda` defaults. Three environment
variables move the whole campaign elsewhere without editing any file:

    LAYOUT_PARTITION=barbun \
    LAYOUT_ACCOUNT=<account> \
    LAYOUT_GRES= \
    LAYOUT_SCRATCH=/arf/scratch/$USER/layout-tuning-barbun \
    ./slurm/run_campaign.sh

`LAYOUT_GRES` set but empty emits `--gres=NONE`, which is what a CPU partition
needs. Omitting the flag would not do: the sbatch scripts carry their own
`#SBATCH --gres=gpu:1`, and a directive stays in force unless the command line
overrides it, so a missing flag still requests a GPU. Leaving `LAYOUT_GRES`
unset keeps the request the cuda partitions demand, since they reject jobs
without one. No experiment uses a GPU either way.

`LAYOUT_SCRATCH` must differ per cluster. The TRUBA clusters mount the same
/arf, so a shared directory would let resume fill the gaps in one curve with
points measured on different client hardware.

Always `--dry-run` first: it prints every sbatch line and submits nothing.

## Data-on-MDT has two boundaries, not one

Confirmed against the Lustre design documentation, not inferred from our
measurements alone.

**The storage boundary is `lod.*.dom_stripesize`.** It is the per-MDT maximum
for a DoM component: default 1 MiB, 64 KiB aligned, 1 GiB ceiling, and a larger
request from `lfs setstripe -E` is silently truncated to it rather than
refused. Below the boundary a file's data lives in the MDT object; above it,
the data goes to the OST components. This is what costs metadata capacity.
Measured on /arf with `slurm/dom_probe.sh`, `-E 1M` is granted in full, so
nothing here is capped -- but the readback stays, because a truncated grant is
invisible without it and would silently move the boundary under every sweep.

**The latency boundary is the reply buffer, and it is unrelated.** DoM's speed
comes from the MDT returning attributes, lock and file data in a single RPC,
with the data carried in the reply buffer. That only happens while the data
fits the space left in that buffer. A file past that point is still on the MDT
and still charged against metadata capacity, but the client must issue a second
RPC to read it. An earlier run with a fully granted 1 MiB component showed read
time step from 47 us at 112 KiB to 1323 us at 128 KiB -- nowhere near 1 MiB,
because the component size was never what governed it.

Files between the two boundaries are inefficient rather than useless, and the
distinction matters. Measured on this filesystem, DoM at 128 KiB is still 1.75x
faster than OST; at 112 KiB it is 3.97x. So the benefit does not vanish past the
cliff, it steps down -- while the MDT capacity consumed keeps growing linearly
with the file. Speedup earned per KiB of MDT falls to about a third of its value
at the cliff, and to a sixth by 256 KiB.

Both boundaries exist for good reasons. The storage boundary is a capacity
decision: an administrator sets how much of the MDT may be spent on file data,
and 1 MiB is a deliberately generous default. The reply-buffer boundary is a
protocol constraint: the open reply is a fixed-size message shared with
attributes and lock information, and past some payload the data simply cannot
ride along. Neither is a bug, and a file in the band still works correctly and
still beats OST.

What the band costs is efficiency, which is precisely this work's subject: not
whether a resource helps, but whether it is still earning what it costs. An
advisor reasoning only about `dom_stripesize` -- the parameter documentation
foregrounds, and the only one a user can set -- would place files there believing
they get the full benefit. `dom_cutoff` brackets both: its sizes span the reply-buffer cliff,
its components span the storage boundary, and every DoM row records the granted
component so the two can be separated afterwards.

### Reading the boundaries on a client

The storage boundary lives on the server (`lod.*.dom_stripesize`) and is not
readable from a client. Read it indirectly: create a DoM file and check what
`lfs getstripe` reports for the first component's `lcme_extent.e_end`. A value
below what `-E` asked for means the MDT truncated the request.

The client-side parameter is `mdc.*.mdc_dom_min_repsize` -- the `mdc_` prefix is
part of the leaf name, so a glob on `mdc.*.dom_min_repsize` finds nothing even
where the parameter exists. It is the minimum reply size the client requests,
not the ceiling on inlined data, so it does not by itself give the latency
boundary. That boundary is where the payload stops fitting the open reply
buffer, and nothing exposes it directly -- `dom_cutoff` finds it by measurement,
which is the only way to see it.

### The client grows its reply buffer, so early reads may differ

Measured on /arf: `mdc.*.mdc_dom_min_repsize` is 8192 on every MDT, while the
read cliff sits near 112-128 KiB. Those are consistent only because the value is
a floor, not a ceiling -- the client asks for at least 8 KiB and enlarges the
request as it observes the sizes actually being read.

That makes the first reads after a fresh mount or a new size potentially
un-inlined while later ones are inlined. `dom_cutoff` reads 300 files per cell
and reports p50, so a handful of un-inlined leading reads cannot move the
headline number; it would visibly shift a mean, which is one more reason the
suite reports percentiles. Anything reading a much smaller set should treat its
first few files as warmup rather than measurement.

## Jobs must run under the scratch filesystem

Some sites reject a job whose working directory is outside scratch:

    sbatch: error: Lutfen islerinizi /arf/scratch/ dizini altinda calistiriniz!
    sbatch: error: Batch job submission failed: Job violates accounting/QOS policy

The workdir defaults to wherever `sbatch` was invoked, and this repository
normally lives in $HOME. `run_campaign.sh` therefore passes
`--chdir=$LAYOUT_SCRATCH`, which is under /arf/scratch by construction.

Two consequences. `SLURM_SUBMIT_DIR` still points at the repository -- `--chdir`
does not change it -- so the scripts continue to find the module tree they copy.
And `--output` is relative to the workdir, so **logs now land in
`$LAYOUT_SCRATCH/logs/`**, not beside the repository. The script prints the path
when it submits.

Submitting by hand needs the same flag:

    sbatch --chdir=$LAYOUT_SCRATCH -p <partition> -A <account> \
        slurm/single.sbatch <experiment>

`run_campaign.sh` runs under `set -e`, so a rejected submit stops the chain
rather than leaving a half-submitted campaign behind.

## Core counts are site-specific too

Partitions enforce their own core quantum, and reject anything else:

    sbatch: error: barbun-cuda kuyruguna gonderilen islerin cekirdek sayisi
                   20 ve katlari olmalidir

kolyoz-cuda wants multiples of 16, barbun-cuda multiples of 20. The sweeps need
"enough" cores rather than an exact number, so both counts are settable:

    LAYOUT_CORES=20 LAYOUT_CORES_WIDE=40 ./slurm/run_campaign.sh

Where a GPU ratio is also enforced, `LAYOUT_GRES_WIDE` has to match the wide
count -- one GPU per 16 cores on kolyoz, and whatever barbun-cuda requires for
40. If the ratio is rejected, the error names it.

`cores_vs_throughput` sweeps powers of two up to the allocation and then the
allocation itself, so a 20-core job measures 1/2/4/8/16/20 rather than stopping
at 16 and never testing what it was actually given.

## Site rules on TRUBA, collected

Each of these appeared only as a submit rejection, never in advance:

    kolyoz-cuda   jobs must request a GPU
    kolyoz-cuda   cores must be a multiple of 16
    kolyoz-cuda   one GPU per 16 cores
    barbun-cuda   cores must be a multiple of 20
    all           the job's working directory must be under /arf/scratch
