"""Experiments that need more than one client node."""

import os, time, shutil

from ..layout import (SCRATCH, OST_PLAIN, OST_WIDE, dom_layout, fresh_dir,
                       working_dir, evict)
from ..io import write_files, read_many, read_until, osts_per_file
from ..probes import profile_reads
from ..runner import experiment, measured, median_by, rows, log


def _rendezvous(tag, rank, nodes, timeout=600):
    """Wait until every client has reached this point, using the shared filesystem.

    The tag carries the node count and the job id: markers left behind by an
    earlier or killed run would otherwise satisfy a later barrier immediately,
    letting some ranks read while others are still writing -- which silently
    destroys the concurrency the experiment exists to create.
    """
    job = os.environ.get("SLURM_JOB_ID", "local")
    marker = f"{SCRATCH}/barrier_{job}_{nodes}_{tag}"
    open(f"{marker}_{rank}", "w").close()
    deadline = time.time() + timeout
    while sum(1 for n in range(nodes) if os.path.exists(f"{marker}_{n}")) < nodes:
        if time.time() > deadline:
            raise RuntimeError(
                f"rank {rank} waited {timeout}s at barrier {tag} for {nodes} clients; "
                "a peer never arrived, so this measurement would not be concurrent")
        time.sleep(0.3)


@experiment("mds_scaling", nodes=2)
def mds_scaling(rank, nodes):
    """DoM reads are almost all metadata work, and metadata has one server.

    Every client reads its own files at once. DoM's single-client advantage
    meets the fact that those operations all land on one metadata server,
    while OST reads spread across many object servers.
    """
    for arm, layout in [("ost", OST_PLAIN), ("dom", dom_layout())]:
        cell = f"mds_scaling/{nodes}/{arm}/{rank}"

        def body(arm=arm, layout=layout):
            with working_dir(f"{SCRATCH}/mds_{arm}_{nodes}_{rank}", layout) as directory:
                paths, _ = write_files(directory, 6000, 32 << 10)
                evict(paths)
                _rendezvous(arm, rank, nodes)
                row = {"arm": arm, "clients": nodes, "rank": rank,
                       "files": len(paths), **profile_reads(paths)}
                row["files_per_s"] = 1e6 / row["total_us_mean"]
            return row

        measured(cell, body)
        seen = rows(f"mds_scaling/{nodes}/{arm}/")
        log(f"mds_scaling rank {rank} {arm:>4} at {nodes} clients: "
            f"{median_by(seen, 'files_per_s', arm=arm):.0f} files/s, "
            f"open p50 {median_by(seen, 'open_us_p50', arm=arm):.0f} us")


@experiment("neighbour_cost", nodes=2)
def neighbour_cost(rank, nodes):
    """What an over-provisioned neighbour costs the job beside it.

    Rank 0 is the victim on a minimal layout and reports its own throughput.
    The other ranks read the same volume, once on a minimal layout and once
    on a wide one, so the difference is what the surplus cost the victim.
    """
    # 16 MiB files, so the neighbour's wide layout genuinely spreads across OSTs.
    # With small files every arm lands on one OST and there is nothing to measure.
    file_size, file_count = 16 << 20, 128
    min_seconds = 3.0
    for label, neighbour_layout in [("minimal", OST_PLAIN), ("wide", OST_WIDE)]:
        cell = f"neighbour_cost/{label}/{rank}"

        def body(label=label, neighbour_layout=neighbour_layout):
            layout = OST_PLAIN if rank == 0 else neighbour_layout
            with working_dir(f"{SCRATCH}/neigh_{label}_{nodes}_{rank}", layout) as directory:
                paths, _ = write_files(directory, file_count, file_size, flush=False)
                evict(paths)
                _rendezvous(label, rank, nodes)
                elapsed, total, passes = read_until(paths, 8, min_seconds)
                row = {"neighbour_layout": label, "rank": rank, "files": len(paths),
                       "role": "victim" if rank == 0 else "neighbour",
                       "read_s": elapsed, "passes": passes,
                       "mib_per_s": total / elapsed / (1 << 20),
                       "osts_per_file": osts_per_file(paths[0]),
                       "ost_objects": len(paths) * osts_per_file(paths[0])}
            return row

        measured(cell, body)
    if rank == 0:
        seen = rows("neighbour_cost/")
        quiet = median_by(seen, "mib_per_s", neighbour_layout="minimal", rank=0)
        for label in ["minimal", "wide"]:
            got = median_by(seen, "mib_per_s", neighbour_layout=label, rank=0)
            log(f"neighbour_cost neighbour={label:>7}: victim {got:.0f} MiB/s"
                + ("" if label == "minimal" else f"  ({got / quiet:.2f}x of the minimal case)"))
