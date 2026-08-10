"""Append-only results with resume, so an interrupted run continues."""

import glob, os, json, platform, time

class Ledger:
    """Append-only results plus a run log, so a killed job resumes where it stopped.

    Every finished cell is written and fsynced immediately. On restart the
    keys already present are skipped, so re-submitting the same job continues
    rather than starting over.
    """

    def __init__(self, path=None, log_path="run.log"):
        # One ledger per job. Concurrent jobs sharing a scratch directory would
        # otherwise interleave appends into one file; the writes are line-atomic
        # so nothing is corrupted, but resume then depends on which jobs happened
        # to run, and a rerun of one experiment cannot be separated from another.
        # Resume still sees every ledger, because done() and rows() read them all.
        job = os.environ.get("SLURM_JOB_ID")
        self.path = path or (f"results_{job}.jsonl" if job else "results.jsonl")
        self.resume_glob = "results*.jsonl"
        self.log_path = log_path
        self.finished = set()
        for done_path in sorted(glob.glob(self.resume_glob)):
            for line in open(done_path):
                try:
                    self.finished.add(json.loads(line)["cell"])
                except (ValueError, KeyError):
                    pass
        self.log(f"ledger opened with {len(self.finished)} cells already done")

    def done(self, cell):
        return cell in self.finished

    def append(self, cell, row):
        # Where a row was measured is part of the measurement. Once results from
        # more than one partition or cluster share a ledger, a row without its
        # origin cannot be compared with any other, and the mistake is only
        # visible long after the run.
        row = dict(row, cell=cell, at=time.strftime("%Y-%m-%d %H:%M:%S"),
                   node=os.environ.get("SLURMD_NODENAME", platform.node()),
                   partition=os.environ.get("SLURM_JOB_PARTITION", ""),
                   job=os.environ.get("SLURM_JOB_ID", ""))
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.finished.add(cell)

    def log(self, message):
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
            f.flush()

    def rows(self, prefix):
        """Rows whose cell key starts with `prefix`.

        An absent file means no cell has finished yet, which is a normal state
        early in a run rather than an error.
        """
        out = []
        for path in sorted(glob.glob(self.resume_glob)):
            for line in open(path):
                row = json.loads(line)
                if row["cell"].startswith(prefix):
                    out.append(row)
        return out
