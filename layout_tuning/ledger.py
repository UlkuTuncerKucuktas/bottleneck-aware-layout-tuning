"""Append-only results with resume, so an interrupted run continues."""

import os, json, time

class Ledger:
    """Append-only results plus a run log, so a killed job resumes where it stopped.

    Every finished cell is written and fsynced immediately. On restart the
    keys already present are skipped, so re-submitting the same job continues
    rather than starting over.
    """

    def __init__(self, path="results.jsonl", log_path="run.log"):
        self.path = path
        self.log_path = log_path
        self.finished = set()
        if os.path.exists(path):
            for line in open(path):
                try:
                    self.finished.add(json.loads(line)["cell"])
                except (ValueError, KeyError):
                    pass
        self.log(f"ledger opened with {len(self.finished)} cells already done")

    def done(self, cell):
        return cell in self.finished

    def append(self, cell, row):
        row = dict(row, cell=cell, at=time.strftime("%Y-%m-%d %H:%M:%S"))
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
        if not os.path.exists(self.path):
            return []
        out = []
        for line in open(self.path):
            row = json.loads(line)
            if row["cell"].startswith(prefix):
                out.append(row)
        return out
