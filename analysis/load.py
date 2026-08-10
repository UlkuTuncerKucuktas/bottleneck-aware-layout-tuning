"""Load one or more result ledgers into a DataFrame.

    from analysis.load import load
    df = load("results/*.jsonl")
    df[df.experiment == "dom_cutoff"].groupby("size_kib").total_us_p50.median()

Every row carries the cell key it came from, split into `experiment` plus the
key parts, so a run assembled from several jobs analyses as one table.
"""

import glob
import json

import pandas as pd


def load(pattern="results*.jsonl", include_failed=False):
    """Every row from every matching ledger, failed cells excluded by default.

    A cell that raised is recorded with failed=True and no measurements, so
    letting those rows into a groupby would put NaNs beside real values. They
    are dropped here and their count reported, because a silent drop and a
    silent inclusion are both worse than a number on stderr.
    """
    rows = []
    for path in sorted(glob.glob(pattern)):
        for line in open(path):
            row = json.loads(line)
            parts = row["cell"].split("/")
            row["experiment"] = parts[0]
            row["cell_parts"] = parts[1:]
            row["source_file"] = path
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no ledger matched {pattern}")
    frame = pd.DataFrame(rows)
    if "failed" in frame.columns:
        failed = frame["failed"].fillna(False).astype(bool)
        if failed.any():
            print(f"note: {int(failed.sum())} failed cell(s) in {pattern}"
                  f"{'' if include_failed else ', excluded'}; "
                  "see the error column for why")
        if not include_failed:
            frame = frame[~failed].drop(columns=["failed", "error"], errors="ignore")
    return frame.reset_index(drop=True)


def phase_breakdown(df, experiment, group):
    """Mean microseconds per phase, so a latency can be attributed to a call."""
    phases = [c for c in df.columns if c.startswith("share_")]
    subset = df[df.experiment == experiment]
    return subset.groupby(group)[phases].mean()
