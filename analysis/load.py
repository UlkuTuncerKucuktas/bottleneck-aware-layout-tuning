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


def load(pattern="results*.jsonl"):
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
    return pd.DataFrame(rows)


def phase_breakdown(df, experiment, group):
    """Mean microseconds per phase, so a latency can be attributed to a call."""
    phases = [c for c in df.columns if c.startswith("share_")]
    subset = df[df.experiment == experiment]
    return subset.groupby(group)[phases].mean()
