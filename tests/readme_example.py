"""Run the README's "adding an experiment" snippet, extracted from the README.

    python tests/readme_example.py

The snippet is read out of README.md rather than copied here, so the two cannot
drift apart: edit the README and this test exercises the edit.
"""

import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

WORK = tempfile.mkdtemp(prefix="layout-tuning-readme-")
os.environ["LAYOUT_SCRATCH"] = os.path.join(WORK, "scratch")

import smoke  # noqa: E402  (reuses the same Lustre stand-ins)
from layout_tuning.runner import main, registry  # noqa: E402


def snippet():
    text = open(os.path.join(ROOT, "README.md")).read()
    section = text[text.index("## Adding an experiment"):]
    return re.search(r"```python\n(.*?)```", section, re.S).group(1)


def run():
    code = snippet()
    print("--- snippet under test, verbatim from README.md ---")
    print(code)

    # The imports the README's layout section implies for an experiment module.
    import shutil
    from layout_tuning.layout import SCRATCH, fresh_dir, evict
    from layout_tuning.io import write_files
    from layout_tuning.probes import profile_reads
    from layout_tuning.runner import experiment, measured

    namespace = {
        "shutil": shutil, "SCRATCH": SCRATCH, "fresh_dir": fresh_dir,
        "evict": evict, "write_files": write_files,
        "profile_reads": profile_reads, "experiment": experiment,
        "measured": measured,
    }
    exec(compile(code, "README.md", "exec"), namespace)

    assert "example_probe" in registry(), "the snippet did not register an experiment"

    # Shrink the write and stand in for the lfs calls, exactly as smoke.py does,
    # but leave the snippet's own lines untouched.
    smoke.patch([
        ("fresh_dir", smoke.make_dir_ignoring_layout),
        ("run", smoke.shell),
        ("write_files", smoke.small_write),
        ("flush_dom_lock", lambda fd: None),
    ])
    namespace["fresh_dir"] = smoke.make_dir_ignoring_layout
    namespace["write_files"] = smoke.small_write
    if not hasattr(os, "posix_fadvise"):
        namespace["evict"] = lambda paths: None

    registry()["example_probe"]["fn"].__globals__.update(namespace)

    os.chdir(WORK)
    from layout_tuning.runner import configure
    configure()
    main(["example_probe"])

    cells = [line for line in open("results.jsonl")]
    assert len(cells) == 2, f"expected 2 cells, got {len(cells)}"
    print(f"\nthe documented snippet registered and produced {len(cells)} cells")
    return 0


if __name__ == "__main__":
    sys.exit(run())
