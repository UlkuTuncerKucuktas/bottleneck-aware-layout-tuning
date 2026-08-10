"""A failing cell must not end the run, and must not leave files behind.

The smoke test only exercises the success path, so it cannot tell whether a
twelve-hour job survives one bad configuration. These checks make a cell fail on
purpose and assert what the run does next.
"""

import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layout_tuning import layout as layout_module
from layout_tuning.runner import configure, measured, ledger, registry, main, experiment


def failing_cell_is_recorded_and_survived(root):
    os.chdir(root)
    configure("_failtest")

    measured("probe/ok", lambda: {"value": 1})
    measured("probe/boom", lambda: (_ for _ in ()).throw(RuntimeError("planned")))
    measured("probe/after", lambda: {"value": 2})

    rows = ledger().rows("probe/")
    by_cell = {r["cell"]: r for r in rows}
    assert set(by_cell) == {"probe/ok", "probe/boom", "probe/after"}, sorted(by_cell)
    assert by_cell["probe/boom"]["failed"] is True
    assert "planned" in by_cell["probe/boom"]["error"]
    assert by_cell["probe/after"]["value"] == 2, "run stopped at the failure"
    print("a failing cell is recorded and the run continues")

    # A recorded failure is not retried, so a systematic failure cannot loop.
    calls = []
    measured("probe/boom", lambda: calls.append(1) or {"value": 3})
    assert not calls, "a recorded failure was retried"
    print("a recorded failure is not retried on resume")


def working_dir_cleans_up_after_a_failure(root):
    path = os.path.join(root, "leaky")
    layout_module.run = lambda cmd: None          # no lfs here
    try:
        with layout_module.working_dir(path, "-c 1") as directory:
            open(os.path.join(directory, "f"), "w").close()
            raise RuntimeError("planned")
    except RuntimeError:
        pass
    assert not os.path.exists(path), "working_dir left its directory behind"
    print("working_dir removes its directory when the cell raises")


def aborting_experiment_does_not_stop_the_others(root):
    os.chdir(root)
    ran = []

    @experiment("_probe_boom")
    def _probe_boom():
        raise RuntimeError("planned abort")

    @experiment("_probe_next")
    def _probe_next():
        ran.append("next")

    code = main(["_probe_boom", "_probe_next"])
    assert ran == ["next"], "the second experiment did not run"
    assert code == 1, f"a run with an aborted experiment exited {code}, expected 1"
    print("an aborting experiment does not stop the ones after it")


def evict_does_not_reopen_the_files(root):
    """The measured read must be the first open since the write.

    Reopening a file before reading it defeats DoM inlining -- measured, not
    assumed -- and this suite lost a 3.9x effect to exactly that for several
    runs. The regression was invisible because the check was a substring search
    against the source, which passed while a duplicated definition later in the
    file shadowed the fixed one. So this asserts on behaviour: evict() must not
    open anything.
    """
    from layout_tuning import layout as L

    names = re.findall(r"^(?:def|class) (\w+)", open(L.__file__).read(), re.M)
    duplicated = sorted(n for n in set(names) if names.count(n) > 1)
    assert not duplicated, f"layout.py defines these twice: {duplicated}"

    directory = os.path.join(root, "evictcheck")
    os.makedirs(directory, exist_ok=True)
    paths = []
    for i in range(3):
        path = os.path.join(directory, f"f{i}")
        with open(path, "wb") as handle:
            handle.write(b"x" * 4096)
        paths.append(path)

    opened = []
    real_open = os.open

    def spy(path, flags, *args, **kwargs):
        opened.append(path)
        return real_open(path, flags, *args, **kwargs)

    real_run = L.run
    L.run = lambda cmd: None
    os.open = spy
    try:
        L.evict(paths)
    finally:
        os.open = real_open
        L.run = real_run

    assert not opened, f"evict() reopened {len(opened)} file(s): {opened[:3]}"
    print("evict() does not reopen the files it was given")


def run():
    root = tempfile.mkdtemp(prefix="layout-tuning-failtest-")
    try:
        failing_cell_is_recorded_and_survived(root)
        working_dir_cleans_up_after_a_failure(root)
        aborting_experiment_does_not_stop_the_others(root)
        evict_does_not_reopen_the_files(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(run())
