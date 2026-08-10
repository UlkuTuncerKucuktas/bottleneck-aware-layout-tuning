"""Layout definitions and the Lustre commands used to apply them."""

import os, sys, time, array, fcntl, shutil, subprocess

# Override with LAYOUT_SCRATCH to run somewhere else.
SCRATCH = os.environ.get("LAYOUT_SCRATCH",
                         f"/arf/scratch/{os.environ.get('USER', 'nobody')}/layout-tuning")
CHUNK = 1 << 20

# Layouts, named the way the paper names them.
OST_PLAIN = "-c 1 -S 1M"
OST_WIDE = "-c -1 -S 1M"


def dom_layout(component="1M"):
    return f"-E {component} -L mdt -E -1 -c 1 -S 1M"

# Flushing a DoM file's write lock is what lets the next reader get the data
# inlined in the open reply. Without it DoM measures the same as OST.
LL_IOC_DATA_VERSION = (2 << 30) | (16 << 16) | (ord('f') << 8) | 218
LL_DV_WR_FLUSH = 1 << 1


def run(cmd):
    subprocess.run(cmd, shell=True, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def capture(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def fresh_dir(path, layout):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    assert os.path.isdir(path), f"{path} is not a directory (setstripe would create a file)"
    run(f"lfs setstripe {layout} {path}")
    return path


class working_dir:
    """A layout-stamped directory that is removed even if the cell fails.

    Every cell writes GiB of files, and a cell that raises used to leave them
    behind: over a long run that fills the filesystem and, worse, leaves a
    directory whose layout a later cell might inherit. As a context manager the
    cleanup happens on the way out either way.

        with working_dir(f"{SCRATCH}/thing", OST_PLAIN) as directory:
            ...
    """

    def __init__(self, path, layout):
        self.path = path
        self.layout = layout

    def __enter__(self):
        return fresh_dir(self.path, self.layout)

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)
        return False


def flush_dom_lock(fd):
    buf = array.array('B', bytes(16))
    buf[12:16] = array.array('B', LL_DV_WR_FLUSH.to_bytes(4, sys.byteorder))
    fcntl.ioctl(fd, LL_IOC_DATA_VERSION, buf, True)


def evict(paths):
    """Drop pages without opening the file for reading (that would take the lock).

    Cache eviction is what makes a read measurement mean anything, so a
    platform without posix_fadvise fails here rather than silently returning
    warm-cache numbers.
    """
    if not hasattr(os, "posix_fadvise"):
        raise RuntimeError(
            "os.posix_fadvise is unavailable, so the page cache cannot be dropped; "
            "these measurements require Linux")
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    run("sync")
    time.sleep(0.5)
