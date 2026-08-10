"""Layout definitions and the Lustre commands used to apply them."""

import os, re, sys, time, array, fcntl, shutil, subprocess

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
    """Release the file's write lock so the next reader can be served inlined.

    The flag lands at offset 12, matching
    `struct ioc_data_version {__u64 idv_version; __u32 idv_layout_version;
    __u32 idv_flags;}`. An older Lustre used two __u64 fields, where the same
    bytes decode as flags=0x200000000 and `flags & LL_DV_WR_FLUSH` is zero: the
    ioctl succeeds, returns a data version, and flushes nothing. Nothing in the
    reply distinguishes the two cases, so the version is checked once against
    the layout the client actually speaks rather than assumed.
    """
    buf = array.array('B', bytes(16))
    buf[12:16] = array.array('B', LL_DV_WR_FLUSH.to_bytes(4, sys.byteorder))
    fcntl.ioctl(fd, LL_IOC_DATA_VERSION, buf, True)


def flush_layout_is_supported():
    """True when this client uses the struct the flush ioctl is built for.

    2.11 introduced the split __u32 layout_version/__u32 flags fields. Below
    that the flush silently does nothing, which turns every DoM measurement
    into an ordinary OST measurement without any error to notice.
    """
    text = capture("lfs --version")
    match = re.search(r"(\d+)\.(\d+)", text)
    if not match:
        return None
    major, minor = int(match.group(1)), int(match.group(2))
    return (major, minor) >= (2, 11)


def drop_client_locks():
    """Release the client's cached LDLM locks. Returns True if it worked.

    Touches no file, so nothing is pre-opened and no lock is re-taken on the way
    out. Needs write access to /proc/fs/lustre, which an unprivileged user may
    not have, so the caller gets a boolean rather than an exception and the run
    can record which kind of eviction it actually got.
    """
    dropped = False
    for namespace in ("*osc*", "*mdc*"):
        result = subprocess.run(
            f"lctl set_param -n ldlm.namespaces.{namespace}.lru_size=clear",
            shell=True, capture_output=True, text=True)
        dropped = dropped or result.returncode == 0
    return dropped


def evict_by_flush(paths):
    """Drop each file's cached data and lock using only the write-flush ioctl.

    The ioctl pushes dirty data to the server and releases the write lock, so
    the next read starts cold -- the same mechanism that makes a freshly
    written file measurable at all. It needs no privilege, unlike the LDLM
    namespace, and it opens the files O_WRONLY, so it never takes the read lock
    that suppresses DoM inlining.

    Whether this leaves the client as cold as an LDLM flush is a question for
    measurement, not assumption: slurm/evict_probe.sh compares the strategies
    and prints the open/read split that distinguishes them.
    """
    for path in paths:
        fd = os.open(path, os.O_WRONLY)
        try:
            flush_dom_lock(fd)
        finally:
            os.close(fd)


def evict(paths):
    """Return the client to a cold state before a measured read.

    Measured on /arf, and this is the whole story for DoM: ANY reopen of the
    file between writing and reading defeats inlining, whatever mode it uses.
    Reading straight after a flushed write gave 404 us open against 44 us read
    (a ratio of 0.11 -- the payload arrived with the open reply). Reopening the
    same files write-only and flushing again gave 252/697, and reopening them
    read-only to drop pages gave 178/545: both ratios above 2.7, meaning the
    data came by a separate RPC. Against the OST control at 1243 us total, the
    unevicted DoM read was 1.88x faster and the reopened ones were not faster
    at all.

    Lustre caches under lock, so releasing the write lock at close already
    surrenders the client's cached pages: a freshly written and flushed file is
    cold without further help. The reopen is not just unnecessary, it caches the
    read lock that stops the server sending data with the next open.

    So for files this process has just written -- which is every file in this
    suite -- eviction is a no-op by design. The privileged LDLM path is kept for
    the case of reading files written by someone else, where the lock may
    genuinely be cached and there is no write descriptor to flush.
    """
    # Deliberately does nothing to the files. See above: the write flush already
    # left them cold, and touching them again would warm the lock. The old
    # posix_fadvise guard is gone with the code that needed it -- keeping a
    # Linux-only check on a function that no longer touches a file would refuse
    # to run for a reason that had stopped being true.
    run("sync")
    time.sleep(0.5)


def evict_foreign(paths):
    """Cold-read files this process did NOT write, if the client permits it.

    Without a write descriptor to flush there is no unprivileged way to release
    a cached lock, so this needs write access to the LDLM namespace and reports
    whether it got it. Nothing in the current suite needs it -- every experiment
    writes its own files -- but a workload replay reading someone else's dataset
    would.
    """
    return drop_client_locks()


