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


def evict(paths):
    """Return the client to a cold state: no cached pages, and no cached locks.

    Dropping pages alone is not enough, and the obvious way of doing it is
    actively harmful. posix_fadvise(DONTNEED) needs an open descriptor, so a
    page-only evictor opens every file immediately before the measured read.
    That open takes the DoM ibits lock and leaves it cached; the next open then
    finds the lock already held, so the server has no reason to send the file's
    data inlined in the open reply -- while the pages it would have used are
    gone. What follows measures a cheap open and an expensive read, which is
    data-on-MDT behaving exactly like an ordinary OST file. That is how this
    suite lost a 3.9x DoM effect an earlier script had measured on the same
    filesystem, the earlier script having never pre-opened anything.

    Locks are therefore dropped through the LDLM namespace instead, which opens
    nothing. The page-drop path remains only as a fallback for when that is not
    permitted; it is reported rather than used silently, because a warm DoM
    measurement reads as a null result rather than as a failure.
    """
    if not hasattr(os, "posix_fadvise"):
        raise RuntimeError(
            "os.posix_fadvise is unavailable, so the page cache cannot be dropped; "
            "these measurements require Linux")

    if not drop_client_locks():
        for path in paths:
            fd = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
    run("sync")
    time.sleep(0.5)
