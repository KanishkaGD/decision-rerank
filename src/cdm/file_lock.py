import fcntl
import os
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def locked(path: str) -> Iterator[None]:
    """
    Exclusive advisory lock (fcntl.flock) on "<path>.lock" for the duration of the block. Guards a read or a
    read-modify-write of the file at `path` against concurrent jobs: results/ may live on a shared network file system,
    which supports flock() across nodes. A separate lock file, so an atomic replace of `path` (write a temporary
    file, then os.replace) never swaps out the inode the lock is held on.
    """
    lock_path = f"{path}.lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
