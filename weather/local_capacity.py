"""Cross-process local training lease and Windows commit headroom."""
from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import time


def commit_free_bytes():
    if os.name!="nt":return None
    class PerformanceInformation(ctypes.Structure):
        _fields_=[("cb",ctypes.c_ulong)]+[(name,ctypes.c_size_t) for name in
            ("CommitTotal","CommitLimit","CommitPeak","PhysicalTotal","PhysicalAvailable",
             "SystemCache","KernelTotal","KernelPaged","KernelNonpaged","PageSize")]+[
            (name,ctypes.c_ulong) for name in ("HandleCount","ProcessCount","ThreadCount")]
    information=PerformanceInformation();information.cb=ctypes.sizeof(information)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(information),information.cb):return None
    return (information.CommitLimit-information.CommitTotal)*information.PageSize


@contextmanager
def local_training_lease(path,poll_seconds=5.,timeout_seconds=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();last_message=started-30
    with path.open("a+b") as handle:
        if handle.seek(0,2)==0:handle.write(b"0");handle.flush()
        while True:
            handle.seek(0)
            try:
                if os.name=="nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except (BlockingIOError,OSError):
                now=time.monotonic()
                if timeout_seconds is not None and now-started>=timeout_seconds:
                    raise TimeoutError("Another local training process holds the lease")
                if now-last_message>=30:
                    print("WAITING_LOCAL_TRAINING_SLOT",flush=True);last_message=now
                time.sleep(poll_seconds)
        try:yield
        finally:
            handle.seek(0)
            if os.name=="nt":msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
