"""ctypes bridge for the compiled Mojo alignment kernel."""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-cutadapt.so")
I = ctypes.c_int64
F = ctypes.c_double


class BuildError(RuntimeError):
    pass


def _mojo() -> list[str]:
    command = os.environ.get("MOJOCUTADAPT_MOJO")
    if command:
        return command.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJOCUTADAPT_MOJO")


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "capi.mojo")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    proc = subprocess.run([os.path.join(ROOT, "build", "build.sh")], capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip())
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        fn = _library.mca_locate
        fn.argtypes = [I, I, I, I, F, I, I, I, I, I, I, I, I, I]
        fn.restype = I
        fn = _library.mca_hamming
        fn.argtypes = [I, I, I, I, F, I, I, I]
        fn.restype = I
    return _library


class Workspace:
    def __init__(self, reference: bytes) -> None:
        self.reference = np.frombuffer(reference, dtype=np.uint8)
        size = len(self.reference) + 1
        self.costs = np.empty(size, dtype=np.int64)
        self.scores = np.empty(size, dtype=np.int64)
        self.origins = np.empty(size, dtype=np.int64)
        self.result = np.empty(6, dtype=np.int64)
        # The work arrays and result buffer are mutated by each call. ctypes
        # releases the GIL around CDLL calls, so one Aligner must not share a
        # workspace concurrently without this lock.
        self.lock = threading.Lock()


def _query_array(query: bytes) -> np.ndarray:
    if not isinstance(query, bytes):
        raise TypeError("query must be bytes")
    return np.frombuffer(query, dtype=np.uint8)


def _validate_call(max_error_rate: float, flags: int | None, indel_cost: int | None,
                   min_overlap: int) -> None:
    if not isinstance(max_error_rate, (int, float)) or not math.isfinite(max_error_rate) or max_error_rate < 0:
        raise ValueError("max_error_rate must be a finite non-negative number")
    if flags is not None and (not isinstance(flags, int) or flags & ~0xF):
        raise ValueError("flags must use only EndSkip bits")
    if indel_cost is not None and (not isinstance(indel_cost, int) or indel_cost <= 0):
        raise ValueError("indel_cost must be a positive integer")
    if not isinstance(min_overlap, int) or min_overlap < 0:
        raise ValueError("min_overlap must be a non-negative integer")


def locate(
    query: bytes,
    max_error_rate: float,
    flags: int,
    indel_cost: int,
    min_overlap: int,
    workspace: Workspace,
) -> tuple[int, int, int, int, int, int] | None:
    _validate_call(max_error_rate, flags, indel_cost, min_overlap)
    ref = workspace.reference
    qry = _query_array(query)
    # Keep qry strongly referenced through the FFI call; frombuffer is a view
    # of the immutable bytes object, not a copied temporary.
    with workspace.lock:
        found = lib().mca_locate(
            ref.ctypes.data, len(ref), qry.ctypes.data, len(qry), max_error_rate, flags,
            0, 0, indel_cost, min_overlap, workspace.costs.ctypes.data, workspace.scores.ctypes.data,
            workspace.origins.ctypes.data, workspace.result.ctypes.data,
        )
        result = tuple(map(int, workspace.result)) if found > 0 else None
    if found < 0:
        raise RuntimeError("Mojo alignment kernel rejected its validated ABI arguments")
    return result


def hamming(
    query: bytes,
    max_error_rate: float,
    suffix: bool,
    min_overlap: int,
    workspace: Workspace,
) -> tuple[int, int, int, int, int, int] | None:
    _validate_call(max_error_rate, None, None, min_overlap)
    ref = workspace.reference
    qry = _query_array(query)
    with workspace.lock:
        found = lib().mca_hamming(
            ref.ctypes.data, len(ref), qry.ctypes.data, len(qry), max_error_rate,
            int(suffix), min_overlap, workspace.result.ctypes.data,
        )
        result = tuple(map(int, workspace.result)) if found > 0 else None
    if found < 0:
        raise RuntimeError("Mojo Hamming kernel rejected its validated ABI arguments")
    return result
