"""Measure the ctypes crossing itself, independent of any kernel work.

`bench/bench.py` times Python-level calls, and the Mojo entry points are C ABI
symbols reached through ctypes. Before attributing a row's cost to a kernel,
this script measures the floor: an exported Mojo symbol that returns without
touching its buffers (~2.3 us on the reference machine). Subtracting it shows
that the small kernels (a 34 nt Hamming comparison) are mostly bridge, which is
why no SIMD change to them shows up in `pixi run bench`.

    pixi run python bench/bridge.py
"""

from __future__ import annotations

import ctypes
import statistics
import time

from mojocutadapt import _lib


def measure(fn, repeats=7, calls=100000):
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(calls):
            fn()
        values.append((time.perf_counter() - start) * 1e6 / calls)
    return statistics.median(values)


def main():
    lib = _lib.lib()
    hamming = lib.mca_hamming
    hamming.argtypes = [_lib.I] * 4 + [_lib.F] + [_lib.I] * 3
    hamming.restype = _lib.I
    workspace = _lib.Workspace(b"ACGT")
    adapter = b"AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC"
    read = b"GATTACA" * 19 + adapter
    big = _lib.Workspace(adapter)

    rows = [
        ("no-op: arguments rejected before any buffer is touched",
         lambda: hamming(0, 0, 0, 0, 1.0, 0, 0, 0)),
        ("no-op: min_overlap above length, returns 0 at once",
         lambda: hamming(workspace.reference_addr, 1, workspace.reference_addr, 1, 1.0, 0, 2,
                         workspace.result_addr)),
        ("mca_hamming, 34 nt adapter vs 136 nt read",
         lambda: hamming(big.reference_addr, len(adapter),
                         ctypes.cast(ctypes.c_char_p(read), ctypes.c_void_p).value, len(read),
                         0.15, 0, 1, big.result_addr)),
    ]
    print("| measurement | us/call |")
    print("| --- | ---: |")
    for name, fn in rows:
        print(f"| {name} | {measure(fn):.3f} |")


if __name__ == "__main__":
    main()
