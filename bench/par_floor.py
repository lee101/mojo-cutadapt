"""Report the `max.algorithm.parallelize` launch/join cost at 1, 2, 4 and 8 workers.

This is the floor any parallel kernel pays before doing useful work, so it is
what decides whether splitting a kernel can win at a given call size. Measured
through the C ABI, the same way `bench/bench.py` reaches the real kernels.

Needs the `max` package and a built probe library:

    pixi add 'max==26.7.0.dev2026092605'
    pixi run mojo build --emit shared-lib bench/par_floor.mojo -o dist/libparallel-floor.so
    pixi run python bench/par_floor.py
"""

from __future__ import annotations

import ctypes
import os
import statistics

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "dist", "libparallel-floor.so")

I = ctypes.c_int64
STRIDE = 32
REPS = 9


def main() -> None:
    if not os.path.exists(LIB):
        raise SystemExit(
            f"{LIB} is missing; build it with the commands in this file's docstring"
        )
    lib = ctypes.CDLL(LIB)
    probe = lib.par_floor
    probe.argtypes = [I, I, I]
    probe.restype = I
    out = np.zeros(8 * STRIDE, dtype=np.int64)
    print("| workers | launch+join ms | 1 ms/worker unit ms | serial ms | speedup |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for workers in (1, 2, 4, 8):
        if probe(workers, REPS, out.ctypes.data) != 0:
            raise SystemExit(f"probe rejected workers={workers}")
        empty = statistics.median(out[0:REPS]) / 1e6
        parallel = statistics.median(out[STRIDE : STRIDE + REPS]) / 1e6
        serial = out[2 * STRIDE - 1] / 1e6
        print(f"| {workers} | {empty:.3f} | {parallel:.1f} | {serial:.1f} | {serial / parallel:.2f}x |")


if __name__ == "__main__":
    main()
