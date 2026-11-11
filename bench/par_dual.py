"""Build both forms, then drive the library from a native C driver (no Python)
and from ctypes, to see whether the ~5 ms floor is a property of the shared
library build or of Python hosting the call.

    pixi run python bench/par_dual.py
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "dist", "libpar-dual.so")
EXE = "/tmp/par-dual-exe"
REPS = 9

NATIVE = r"""
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
typedef int64_t (*fn_t)(int64_t, int64_t);
int main(void) {
    void *h = dlopen("%s", RTLD_NOW);
    if (!h) { printf("dlopen failed: %%s\n", dlerror()); return 1; }
    fn_t f = (fn_t)dlsym(h, "par_dual");
    if (!f) { printf("dlsym failed\n"); return 1; }
    static int64_t buf[64];
    for (int w = 1; w <= 8; w *= 2) {
        f(w, (int64_t)buf);
        printf("  workers %%d:", w);
        for (int k = 5; k < 5 + %d; k++) printf(" %%lld", (long long)(buf[k] / 1000));
        printf("\n");
    }
    return 0;
}
"""


def build() -> None:
    subprocess.run(["pixi", "run", "mojo", "build", "bench/par_dual.mojo", "-o", EXE], check=True)
    with tempfile.NamedTemporaryFile("w", suffix=".mojo", delete=False) as fh:
        fh.write(open(os.path.join(ROOT, "bench", "par_dual.mojo")).read().split("def main():")[0])
        lib_source = fh.name
    subprocess.run(
        ["pixi", "run", "mojo", "build", "--emit", "shared-lib", lib_source, "-o", LIB], check=True
    )
    os.unlink(lib_source)


def main() -> None:
    build()
    print("executable, us/call:")
    print(subprocess.run([EXE], capture_output=True, text=True, check=True).stdout, end="")

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "driver.c")
        driver = os.path.join(tmp, "driver")
        with open(src, "w") as fh:
            fh.write(NATIVE % (LIB, REPS))
        subprocess.run(["cc", "-O2", "-o", driver, src, "-ldl"], check=True)
        print("shared library, native C driver, us/call:")
        print(subprocess.run([driver], capture_output=True, text=True, check=True).stdout, end="")

    lib = ctypes.CDLL(LIB)
    p = lib.par_dual
    p.argtypes = [ctypes.c_int64] * 2
    buf = np.zeros(64, dtype=np.int64)
    print("shared library, Python/ctypes driver, us/call:")
    for w in (1, 2, 4, 8):
        p(w, buf.ctypes.data)
        print("  workers", w, ":", [int(v) // 1000 for v in buf[5 : 5 + REPS]])


if __name__ == "__main__":
    main()
