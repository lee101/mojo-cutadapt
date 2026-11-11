# Hot-loop analysis (SIMD phase)

This file did not exist when the SIMD phase began; it records what was found
and measured so the numbers in the README can be re-derived.

## What runs in a call

Two exported kernels carry the compute:

| kernel | loops in it | dependency | measured cost |
| --- | --- | --- | --- |
| `mca_locate` | DP column init, DP row fill, top-row scan, final column scan | the row fill is a strict diagonal chain | 8.8 us raw for a 34 nt adapter vs 139 nt read |
| `mca_hamming` | vectorised byte-compare + mismatch reduce | independent within a vector | 3.7 us raw for a 34 nt adapter vs 136 nt read |
| either | — | — | 2.3 us ctypes floor (`bench/bridge.py`) |

The ctypes floor is measured by calling an exported symbol that returns before
touching any buffer: 2.28 us. Anything smaller than ~2.3 us of kernel time
cannot move a `pixi run bench` row, and a 34 nt Hamming comparison is only
about 1.4 us above the floor.

## Loops that can be vectorised

- **DP column initialisation** (`mca_locate`, the `iota` loop): independent per
  column, and already written with `load`/`store` at
  `simdwidth_of[DType.float64]()` width (4 int64 lanes) plus a scalar tail.
- **The Hamming mismatch reduce** (`mca_hamming`): 32 u8 lanes, unaligned loads
  (`alignment=1`), a scalar tail, and an early-out that was already fused into
  the vector body so the bytes are read once.

## Loops that cannot

- **The DP row fill** (`_fill_row`): every cell consumes the cell diagonally
  before it (`carried_cost`/`carried_score`/`carried_origin`), so iterations
  depend on each other. The compiler-side answer is the existing fixed 8-cell
  `comptime for` strip, not a SIMD lane vector.
- **Top-row scan and final column scan** (`mca_locate`): these reduce over the
  DP column (`best_cost`, `best_score`, tie order). They look like textbook
  `reduce_add`/`reduce_min` candidates, but two things block it:
  1. The acceptance test compares each column against the *constant*
     `stale_origin` (whatever origin the last DP row wrote), which the shipped
     kernel has preserved since it was first frozen. A floating-point
     reduction would not reproduce that ordering.
  2. The two scans are **dependent, not adjacent**: the row scan and the
     column scan both read and write the same column cells. Fusing the column
     scan into the row loop (as an experiment did) changes the data the column
     scan sees, because the row loop keeps overwriting those cells after the
     column pass originally ran. It is a correctness bug, not an optimisation.

## Alignment facts

The work buffers are `np.empty(n, dtype=np.int64)`; 117 of 299 small sizes are
**not** 32-byte aligned on this machine. Any wide int64 load must be
`alignment=1`. The ASCII query and reference buffers are arbitrary bytes, so
the u8 Hamming loads are `alignment=1` as well.

## Measured experiments that were not kept

Same machine, same harness (`pixi run bench`, five runs, medians):

| experiment | result | kept? |
| --- | --- | --- |
| Base kernel, before this phase | 3' row 11.1-13.3 us; prefix 1.2-2.2 us; suffix 1.3-2.4 us (five runs) | — |
| Fuse the final column scan into the DP row loop | not auto-vectorisable; final kernel 8.79 us vs 8.67 us unaltered, i.e. **slower**, and it needed bit-for-bit care to stay identical | no |
| Hamming counting pass, widths 4/8/16/32 u8 lanes, incl. a disjoint-buffer form | 2.42 us flat across all widths, same as the existing computed width; disjoint form 2.32 us on the longer read, i.e. inside the ~2.3 us bridge floor and noise | no |

The Hamming widths were measured on standalone exported clones of the counting
loop (`_errors[False, W]` for W = 4, 8, 16, 32) plus a `DISJOINT=True` variant,
driven from Python with 200000 calls each, so the comparison was not affected by
the width the kernel happens to select.

## Parallelism (PARALLEL phase): what was measured, and why nothing shipped

The question was whether any of the remaining work can be split across cores.
Two numbers decide it: how big one call is, and what a `parallelize` call costs
on this host before it does any work.

**Per-call size of the shipped kernel** (`Aligner(adapter, 0.15, BACK)` with a
`locate()` on a read of 139, 1000 and 4000 nt; median of five runs, no code
changed for this table):

| adapter nt | 139 nt read | 1000 nt read | 4000 nt read |
| ---: | ---: | ---: | ---: |
| 34 | 10.0 us | 16.2 us | 36.6 us |
| 64 | 13.7 us | 18.8 us | 37.5 us |
| 128 | 32.2 us | 37.4 us | 54.8 us |
| 512 | 57.3 us | 380.6 us | 397.7 us |
| 1024 | 80.8 us | 1403.7 us | 1477.8 us |
| 2048 | 130.3 us | 1778.3 us | 5801.1 us |

The benchmark row and every real Cutadapt adapter is in the 34 nt band, i.e.
around 10 us per call, of which `bench/bridge.py` measures 2.3 us as the ctypes
floor.

**Cost of one `parallelize` call** (`bench/par_floor.mojo`, built separately so
the shipped library is untouched, called through ctypes like the real kernel;
median of nine calls after two warm-up calls; three runs on the same host):

| workers | launch+join ms (empty work) | ~1 ms per worker, parallel ms | same work serial ms | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | inline, no threads | 1.1-1.3 | 1.1-1.3 | ~1.0x |
| 2 | 4.8-5.8 | 6.0-6.7 | 2.2-2.6 | 0.36-0.41x |
| 4 | 4.7-5.4 | 5.8-6.9 | 4.3-5.4 | 0.68-0.83x |
| 8 | 4.6-5.3 | 5.9-6.3 | 8.8-9.6 | 1.49-1.57x |

Launch and join cost roughly 5 ms per call as soon as more than one worker is
requested, and it does not fall as the work grows or as the pool is reused.
Even with a perfectly parallel kernel, 2 and 4 workers lose at every size a
real caller passes: 5 ms of overhead against a 10 us call. The 8-worker row
only turns positive because the serial baseline in that row is ~9 ms of work,
which no adapter of realistic length produces.

That 5 ms is a property of the *library* build, not of Python. The identical
`parallelize` workload costs ~21-35 us per call when compiled as an executable,
but ~5.1-5.6 ms per call when compiled as a shared library — and it is just as
slow when the library is driven from a native C `main` (`dlopen`/`dlsym`, no
Python in the process):

| build of the same source | workers 2 | workers 4 | workers 8 |
| --- | ---: | ---: | ---: |
| executable | 14-23 us | 20-34 us | 23-35 us |
| shared library, native C driver | 5.18-5.42 ms | 5.17-5.47 ms | 4.95-5.44 ms |
| shared library, ctypes driver | 5.05-5.27 ms | 4.85-5.77 ms | 5.05-6.35 ms |

The kernel ships as a shared library, so the library row is the one that
applies: whichever thread-pool runtime the `max` build links into a shared
object takes milliseconds per dispatch on this host. A parallel kernel here
would pay that on every `locate` call.

**Why the DP cannot be split within a call anyway.** In `_fill_row`, cell
`(i, j)` reads `(i-1, j)`, `(i, j-1)` and `(i-1, j-1)`, so every cell in a row
depends on its left neighbour and on the whole previous row. The legal unit is
an anti-diagonal wavefront, which needs one barrier per diagonal — about
`reference_length + query_length` barriers for ~1000 cells on the benchmark
row. The band shrink (`while last >= 0 and costs[last] > k: last -= 1`) is a
genuine scalar loop-carried dependency that also feeds the final column scan,
so a chunked-over-rows split would change the result, not just the order.

**Decision: no parallel path shipped.** At 10 us per call against a ~5 ms
launch floor there is no threshold that makes `parallelize` a win for a real
caller, so the kernel stays serial and the gate is unchanged. The measurement
harness is left in place (`bench/par_floor.mojo`, `bench/par_floor.py`) so the
number can be re-taken if the toolchain or the host changes. It needs the `max`
package to build; that dependency is not in `pixi.toml` because the shipped
library does not import `max`.

## Verification used for the experiments

`tests/_reference.py` is a Python transcription of the DP recurrence used for
differential testing only. `tests/test_kernel_tables.py` compares the full
cost/score/origin column of the kernel against it for every flag combination
(16) and lengths {0, 1, 3, 4, 5, 31, 32, 33} — the values that exercise the
zero-lane, scalar-only, `W-1`, `W`, `W+1` and long odd-tail cases of both SIMD
sites. A candidate that returns the right answer but corrupts the table fails
here; that is how the fused-scan experiment was caught.
