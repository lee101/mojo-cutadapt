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

## Verification used for the experiments

`tests/_reference.py` is a Python transcription of the DP recurrence used for
differential testing only. `tests/test_kernel_tables.py` compares the full
cost/score/origin column of the kernel against it for every flag combination
(16) and lengths {0, 1, 3, 4, 5, 31, 32, 33} — the values that exercise the
zero-lane, scalar-only, `W-1`, `W`, `W+1` and long odd-tail cases of both SIMD
sites. A candidate that returns the right answer but corrupts the table fails
here; that is how the fused-scan experiment was caught.
