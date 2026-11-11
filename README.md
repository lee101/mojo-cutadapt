# mojo-cutadapt

`mojo-cutadapt` is a standalone Mojo port of Cutadapt's compute-bound adapter
alignment core. It provides a Python API shaped like Cutadapt's alignment and
single-adapter classes, while executing ordinary A/C/G/T hybrid semiglobal
alignment through a small ctypes bridge into one compiled Mojo library.

## Covered subset

- `Aligner` and `EndSkip`, including Cutadapt's error-rate, origin, and
  semiglobal endpoint-selection rules.
- `PrefixComparer` and `SuffixComparer` for anchored no-indel matching.
- `FrontAdapter`, `BackAdapter`, `AnywhereAdapter`, `PrefixAdapter`, and
  `SuffixAdapter`, including `match_to()`, match coordinates, and trim slices.
- IUPAC wildcard alignment, implemented in the Python compatibility path.

This does not include Cutadapt's CLI, file/pipeline machinery, k-mer filtering,
linked adapters, statistics/reporting, quality trimming, or multiplexed adapter
indexes. Those are orchestration and I/O features rather than the alignment
kernel targeted here.

## Install

```bash
pixi install
pixi run build
```

The package is imported from the checkout by the Pixi environment:

```python
from mojocutadapt import BackAdapter

adapter = BackAdapter("AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC")
match = adapter.match_to("GATTACAGATTACAAGATCGGAAGAGCACACGTCTGAACTCCAGTCAC")
print(match.remainder_interval())
```

## How it works

`src/capi.mojo` is the sole compilation unit. Its C ABI accepts byte-buffer
addresses and caller-owned `Int` work buffers. The kernel validates the raw
arguments before dereferencing them; the Python bridge passes the immutable
ASCII query buffer zero-copy, caches the NumPy workspace addresses, and locks
each mutable workspace. The kernel holds one
dynamic-programming column per adapter base: cost, score, and signed origin.
This uses linear extra memory and preserves Cutadapt's hybrid cost/score and
endpoint-selection behavior.

`python/mojocutadapt/_lib.py` exposes that ABI with ctypes. Uppercase A/C/G/T
alignments take the Mojo path; IUPAC wildcard requests take a faithful Python
compatibility implementation because set-valued base matching has different
data representation requirements. The public adapter classes compose the same
aligner and return compatible coordinate-bearing match objects.

## Verification and benchmark

`pixi run test` compares the covered API to the installed Cutadapt extension:
fixed vectors, randomized non-wildcard alignments across flags and costs,
wildcard cases, SIMD initialization and Hamming tails, adapter trimming
outcomes, and raw-ABI argument rejection. Each API item in the covered-subset
list has an exercising parity or regression test.

Measured with `pixi run bench` on x86_64 (Intel Xeon E5-2697 v4), Python 3.13.14.
"before" is the previous kernel and "after" is this one, each run five times and
taken as the median:

| kernel | before | after | change | cutadapt 5.2 | relative |
| --- | ---: | ---: | ---: | ---: | ---: |
| 34 nt 3' adapter vs 139 nt read | 14.3 us | 11.5 us | -20% | 6.2 us | 1.9x slower |
| 34 nt prefix no-indel comparer vs 139 nt read | 1.2 us | 1.2 us | unchanged | 0.3 us | unchanged |
| 34 nt suffix no-indel comparer vs 139 nt read | 1.3 us | 1.3 us | unchanged | 0.7 us | unchanged |

The comparer rows are unchanged because a perfect prefix or suffix match returns
before the FFI boundary, and a mismatch only reaches the SIMD Hamming kernel.
The 3' row improved in an earlier phase as follows:

- The DP interior is unrolled in fixed 8-cell strips with `comptime for`, and the
  band end is hoisted to the row. The scalar remainder keeps the existing
  loop-carried semantics.
- The diagonal cell is carried in registers across a row, and the two neighbour
  values are loaded only where the match, deletion, or insertion branch that
  needs each of them is taken, rather than unconditionally every iteration.
- The init and selection phases now use `unsafe_load`/`unsafe_store` on indices
  the existing loops already bound; there are no new index assumptions.

### SIMD phase

This phase tried to vectorise the remaining hot loops and found that the
measurable ones were already vectorised, while the rest cannot be:

- The DP column initialisation and the Hamming mismatch count were already
  SIMD with scalar tails (widths 4 int64 lanes and 32 u8 lanes on this machine).
- The DP row fill is a strict diagonal chain, so its iterations depend on each
  other; the existing 8-cell `comptime for` strip is what the compiler can do.
- The top-row and column scans are not two adjacent passes: fusing them makes
  the column pass read cells the row pass keeps rewriting. The experiment was
  built, caught by a new whole-table differential test, and reverted.

Nothing was kept, so there is no before/after difference to report. Measured on
the same machine (median of five `pixi run bench` runs, raw library):

| kernel | before | after | change |
| --- | ---: | ---: | ---: |
| 34 nt 3' adapter vs 139 nt read | 11.2 us | 11.2 us | unchanged |
| 34 nt prefix no-indel comparer vs 139 nt read | 1.3 us | 1.3 us | unchanged |
| 34 nt suffix no-indel comparer vs 139 nt read | 1.4 us | 1.4 us | unchanged |

`bench/bridge.py` measures the ctypes floor at 2.30 us per call, using a Mojo
entry point that returns before touching a buffer. A 34 nt Hamming comparison
costs about 3.75 us in total, so roughly 60% of that row is the Python/Mojo
crossing and no vector-width change can show up in it. Experiments and their
measurements are recorded in `bench/analysis.md`.

Two techniques from this phase did not pay and were reverted: multi-accumulator
reductions (the recurrence is a strict diagonal chain, so there is nothing to
accumulate in parallel) and dropping bounds checks (the compiler's checks are
already folded into the loop bounds). The result is bit-identical to the previous
kernel; the summation order did not change, so there is no last-bit caveat.

### PARALLEL phase

This phase looked for work to split across cores and found none that wins, so
nothing was kept: the kernel is still the serial one from the SIMD phase, and
there is no before/after difference to report.

Measured on the same machine, no code changed (median of five runs):

| adapter nt | 139 nt read | 1000 nt read | 4000 nt read |
| ---: | ---: | ---: | ---: |
| 34 (the benchmark row) | 10.0 us | 16.2 us | 36.6 us |
| 128 | 32.2 us | 37.4 us | 54.8 us |
| 512 | 57.3 us | 380.6 us | 397.7 us |
| 1024 | 80.8 us | 1403.7 us | 1477.8 us |
| 2048 | 130.3 us | 1778.3 us | 5801.1 us |

One `parallelize` call costs this host about 5 ms before it does any work, at
any worker count from 2 to 8 (3 runs; `bench/par_floor.py`):

| workers | launch+join (empty work) | 1 ms per worker | same work serial | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | inline, no threads | 1.1-1.3 ms | 1.1-1.3 ms | ~1.0x |
| 2 | 4.8-5.8 ms | 6.0-6.7 ms | 2.2-2.6 ms | 0.36-0.41x |
| 4 | 4.7-5.4 ms | 5.8-6.9 ms | 4.3-5.4 ms | 0.68-0.83x |
| 8 | 4.6-5.3 ms | 5.9-6.3 ms | 8.8-9.6 ms | 1.49-1.57x |

So the gain does not just flatten at 2 and 4 threads, it reverses: 5 ms of
launch and join against a 10 us call. The 8-worker row only looks positive
because its serial baseline is 9 ms of work, which no real adapter produces.
There is therefore no size threshold above which a parallel path beats the
serial one for a caller of this package, which is why nothing shipped.

The 5 ms is a property of the shared-library build, not of Python. The same
`parallelize` workload compiled as an executable costs 21-35 us per call; as a
shared library it costs 5.1-5.6 ms per call, and a native C `dlopen` driver
sees the same 5 ms without Python in the process:

| build of the same source | workers 2 | workers 4 | workers 8 |
| --- | ---: | ---: | ---: |
| executable | 14-23 us | 20-34 us | 23-35 us |
| shared library, native C driver | 5.18-5.42 ms | 5.17-5.47 ms | 4.95-5.44 ms |
| shared library, ctypes driver | 5.05-5.27 ms | 4.85-5.77 ms | 5.05-6.35 ms |

This package ships a shared library, so the last two rows are the ones that
would apply to a parallel kernel.

The dependency structure says the same thing. In the DP row fill, cell
`(i, j)` reads `(i-1, j)`, `(i, j-1)` and `(i-1, j-1)`, so no two cells in a
row are independent and row `j` needs all of row `j-1`. The only legal split is
an anti-diagonal wavefront, which costs one barrier per diagonal (about
`reference_length + query_length` of them for ~1000 cells), and the band shrink
is a scalar loop-carried dependency that feeds the final column scan, so
splitting by row would change results rather than order.

Because no parallel path shipped, there is no serial-vs-parallel test to add;
the existing whole-table differential test in `tests/test_kernel_tables.py`
still pins the kernel's output. `bench/par_floor.mojo`/`.py` re-take the launch
measurement and `bench/par_dual.mojo`/`.py` re-take the executable-vs-library
comparison; building them needs `pixi add 'max==26.7.0.dev2026092605'`, which is
not a dependency of the shipped library.

### Tests added

`tests/test_parity.py` now covers the SIMD tail boundaries for both vectorised
sites — reference lengths 0, 1, 3, 4, 5 and 33, i.e. zero lanes, scalar-only,
`W-1`, `W`, `W+1`, and a long odd length — across every flag combination, plus
mismatching comparers that are long enough to reach the kernel.

`tests/test_kernel_tables.py` is new and differs from the parity suite: instead
of comparing final answers, it compares the whole cost/score/origin DP column
against `tests/_reference.py`, a Python transcription, for all 16 flag
combinations and those lengths. A kernel that returns the right answer from a
corrupted table fails it, which is what caught the fused-scan experiment.

The independent DP-column initialization is SIMD-vectorized with a scalar tail.
The general recurrence has loop-carried dependencies, so one alignment cannot
be usefully parallelized across cells. Anchored no-indel comparers return exact
prefix or suffix matches without crossing the FFI boundary; mismatch counting
uses SIMD Hamming comparison with unaligned loads and a scalar tail. A
thread-launch threshold was measured rather than assumed and did not pay off;
see the PARALLEL phase section above for the numbers.
GPU execution is intentionally not used: these small, memory-bound kernels are
well below two arithmetic operations per byte and cannot offset device transfer
and launch costs. Re-run `pixi run bench` on your machine for local measurements.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

MIT.
