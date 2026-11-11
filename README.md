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
The 3' row improves as follows:

- The DP interior is unrolled in fixed 8-cell strips with `comptime for`, and the
  band end is hoisted to the row. The scalar remainder keeps the existing
  loop-carried semantics.
- The diagonal cell is carried in registers across a row, and the two neighbour
  values are loaded only where the match, deletion, or insertion branch that
  needs each of them is taken, rather than unconditionally every iteration.
- The init and selection phases now use `unsafe_load`/`unsafe_store` on indices
  the existing loops already bound; there are no new index assumptions.

Two techniques from this phase did not pay and were reverted: multi-accumulator
reductions (the recurrence is a strict diagonal chain, so there is nothing to
accumulate in parallel) and dropping bounds checks (the compiler's checks are
already folded into the loop bounds). The result is bit-identical to the previous
kernel; the summation order did not change, so there is no last-bit caveat.

The independent DP-column initialization is SIMD-vectorized with a scalar tail.
The general recurrence has loop-carried dependencies, so one alignment cannot
be usefully parallelized across cells. Anchored no-indel comparers return exact
prefix or suffix matches without crossing the FFI boundary; mismatch counting
uses SIMD Hamming comparison with unaligned loads and a scalar tail. There is no
batch kernel with enough independent work to justify a thread-launch threshold.
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
