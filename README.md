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
arguments before dereferencing them; the Python bridge keeps NumPy views alive
through the call and locks each mutable workspace. The kernel holds one
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
wildcard cases, SIMD-tail anchored comparers, adapter trimming outcomes, and
raw-ABI argument rejection. Each API item in the covered-subset list has an
exercising parity or regression test.

Measured with `pixi run bench` on x86_64, Python 3.13.14:

| kernel | mojo-cutadapt | cutadapt 5.2 | relative |
| --- | ---: | ---: | ---: |
| 34 nt 3' adapter vs 139 nt read | 25.1 us | 6.3 us | 3.95x slower |
| 34 nt prefix no-indel comparer vs 139 nt read | 10.9 us | 0.3 us | 40.90x slower |
| 34 nt suffix no-indel comparer vs 139 nt read | 10.9 us | 0.7 us | 15.80x slower |

The general semiglobal DP recurrence has loop-carried dependencies, so one
alignment cannot be usefully SIMD-vectorized or parallelized across cells.
The anchored no-indel comparers use SIMD Hamming comparison with an unaligned
load-safe scalar tail. GPU execution is intentionally not used: these small,
memory-bound alignments do not have enough independent arithmetic to offset
device-transfer and launch costs. Re-run `pixi run bench` on your machine for
local measurements.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

MIT.
