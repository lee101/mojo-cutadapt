"""Benchmark the Mojo alignment path against Cutadapt's compiled extension."""

from __future__ import annotations

import platform
import statistics
import time

from cutadapt._align import (
    Aligner as UpstreamAligner,
    PrefixComparer as UpstreamPrefixComparer,
    SuffixComparer as UpstreamSuffixComparer,
)

from mojocutadapt import Aligner, EndSkip, PrefixComparer, SuffixComparer


def measure(fn, repeats=5, calls=2000):
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(calls):
            fn()
        values.append((time.perf_counter() - start) * 1e6 / calls)
    return statistics.median(values)


def main():
    adapter = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC"
    read = "GATTACAGATTACAGATTACA" * 5 + adapter
    mojo = Aligner(adapter, 0.15, EndSkip.QUERY_START | EndSkip.QUERY_STOP | EndSkip.REFERENCE_END, min_overlap=3)
    upstream = UpstreamAligner(adapter, 0.15, int(EndSkip.QUERY_START | EndSkip.QUERY_STOP | EndSkip.REFERENCE_END), min_overlap=3)
    mojo.locate(read)
    prefix_read = adapter + "GATTACAGATTACAGATTACA" * 5
    suffix_read = "GATTACAGATTACAGATTACA" * 5 + adapter
    mojo_prefix = PrefixComparer(adapter, 0.15, min_overlap=3)
    upstream_prefix = UpstreamPrefixComparer(adapter, 0.15, min_overlap=3)
    mojo_suffix = SuffixComparer(adapter, 0.15, min_overlap=3)
    upstream_suffix = UpstreamSuffixComparer(adapter, 0.15, min_overlap=3)
    cases = [
        ("34 nt 3' adapter vs 139 nt read", lambda: mojo.locate(read), lambda: upstream.locate(read)),
        ("34 nt prefix no-indel comparer vs 139 nt read",
         lambda: mojo_prefix.locate(prefix_read), lambda: upstream_prefix.locate(prefix_read)),
        ("34 nt suffix no-indel comparer vs 139 nt read",
         lambda: mojo_suffix.locate(suffix_read), lambda: upstream_suffix.locate(suffix_read)),
    ]
    print(f"Machine: {platform.processor() or platform.machine()}, Python {platform.python_version()}")
    print()
    print("| kernel | mojo-cutadapt | cutadapt 5.2 | relative |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours_fn, theirs_fn in cases:
        ours = measure(ours_fn)
        theirs = measure(theirs_fn)
        relation = f"{theirs / ours:.2f}x" if ours <= theirs else f"{ours / theirs:.2f}x slower"
        print(f"| {name} | {ours:.1f} us | {theirs:.1f} us | {relation} |")


if __name__ == "__main__":
    main()
