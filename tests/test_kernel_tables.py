"""Differential checks against a Python transcription of the kernel.

The parity suite compares answers with Cutadapt. These tests compare the DP
state itself, so a wrong recurrence that still lands on the same answer in the
sampled cases cannot hide: every row of the cost/score/origin column is checked
against an independent transcription for every flag combination and every
comparer layout.
"""

from __future__ import annotations

import ctypes
import random

import numpy as np
import pytest

from mojocutadapt import _lib
from mojocutadapt.align import PrefixComparer, SuffixComparer

from _reference import dp_tables

I = _lib.I
F = _lib.F

LENGTHS = [0, 1, 3, 4, 5, 31, 32, 33]


@pytest.fixture(scope="module")
def raw_locate():
    lib = _lib.lib()
    fn = lib.mca_locate
    fn.argtypes = [I, I, I, I, F, I, I, I, I, I, I, I, I, I]
    fn.restype = I
    return fn


def _workspace(reference: str):
    return _lib.Workspace(reference.encode())


def _call_locate(fn, reference: str, query: str, rate: float, flags: int, indel: int, overlap: int):
    ws = _workspace(reference)
    q = query.encode()
    found = fn(
        ws.reference_addr, len(reference),
        ctypes.cast(ctypes.c_char_p(q), ctypes.c_void_p).value, len(q),
        rate, flags, 0, 0, indel, overlap,
        ws.costs_addr, ws.scores_addr, ws.origins_addr, ws.result_addr,
    )
    return found, ws


@pytest.mark.parametrize("flags", range(16))
@pytest.mark.parametrize("length", LENGTHS)
def test_dp_tables_match_reference(raw_locate, flags, length):
    rng = random.Random(flags * 1000 + length)
    # Sanitising an empty reference would make the kernel reject the call on
    # argument grounds before touching the tables, so skip that length here.
    if length == 0:
        pytest.skip("empty reference is rejected before the row loop")
    reference = "".join(rng.choice("ACGT") for _ in range(length))
    query = "".join(rng.choice("ACGT") for _ in range(length + rng.randrange(0, 5)))
    for rate, indel, overlap in [(0.0, 1, 1), (0.25, 2, 1), (0.5, 1, 2), (1.0, 5, 0)]:
        found, ws = _call_locate(raw_locate, reference, query, rate, flags, indel, overlap)
        costs, scores, origins, _, _ = dp_tables(reference, query, rate, flags, indel, overlap)
        # The kernel leaves the column as the row loop left it; a rejected call
        # still writes it, so compare the tables either way.
        assert ws.costs.tolist()[: len(reference) + 1] == costs, (flags, length, rate, indel, overlap)
        assert ws.scores.tolist()[: len(reference) + 1] == scores
        assert ws.origins.tolist()[: len(reference) + 1] == origins


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("mismatches", [1, 2])
def test_hamming_counts_match_reference(length, mismatches):
    """Every comparer layout must count exactly, including the scalar tails."""
    if length < 1:
        pytest.skip("nothing to compare")
    reference = ("ACGTTGCAAGCT" * 8)[:length]
    query = list(reference)
    for position in [length - 1 - k for k in range(min(mismatches, length))]:
        query[position] = "A" if query[position] != "A" else "C"
    query = "".join(query)
    expected_errors = sum(a != b for a, b in zip(reference, query))
    # rate chosen to admit the count, min_overlap small enough to reach the kernel
    for rate in (0.0, 0.5, 1.0):
        limit = int(len(reference) * rate)
        for ours in (PrefixComparer, SuffixComparer):
            result = ours(reference, rate, min_overlap=1).locate(query)
            if expected_errors > limit:
                assert result is None
            else:
                assert result is not None
                assert result[5] == expected_errors, (ours.__name__, length, rate)
                assert result[4] == length - 2 * expected_errors
