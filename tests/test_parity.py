import random
import math

import pytest
from cutadapt._align import Aligner as UpstreamAligner
from cutadapt._align import PrefixComparer as UpstreamPrefixComparer
from cutadapt._align import SuffixComparer as UpstreamSuffixComparer
from cutadapt.adapters import AnywhereAdapter as UpstreamAnywhereAdapter
from cutadapt.adapters import BackAdapter as UpstreamBackAdapter
from cutadapt.adapters import FrontAdapter as UpstreamFrontAdapter
from cutadapt.adapters import PrefixAdapter as UpstreamPrefixAdapter
from cutadapt.adapters import SuffixAdapter as UpstreamSuffixAdapter

from mojocutadapt import Aligner, EndSkip, PrefixComparer, SuffixComparer
from mojocutadapt.adapters import AnywhereAdapter, BackAdapter, FrontAdapter, PrefixAdapter, SuffixAdapter
from mojocutadapt import _lib


@pytest.mark.parametrize("flags", [0, 2, 4, 6, 8, 9, 11, 14, 15])
@pytest.mark.parametrize("reference,query,rate,min_overlap", [
    ("ACGT", "TTACGA", 0.5, 1),
    ("ACGT", "ACGTT", 0.5, 1),
    ("TTTACG", "CTCCCGTCACTTGCGTTT", 0.5, 1),
    ("GAGTGTATCCC", "GGTAGCGCAC", 1.0, 1),
    ("ACGT", "", 1.0, 1),
])
def test_alignment_vectors_match_upstream(flags, reference, query, rate, min_overlap):
    assert Aligner(reference, rate, flags, min_overlap=min_overlap).locate(query) == UpstreamAligner(reference, rate, flags, min_overlap=min_overlap).locate(query)


def test_random_nonwildcard_alignment_parity():
    rng = random.Random(31)
    for _ in range(1500):
        reference = "".join(rng.choice("ACGT") for _ in range(rng.randrange(1, 18)))
        query = "".join(rng.choice("ACGT") for _ in range(rng.randrange(0, 28)))
        flags = rng.randrange(16)
        rate = rng.choice([0.0, 0.1, 0.25, 0.5, 1.0])
        indel_cost = rng.choice([1, 2, 5])
        overlap = rng.randrange(1, len(reference) + 1)
        actual = Aligner(reference, rate, flags, indel_cost=indel_cost, min_overlap=overlap).locate(query)
        expected = UpstreamAligner(reference, rate, flags, indel_cost=indel_cost, min_overlap=overlap).locate(query)
        assert actual == expected


# Lengths chosen so every SIMD path runs its zero, scalar-only, one-lane-short,
# exactly-one-vector, one-lane-over, and long odd-tail cases. W is 4 for the
# int64 DP buffers and the float64-width u8 Hamming loads on the pinned build.
SIMD_TAIL_LENGTHS = [0, 1, 3, 4, 5, 33]


@pytest.mark.parametrize("flags", [0, 1, 2, 3, 4, 8, 12, 15])
@pytest.mark.parametrize("length", SIMD_TAIL_LENGTHS)
def test_simd_initialization_tail_lengths_match_upstream(flags, length):
    reference = ("ACGTTGCAAGCT" * 4)[:length]
    query = ("TTACGTTGCAAGCTTACGA" * 2)[: max(0, length - 1)]
    for rate, overlap in [(0.25, 1), (0.5, max(1, length // 2)), (1.0, 0)]:
        actual = Aligner(reference, rate, flags, min_overlap=overlap).locate(query)
        expected = UpstreamAligner(reference, rate, flags, min_overlap=overlap).locate(query)
        assert actual == expected


@pytest.mark.parametrize("flags", [0, 1, 2, 3])
def test_simd_initialization_tail_matches_upstream(flags):
    reference = "ACGTACGT"
    query = "TTACGTACGTT"
    actual = Aligner(reference, 0.25, flags, min_overlap=3).locate(query)
    expected = UpstreamAligner(reference, 0.25, flags, min_overlap=3).locate(query)
    assert actual == expected


@pytest.mark.parametrize("length", SIMD_TAIL_LENGTHS)
@pytest.mark.parametrize("mismatches", [1, 2])
def test_simd_hamming_tail_lengths(length, mismatches):
    if length < 2:
        pytest.skip("no tail to mismatch in")
    reference = ("ACGTTGCAAGCT" * 4)[:length]
    # Put the mismatches past the last whole vector so the scalar tail counts
    # them; the suffixes then differ, which is what reaches the kernel instead
    # of taking the exact-match shortcut in Python.
    positions = [length - 1 - k for k in range(min(mismatches, length))]
    query = list(reference)
    for position in positions:
        query[position] = "A" if query[position] != "A" else "C"
    query = "".join(query)
    for ours, upstream in [(PrefixComparer, UpstreamPrefixComparer),
                           (SuffixComparer, UpstreamSuffixComparer)]:
        for rate in (0.0, 0.5, 1.0):
            actual = ours(reference, rate, min_overlap=1).locate(query)
            expected = upstream(reference, rate, min_overlap=1).locate(query)
            assert actual == expected
    # A query longer than the reference exercises the suffix offset
    # (reference_start = reference_length - length) with a real tail.
    longer = "TT" + query
    for rate in (0.0, 0.5, 1.0):
        actual = SuffixComparer(reference, rate, min_overlap=1).locate(longer)
        expected = UpstreamSuffixComparer(reference, rate, min_overlap=1).locate(longer)
        assert actual == expected


@pytest.mark.parametrize("ours,upstream,query", [
    (PrefixComparer, UpstreamPrefixComparer, "ACGTACGTT"),
    (SuffixComparer, UpstreamSuffixComparer, "TTACGTACGTA"),
])
@pytest.mark.parametrize("reference", ["ACGTACG", "ACGTACGTA"])
def test_simd_hamming_tail(ours, upstream, query, reference):
    actual = ours(reference, 0.5, min_overlap=3).locate(query)
    expected = upstream(reference, 0.5, min_overlap=3).locate(query)
    assert actual == expected


@pytest.mark.parametrize("reference,query,flags", [
    ("ATNG", "ATCGTT", EndSkip.SEMIGLOBAL),
    ("ARCG", "AGCG", EndSkip.SEMIGLOBAL),
    ("NNAC", "TTAC", EndSkip.QUERY_START | EndSkip.QUERY_STOP | EndSkip.REFERENCE_END),
])
def test_iupac_wildcard_parity(reference, query, flags):
    actual = Aligner(reference, 0.5, flags, wildcard_ref=True, min_overlap=1).locate(query)
    expected = UpstreamAligner(reference, 0.5, flags, wildcard_ref=True, min_overlap=1).locate(query)
    assert actual == expected


@pytest.mark.parametrize("ours,upstream", [
    (PrefixComparer, UpstreamPrefixComparer),
    (SuffixComparer, UpstreamSuffixComparer),
])
def test_hamming_comparers_match_upstream(ours, upstream):
    for reference, query in [("ACGT", "ACCTGG"), ("ACGT", "TTACGT"), ("ATNG", "ATCG")]:
        for wildcards in (False, True):
            actual = ours(reference, 0.5, wildcard_ref=wildcards, min_overlap=1).locate(query)
            expected = upstream(reference, 0.5, wildcard_ref=wildcards, min_overlap=1).locate(query)
            assert actual == expected


def test_random_hamming_comparer_parity():
    rng = random.Random(847)
    for _ in range(1000):
        reference = "".join(rng.choice("ACGT") for _ in range(rng.randrange(1, 25)))
        query = "".join(rng.choice("ACGT") for _ in range(rng.randrange(0, 31)))
        rate = rng.choice([0.0, 0.1, 0.25, 0.5, 1.0])
        overlap = rng.randrange(1, len(reference) + 1)
        for ours, upstream in [(PrefixComparer, UpstreamPrefixComparer), (SuffixComparer, UpstreamSuffixComparer)]:
            actual = ours(reference, rate, min_overlap=overlap).locate(query)
            expected = upstream(reference, rate, min_overlap=overlap).locate(query)
            assert actual == expected


@pytest.mark.parametrize("ours,upstream,sequence,read,kwargs", [
    (FrontAdapter, UpstreamFrontAdapter, "ACGT", "ACGTTTAG", {}),
    (BackAdapter, UpstreamBackAdapter, "ACGT", "TTAGACGT", {}),
    (AnywhereAdapter, UpstreamAnywhereAdapter, "ACGT", "TTACGTAA", {}),
    (PrefixAdapter, UpstreamPrefixAdapter, "ACGT", "ACCTTAG", {"indels": False, "max_errors": 0.5}),
    (SuffixAdapter, UpstreamSuffixAdapter, "ACGT", "TTAGACCT", {"indels": False, "max_errors": 0.5}),
    (BackAdapter, UpstreamBackAdapter, "ATNG", "GGATCG", {"max_errors": 0.25}),
])
def test_adapter_match_and_trim_parity(ours, upstream, sequence, read, kwargs):
    mine = ours(sequence, name="adapter", **kwargs).match_to(read)
    theirs = upstream(sequence, name="adapter", **kwargs).match_to(read)
    assert (mine is None) == (theirs is None)
    if mine is not None:
        assert (mine.astart, mine.astop, mine.rstart, mine.rstop, mine.score, mine.errors) == (theirs.astart, theirs.astop, theirs.rstart, theirs.rstop, theirs.score, theirs.errors)
        assert read[mine.trim_slice()] == read[theirs.trim_slice()]


@pytest.mark.parametrize("kwargs", [
    {"max_error_rate": -0.1},
    {"max_error_rate": math.nan},
    {"max_error_rate": math.inf},
    {"flags": 16},
    {"indel_cost": 0},
    {"min_overlap": -1},
])
def test_aligner_rejects_invalid_ffi_parameters(kwargs):
    with pytest.raises(ValueError):
        Aligner("ACGT", kwargs.pop("max_error_rate", 0.1), **kwargs)


def test_hamming_comparer_uses_simd_path_for_long_mismatching_reads():
    """A long read with N mismatches reports exactly N and prefers the longest."""
    adapter = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC"
    read = "GATTACA" * 19 + adapter[:3]
    for position in (0, 7, len(read) - 1):
        mutated = list(read)
        mutated[position] = "A" if mutated[position] != "A" else "C"
        mutated = "".join(mutated)
        for ours, upstream in [(PrefixComparer, UpstreamPrefixComparer),
                               (SuffixComparer, UpstreamSuffixComparer)]:
            actual = ours(adapter, 1.0, min_overlap=1).locate(mutated)
            expected = upstream(adapter, 1.0, min_overlap=1).locate(mutated)
            # The kernel is reached (the read is long and mismatching), and its
            # answer must match the reference implementation exactly.
            assert actual == expected
            assert actual is not None

    # Longer read than adapter, suffix layout: reference is offset to its end.
    longer = read + adapter
    for ours, upstream in [(PrefixComparer, UpstreamPrefixComparer),
                           (SuffixComparer, UpstreamSuffixComparer)]:
        assert ours(adapter, 0.5, min_overlap=1).locate(longer) == \
            upstream(adapter, 0.5, min_overlap=1).locate(longer)


def test_aligner_rejects_non_ascii_before_the_ffi_boundary():
    with pytest.raises(ValueError, match="ASCII"):
        Aligner("ACGé", 0.1)
    aligner = Aligner("ACGT", 0.1)
    with pytest.raises(ValueError, match="ASCII"):
        aligner.locate("ACGé")


def test_c_abi_rejects_invalid_raw_arguments_without_dereference():
    # Raw addresses are deliberately zero. A negative status demonstrates that
    # mca_locate validates them before turning them into Mojo pointers.
    assert _lib.lib().mca_locate(*([0] * 14)) == -1
    assert _lib.lib().mca_hamming(*([0] * 8)) == -1
