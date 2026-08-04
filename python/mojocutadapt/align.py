"""Cutadapt-compatible adapter alignment backed by Mojo."""

from __future__ import annotations

from enum import IntFlag
import math
from typing import Optional

from . import _lib


class EndSkip(IntFlag):
    REFERENCE_START = 1
    QUERY_START = 2
    REFERENCE_END = 4
    QUERY_STOP = 8
    SEMIGLOBAL = 15


_IUPAC = {
    "A": 1, "C": 2, "G": 4, "T": 8, "U": 8, "R": 5, "Y": 10,
    "S": 6, "W": 9, "K": 12, "M": 3, "B": 14, "D": 13, "H": 11,
    "V": 7, "N": 15, "X": 15,
}


def _equal(a: str, b: str, wildcard_ref: bool, wildcard_query: bool) -> bool:
    if not (wildcard_ref or wildcard_query):
        return a == b
    left = _IUPAC.get(a, 0) if wildcard_ref else _IUPAC.get(a, 0) if a in "ACGTU" else 0
    right = _IUPAC.get(b, 0) if wildcard_query else _IUPAC.get(b, 0) if b in "ACGTU" else 0
    return bool(left & right)


def _python_locate(reference: str, query: str, rate: float, flags: int, wildcard_ref: bool,
                   wildcard_query: bool, indel_cost: int, min_overlap: int):
    m, n = len(reference), len(query)
    start_ref, start_query = bool(flags & 1), bool(flags & 2)
    stop_ref, stop_query = bool(flags & 4), bool(flags & 8)
    k = int(rate * m)
    max_n = n if start_query else min(n, m + k)
    min_n = 0 if stop_query else max(0, n - m - k)
    costs, scores, origins = [0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(m + 1):
        if not start_ref and not start_query:
            costs[i], scores[i], origins[i] = max(i, min_n) * indel_cost, -2 * i, 0
        elif start_ref and not start_query:
            costs[i], origins[i] = min_n * indel_cost, min(0, min_n - i)
        elif not start_ref and start_query:
            costs[i], scores[i], origins[i] = i * indel_cost, -2 * i, max(0, min_n - i)
        else:
            costs[i], origins[i] = min(i, min_n) * indel_cost, min_n - i
    sentinel = m + n + 1
    best_cost, best_score, best_origin, best_ref_stop, best_query_stop = sentinel, 0, 0, m, n
    last = m if start_ref else min(m, k + 1)
    last_filled = stale_origin = 0
    def effective(stop: int, origin: int) -> int:
        length = stop + min(origin, 0)
        return length - reference[stop - length:stop].count("N") if wildcard_ref else length
    for j in range(min_n + 1, max_n + 1):
        diag_cost, diag_score, diag_origin = costs[0], scores[0], origins[0]
        if start_query:
            origins[0] += 1
        else:
            costs[0], scores[0] = costs[0] + indel_cost, scores[0] - 2
        for i in range(1, last + 1):
            old_cost, old_score, old_origin = costs[i], scores[i], origins[i]
            if _equal(reference[i - 1], query[j - 1], wildcard_ref, wildcard_query):
                cost, score, origin = diag_cost, diag_score + 1, diag_origin
            else:
                cd, ci, cdel = diag_cost + 1, old_cost + indel_cost, costs[i - 1] + indel_cost
                if cd <= cdel and cd <= ci:
                    cost, score, origin = cd, diag_score - 1, diag_origin
                elif cdel <= ci:
                    cost, score, origin = cdel, scores[i - 1] - 2, origins[i - 1]
                else:
                    cost, score, origin = ci, old_score - 2, old_origin
            diag_cost, diag_score, diag_origin = old_cost, old_score, old_origin
            costs[i], scores[i], origins[i] = cost, score, origin
            stale_origin = origin
        last_filled = last
        while last >= 0 and costs[last] > k:
            last -= 1
        if last < m:
            last += 1
        elif stop_query:
            cost, score, origin = costs[m], scores[m], origins[m]
            length, current_effective = m + min(origin, 0), effective(m, origin)
            best_length = m + min(best_origin, 0)
            if length >= min_overlap and cost <= current_effective * rate and (best_cost == sentinel or (origin <= best_origin + m // 2 and score > best_score) or (length > best_length and score > best_score)):
                best_cost, best_score, best_origin, best_ref_stop, best_query_stop = cost, score, origin, m, j
                if cost == 0 and origin >= 0:
                    break
    if max_n == n:
        for i in reversed(range(0 if stop_ref else m, last_filled + 1)):
            cost, score, origin = costs[i], scores[i], origins[i]
            length, current_effective = i + min(origin, 0), effective(i, origin)
            best_length = best_ref_stop + min(best_origin, 0)
            if length >= min_overlap and cost <= current_effective * rate and (best_cost == sentinel or (stale_origin <= best_origin + m // 2 and score > best_score) or (length > best_length and score > best_score)):
                best_cost, best_score, best_origin, best_ref_stop, best_query_stop = cost, score, origin, i, n
    if best_cost == sentinel:
        return None
    return (0, best_ref_stop, best_origin, best_query_stop, best_score, best_cost) if best_origin >= 0 else (-best_origin, best_ref_stop, 0, best_query_stop, best_score, best_cost)


class Aligner:
    def __init__(self, reference: str, max_error_rate: float, flags: int = 15,
                 wildcard_ref: bool = False, wildcard_query: bool = False,
                 indel_cost: int = 1, min_overlap: int = 1):
        if not isinstance(reference, str):
            raise TypeError("reference must be a str")
        if not isinstance(max_error_rate, (int, float)) or not math.isfinite(max_error_rate) or max_error_rate < 0:
            raise ValueError("max_error_rate must be a finite non-negative number")
        if not isinstance(flags, (int, IntFlag)) or int(flags) & ~0xF:
            raise ValueError("flags must use only EndSkip bits")
        if not isinstance(indel_cost, int) or indel_cost <= 0:
            raise ValueError("indel_cost must be a positive integer")
        if not isinstance(min_overlap, int) or min_overlap < 0:
            raise ValueError("min_overlap must be a non-negative integer")
        self.reference = reference.upper()
        self.max_error_rate = max_error_rate
        self.flags = int(flags)
        self.wildcard_ref = wildcard_ref
        self.wildcard_query = wildcard_query
        self.indel_cost = indel_cost
        self.min_overlap = min_overlap
        try:
            self._reference_bytes = self.reference.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("reference must contain only ASCII characters") from error
        self._workspace = _lib.Workspace(self._reference_bytes)

    @property
    def effective_length(self) -> int:
        return len(self.reference) - self.reference.count("N") if self.wildcard_ref else len(self.reference)

    def __repr__(self) -> str:
        return (f"Aligner(reference={self.reference!r}, max_error_rate={self.max_error_rate}, "
                f"flags={self.flags}, wildcard_ref={self.wildcard_ref}, "
                f"wildcard_query={self.wildcard_query}, indel_cost={self.indel_cost}, "
                f"min_overlap={self.min_overlap})")

    def enable_debug(self) -> None:
        return None

    def locate(self, query: str) -> Optional[tuple[int, int, int, int, int, int]]:
        if not isinstance(query, str):
            raise TypeError("query must be a str")
        query = query.upper()
        if self.wildcard_ref or self.wildcard_query:
            return _python_locate(self.reference, query, self.max_error_rate, self.flags,
                                  self.wildcard_ref, self.wildcard_query, self.indel_cost, self.min_overlap)
        try:
            query_bytes = query.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("query must contain only ASCII characters") from error
        return _lib.locate(query_bytes, self.max_error_rate,
                           self.flags, self.indel_cost, self.min_overlap, self._workspace)


class PrefixComparer(Aligner):
    def __init__(self, reference: str, max_error_rate: float, wildcard_ref: bool = False,
                 wildcard_query: bool = False, min_overlap: int = 1):
        super().__init__(reference, max_error_rate, EndSkip.QUERY_STOP, wildcard_ref,
                         wildcard_query, 1_000_000, min_overlap)

    def locate(self, query: str) -> Optional[tuple[int, int, int, int, int, int]]:
        if not isinstance(query, str):
            raise TypeError("query must be a str")
        query = query.upper()
        if self.wildcard_ref or self.wildcard_query:
            return Aligner.locate(self, query)
        try:
            query_bytes = query.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("query must contain only ASCII characters") from error
        return _lib.hamming(query_bytes, self.max_error_rate,
                            False, self.min_overlap, self._workspace)


class SuffixComparer(PrefixComparer):
    def __init__(self, reference: str, max_error_rate: float, wildcard_ref: bool = False,
                 wildcard_query: bool = False, min_overlap: int = 1):
        Aligner.__init__(self, reference, max_error_rate, EndSkip.QUERY_START, wildcard_ref,
                         wildcard_query, 1_000_000, min_overlap)

    def locate(self, query: str) -> Optional[tuple[int, int, int, int, int, int]]:
        if not isinstance(query, str):
            raise TypeError("query must be a str")
        query = query.upper()
        if self.wildcard_ref or self.wildcard_query:
            return Aligner.locate(self, query)
        try:
            query_bytes = query.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("query must contain only ASCII characters") from error
        return _lib.hamming(query_bytes, self.max_error_rate,
                            True, self.min_overlap, self._workspace)
