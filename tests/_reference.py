"""Reference-quality Python DP for differential testing only.

This is a transcription of the cell recurrence, kept independent of the Mojo
kernel so a test can compare counts and tables rather than just the final
answer. It is deliberately slow; it is not imported by the package.

NOTE: this models the *current* fused kernel. The shipped kernel is not the
classic Cutadapt semiglobal recurrence: its column-acceptance term compares
every column against one constant origin (`stale_origin`), copied from the
frozen previous kernel. Any differential test must reproduce that or it will
report the kernel as wrong where it is merely frozen. See bench/analysis.md.
"""

from __future__ import annotations


def dp_tables(reference: str, query: str, rate: float, flags: int, indel_cost: int, min_overlap: int):
    """Return (costs, scores, origins) after the row loop, for differential checks."""
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
    last = m if start_ref else min(m, k + 1)
    stale_origin = 0
    last_filled = 0
    for j in range(min_n + 1, max_n + 1):
        diag_cost, diag_score, diag_origin = costs[0], scores[0], origins[0]
        if start_query:
            origins[0] += 1
        else:
            costs[0], scores[0] = costs[0] + indel_cost, scores[0] - 2
        for i in range(1, last + 1):
            old_cost, old_score, old_origin = costs[i], scores[i], origins[i]
            if reference[i - 1] == query[j - 1]:
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
    return costs, scores, origins, last_filled, stale_origin
