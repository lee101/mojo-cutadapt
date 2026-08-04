"""C ABI for Cutadapt's hybrid semiglobal alignment kernel."""

from std.sys.info import simd_width_of as simdwidthof

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int, AnyOrigin[mut=True]]


@export("mca_locate")
def mca_locate(
    reference_addr: Int,
    reference_length: Int,
    query_addr: Int,
    query_length: Int,
    max_error_rate: Float64,
    flags: Int,
    wildcard_ref: Int,
    wildcard_query: Int,
    indel_cost: Int,
    min_overlap: Int,
    costs_addr: Int,
    scores_addr: Int,
    origins_addr: Int,
    result_addr: Int,
) abi("C") -> Int:
    # This entry point is also callable without the Python wrapper. Check every
    # property we can validate before constructing or dereferencing a pointer.
    # The caller owns all buffers, so their capacities cannot be established
    # from a raw C address; the documented contract requires reference_length
    # + 1 Ints for each work buffer and 6 Ints for result.
    if reference_length < 0 or query_length < 0 or reference_addr <= 0 or result_addr <= 0:
        return -1
    if query_length > 0 and query_addr <= 0:
        return -1
    if costs_addr <= 0 or scores_addr <= 0 or origins_addr <= 0:
        return -1
    if max_error_rate < 0.0 or max_error_rate != max_error_rate or indel_cost <= 0 or min_overlap < 0:
        return -1
    if (flags & ~15) != 0 or wildcard_ref != 0 or wildcard_query != 0:
        return -1
    var reference = BPtr(unsafe_from_address=reference_addr)
    var query = BPtr(unsafe_from_address=query_addr)
    var costs = IPtr(unsafe_from_address=costs_addr)
    var scores = IPtr(unsafe_from_address=scores_addr)
    var origins = IPtr(unsafe_from_address=origins_addr)
    var result = IPtr(unsafe_from_address=result_addr)
    var start_in_reference = (flags & 1) != 0
    var start_in_query = (flags & 2) != 0
    var stop_in_reference = (flags & 4) != 0
    var stop_in_query = (flags & 8) != 0
    var k = Int(max_error_rate * Float64(reference_length))
    var max_n = query_length
    var min_n = 0
    if not start_in_query:
        max_n = reference_length + k
        if max_n > query_length:
            max_n = query_length
    if not stop_in_query:
        min_n = query_length - reference_length - k
        if min_n < 0:
            min_n = 0

    for i in range(reference_length + 1):
        if not start_in_reference and not start_in_query:
            scores[i] = -2 * i
            costs[i] = (i if i > min_n else min_n) * indel_cost
            origins[i] = 0
        elif start_in_reference and not start_in_query:
            scores[i] = 0
            costs[i] = min_n * indel_cost
            origins[i] = 0 if min_n >= i else min_n - i
        elif not start_in_reference and start_in_query:
            scores[i] = -2 * i
            costs[i] = i * indel_cost
            origins[i] = min_n - i if min_n >= i else 0
        else:
            scores[i] = 0
            costs[i] = (i if i < min_n else min_n) * indel_cost
            origins[i] = min_n - i

    var best_cost = reference_length + query_length + 1
    var best_score = 0
    var best_origin = 0
    var best_ref_stop = reference_length
    var best_query_stop = query_length
    var last = k + 1
    if last > reference_length:
        last = reference_length
    if start_in_reference:
        last = reference_length
    var last_filled = 0
    var stale_origin = 0

    for j in range(min_n + 1, max_n + 1):
        var diag_cost = costs[0]
        var diag_score = scores[0]
        var diag_origin = origins[0]
        if start_in_query:
            origins[0] += 1
        else:
            costs[0] += indel_cost
            scores[0] -= 2

        for i in range(1, last + 1):
            var old_cost = costs[i]
            var old_score = scores[i]
            var old_origin = origins[i]
            var is_match = reference[i - 1] == query[j - 1]
            var chosen_cost = 0
            var chosen_score = 0
            var chosen_origin = 0
            if is_match:
                chosen_cost = diag_cost
                chosen_score = diag_score + 1
                chosen_origin = diag_origin
            else:
                var cost_diag = diag_cost + 1
                var cost_insertion = old_cost + indel_cost
                var cost_deletion = costs[i - 1] + indel_cost
                if cost_diag <= cost_deletion and cost_diag <= cost_insertion:
                    chosen_cost = cost_diag
                    chosen_score = diag_score - 1
                    chosen_origin = diag_origin
                elif cost_deletion <= cost_insertion:
                    chosen_cost = cost_deletion
                    chosen_score = scores[i - 1] - 2
                    chosen_origin = origins[i - 1]
                else:
                    chosen_cost = cost_insertion
                    chosen_score = old_score - 2
                    chosen_origin = old_origin
            diag_cost = old_cost
            diag_score = old_score
            diag_origin = old_origin
            costs[i] = chosen_cost
            scores[i] = chosen_score
            origins[i] = chosen_origin
            stale_origin = chosen_origin

        last_filled = last
        while last >= 0 and costs[last] > k:
            last -= 1
        if last < reference_length:
            last += 1
        elif stop_in_query:
            var length = reference_length + (origins[reference_length] if origins[reference_length] < 0 else 0)
            var cost = costs[reference_length]
            var score = scores[reference_length]
            var origin = origins[reference_length]
            var acceptable = length >= min_overlap and Float64(cost) <= Float64(length) * max_error_rate
            var best_length = reference_length + (best_origin if best_origin < 0 else 0)
            if acceptable and (best_cost == reference_length + query_length + 1 or (origin <= best_origin + reference_length // 2 and score > best_score) or (length > best_length and score > best_score)):
                best_score = score
                best_cost = cost
                best_origin = origin
                best_ref_stop = reference_length
                best_query_stop = j
                if cost == 0 and origin >= 0:
                    break

    if max_n == query_length:
        var first_i = 0 if stop_in_reference else reference_length
        for i in range(first_i, last_filled + 1):
            var reverse_i = last_filled - (i - first_i)
            var length = reverse_i + (origins[reverse_i] if origins[reverse_i] < 0 else 0)
            var cost = costs[reverse_i]
            var score = scores[reverse_i]
            var acceptable = length >= min_overlap and Float64(cost) <= Float64(length) * max_error_rate
            var best_length = best_ref_stop + (best_origin if best_origin < 0 else 0)
            if acceptable and (best_cost == reference_length + query_length + 1 or (stale_origin <= best_origin + reference_length // 2 and score > best_score) or (length > best_length and score > best_score)):
                best_score = score
                best_cost = cost
                best_origin = origins[reverse_i]
                best_ref_stop = reverse_i
                best_query_stop = query_length

    if best_cost == reference_length + query_length + 1:
        return 0
    if best_origin >= 0:
        result[0] = 0
        result[2] = best_origin
    else:
        result[0] = -best_origin
        result[2] = 0
    result[1] = best_ref_stop
    result[3] = best_query_stop
    result[4] = best_score
    result[5] = best_cost
    return 1


@export("mca_hamming")
def mca_hamming(
    reference_addr: Int,
    reference_length: Int,
    query_addr: Int,
    query_length: Int,
    max_error_rate: Float64,
    suffix: Int,
    min_overlap: Int,
    result_addr: Int,
) abi("C") -> Int:
    if reference_length < 0 or query_length < 0 or reference_addr <= 0 or result_addr <= 0:
        return -1
    if query_length > 0 and query_addr <= 0:
        return -1
    if max_error_rate < 0.0 or max_error_rate != max_error_rate or min_overlap < 0:
        return -1
    var reference = BPtr(unsafe_from_address=reference_addr)
    var query = BPtr(unsafe_from_address=query_addr)
    var result = IPtr(unsafe_from_address=result_addr)
    var length = reference_length if reference_length < query_length else query_length
    if length < min_overlap:
        return 0
    var reference_start = 0
    var query_start = 0
    if suffix != 0:
        reference_start = reference_length - length
        query_start = query_length - length
    comptime W = simdwidthof[DType.float64]()
    var max_errors = Int(Float64(reference_length) * max_error_rate)
    var errors = 0
    var i = 0
    var one = SIMD[DType.uint8, W](UInt8(1))
    var zero = SIMD[DType.uint8, W](UInt8(0))
    while i + W <= length:
        var matches = reference.load[width=W, alignment=1](reference_start + i).eq(
            query.load[width=W, alignment=1](query_start + i)
        )
        if not matches.reduce_and():
            errors += Int(matches.select(zero, one).reduce_add()[0])
            if errors > max_errors:
                return 0
        i += W
    while i < length:
        if reference[reference_start + i] != query[query_start + i]:
            errors += 1
            if errors > max_errors:
                return 0
        i += 1
    result[0] = reference_start
    result[1] = reference_start + length
    result[2] = query_start
    result[3] = query_start + length
    result[4] = length - 2 * errors
    result[5] = errors
    return 1
