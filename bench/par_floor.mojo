"""Measure what one `max.algorithm.parallelize` call costs on this host.

Built as its own shared library so the shipped kernel is untouched and the
package itself needs no `max` dependency.

Two workloads per worker count:

* `empty` - one store per work item: the launch and join cost alone.
* `unit`  - each worker runs a fixed CPU-bound loop (~1 ms); the reported time
            is what a perfectly parallel kernel would pay for that work.

Reported as the median of the last `reps` calls, after two warm-up calls, so a
one-off lazy thread-pool creation is not counted as per-call cost.

Building it needs the `max` package (MOJO_NOTES.md section 5b):

    pixi add 'max==26.7.0.dev2026092605'
    pixi run mojo build --emit shared-lib bench/par_floor.mojo -o dist/libparallel-floor.so
"""
from max.algorithm import parallelize
from std.time import perf_counter_ns

comptime Loops = 1_000_000


@export("par_floor")
def par_floor(workers: Int, reps: Int, out_addr: Int) abi("C") -> Int:
    if workers < 1 or workers > 16 or reps < 1 or out_addr <= 0:
        return -1
    var out = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=out_addr)
    var stride = 32

    def store_work(i: Int) {imm}:
        # A dedicated region: the empty workload must not overwrite the timings.
        out.unsafe_store(4 * stride + i, i)

    def unit_work(i: Int) {imm}:
        var x = 0.0
        for k in range(Loops):
            x += Float64(k) * 0.5 + Float64(i)
        if x < -1.0:
            print(x)

    # Empty: launch + join only.
    for k in range(2):
        parallelize(store_work, 8, workers)
    for k in range(reps):
        var t0 = perf_counter_ns()
        parallelize(store_work, 8, workers)
        var t1 = perf_counter_ns()
        out.unsafe_store(k, Int(t1 - t0))

    # One unit per worker: what a genuinely parallel kernel would pay.
    for k in range(2):
        parallelize(unit_work, workers, workers)
    for k in range(reps):
        var t0 = perf_counter_ns()
        parallelize(unit_work, workers, workers)
        var t1 = perf_counter_ns()
        out.unsafe_store(stride + k, Int(t1 - t0))

    # The same units serially, for the speedup denominator.
    var t2 = perf_counter_ns()
    for k in range(workers):
        unit_work(k)
    var t3 = perf_counter_ns()
    out.unsafe_store(2 * stride - 1, Int(t3 - t2))
    return 0
