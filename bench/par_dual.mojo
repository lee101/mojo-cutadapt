"""Same parallelize workload, built two ways, to locate where a cost comes from.

Built as an executable it costs ~30 us per `parallelize` call; built as a shared
library it costs ~5 ms. `bench/par_dual.py` drives both and checks whether the
difference is the library build or Python hosting the call.

    pixi run mojo build                    bench/par_dual.mojo -o /tmp/par_dual
    pixi run mojo build --emit shared-lib  bench/par_dual.mojo -o dist/libpar-dual.so
    pixi run python bench/par_dual.py
"""
from max.algorithm import parallelize
from std.memory import alloc
from std.time import perf_counter_ns

comptime Reps = 9


def run(workers: Int, buf_addr: Int) -> Int:
    var buf = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=buf_addr)

    def work(i: Int) {imm}:
        buf.unsafe_store(i, i)

    for k in range(2):
        parallelize(work, 8, workers)
    for k in range(Reps):
        var t0 = perf_counter_ns()
        parallelize(work, 8, workers)
        var t1 = perf_counter_ns()
        # Results land at 5.. so slot 0 (written by `work`) cannot clobber them.
        buf.unsafe_store(5 + k, Int(t1 - t0))
    return 0


@export("par_dual")
def par_dual(workers: Int, buf_addr: Int) abi("C") -> Int:
    if workers < 1 or buf_addr <= 0:
        return -1
    return run(workers, buf_addr)


def main():
    var raw = alloc[Int](64)
    for w in [1, 2, 4, 8]:
        run(w, Int(UnsafePointer(raw)))
        var line = String("workers ") + String(w) + ":"
        for k in range(5, 5 + Reps):
            line += " " + String(raw.unsafe_load(k) // 1000)
        print(line)
