\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# General Performance Guidelines {#general_perf_guide}

\tableofcontents

Below are some general guidelines for achieving decent performance.

- One good measurement is worth more than a million expert opinions.
  Our testing framework [Catch2](https://github.com/catchorg/Catch2) supports
  benchmarks, so we encourage you to add benchmarks to your tests. See the
  [Catch2 benchmarks documentation](https://github.com/catchorg/Catch2/blob/devel/docs/benchmarks.md)
  for instructions. Essentially, add a `BENCHMARK` to your test case and run
  the test executable (such as `./bin/Test_LinearOperators`).
  Note that we skip benchmarks during automated unit testing with `ctest`
  because benchmarks are only meaningful in a controlled environment (such as a
  specific machine or architecture). You can keep track of the benchmark results
  you ran on specific machines in a comment in the test case (until we have a
  better way of keeping track of benchmark results).

  Catch2's benchmarking is not as feature-rich as Google Benchmark. We have a
  `Benchmark` executable that uses Google Benchmark so one can compare
  different implementations and see how they perform. This executable is only
  available in release builds.
- Reduce memory allocations. On all modern hardware (many core CPUs, GPUs, and
  FPGAs), memory is almost always the bottleneck. Memory allocations are
  especially expensive since this is a quasi-serial process: the OS has to
  manage memory allocations for _all_ running threads and processes. SpECTRE has
  various classes to optimize this. For example, there are `Variables`,
  `TempBuffer`, and `DynamicBuffer` that allow making large contiguous memory
  allocations that are then used for individual tensor components.
