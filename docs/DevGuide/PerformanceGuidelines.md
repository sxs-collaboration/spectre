\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# General Performance Guidelines {#general_perf_guide}

\tableofcontents

Below are some general guidelines for achieving decent performance.

- One good measurement is worth more than a million expert opinions. We have a
  `Benchmark` executable that uses Google Benchmark so one can compare different
  implementations and see how they perform. This executable is only available in
  release builds.
- Reduce memory allocations. On all modern hardware (many core CPUs, GPUs, and
  FPGAs), memory is almost always the bottleneck. Memory allocations are
  especially expensive since this is a quasi-serial process: the OS has to
  manage memory allocations for _all_ running threads and processes. SpECTRE has
  various classes to optimize this. For example, there are `Variables`,
  `TempBuffer`, and `DynamicBuffer` that allow making large contiguous memory
  allocations that are then used for individual tensor components.
