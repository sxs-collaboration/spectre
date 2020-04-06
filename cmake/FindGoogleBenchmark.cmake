# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find GOOGLE_BENCHMARK: https://github.com/google/benchmark
# If not in one of the default paths specify
# -DGOOGLE_BENCHMARK_ROOT=/path/to/GOOGLE_BENCHMARK to search there as well.

if(NOT GOOGLE_BENCHMARK_ROOT)
  # Need to set to empty to avoid warnings with --warn-uninitialized
  set(GOOGLE_BENCHMARK_ROOT "")
  set(GOOGLE_BENCHMARK_ROOT $ENV{GOOGLE_BENCHMARK_ROOT})
endif()

if(NOT GoogleBenchmark_ROOT)
  # Need to set to empty to avoid warnings with --warn-uninitialized
  set(GoogleBenchmark_ROOT "")
  set(GoogleBenchmark_ROOT $ENV{GoogleBenchmark_ROOT})
endif()

find_path(GoogleBenchmark_INCLUDE_DIRS benchmark.h
    PATH_SUFFIXES include/benchmark
    HINTS ${GOOGLE_BENCHMARK_ROOT})

find_library(GoogleBenchmark_LIBRARIES
    NAMES benchmark
    PATH_SUFFIXES lib64 lib
    HINTS ${GOOGLE_BENCHMARK_ROOT} ${GoogleBenchmark_ROOT})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GoogleBenchmark
  FOUND_VAR GoogleBenchmark_FOUND
  REQUIRED_VARS GoogleBenchmark_INCLUDE_DIRS GoogleBenchmark_LIBRARIES)
mark_as_advanced(GoogleBenchmark_INCLUDE_DIRS GoogleBenchmark_LIBRARIES)
