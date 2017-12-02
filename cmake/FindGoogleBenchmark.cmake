# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find GOOGLE_BENCHMARK: https://github.com/google/benchmark
# If not in one of the default paths specify
# -DGOOGLE_BENCHMARK_ROOT=/path/to/GOOGLE_BENCHMARK to search there as well.

find_path(GOOGLE_BENCHMARK_INCLUDE_DIRS benchmark.h
    PATH_SUFFIXES include/benchmark
    HINTS ${GOOGLE_BENCHMARK_ROOT})

find_library(GOOGLE_BENCHMARK_LIBRARIES
    NAMES benchmark
    PATH_SUFFIXES lib64 lib
    HINTS ${GOOGLE_BENCHMARK_ROOT})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GOOGLE_BENCHMARK
    DEFAULT_MSG GOOGLE_BENCHMARK_INCLUDE_DIRS GOOGLE_BENCHMARK_LIBRARIES)
mark_as_advanced(GOOGLE_BENCHMARK_INCLUDE_DIRS GOOGLE_BENCHMARK_LIBRARIES)
