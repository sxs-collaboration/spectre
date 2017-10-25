# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(benchmark QUIET)

if (${benchmark_FOUND})
  message(STATUS "Google Benchmark version: ${benchmark_VERSION}")
endif()
