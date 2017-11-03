# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(benchmark QUIET)

if (${benchmark_FOUND})
  message(STATUS "Google Benchmark version: ${benchmark_VERSION}")

  file(APPEND
    "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
    "Google Benchmark Version:  ${benchmark_VERSION}\n"
    )

endif()
