# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(GoogleBenchmark QUIET)

if (${GOOGLE_BENCHMARK_FOUND})
  spectre_include_directories(${GOOGLE_BENCHMARK_INCLUDE_DIRS})
  set(SPECTRE_LIBRARIES "${SPECTRE_LIBRARIES};${GOOGLE_BENCHMARK_LIBRARIES}")

  message(STATUS "Google Benchmark libs: " ${GOOGLE_BENCHMARK_LIBRARIES})
  message(STATUS "Google Benchmark incl: " ${GOOGLE_BENCHMARK_INCLUDE_DIRS})

  file(APPEND
    "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
    "Google Benchmark Found\n"
    )
endif()
