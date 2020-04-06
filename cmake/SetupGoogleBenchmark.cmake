# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(GoogleBenchmark QUIET)


if (${GoogleBenchmark_FOUND})
  message(STATUS "Google Benchmark libs: " ${GoogleBenchmark_LIBRARIES})
  message(STATUS "Google Benchmark incl: " ${GoogleBenchmark_INCLUDE_DIRS})

  file(APPEND
    "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
    "Google Benchmark Found\n"
    )

  add_library(GoogleBenchmark INTERFACE IMPORTED)
  set_property(TARGET GoogleBenchmark PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${GoogleBenchmark_INCLUDE_DIRS})
  set_property(TARGET GoogleBenchmark PROPERTY
    INTERFACE_LINK_LIBRARIES ${GoogleBenchmark_LIBRARIES})
  set_property(
    GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
    GoogleBenchmark
    )
endif()
