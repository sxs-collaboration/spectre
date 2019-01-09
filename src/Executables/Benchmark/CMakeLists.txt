# Distributed under the MIT License.
# See LICENSE.txt for details.

# Since benchmarking is only interesting in release mode the executable isn't
# added for Debug builds. Charm++'s main function is overridden with the main
# from the Google Benchmark library. The executable is not added to the `all` make
# target since it is only interesting in specific circumstances.
if("${GOOGLE_BENCHMARK_FOUND}" AND NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(executable Benchmark)

  add_spectre_executable(
    ${executable}
    EXCLUDE_FROM_ALL
    Benchmark.cpp
    )

  # Add specific libraries needed for the benchmark you are interested in.
  target_link_libraries(
    ${executable}
    benchmark
    Domain
    CoordinateMaps
    Spectral
    ${SPECTRE_LIBRARIES}
    )

  set_target_properties(
    ${executable}
    PROPERTIES LINK_FLAGS "-nomain-module -nomain"
    )
endif()
