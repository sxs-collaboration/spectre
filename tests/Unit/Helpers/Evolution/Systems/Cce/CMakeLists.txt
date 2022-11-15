# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(AnalyticSolutions)

set(LIBRARY "CceHelpers")

set(LIBRARY_SOURCES
  CceComputationTestHelpers.cpp
  )

add_spectre_library(${LIBRARY} ${SPECTRE_TEST_LIBS_TYPE} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Cce
  DataStructures
  H5
  Spectral
  Utilities
  )
