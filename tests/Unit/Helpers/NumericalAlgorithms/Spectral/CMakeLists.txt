# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "SpectralHelpers")

set(LIBRARY_SOURCES
  DiskTestFunctions.cpp
  FourierTestFunctions.cpp
  PolynomialTestFunctions.cpp
  )

add_spectre_library(${LIBRARY} ${SPECTRE_TEST_LIBS_TYPE} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Catch2::Catch2
  DataStructures
  DataStructuresHelpers
  Spectral
  Utilities
  )
