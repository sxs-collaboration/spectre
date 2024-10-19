# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Ringdown")

set(LIBRARY_SOURCES
  Test_StrahlkorperCoefsInRingdownDistortedFrame.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataStructures
  DomainCreators
  Domain
  H5
  Ringdown
  Spectral
  SphericalHarmonics
  SphericalHarmonicsIO
  Utilities
  )

add_subdirectory(Python)
