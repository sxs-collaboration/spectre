# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Ringdown")

set(LIBRARY_SOURCES
  Test_MinimumAhCExcisionRadius.cpp
  Test_StrahlkorperCoefsAndCenters.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataStructures
  DomainCreators
  Domain
  GeneralRelativity
  H5
  Ringdown
  Spectral
  SphericalHarmonics
  SphericalHarmonicsIO
  Utilities
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyCoordinateMaps
  PyDomain
  PyDomainTimeDependentOptions
  )

add_subdirectory(Python)
