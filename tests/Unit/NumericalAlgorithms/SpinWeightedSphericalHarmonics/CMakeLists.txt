# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_SpinWeightedSphericalHarmonics")

set(LIBRARY_SOURCES
  Test_ComplexDataView.cpp
  Test_SwshCoefficients.cpp
  Test_SwshCollocation.cpp
  Test_SwshDerivatives.cpp
  Test_SwshFiltering.cpp
  Test_SwshInterpolation.cpp
  Test_SwshTags.cpp
  Test_SwshTestHelpers.cpp
  Test_SwshTransform.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  DataStructures
  DiscontinuousGalerkin
  Domain
  ErrorHandling
  LinearOperators
  Spectral
  SpinWeightedSphericalHarmonics
  SpinWeightedSphericalHarmonicsHelpers
  Utilities
  )
