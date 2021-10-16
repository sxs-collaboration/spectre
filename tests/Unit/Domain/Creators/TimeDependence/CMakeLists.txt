# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DomainTimeDependence")

set(LIBRARY_SOURCES
  Test_CompositionUniformTranslation.cpp
  Test_CompositionCubicScaleAndUniformRotationAboutZAxis.cpp
  Test_CubicScale.cpp
  Test_None.cpp
  Test_SphericalCompression.cpp
  Test_UniformRotationAboutZAxis.cpp
  Test_UniformTranslation.cpp
  )

add_test_library(
  ${LIBRARY}
  "Domain/Creators/TimeDependence"
  "${LIBRARY_SOURCES}"
  "DataStructuresHelpers;DomainHelpers;DomainTimeDependence;Options"
  )
