# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DomainTimeDependence")

set(LIBRARY_SOURCES
  Test_Composition.cpp
  Test_UniformTranslation.cpp
  )

add_test_library(
  ${LIBRARY}
  "Domain/Creators/TimeDependence"
  "${LIBRARY_SOURCES}"
  "DomainTimeDependence;TestUtilities;Test_Domain"
  )
