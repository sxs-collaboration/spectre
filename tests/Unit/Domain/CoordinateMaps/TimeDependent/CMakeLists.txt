# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_CoordinateMapsTimeDependent")

set(LIBRARY_SOURCES
  Test_CubicScale.cpp
  Test_ProductMaps.cpp
  Test_Rotation.cpp
  Test_SphericalCompression.cpp
  Test_Translation.cpp
  )

add_test_library(
  ${LIBRARY}
  "Domain/CoordinateMaps/TimeDependent"
  "${LIBRARY_SOURCES}"
  "CoordinateMaps;FunctionsOfTime"
  )
