# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_CoordinateMapsTimeDependent")

set(LIBRARY_SOURCES
  Test_CubicScale.cpp
  Test_ProductMaps.cpp
  Test_Rotation.cpp
  Test_RotationMatrixHelpers.cpp
  Test_RotScaleTrans.cpp
  Test_Shape.cpp
  Test_SphericalCompression.cpp
  Test_Translation.cpp
  )

add_subdirectory(ShapeMapTransitionFunctions)

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataStructures
  Domain
  DataStructuresHelpers
  MathFunctions
  FunctionsOfTime
  SphericalHarmonics
  Spectral
  Utilities
)
