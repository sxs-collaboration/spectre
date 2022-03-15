# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_CoordinateMaps")

set(LIBRARY_SOURCES
  Test_Affine.cpp
  Test_BulgedCube.cpp
  Test_CoordinateMap.cpp
  Test_CylindricalEndcap.cpp
  Test_CylindricalFlatEndcap.cpp
  Test_CylindricalFlatSide.cpp
  Test_CylindricalSide.cpp
  Test_DiscreteRotation.cpp
  Test_Distribution.cpp
  Test_EquatorialCompression.cpp
  Test_Equiangular.cpp
  Test_Frustum.cpp
  Test_Identity.cpp
  Test_Interval.cpp
  Test_KerrHorizonConforming.cpp
  Test_ProductMaps.cpp
  Test_Rotation.cpp
  Test_SpecialMobius.cpp
  Test_Tags.cpp
  Test_TimeDependentHelpers.cpp
  Test_UniformCylindricalEndcap.cpp
  Test_Wedge2D.cpp
  Test_Wedge3D.cpp
  )

add_test_library(
  ${LIBRARY}
  "Domain/CoordinateMaps"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataStructures
  DataStructuresHelpers
  Domain
  DomainStructure
  ErrorHandling
  FunctionsOfTime
  GeneralRelativitySolutions
  RootFinding
  Spectral
  Utilities
)

add_subdirectory(TimeDependent)
