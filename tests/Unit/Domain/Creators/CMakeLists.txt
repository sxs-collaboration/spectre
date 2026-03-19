# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DomainCreators")

set(LIBRARY_SOURCES
  Test_AlignedLattice.cpp
  Test_AngularDisk.cpp
  Test_BinaryCompactObject.cpp
  Test_Brick.cpp
  Test_CartoonCylinder.cpp
  Test_CartoonSphere1D.cpp
  Test_CartoonSphere2D.cpp
  Test_Cylinder.cpp
  Test_CylindricalBinaryCompactObject.cpp
  Test_Disk.cpp
  Test_ExpandOverBlocks.cpp
  Test_FrustalCloak.cpp
  Test_Interval.cpp
  Test_NonconformingSphericalShells.cpp
  Test_Rectangle.cpp
  Test_RotatedBricks.cpp
  Test_RotatedIntervals.cpp
  Test_RotatedRectangles.cpp
  Test_Sphere.cpp
  Test_SphericalShells.cpp
  Test_Tags.cpp
  )

add_subdirectory(TimeDependentOptions)

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Domain
  DomainBoundaryConditions
  DomainBoundaryConditionsHelpers
  DomainCreators
  DomainHelpers
  GeneralRelativity
  H5
  Informer
  SphericalHarmonics
  SphericalHarmonicsIO
  )

add_subdirectory(Python)
add_subdirectory(TimeDependence)
