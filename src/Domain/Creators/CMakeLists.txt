# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DomainCreators)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AlignedLattice.cpp
  AngularCylinder.cpp
  AngularDisk.cpp
  BinaryCompactObject.cpp
  CartoonCylinder.cpp
  CartoonSphere1D.cpp
  CartoonSphere2D.cpp
  Cylinder.cpp
  CylindricalBinaryCompactObject.cpp
  Disk.cpp
  ExpandOverBlocks.cpp
  FrustalCloak.cpp
  NonconformingSphericalShells.cpp
  Rectilinear.cpp
  RegisterDerivedWithCharm.cpp
  RotatedBricks.cpp
  RotatedIntervals.cpp
  RotatedRectangles.cpp
  ShellDistribution.cpp
  Sphere.cpp
  SphericalShells.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AlignedLattice.hpp
  AngularCylinder.hpp
  AngularDisk.hpp
  BinaryCompactObject.hpp
  CartoonCylinder.hpp
  CartoonSphere1D.hpp
  CartoonSphere2D.hpp
  Cylinder.hpp
  CylindricalBinaryCompactObject.hpp
  Disk.hpp
  DomainCreator.hpp
  ExpandOverBlocks.hpp
  ExpandOverBlocks.tpp
  Factory.hpp
  Factory1D.hpp
  Factory2D.hpp
  Factory3D.hpp
  FrustalCloak.hpp
  NonconformingSphericalShells.hpp
  OptionTags.hpp
  Rectilinear.hpp
  RegisterDerivedWithCharm.hpp
  RotatedBricks.hpp
  RotatedIntervals.hpp
  RotatedRectangles.hpp
  ShellDistribution.hpp
  Sphere.hpp
  SphericalShells.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DomainStructure
  H5
  PUBLIC
  CoordinateMaps
  DataStructures
  Domain
  DomainBoundaryConditions
  DomainTimeDependence
  GeneralRelativity
  SphericalHarmonics
  SphericalHarmonicsIO
  INTERFACE
  ErrorHandling
  Options
  )

add_subdirectory(Python)
add_subdirectory(Tags)
add_subdirectory(TimeDependence)
add_subdirectory(TimeDependentOptions)
