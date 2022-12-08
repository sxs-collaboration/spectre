# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DomainCreators)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AlignedLattice.cpp
  BinaryCompactObject.cpp
  Brick.cpp
  Cylinder.cpp
  CylindricalBinaryCompactObject.cpp
  Disk.cpp
  ExpandOverBlocks.cpp
  FrustalCloak.cpp
  Interval.cpp
  Rectangle.cpp
  RegisterDerivedWithCharm.cpp
  RotatedBricks.cpp
  RotatedIntervals.cpp
  RotatedRectangles.cpp
  Shell.cpp
  Sphere.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AlignedLattice.hpp
  BinaryCompactObject.hpp
  BinaryCompactObjectHelpers.hpp
  Brick.hpp
  Cylinder.hpp
  CylindricalBinaryCompactObject.hpp
  Disk.hpp
  DomainCreator.hpp
  ExpandOverBlocks.hpp
  Factory.hpp
  Factory1D.hpp
  Factory2D.hpp
  Factory3D.hpp
  FrustalCloak.hpp
  Interval.hpp
  Rectangle.hpp
  RegisterDerivedWithCharm.hpp
  RotatedBricks.hpp
  RotatedIntervals.hpp
  RotatedRectangles.hpp
  Shell.hpp
  Sphere.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DomainStructure
  PUBLIC
  CoordinateMaps
  DataStructures
  DomainBoundaryConditions
  DomainTimeDependence
  INTERFACE
  Domain
  ErrorHandling
  Options
  )

add_subdirectory(Python)
add_subdirectory(TimeDependence)
