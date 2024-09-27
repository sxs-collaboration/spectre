# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DomainCreators)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AlignedLattice.cpp
  BinaryCompactObject.cpp
  Cylinder.cpp
  CylindricalBinaryCompactObject.cpp
  Disk.cpp
  ExpandOverBlocks.cpp
  FrustalCloak.cpp
  Rectilinear.cpp
  RegisterDerivedWithCharm.cpp
  RotatedBricks.cpp
  RotatedIntervals.cpp
  RotatedRectangles.cpp
  Sphere.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AlignedLattice.hpp
  BinaryCompactObject.hpp
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
  OptionTags.hpp
  Rectilinear.hpp
  RegisterDerivedWithCharm.hpp
  RotatedBricks.hpp
  RotatedIntervals.hpp
  RotatedRectangles.hpp
  Sphere.hpp
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
  GeneralRelativitySolutions
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
