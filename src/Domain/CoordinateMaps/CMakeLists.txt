# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY CoordinateMaps)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Affine.cpp
  BulgedCube.cpp
  DiscreteRotation.cpp
  EquatorialCompression.cpp
  Equiangular.cpp
  Frustum.cpp
  Identity.cpp
  Rotation.cpp
  SpecialMobius.cpp
  Wedge2D.cpp
  Wedge3D.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Affine.hpp
  BulgedCube.hpp
  CoordinateMap.hpp
  CoordinateMap.tpp
  CoordinateMapHelpers.hpp
  DiscreteRotation.hpp
  EquatorialCompression.hpp
  Equiangular.hpp
  Frustum.hpp
  Identity.hpp
  MapInstantiationMacros.hpp
  ProductMaps.hpp
  ProductMaps.tpp
  Rotation.hpp
  SpecialMobius.hpp
  Tags.hpp
  TimeDependentHelpers.hpp
  Wedge2D.hpp
  Wedge3D.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  PUBLIC
  Boost::boost
  DataStructures
  DomainStructure
  ErrorHandling
  FunctionsOfTime
  )

add_subdirectory(TimeDependent)
