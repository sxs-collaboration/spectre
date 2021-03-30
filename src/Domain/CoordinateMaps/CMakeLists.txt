# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY CoordinateMaps)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Affine.cpp
  BulgedCube.cpp
  CylindricalEndcap.cpp
  CylindricalFlatEndcap.cpp
  CylindricalSide.cpp
  DiscreteRotation.cpp
  EquatorialCompression.cpp
  Equiangular.cpp
  FocallyLiftedEndcap.cpp
  FocallyLiftedFlatEndcap.cpp
  FocallyLiftedMap.cpp
  FocallyLiftedMapHelpers.cpp
  FocallyLiftedSide.cpp
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
  CylindricalEndcap.hpp
  CylindricalFlatEndcap.hpp
  CylindricalSide.hpp
  DiscreteRotation.hpp
  EquatorialCompression.hpp
  Equiangular.hpp
  FocallyLiftedEndcap.hpp
  FocallyLiftedFlatEndcap.hpp
  FocallyLiftedMap.hpp
  FocallyLiftedMapHelpers.hpp
  FocallyLiftedSide.hpp
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
  GSL::gsl
  RootFinding
  )

add_subdirectory(TimeDependent)
