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
  CylindricalFlatSide.cpp
  CylindricalSide.cpp
  DiscreteRotation.cpp
  Distribution.cpp
  EquatorialCompression.cpp
  Equiangular.cpp
  FocallyLiftedEndcap.cpp
  FocallyLiftedFlatEndcap.cpp
  FocallyLiftedFlatSide.cpp
  FocallyLiftedMap.cpp
  FocallyLiftedMapHelpers.cpp
  FocallyLiftedSide.cpp
  Frustum.cpp
  Identity.cpp
  Interval.cpp
  KerrHorizonConforming.cpp
  Rotation.cpp
  SpecialMobius.cpp
  Wedge.cpp
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
  CylindricalFlatSide.hpp
  CylindricalSide.hpp
  DiscreteRotation.hpp
  Distribution.hpp
  EquatorialCompression.hpp
  Equiangular.hpp
  FocallyLiftedEndcap.hpp
  FocallyLiftedFlatEndcap.hpp
  FocallyLiftedFlatSide.hpp
  FocallyLiftedMap.hpp
  FocallyLiftedMapHelpers.hpp
  FocallyLiftedSide.hpp
  Frustum.hpp
  Identity.hpp
  Interval.hpp
  KerrHorizonConforming.hpp
  MapInstantiationMacros.hpp
  ProductMaps.hpp
  ProductMaps.tpp
  Rotation.hpp
  SpecialMobius.hpp
  Tags.hpp
  TimeDependentHelpers.hpp
  Wedge.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  PUBLIC
  ApparentHorizons
  Boost::boost
  DataStructures
  DomainStructure
  ErrorHandling
  FunctionsOfTime
  GSL::gsl
  Parallel
  RootFinding
  )

add_subdirectory(TimeDependent)
