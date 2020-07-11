# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  Evolution
  PRIVATE
  DgDomain.cpp
  )

spectre_target_headers(
  Evolution
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ConservativeSystem.hpp
  DgDomain.hpp
  DiscontinuousGalerkin.hpp
  Evolution.hpp
  GrTagsForHydro.hpp
  InitialData.hpp
  Limiter.hpp
  NonconservativeSystem.hpp
  SetVariables.hpp
  Tags.hpp
  )
