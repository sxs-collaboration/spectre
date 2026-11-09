# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GrSelfForceEvents)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ObserveFlux.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ObserveFlux.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Charmxx::pup
  DataStructures
  Domain
  DomainStructure
  Elliptic
  EventsAndTriggers
  GrSelfForceAnalyticData
  Observer
  Options
  Parallel
  Serialization
  Spectral
  Utilities
  PRIVATE
  DiscontinuousGalerkin
  LinearOperators
  )
