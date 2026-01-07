# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarSelfForceActions)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  InitializeEffectiveSource.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  DataStructures
  DiscontinuousGalerkin
  Domain
  Elliptic
  Parallel
  ParallelAmr
  ScalarSelfForce
  ScalarSelfForceAnalyticData
  Utilities
  )
