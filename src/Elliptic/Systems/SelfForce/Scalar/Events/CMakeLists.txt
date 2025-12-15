# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarSelfForceEvents)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ObserveSelfForce.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ObserveSelfForce.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Interpolation
  PUBLIC
  DataStructures
  Domain
  Elliptic
  Options
  ScalarSelfForceAnalyticData
  Serialization
  Utilities
  )
