# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarSelfForceAnalyticData)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  CircularOrbit.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CircularOrbit.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  EffectiveSource
  ErrorHandling
  GeneralRelativity
  Options
  Utilities
  )
