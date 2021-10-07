# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ControlSystemApparentHorizons)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Measurements.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  ApparentHorizons
  ControlSystem
  DataStructures
  GeneralRelativity
  GeneralizedHarmonic
  ParallelInterpolation
  Utilities
  )
