# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ControlSystemMeasurements)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BNSCenterOfMass.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BNSCenterOfMass.hpp
  BothHorizons.hpp
  CharSpeed.hpp
  SingleHorizon.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ApparentHorizons
  ControlSystem
  DataStructures
  GeneralRelativity
  GeneralizedHarmonic
  ParallelInterpolation
  Utilities
  )
