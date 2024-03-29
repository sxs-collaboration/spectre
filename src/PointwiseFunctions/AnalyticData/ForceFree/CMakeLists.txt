# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ForceFreeAnalyticData)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  FfeBreakdown.cpp
  MagnetosphericWald.cpp
  RotatingDipole.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticData.hpp
  Factory.hpp
  FfeBreakdown.hpp
  MagnetosphericWald.hpp
  RotatingDipole.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  GeneralRelativitySolutions
  Options
  )
