# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY RelativisticEulerSolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  FishboneMoncriefDisk.cpp
  SmoothFlow.cpp
  TovStar.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  FishboneMoncriefDisk.hpp
  SmoothFlow.hpp
  Solutions.hpp
  TovStar.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  ErrorHandling
  GeneralRelativitySolutions
  Hydro
  Options
  )
