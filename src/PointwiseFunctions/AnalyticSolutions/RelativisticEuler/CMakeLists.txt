# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY RelativisticEulerSolutions)

set(LIBRARY_SOURCES
  FishboneMoncriefDisk.cpp
  SmoothFlow.cpp
  TovStar.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

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
