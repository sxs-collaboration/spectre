# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY RelativisticEulerSolutions)

set(LIBRARY_SOURCES
  FishboneMoncriefDisk.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE GeneralRelativitySolutions
  INTERFACE Hydro
  )
