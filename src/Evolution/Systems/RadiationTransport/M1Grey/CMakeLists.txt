# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY M1Grey)

set(LIBRARY_SOURCES
  Fluxes.cpp
  M1Closure.cpp
  Sources.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE GeneralRelativity
  INTERFACE Hydro
  INTERFACE RootFinding
  INTERFACE Utilities
  )
