# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Burgers)

set(LIBRARY_SOURCES
  Equations.cpp
  Fluxes.cpp
  Minmod.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )
