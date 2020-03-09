# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Limiters)

set(LIBRARY_SOURCES
  HwenoModifiedSolution.cpp
  Minmod.cpp
  MinmodHelpers.cpp
  MinmodTci.cpp
  MinmodType.cpp
  WenoGridHelpers.cpp
  WenoHelpers.cpp
  WenoOscillationIndicator.cpp
  WenoType.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE Domain
  INTERFACE Interpolation  # for WENO
  INTERFACE LinearOperators
  )
