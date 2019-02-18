# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY NewtonianEuler)

set(LIBRARY_SOURCES
  Characteristics.cpp
  ConservativeFromPrimitive.cpp
  Fluxes.cpp
  PrimitiveFromConservative.cpp
  SoundSpeedSquared.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

add_subdirectory(Sources)

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE Hydro
  )
