# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Valencia)

set(LIBRARY_SOURCES
  Characteristics.cpp
  ConservativeFromPrimitive.cpp
  Equations.cpp
  Fluxes.cpp
  PrimitiveFromConservative.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE GeneralRelativity
  )
