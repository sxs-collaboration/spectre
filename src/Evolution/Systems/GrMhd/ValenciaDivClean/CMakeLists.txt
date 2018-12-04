# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ValenciaDivClean)

set(LIBRARY_SOURCES
  Characteristics.cpp
  ConservativeFromPrimitive.cpp
  FixConservatives.cpp
  Fluxes.cpp
  Minmod.cpp
  NewmanHamlin.cpp
  PalenzuelaEtAl.cpp
  PrimitiveFromConservative.cpp
  Sources.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE GeneralRelativity
  INTERFACE Hydro
  INTERFACE SlopeLimiters
  INTERFACE Valencia
  INTERFACE VariableFixing
  )
