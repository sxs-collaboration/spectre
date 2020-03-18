# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY NewtonianEulerNumericalFluxes)

set(LIBRARY_SOURCES
  Hllc.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC DataStructures
  INTERFACE ErrorHandling
  )
