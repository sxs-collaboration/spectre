# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY NewtonianEulerSolutions)

set(LIBRARY_SOURCES
    IsentropicVortex.cpp
    )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE NewtonianEuler
  )
