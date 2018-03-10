# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Valencia)

set(LIBRARY_SOURCES
  ConservativeFromPrimitive.cpp
  Fluxes.cpp
  )

add_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )
