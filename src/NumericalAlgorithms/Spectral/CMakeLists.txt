# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Spectral)

set(LIBRARY_SOURCES
    Legendre.cpp
    Projection.cpp
    Spectral.cpp
    )

add_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )
