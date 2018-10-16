# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GrMhdSolutions)

set(LIBRARY_SOURCES
    AlfvenWave.cpp
    SmoothFlow.cpp
    )

add_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE Hydro
  )
