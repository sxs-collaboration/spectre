# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY LinearSolver)

set(LIBRARY_SOURCES
    IterationId.cpp
    )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE Utilities
  )
