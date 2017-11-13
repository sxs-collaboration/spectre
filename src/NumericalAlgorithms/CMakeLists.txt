# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY NumericalAlgorithms)

set(LIBRARY_SOURCES)

add_subdirectory(RootFinding)
add_subdirectory(Spectral)

add_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )
