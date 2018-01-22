# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EinsteinSolutions)

set(LIBRARY_SOURCES
    KerrSchild.cpp
    Minkowski.cpp
    )

add_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )
