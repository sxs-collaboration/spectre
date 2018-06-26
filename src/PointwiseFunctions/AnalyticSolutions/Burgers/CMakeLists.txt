# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY BurgersSolutions)

set(LIBRARY_SOURCES
  Bump.cpp
  Linear.cpp
  )

add_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )
