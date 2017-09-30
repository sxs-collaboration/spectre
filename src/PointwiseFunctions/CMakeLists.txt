# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY PointwiseFunctions)

set(LIBRARY_SOURCES)

add_subdirectory(MathFunctions)

add_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE Utilities
  )
