# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Informer)

set(LIBRARY_SOURCES
    Informer.cpp
    Verbosity.cpp)

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE ErrorHandling
  )
