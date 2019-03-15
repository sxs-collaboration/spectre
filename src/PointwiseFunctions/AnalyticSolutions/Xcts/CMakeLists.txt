# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY XctsSolutions)

set(LIBRARY_SOURCES
  ConstantDensityStar.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE Utilities
  )
