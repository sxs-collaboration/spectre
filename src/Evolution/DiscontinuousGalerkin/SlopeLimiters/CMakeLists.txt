# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SlopeLimiters)

set(LIBRARY_SOURCES
  MinmodHelpers.cpp
  MinmodTci.cpp
  MinmodType.cpp
  WenoGridHelpers.cpp
  WenoHelpers.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE Domain
  INTERFACE LinearOperators
  )
