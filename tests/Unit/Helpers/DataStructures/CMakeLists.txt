# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "DataStructuresHelpers")

set(LIBRARY_SOURCES
  RandomUnitNormal.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Utilities
  )
