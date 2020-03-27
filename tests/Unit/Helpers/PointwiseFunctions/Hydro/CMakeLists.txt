# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "HydroHelpers")

set(LIBRARY_SOURCES
  TestHelpers.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Hydro

  PRIVATE
  DataStructuresHelpers

  PUBLIC
  DataStructures
  Utilities
  )
