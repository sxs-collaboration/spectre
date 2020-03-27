# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "ApparentHorizonsHelpers")

set(LIBRARY_SOURCES
  StrahlkorperGrTestHelpers.cpp
  YlmTestFunctions.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ApparentHorizons
  DataStructures
  Utilities
  )
