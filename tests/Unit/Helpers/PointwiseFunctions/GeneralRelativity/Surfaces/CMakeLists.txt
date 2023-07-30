# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "StrahlkorperGrHelpers")

set(LIBRARY_SOURCES
  StrahlkorperGrTestHelpers.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ApparentHorizons
  DataStructures
  SphericalHarmonics
  Utilities
  )
