# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "SphericalHarmonicsHelpers")

set(LIBRARY_SOURCES
  YlmTestFunctions.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  SphericalHarmonics
  Utilities
  )
