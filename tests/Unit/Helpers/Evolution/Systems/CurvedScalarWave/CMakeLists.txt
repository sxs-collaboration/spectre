# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "CurvedScalarWaveHelpers")

set(LIBRARY_SOURCES
  TestHelpers.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Utilities

  PUBLIC
  DataStructures
  )
