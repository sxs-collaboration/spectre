# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "SpectralHelpers")

set(LIBRARY_SOURCES
  SwshTestHelpers.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Libsharp
  Spectral
  Utilities

  PRIVATE
  Boost::boost
  )
