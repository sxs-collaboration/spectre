# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "SpinWeightedSphericalHarmonicsHelpers")

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
  SpinWeightedSphericalHarmonics
  Utilities

  PRIVATE
  Boost::boost
  )
