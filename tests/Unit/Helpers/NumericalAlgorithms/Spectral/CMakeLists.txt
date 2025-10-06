# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "SpectralHelpers")

set(LIBRARY_SOURCES
  FourierTestFunctions.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Spectral
  SphericalHarmonics
  Utilities
  )
