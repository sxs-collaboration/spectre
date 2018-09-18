# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DiscontinuousGalerkin)

set(LIBRARY_SOURCES
    MortarHelpers.cpp
    )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE Spectral
  )
