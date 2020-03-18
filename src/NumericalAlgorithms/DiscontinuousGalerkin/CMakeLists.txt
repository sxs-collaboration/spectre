# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DiscontinuousGalerkin)

set(LIBRARY_SOURCES
  Formulation.cpp
  MortarHelpers.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  Spectral
  )
