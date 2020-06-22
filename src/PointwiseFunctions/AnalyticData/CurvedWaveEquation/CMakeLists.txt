# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY CurvedWaveEquationAnalyticData)

set(LIBRARY_SOURCES
  PlaneWave3DKerrSchild.cpp
  RegularSphericalWave3DKerrSchild.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE GeneralRelativity
  INTERFACE GeneralRelativitySolutions
  INTERFACE Spectral
  INTERFACE Utilities
  )
