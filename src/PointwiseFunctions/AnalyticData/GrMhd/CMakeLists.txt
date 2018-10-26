# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GrMhdAnalyticData)

set(LIBRARY_SOURCES
  BondiHoyleAccretion.cpp
  CylindricalBlastWave.cpp
  MagnetizedFmDisk.cpp
  OrszagTangVortex.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE Hydro
  INTERFACE RelativisticEulerSolutions
  )
