# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GrMhdAnalyticData)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BlastWave.cpp
  BondiHoyleAccretion.cpp
  MagneticFieldLoop.cpp
  MagneticRotor.cpp
  MagnetizedFmDisk.cpp
  OrszagTangVortex.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticData.hpp
  BlastWave.hpp
  BondiHoyleAccretion.hpp
  MagneticFieldLoop.hpp
  MagneticRotor.hpp
  MagnetizedFmDisk.hpp
  OrszagTangVortex.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Hydro
  Options
  RelativisticEulerSolutions
  )
