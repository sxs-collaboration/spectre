# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GrMhdAnalyticData)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BlastWave.cpp
  BondiHoyleAccretion.cpp
  CcsnCollapse.cpp
  KhInstability.cpp
  MagneticFieldLoop.cpp
  MagneticRotor.cpp
  MagnetizedFmDisk.cpp
  MagnetizedTovStar.cpp
  OrszagTangVortex.cpp
  RiemannProblem.cpp
  SlabJet.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticData.hpp
  BlastWave.hpp
  BondiHoyleAccretion.hpp
  CcsnCollapse.hpp
  KhInstability.hpp
  MagneticFieldLoop.hpp
  MagneticRotor.hpp
  MagnetizedFmDisk.hpp
  MagnetizedTovStar.hpp
  OrszagTangVortex.hpp
  RiemannProblem.hpp
  SlabJet.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Hydro
  Options
  RelativisticEulerSolutions
  Serialization
  )

if (TARGET SpEC::Exporter)
  spectre_target_sources(
    ${LIBRARY}
    PRIVATE
    SpecInitialData.cpp
    )
  spectre_target_headers(
    ${LIBRARY}
    INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
    HEADERS
    SpecInitialData.hpp
    )
  target_link_libraries(
    ${LIBRARY}
    PUBLIC
    GeneralRelativityAnalyticData
    SpEC::Exporter
    )
endif()

add_subdirectory(InitialMagneticFields)
