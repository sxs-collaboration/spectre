# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GrMhdAnalyticData")

set(LIBRARY_SOURCES
  InitialMagneticFields/Test_Poloidal.cpp
  InitialMagneticFields/Test_Toroidal.cpp
  Test_BlastWave.cpp
  Test_BondiHoyleAccretion.cpp
  Test_CcsnCollapse.cpp
  Test_KhInstability.cpp
  Test_MagneticFieldLoop.cpp
  Test_MagneticRotor.cpp
  Test_MagnetizedFmDisk.cpp
  Test_MagnetizedTovStar.cpp
  Test_OrszagTangVortex.cpp
  Test_RiemannProblem.cpp
  Test_SlabJet.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticData/GrMhd"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  ErrorHandling
  GrMhdAnalyticData
  Hydro
  Options
  Utilities
)
