# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GrMhdAnalyticData")

set(LIBRARY_SOURCES
  Test_BlastWave.cpp
  Test_BondiHoyleAccretion.cpp
  Test_MagneticFieldLoop.cpp
  Test_MagneticRotor.cpp
  Test_MagnetizedFmDisk.cpp
  Test_MagnetizedTovStar.cpp
  Test_OrszagTangVortex.cpp
  Test_RiemannProblem.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticData/GrMhd"
  "${LIBRARY_SOURCES}"
  "GrMhdAnalyticData;Options;Utilities"
  )
