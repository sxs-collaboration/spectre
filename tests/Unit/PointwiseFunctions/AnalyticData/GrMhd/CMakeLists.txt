# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GrMhdAnalyticData")

set(LIBRARY_SOURCES
  Test_BondiHoyleAccretion.cpp
  Test_CylindricalBlastWave.cpp
  Test_MagnetizedFmDisk.cpp
  Test_OrszagTangVortex.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticData/GrMhd"
  "${LIBRARY_SOURCES}"
  "GrMhdAnalyticData;Utilities"
  )
