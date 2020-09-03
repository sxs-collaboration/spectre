# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_AnalyticData")

set(LIBRARY_SOURCES
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticData"
  "${LIBRARY_SOURCES}"
  "AnalyticData;DataStructures;DataStructuresHelpers"
  )

add_subdirectory(Burgers)
add_subdirectory(CurvedWaveEquation)
add_subdirectory(GrMhd)
add_subdirectory(NewtonianEuler)
