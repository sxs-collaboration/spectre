# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_AnalyticData")

set(LIBRARY_SOURCES
  Test_AnalyticData.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticData"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  AnalyticData
  AnalyticSolutions
  DataStructures
  DataStructuresHelpers
  )

add_subdirectory(Burgers)
add_subdirectory(CurvedWaveEquation)
add_subdirectory(ForceFree)
add_subdirectory(GeneralRelativity)
add_subdirectory(GrMhd)
add_subdirectory(NewtonianEuler)
add_subdirectory(Punctures)
add_subdirectory(Xcts)
