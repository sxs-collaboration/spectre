# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_RelativisticEulerSolutions")

set(LIBRARY_SOURCES
  Test_FishboneMoncriefDisk.cpp
  Test_SmoothFlow.cpp
  Test_TovStar.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  DomainCreators
  Hydro
  RelativisticEulerSolutions
  Options
  Spectral
  Utilities
  ValenciaDivClean
)
