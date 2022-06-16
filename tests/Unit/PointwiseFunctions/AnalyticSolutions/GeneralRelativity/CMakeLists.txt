# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GeneralRelativitySolutions")

set(LIBRARY_SOURCES
  Test_GaugeWave.cpp
  Test_HarmonicSchwarzschild.cpp
  Test_KerrHorizon.cpp
  Test_KerrSchild.cpp
  Test_Minkowski.cpp
  Test_SphericalKerrSchild.cpp
  Test_WrappedGr.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  GeneralRelativity
  GeneralRelativitySolutions
  GeneralizedHarmonic
  GrSolutionsTestHelpers
  Hydro
  Options
  Spectral
  Utilities
  )
