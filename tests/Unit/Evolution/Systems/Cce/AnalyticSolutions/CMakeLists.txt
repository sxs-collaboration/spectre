# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Cce_AnalyticSolutions")

set(LIBRARY_SOURCES
  Test_BouncingBlackHole.cpp
  Test_GaugeWave.cpp
  Test_LinearizedBondiSachs.cpp
  Test_RobinsonTrautman.cpp
  Test_RotatingSchwarzschild.cpp
  Test_TeukolskyWave.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Cce
  CceAnalyticSolutionsHelpers
  DataStructures
  GeneralizedHarmonic
  GeneralRelativity
  Utilities
  SpinWeightedSphericalHarmonics
  )
