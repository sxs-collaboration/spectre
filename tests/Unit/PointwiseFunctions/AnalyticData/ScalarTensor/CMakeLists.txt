# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ScalarTensorAnalyticData")

set(LIBRARY_SOURCES
  Test_KerrSphericalHarmonic.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticData/ScalarTensor"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  ErrorHandling
  GeneralRelativitySolutions
  GhScalarTensorAnalyticData
  ScalarTensorAnalyticData
  Options
)
