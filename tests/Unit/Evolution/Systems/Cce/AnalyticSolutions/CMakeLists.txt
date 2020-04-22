# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Cce_AnalyticSolutions")

set(LIBRARY_SOURCES
  Test_BouncingBlackHole.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/Cce/AnalyticSolutions/"
  "${LIBRARY_SOURCES}"
  "CceAnalyticSolutions"
  )
