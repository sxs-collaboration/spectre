# Distributed under the MIT License.
# See LICENSE.txt for details.

set(TEST_DIR
  PointwiseFunctions/AnalyticSolutions/EinsteinSolutions)

set(SPECTRE_UNIT_EINSTEIN_SOLUTION
  ${TEST_DIR}/Test_KerrSchild.cpp
  ${TEST_DIR}/Test_Minkowski.cpp
  ${TEST_DIR}/VerifyEinsteinSolution.cpp
  )

set(SPECTRE_UNIT_ANALYTIC_SOLUTIONS
  "${SPECTRE_UNIT_ANALYTIC_SOLUTIONS};${SPECTRE_UNIT_EINSTEIN_SOLUTION}"
  PARENT_SCOPE)
