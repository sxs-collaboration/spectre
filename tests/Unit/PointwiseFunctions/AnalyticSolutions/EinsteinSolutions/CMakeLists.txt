# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_EINSTEIN_SOLUTION
  PointwiseFunctions/AnalyticSolutions/EinsteinSolutions/Test_Minkowski.cpp
  )

set(SPECTRE_UNIT_ANALYTIC_SOLUTIONS
  "${SPECTRE_UNIT_ANALYTIC_SOLUTIONS};${SPECTRE_UNIT_EINSTEIN_SOLUTION}"
  PARENT_SCOPE)
