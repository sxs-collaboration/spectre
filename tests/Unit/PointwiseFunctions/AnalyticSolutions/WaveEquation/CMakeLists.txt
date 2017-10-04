# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_WAVE_EQUATION
  PointwiseFunctions/AnalyticSolutions/WaveEquation/Test_PlaneWave.cpp
  )

set(SPECTRE_UNIT_ANALYTIC_SOLUTIONS
  "${SPECTRE_UNIT_ANALYTIC_SOLUTIONS};${SPECTRE_UNIT_WAVE_EQUATION}"
  PARENT_SCOPE)
