# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_SPECTRAL
  NumericalAlgorithms/Spectral/Test_LegendreGaussLobatto.cpp
  )

set(SPECTRE_UNIT_NUMERICAL_ALGORITHMS
  "${SPECTRE_UNIT_NUMERICAL_ALGORITHMS};${SPECTRE_UNIT_SPECTRAL}"
  PARENT_SCOPE)
