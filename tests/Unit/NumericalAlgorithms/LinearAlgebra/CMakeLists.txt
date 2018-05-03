# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_LinearAlgebra")

set(LIBRARY_SOURCES
  Test_FindGeneralizedEigenvalues.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/LinearAlgebra/"
  "${LIBRARY_SOURCES}"
  "LinearAlgebra"
  )
