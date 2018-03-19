# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NumericalInterpolation")

set(LIBRARY_SOURCES
  Test_LagrangePolynomial.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/Interpolation/"
  "${LIBRARY_SOURCES}"
  "" # Header-only, link dependencies included from testing lib
  )
