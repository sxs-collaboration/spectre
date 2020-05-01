# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NumericalIntegration")

set(LIBRARY_SOURCES
  Test_GslQuadAdaptive.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/Integration/"
  "${LIBRARY_SOURCES}"
  "Integration"
  )
