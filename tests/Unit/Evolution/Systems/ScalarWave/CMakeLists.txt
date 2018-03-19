# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Scalarwave")

set(LIBRARY_SOURCES
  Test_Equations.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/ScalarWave/"
  "${LIBRARY_SOURCES}"
  "ScalarWave"
  )
