# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Convergence")

set(LIBRARY_SOURCES
  Test_Criteria.cpp
  Test_HasConverged.cpp
  Test_Reason.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/Convergence/"
  "${LIBRARY_SOURCES}"
  "Convergence;DataStructures"
  )
