# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_LinearSolver")

set(LIBRARY_SOURCES
  Test_InnerProduct.cpp
  Test_Tags.cpp
  )

add_subdirectory(Actions)
add_subdirectory(ConjugateGradient)
add_subdirectory(Gmres)

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/LinearSolver/"
  "${LIBRARY_SOURCES}"
  "Convergence;DataStructures"
  )
