# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelLinearSolver")

set(LIBRARY_SOURCES
  Actions/Test_TerminateIfConverged.cpp
  Test_InnerProduct.cpp
  Test_Tags.cpp
  )

add_subdirectory(ConjugateGradient)
add_subdirectory(Gmres)

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/LinearSolver/"
  "${LIBRARY_SOURCES}"
  "Convergence;DataStructures;Options;ParallelLinearSolver"
  )

add_dependencies(
  ${LIBRARY}
  module_ConstGlobalCache
  )
