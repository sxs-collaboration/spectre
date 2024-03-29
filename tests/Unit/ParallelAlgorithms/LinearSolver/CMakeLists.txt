# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelLinearSolver")

set(LIBRARY_SOURCES
  Test_Tags.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Convergence
  DataStructures
  Options
  ParallelLinearSolver
  )

set(INTEGRATION_TEST_LINK_LIBRARIES
  Charmxx::main
  DataStructures
  ErrorHandling
  Informer
  LinearSolverHelpers
  Observer
  Parallel
  ParallelLinearSolver
  )

set(DISTRIBUTED_INTEGRATION_TEST_LINK_LIBRARIES
  ${INTEGRATION_TEST_LINK_LIBRARIES}
  Domain
  DomainBoundaryConditionsHelpers
  DomainCreators
  )

add_subdirectory(Actions)
add_subdirectory(AsynchronousSolvers)
add_subdirectory(ConjugateGradient)
add_subdirectory(Gmres)
add_subdirectory(Multigrid)
add_subdirectory(Richardson)
add_subdirectory(Schwarz)
