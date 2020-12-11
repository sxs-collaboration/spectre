# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelNewtonRaphson")

set(LIBRARY_SOURCES
  Test_LineSearch.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/NonlinearSolver/NewtonRaphson"
  "${LIBRARY_SOURCES}"
  "ParallelNonlinearSolver"
  )

# This function is adapted from
# tests/Unit/ParallelAlgorithms/LinearSolver/CMakeLists.txt
function(add_nonlinear_solver_algorithm_test TEST_NAME)
  set(EXECUTABLE_NAME Test_${TEST_NAME})
  set(TEST_IDENTIFIER Integration.NonlinearSolver.${TEST_NAME})

  add_spectre_executable(
    ${EXECUTABLE_NAME}
    ${EXECUTABLE_NAME}.cpp
    )

  add_dependencies(
    ${EXECUTABLE_NAME}
    module_GlobalCache
    module_Main
    )

  target_link_libraries(
    ${EXECUTABLE_NAME}
    PRIVATE
    # Link against Boost::program_options for now until we have proper
    # dependency handling for header-only libs
    Boost::program_options
    Convergence
    DataStructures
    ErrorHandling
    Informer
    IO
    ParallelLinearSolver
    ParallelNonlinearSolver
    )

  add_dependencies(test-executables ${EXECUTABLE_NAME})

  add_test(
    NAME "\"${TEST_IDENTIFIER}\""
    COMMAND ${CMAKE_BINARY_DIR}/bin/${EXECUTABLE_NAME} --input-file
    ${CMAKE_CURRENT_SOURCE_DIR}/${EXECUTABLE_NAME}.yaml
    )

  set_tests_properties(
    "\"${TEST_IDENTIFIER}\""
    PROPERTIES
    TIMEOUT 5
    LABELS "integration"
    ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
endfunction()

add_nonlinear_solver_algorithm_test("NewtonRaphsonAlgorithm")
