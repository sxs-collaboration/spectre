# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelLinearSolver")

set(LIBRARY_SOURCES
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/LinearSolver/"
  "${LIBRARY_SOURCES}"
  "Convergence;DataStructures;Options;ParallelLinearSolver"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )

function(add_linear_solver_algorithm_test TEST_NAME)
  set(EXECUTABLE_NAME Test_${TEST_NAME})
  set(TEST_IDENTIFIER Integration.LinearSolver.${TEST_NAME})

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
    PUBLIC
    # Link against Boost::program_options for now until we have proper
    # dependency handling for header-only libs
    Boost::program_options
    DataStructures
    ErrorHandling
    IO
    Informer
    Parallel
    ParallelLinearSolver
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

add_subdirectory(AsynchronousSolvers)
add_subdirectory(ConjugateGradient)
add_subdirectory(Gmres)
add_subdirectory(Richardson)
add_subdirectory(Schwarz)
