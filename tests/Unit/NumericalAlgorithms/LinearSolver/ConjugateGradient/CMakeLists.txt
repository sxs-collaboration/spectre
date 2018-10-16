# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ConjugateGradient")

set(LIBRARY_SOURCES
  Test_ElementActions.cpp
  Test_ResidualMonitorActions.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/LinearSolver/ConjugateGradient"
  "${LIBRARY_SOURCES}"
  "DataStructures;LinearSolver"
  )

# This code is adapted from Parallel/CMakeLists.txt

function(add_algorithm_test TEST_NAME)
  set(EXECUTABLE_NAME Test_${TEST_NAME})
  set(TEST_IDENTIFIER Integration.LinearSolver.${TEST_NAME})

  add_executable(
    ${EXECUTABLE_NAME}
    ${EXECUTABLE_NAME}.cpp
    )

  add_dependencies(
    ${EXECUTABLE_NAME}
    module_ConstGlobalCache
    module_Main
    )

  target_link_libraries(
    ${EXECUTABLE_NAME}
    ErrorHandling
    Informer
    DataStructures
    LinearSolver
    ${SPECTRE_LIBRARIES}
    )

  add_dependencies(test-executables ${EXECUTABLE_NAME})

  add_test(
    NAME "\"${TEST_IDENTIFIER}\""
    COMMAND ${CMAKE_BINARY_DIR}/bin/${EXECUTABLE_NAME} --input-file ${CMAKE_SOURCE_DIR}/tests/Unit/NumericalAlgorithms/LinearSolver/ConjugateGradient/${EXECUTABLE_NAME}.input
    )

  set_tests_properties(
    "\"${TEST_IDENTIFIER}\""
    PROPERTIES
    TIMEOUT 30
    LABELS "integration"
    ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
endfunction()

add_algorithm_test("ConjugateGradientAlgorithm")
add_algorithm_test("DistributedConjugateGradientAlgorithm")
