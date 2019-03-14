# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Gmres")

set(LIBRARY_SOURCES
  Test_ElementActions.cpp
  Test_ResidualMonitorActions.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/LinearSolver/Gmres"
  "${LIBRARY_SOURCES}"
  "Convergence;DataStructures;IO"
  )

# This code is adapted from Parallel/CMakeLists.txt

function(add_algorithm_test TEST_NAME)
  set(EXECUTABLE_NAME Test_${TEST_NAME})
  set(TEST_IDENTIFIER Integration.LinearSolver.${TEST_NAME})

  add_spectre_executable(
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
    Convergence
    DataStructures
    ErrorHandling
    Informer
    IO
    ${SPECTRE_LIBRARIES}
    )

  add_dependencies(test-executables ${EXECUTABLE_NAME})

  add_test(
    NAME "\"${TEST_IDENTIFIER}\""
    COMMAND ${CMAKE_BINARY_DIR}/bin/${EXECUTABLE_NAME} --input-file
    ${CMAKE_CURRENT_SOURCE_DIR}/${EXECUTABLE_NAME}.input
    )

  set_tests_properties(
    "\"${TEST_IDENTIFIER}\""
    PROPERTIES
    TIMEOUT 5
    LABELS "integration"
    ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
endfunction()

add_algorithm_test("GmresAlgorithm")
add_algorithm_test("DistributedGmresAlgorithm")
