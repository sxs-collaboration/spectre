# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DataImporter")

set(LIBRARY_SOURCES
  Test_Tags.cpp
  Test_VolumeDataReaderActions.cpp
  )

add_test_library(
  ${LIBRARY}
  "IO/Importers"
  "${LIBRARY_SOURCES}"
  "Domain;IO;Options"
  )

add_dependencies(
  ${LIBRARY}
  module_ConstGlobalCache
  )

function(add_algorithm_test TEST_NAME DIM)
  set(HPP_NAME Test_${TEST_NAME})
  set(EXECUTABLE_NAME ${HPP_NAME}${DIM}D)
  set(TEST_IDENTIFIER Integration.Importers.${TEST_NAME}${DIM}D)

  add_spectre_parallel_executable(
    ${EXECUTABLE_NAME}
    ${HPP_NAME}
    IO/Importers
    Metavariables<${DIM}>
    "DataStructures;Domain;ErrorHandling;Informer;IO;Options"
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

add_algorithm_test("VolumeDataReaderAlgorithm" 1)
add_algorithm_test("VolumeDataReaderAlgorithm" 2)
add_algorithm_test("VolumeDataReaderAlgorithm" 3)
