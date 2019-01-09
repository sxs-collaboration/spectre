# Distributed under the MIT License.
# See LICENSE.txt for details.

function(add_algorithm_test EXECUTABLE_NAME)
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
    ErrorHandling
    Informer
    Utilities
    ${SPECTRE_LIBRARIES}
    )

  add_dependencies(test-executables ${EXECUTABLE_NAME})
endfunction()

add_algorithm_test(Test_AlgorithmCore)
add_algorithm_test(Test_AlgorithmBadBoxApply)
add_algorithm_test(Test_AlgorithmNestedApply1)
add_algorithm_test(Test_AlgorithmNestedApply2)
add_algorithm_test(Test_AlgorithmParallel)
add_algorithm_test(Test_AlgorithmNodelock)
add_algorithm_test(Test_AlgorithmReduction)

# Test ConstGlobalCache
add_charm_module(Test_ConstGlobalCache)

add_dependencies(
  module_Test_ConstGlobalCache
  module_ConstGlobalCache
  )

add_spectre_executable(
  Test_ConstGlobalCache
  Test_ConstGlobalCache.cpp
  )

add_dependencies(
  Test_ConstGlobalCache
  module_Test_ConstGlobalCache
  )

target_link_libraries(
  Test_ConstGlobalCache
  ErrorHandling
  ${SPECTRE_LIBRARIES}
  )

add_dependencies(test-executables Test_ConstGlobalCache)

spectre_add_catch_tests(Test_ConstGlobalCache "")

# Add integration tests. For tests that result in a failure it is necessary to
# redirect output from stderr to stdout. However, it was necessary at least on
# some systems to do this redirect inside a shell command.
find_program(SHELL_EXECUTABLE "sh")
if ("${SHELL_EXECUTABLE}" STREQUAL "")
  message(FATAL_ERROR
    "Could not find 'sh' shell to execute failure tests of algorithms")
endif()

function(add_algorithm_test TEST_NAME REGEX_TO_MATCH)
  add_test(
    NAME "\"Integration.Parallel.${TEST_NAME}\""
    COMMAND
    ${SHELL_EXECUTABLE}
    -c
    "${CMAKE_BINARY_DIR}/bin/Test_${TEST_NAME} 2>&1"
    )

  if("${REGEX_TO_MATCH}" STREQUAL "")
    set_tests_properties(
      "\"Integration.Parallel.${TEST_NAME}\""
      PROPERTIES
      TIMEOUT 10
      LABELS "integration"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
  else()
    set_tests_properties(
      "\"Integration.Parallel.${TEST_NAME}\""
      PROPERTIES
      PASS_REGULAR_EXPRESSION
      "${REGEX_TO_MATCH}"
      TIMEOUT 10
      LABELS "integration"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
  endif()
endfunction()

add_algorithm_test("AlgorithmCore" "")
add_algorithm_test(
  "AlgorithmBadBoxApply"
  "Cannot call apply function of 'error_size_zero' with DataBox \
type 'db::DataBox<brigand::list<> >'")
add_algorithm_test(
  "AlgorithmNestedApply1"
  "Already performing an Action and cannot execute additional Actions \
from inside of an Action. This is only possible if the simple_action \
function is not invoked via a proxy, which we do not allow.")
add_algorithm_test(
  "AlgorithmNestedApply2"
  "Already performing an Action and cannot execute additional Actions \
from inside of an Action. This is only possible if the simple_action \
function is not invoked via a proxy, which we do not allow.")
add_algorithm_test("AlgorithmParallel" "")
add_algorithm_test("AlgorithmReduction" "")
add_algorithm_test("AlgorithmNodelock" "")

# Tests that do not require their own Chare setup and can work with the
# unit tests
set(PARALLEL_TESTS
  Parallel/Test_ConstGlobalCacheDataBox.cpp
  Parallel/Test_GotoAction.cpp
  Parallel/Test_Parallel.cpp
  Parallel/Test_ParallelComponentHelpers.cpp
  Parallel/Test_PupStlCpp11.cpp
  Parallel/Test_TypeTraits.cpp
  )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${PARALLEL_TESTS}" PARENT_SCOPE)
