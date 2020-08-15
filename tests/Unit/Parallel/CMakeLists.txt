# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Actions)

function(add_algorithm_test EXECUTABLE_NAME)
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
    ErrorHandling
    Informer
    Options
    Parallel
    Utilities
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

# Test GlobalCache
add_charm_module(Test_GlobalCache)

add_dependencies(
  module_Test_GlobalCache
  module_GlobalCache
  )

add_spectre_executable(
  Test_GlobalCache
  Test_GlobalCache.cpp
  )

add_dependencies(
  Test_GlobalCache
  module_Test_GlobalCache
  )

target_link_libraries(
  Test_GlobalCache
  PRIVATE
  ErrorHandling
  Informer
  Parallel
  )

add_dependencies(test-executables Test_GlobalCache)

spectre_add_catch_tests(Test_GlobalCache "")

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
type 'db::DataBox<brigand::list<")
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
set(LIBRARY "Test_Parallel")

set(LIBRARY_SOURCES
  Test_GlobalCacheDataBox.cpp
  Test_InboxInserters.cpp
  Test_NodeLock.cpp
  Test_Parallel.cpp
  Test_ParallelComponentHelpers.cpp
  Test_PupStlCpp11.cpp
  Test_TypeTraits.cpp
  )

add_test_library(
  ${LIBRARY}
  "Parallel"
  "${LIBRARY_SOURCES}"
  "Options"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
