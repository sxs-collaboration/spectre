# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Actions)

add_subdirectory(PhaseControl)

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
    PhaseControl
    SystemUtilities
    Utilities
    )

  add_dependencies(unit-tests ${EXECUTABLE_NAME})
endfunction()

add_algorithm_test(Test_AlgorithmCore)
add_algorithm_test(Test_AlgorithmBadBoxApply)
add_algorithm_test(Test_AlgorithmGlobalCache)
add_algorithm_test(Test_AlgorithmLocalSyncAction)
add_algorithm_test(Test_AlgorithmNestedApply1)
add_algorithm_test(Test_AlgorithmNestedApply2)
add_algorithm_test(Test_AlgorithmParallel)
add_algorithm_test(Test_AlgorithmPhaseControl)
add_algorithm_test(Test_PhaseChangeMain)
add_algorithm_test(Test_AlgorithmNodelock)
add_algorithm_test(Test_AlgorithmReduction)
add_algorithm_test(Test_GlobalCache)
add_algorithm_test(Test_SectionReductions)

# Test GlobalCache
add_charm_module(Test_GlobalCache)

add_dependencies(
  module_Test_GlobalCache
  module_GlobalCache
  )

add_dependencies(
  Test_GlobalCache
  module_Test_GlobalCache
  )

# Add unit tests. For tests that result in a failure it is necessary to
# redirect output from stderr to stdout. However, it was necessary at least on
# some systems to do this redirect inside a shell command.
find_program(SHELL_EXECUTABLE "sh")
if ("${SHELL_EXECUTABLE}" STREQUAL "")
  message(FATAL_ERROR
    "Could not find 'sh' shell to execute failure tests of algorithms")
endif()

function(add_algorithm_test TEST_NAME REGEX_TO_MATCH)
  add_test(
    NAME Unit.Parallel.${TEST_NAME}
    COMMAND
    ${SHELL_EXECUTABLE}
    -c
    "${CMAKE_BINARY_DIR}/bin/Test_${TEST_NAME} 2>&1"
    )

  if("${REGEX_TO_MATCH}" STREQUAL "")
    set_tests_properties(
      Unit.Parallel.${TEST_NAME}
      PROPERTIES
      FAIL_REGULAR_EXPRESSION "ERROR"
      TIMEOUT 10
      LABELS "unit"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
  else()
    set_tests_properties(
      Unit.Parallel.${TEST_NAME}
      PROPERTIES
      PASS_REGULAR_EXPRESSION
      "${REGEX_TO_MATCH}"
      TIMEOUT 10
      LABELS "unit"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
  endif()
endfunction()

function(add_algorithm_test_with_input_file TEST_NAME INPUT_FILE_DIR)
  add_test(
    NAME "\"Unit.Parallel.${TEST_NAME}\""
    COMMAND
    ${SHELL_EXECUTABLE}
    -c
    "${CMAKE_BINARY_DIR}/bin/Test_${TEST_NAME} --input-file \
     ${CMAKE_SOURCE_DIR}/tests/${INPUT_FILE_DIR}/Test_${TEST_NAME}.yaml 2>&1"
    )
  set_tests_properties(
    "\"Unit.Parallel.${TEST_NAME}\""
    PROPERTIES
    TIMEOUT 10
    LABELS "unit"
    ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
endfunction()

# Note that we have a 'balance' here that happens 2.5e4 of times per second just
# to be completely certain that the desired number of migrations occur during
# the test (2). This tends to have a lot of fluctuation depending on how quickly
# the machine manages to execute this fairly short test, and the current balance
# frequency has been roughly tuned to a) very consistently get at least 5
# balances (usually gets several dozen), and b) complete the test in a
# reasonable duration. Balancing too frequently can mean the test time explodes
# due to balancing so much that the test computation doesn't get much of a
# chance to execute.
function(add_algorithm_test_with_balancing TEST_NAME EXECUTABLE_NAME REGEX_TO_MATCH)
  add_test(
    NAME "\"Unit.Parallel.${TEST_NAME}\""
    COMMAND
    ${SHELL_EXECUTABLE}
    -c "${CMAKE_BINARY_DIR}/bin/Test_${EXECUTABLE_NAME} +p2 +balancer \
     RotateLB +LBPeriod 4.0e-5 +LBDebug 3 2>&1"
    )

  if("${REGEX_TO_MATCH}" STREQUAL "")
    set_tests_properties(
      "\"Unit.Parallel.${TEST_NAME}\""
      PROPERTIES
      FAIL_REGULAR_EXPRESSION "ERROR"
      TIMEOUT 10
      LABELS "unit"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
  else()
    set_tests_properties(
      "\"Unit.Parallel.${TEST_NAME}\""
      PROPERTIES
      PASS_REGULAR_EXPRESSION
      "${REGEX_TO_MATCH}"
      FAIL_REGULAR_EXPRESSION "ERROR"
      TIMEOUT 10
      LABELS "unit"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
  endif()
endfunction()

add_algorithm_test("AlgorithmCore" "")
add_algorithm_test(
  "AlgorithmBadBoxApply"
  "Cannot call apply function of 'error_size_zero' with DataBox \
type 'db::DataBox<brigand::list<")
add_algorithm_test("AlgorithmLocalSyncAction" "")
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
add_algorithm_test("PhaseChangeMain" "")
add_algorithm_test("AlgorithmReduction" "")
add_algorithm_test("AlgorithmNodelock" "")
add_algorithm_test("GlobalCache" "")
add_algorithm_test("SectionReductions"
  "Element 0 received reduction: Counted 2 odd elements.")
add_algorithm_test_with_balancing("AlgorithmParallelWithBalancing"
  "AlgorithmParallel" "Objects migrating: 14")

add_algorithm_test_with_input_file("AlgorithmGlobalCache" "Unit/Parallel")
add_algorithm_test_with_input_file("AlgorithmPhaseControl" "Unit/Parallel")

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
  Test_PupStlCpp17.cpp
  Test_TypeTraits.cpp
  )

add_subdirectory(Tags)

add_test_library(
  ${LIBRARY}
  "Parallel"
  "${LIBRARY_SOURCES}"
  "Options;SystemUtilities"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
