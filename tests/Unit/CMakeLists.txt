# Distributed under the MIT License.
# See LICENSE.txt for details.

set(executable RunTests)

set(SPECTRE_TESTS "TestFramework.cpp")

add_subdirectory(DataStructures)
add_subdirectory(Domain)
add_subdirectory(ErrorHandling)
add_subdirectory(Numerical)
add_subdirectory(Parallel)
add_subdirectory(Time)
add_subdirectory(Utilities)

set(SPECTRE_TESTS "${SPECTRE_TESTS};Test_TestHelpers.cpp")

add_charm_mainmodule(${executable})

add_executable(
    ${executable}
    ${executable}.cpp
    ${executable}.decl.h
    ${executable}.def.h
    ${SPECTRE_TESTS}
)

target_link_libraries(
    ${executable}
    ${SPECTRE_LIBRARIES}
)

spectre_add_catch_tests(${executable})

# Setup code coverage for unit tests
if(COVERAGE)
  # Setup unit test coverage target.
  setup_target_for_coverage(
      Unit
      ${CMAKE_BINARY_DIR}/docs/html
      unit-test-coverage
      ${CMAKE_CTEST_COMMAND}
      DEPENDS ${executable}
      IGNORE_COV
      '${CMAKE_BINARY_DIR}/Informer/InfoFromBuild.cpp'
      '${CMAKE_SOURCE_DIR}/tests/*'
  )
endif()

