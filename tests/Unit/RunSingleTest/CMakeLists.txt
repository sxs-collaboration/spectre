# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE "RunSingleTest")

# RunSingleTest can be used to compile only a subset of the code base
# for testing purposes. Since we have switched to building individual test
# executables rather than one giant `RunTests` executable, this is probably
# not needed anymore and can be removed after some testing.
#
# In order to run the tests in one or more test source files without
# compiling the whole source tree:
# 1. Update the cpp files built by replacing `Test_DataVector.cpp`
#    in `add_spectre_executable` below.
# 2. Update the first `target_link_libraries` below to link the required
#    SpECTRE libraries.
# 3. Run `./bin/RunSingleTest` passing as arguments the test(s) to run.
#    If no test names are given then all tests in the built source files
#    are run. If any of them are error tests the executable will terminate
#    when reaching the error.
set(TEST_SOURCES
  # The following lines list the source files with tests in them to be
  # compiled into the executable.
  ../DataStructures/Test_DataVector.cpp
  )
add_test_library(
  ${EXECUTABLE}
  ""
  "${TEST_SOURCES}"
  ""
  )

# Modify which libraries to link for your test case.
target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  DataStructures
  )

# Core required libraries
target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  # Link against Boost::program_options for now until we have proper
  # dependency handling for header-only libs
  Boost::program_options
  Charmxx::main
  ErrorHandling
  Framework
  Informer
  SystemUtilities
  )
