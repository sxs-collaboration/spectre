# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE "RunSingleTest")

# RunSingleTest can be used to compile only a subset of the code base
# for testing purposes. It uses the RunTests infrastructure with a slightly
# different implementation in the cpp file in order to not depend on all
# the test libraries. Note that while typically the `RunSingleTest`
# executable will be used for quick write/build/run cycles of a single test,
# more than one test and source file can be compiled. The main use of
# `RunSingleTest` is to decrease turnaround time when changing very low-level
# code such as DataVector or certain parts of Utilities that cause most or
# all of the source tree to be rebuilt. Rather than having to wait for the
# source tree rebuild before running the tests via `RunTests` the individual
# test source files containing the test(s) of interest can be rebuilt.
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
add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  ${EXECUTABLE}.cpp

  # The following lines list the source files with tests in them to be
  # compiled into the executable.
  ../DataStructures/Test_DataVector.cpp
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
  Framework
  Informer
  )

add_dependencies(
  ${EXECUTABLE}
  module_ConstGlobalCache
  module_Main
  module_RunTests
  )
