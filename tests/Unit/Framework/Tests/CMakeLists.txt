# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Framework")

set(LIBRARY_SOURCES
  Test_ActionTesting.cpp
  Test_Pypp.cpp
  Test_TestCreation.cpp
  Test_TestHelpers.cpp
  Test_TestingFramework.cpp
  )

add_test_library(
  ${LIBRARY}
  "Framework/Tests"
  "${LIBRARY_SOURCES}"
  "Boost::boost;DataStructures;ErrorHandling;Python::NumPy;\
Python::Python;Python::SciPy"
  )

add_dependencies(
  ${LIBRARY}
  module_ConstGlobalCache
  )
