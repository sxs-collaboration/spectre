# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Framework")

set(LIBRARY_SOURCES
  Test_ActionTesting.cpp
  Test_Pypp.cpp
  Test_PyppAnalyticSolution.cpp
  Test_PyppRandomValues.cpp
  Test_TestCreation.cpp
  Test_TestHelpers.cpp
  Test_TestingFramework.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  DataStructures
  SystemUtilities
  )
