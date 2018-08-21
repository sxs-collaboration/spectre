# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_TestUtilities")

set(LIBRARY_SOURCES
  Test_MakeWithRandomValues.cpp
  Test_RandomUnitNormal.cpp
  )

add_test_library(
  ${LIBRARY}
  "TestUtilities"
  "${LIBRARY_SOURCES}"
  "DataStructures;TestUtilities"
  )
