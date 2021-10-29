# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Helpers")

set(LIBRARY_SOURCES
  Test_MakeWithRandomValues.cpp
  Test_RandomUnitNormal.cpp
  Test_MakeRandomVectorInMagnitudeRange.cpp
  )

add_test_library(
  ${LIBRARY}
  "Helpers/Tests"
  "${LIBRARY_SOURCES}"
  "DataStructures;DataStructuresHelpers"
  )

add_subdirectory(PointwiseFunctions)
