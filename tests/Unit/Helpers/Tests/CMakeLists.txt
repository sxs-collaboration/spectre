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
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DataStructuresHelpers
  GeneralRelativityHelpers
  )

add_subdirectory(Domain)
add_subdirectory(IO)
add_subdirectory(PointwiseFunctions)
