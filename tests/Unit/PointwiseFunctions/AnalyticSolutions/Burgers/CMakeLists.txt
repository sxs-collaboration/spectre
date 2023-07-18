# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_BurgersSolutions")

set(LIBRARY_SOURCES
  Test_Bump.cpp
  Test_Linear.cpp
  Test_Step.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/Burgers/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  BurgersSolutions
  Burgers
  Utilities
)
