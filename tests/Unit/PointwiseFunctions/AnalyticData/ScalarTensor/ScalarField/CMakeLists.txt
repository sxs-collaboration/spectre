# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_STAnalyticDataScalarField")

set(LIBRARY_SOURCES
  Test_Inverser.cpp
  Test_SuperposedInverser.cpp
  Test_Zero.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  ErrorHandling
  Framework
  STAnalyticDataScalarField
)
