# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Actions)
add_subdirectory(DiscontinuousGalerkin)
add_subdirectory(Systems)
add_subdirectory(Triggers)

set(LIBRARY "Test_Elliptic")

set(LIBRARY_SOURCES
  Test_FirstOrderComputeTags.cpp
  Test_FirstOrderOperator.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Elliptic;LinearOperators;Utilities"
  )
