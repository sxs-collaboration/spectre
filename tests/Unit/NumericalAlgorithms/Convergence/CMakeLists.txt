# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Convergence")

set(LIBRARY_SOURCES
  Test_Criteria.cpp
  Test_HasConverged.cpp
  Test_Reason.cpp
  Test_Tags.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  Convergence
  DataStructures
  )
