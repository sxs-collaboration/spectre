# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Actions)
add_subdirectory(BoundaryConditions)
add_subdirectory(DiscontinuousGalerkin)
add_subdirectory(Python)
add_subdirectory(SubdomainPreconditioners)
add_subdirectory(Systems)
add_subdirectory(Triggers)
add_subdirectory(Utilities)

set(LIBRARY "Test_Elliptic")

set(LIBRARY_SOURCES
  Test_Tags.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataBoxTestHelpers
  Elliptic
  )
