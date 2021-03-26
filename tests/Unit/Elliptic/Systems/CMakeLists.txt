# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticSystems")

set(LIBRARY_SOURCES
  Test_GetSourcesComputer.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Systems/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  EllipticSystems
  )

add_subdirectory(Elasticity)
add_subdirectory(Poisson)
add_subdirectory(Xcts)
