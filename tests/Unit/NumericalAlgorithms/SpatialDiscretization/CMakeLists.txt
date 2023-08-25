# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_SpatialDiscretization")


set(LIBRARY_SOURCES
  Test_Basis.cpp
  Test_Mesh.cpp
  Test_Quadrature.cpp
 )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ErrorHandling
  Options
  SpatialDiscretization
  Spectral
  Utilities
  )
