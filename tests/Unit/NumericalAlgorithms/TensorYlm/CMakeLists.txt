# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_TensorYlm")

set(LIBRARY_SOURCES
  Test_CartToSphere.cpp
  Test_Filter.cpp
  Test_SphereToCart.cpp
  Test_WignerThreeJ.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  SphericalHarmonics
  TensorYlm
  Utilities
  )
