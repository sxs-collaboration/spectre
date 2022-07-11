# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GhConstraintDamping")

set(LIBRARY_SOURCES
  Test_Constant.cpp
  Test_GaussianPlusConstant.cpp
  Test_TimeDependentTripleGaussian.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  GhConstraintDamping
  GhConstraintDampingHelpers
  Parallel
  Utilities
)
