# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GhConstraintDamping")

set(LIBRARY_SOURCES
  Test_GaussianPlusConstant.cpp
  Test_TimeDependentTripleGaussian.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/"
  "${LIBRARY_SOURCES}"
  "GhConstraintDamping;Utilities"
  )
