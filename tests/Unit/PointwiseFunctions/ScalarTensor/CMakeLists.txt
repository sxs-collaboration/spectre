# Distributed under the MIT License.
# See LICENSE.txt for details.
set(LIBRARY "Test_ScalarTensorPointwise")

set(LIBRARY_SOURCES
  Test_ConstraintDampingTags.cpp
  Test_ConstraintGammas.cpp
  Test_RampUpFunction.cpp
  Test_ScalarCharge.cpp
  Test_ScalarSources.cpp
  Test_SourceTags.cpp
  Test_StressEnergy.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")
target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ConstraintDamping
  Framework
  ScalarTensor
  ScalarTensorPointwise
  )
