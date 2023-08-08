# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ScalarTensor")

set(LIBRARY_SOURCES
  BoundaryConditions/Test_ProductOfConditions.cpp
  BoundaryCorrections/Test_ProductOfCorrections.cpp
  Test_Characteristics.cpp
  Test_Sources.cpp
  Test_StressEnergy.cpp
  Test_Tags.cpp
  Test_TimeDerivative.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/ScalarTensor"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Framework
  GeneralRelativityHelpers
  ScalarTensor
  )
