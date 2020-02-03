# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_CurvedScalarWave")

set(LIBRARY_SOURCES
  Test_Constraints.cpp
  TestHelpers.cpp
  Test_Characteristics.cpp
  Test_DuDt.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/CurvedScalarWave/"
  "${LIBRARY_SOURCES}"
  "CurvedScalarWave;Test_GeneralRelativity"
  )
