# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(GaugeSourceFunctions)

set(LIBRARY "Test_GeneralizedHarmonic")

set(LIBRARY_SOURCES
  Test_Characteristics.cpp
  Test_Constraints.cpp
  Test_DuDt.cpp
  Test_Fluxes.cpp
  Test_Tags.cpp
  Test_UpwindPenaltyCorrection.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/GeneralizedHarmonic/"
  "${LIBRARY_SOURCES}"
  "GeneralRelativityHelpers;GeneralizedHarmonic;Test_GeneralRelativity"
  )
