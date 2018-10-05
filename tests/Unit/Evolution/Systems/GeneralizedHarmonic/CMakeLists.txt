# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GeneralizedHarmonic")

set(LIBRARY_SOURCES
  Test_Constraints.cpp
  Test_DuDt.cpp
  Test_Fluxes.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/GeneralizedHarmonic/"
  "${LIBRARY_SOURCES}"
  "GeneralizedHarmonic"
  )
