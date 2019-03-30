# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_SlopeLimiters")

set(LIBRARY_SOURCES
  Test_Krivodonova.cpp
  Test_LimiterActions.cpp
  Test_LimiterActionsWithMinmod.cpp
  Test_Minmod.cpp
  Test_MinmodTci.cpp
  Test_MinmodType.cpp
  Test_WenoGridHelpers.cpp
  Test_WenoHelpers.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/DiscontinuousGalerkin/SlopeLimiters/"
  "${LIBRARY_SOURCES}"
  "SlopeLimiters"
  )
