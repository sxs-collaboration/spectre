# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GhValenciaDivClean")

set(LIBRARY_SOURCES
  Test_Tags.cpp
  Test_StressEnergy.cpp
  Test_TimeDerivativeTerms.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/GrMhd/GhValenciaDivClean"
  "${LIBRARY_SOURCES}"
  "Framework;GhValenciaDivClean"
  "DomainHelpers;Framework;GeneralRelativityHelpers;GeneralizedHarmonic;ValenciaDivClean;GhValenciaDivClean"
  )
