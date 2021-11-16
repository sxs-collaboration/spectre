# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Ccz4")

set(LIBRARY_SOURCES
  Test_ATilde.cpp
  Test_Christoffel.cpp
  Test_DerivChristoffel.cpp
  Test_DerivLapse.cpp
  Test_DerivZ4Constraint.cpp
  Test_Ricci.cpp
  Test_RicciScalarPlusDivergenceZ4Constraint.cpp
  Test_Tags.cpp
  Test_TempTags.cpp
  Test_Z4Constraint.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/Ccz4/"
  "${LIBRARY_SOURCES}"
  "Ccz4;DataBoxTestHelpers;DataStructures;Domain;GeneralRelativity"
  "GeneralRelativitySolutions;LinearOperators;Spectral;Utilities"
  )
