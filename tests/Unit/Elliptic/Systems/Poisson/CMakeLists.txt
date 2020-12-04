# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Poisson")

set(LIBRARY_SOURCES
  Test_Equations.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Systems/Poisson/"
  "${LIBRARY_SOURCES}"
  "DataStructures;GeneralRelativity;LinearOperators;Poisson;Utilities"
  )
