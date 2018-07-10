# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Poisson")

set(LIBRARY_SOURCES
  Test_PoissonTags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Systems/Poisson/"
  "${LIBRARY_SOURCES}"
  ""
  )
