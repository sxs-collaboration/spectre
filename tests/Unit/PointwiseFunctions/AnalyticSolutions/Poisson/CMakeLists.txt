# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_PoissonSolutions")

set(LIBRARY_SOURCES
  Test_Lorentzian.cpp
  Test_MathFunction.cpp
  Test_Moustache.cpp
  Test_ProductOfSinusoids.cpp
  Test_Zero.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataStructures
  Domain
  Poisson
  PoissonSolutions
  Spectral
  Utilities
)
