# Distributed under the MIT License.
# See LICENSE.txt for details.


set(LIBRARY "Test_WaveEquation")

# Nils Deppe: The SemidiscretizedDg test is commented out because switching
# it over to the ComputeTimeDerivate and ApplyBoundaryCorrections actions
# is too time consuming at the moment (Mar. 25, 2021). We should do one of
# the following:
# - switch it over later
# - switch to a pypp test that we use as a regression test
set(LIBRARY_SOURCES
  Test_PlaneWave.cpp
  Test_RegularSphericalWave.cpp
  # Test_SemidiscretizedDg.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  WaveEquationSolutions
  MathFunctions
  Utilities
  )
