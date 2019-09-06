# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NewtonianEulerSources")

set(LIBRARY_SOURCES
  Test_UniformAcceleration.cpp
  Test_VortexPerturbation.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/NewtonianEuler/Sources"
  "${LIBRARY_SOURCES}"
  "NewtonianEulerSources"
  )
