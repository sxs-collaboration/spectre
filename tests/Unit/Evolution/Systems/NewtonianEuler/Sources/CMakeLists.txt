# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NewtonianEulerSources")

set(LIBRARY_SOURCES
  Test_LaneEmdenGravitationalField.cpp
  Test_UniformAcceleration.cpp
  Test_VortexPerturbation.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  NewtonianEulerSources
  NewtonianEulerSolutions
  Utilities
  )
