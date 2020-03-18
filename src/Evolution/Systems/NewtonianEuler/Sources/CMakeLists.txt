# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY NewtonianEulerSources)

set(LIBRARY_SOURCES
  LaneEmdenGravitationalField.cpp
  UniformAcceleration.cpp
  VortexPerturbation.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC DataStructures
  INTERFACE ErrorHandling
  )
