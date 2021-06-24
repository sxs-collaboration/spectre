# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY NewtonianEulerSources)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  LaneEmdenGravitationalField.cpp
  UniformAcceleration.cpp
  VortexPerturbation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  LaneEmdenGravitationalField.hpp
  NoSource.hpp
  Source.hpp
  UniformAcceleration.hpp
  VortexPerturbation.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  NewtonianEulerSolutions
  Utilities
  )
