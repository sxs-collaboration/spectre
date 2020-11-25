# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY NewtonianEulerSolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  IsentropicVortex.cpp
  LaneEmdenStar.cpp
  RiemannProblem.cpp
  SmoothFlow.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  IsentropicVortex.hpp
  LaneEmdenStar.hpp
  RiemannProblem.hpp
  SmoothFlow.hpp
  Solutions.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Hydro
  HydroSolutions
  NewtonianEuler
  Options
  )
