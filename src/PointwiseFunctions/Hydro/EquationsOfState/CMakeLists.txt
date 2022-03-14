# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  DarkEnergyFluid.cpp
  IdealFluid.cpp
  PolytropicFluid.cpp
  Spectral.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DarkEnergyFluid.hpp
  EquationOfState.hpp
  Factory.hpp
  IdealFluid.hpp
  PolytropicFluid.hpp
  Spectral.hpp
  )

add_subdirectory(Python)
