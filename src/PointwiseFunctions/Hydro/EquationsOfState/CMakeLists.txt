# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  Hydro
  PRIVATE
  DarkEnergyFluid.cpp
  IdealFluid.cpp
  PolytropicFluid.cpp
  )

spectre_target_headers(
  Hydro
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DarkEnergyFluid.hpp
  EquationOfState.hpp
  IdealFluid.hpp
  PolytropicFluid.hpp
  )

add_subdirectory(Python)
