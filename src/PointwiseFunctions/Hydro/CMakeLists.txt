# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Hydro)

set(LIBRARY_SOURCES
  EquationsOfState/DarkEnergyFluid.cpp
  EquationsOfState/IdealFluid.cpp
  EquationsOfState/PolytropicFluid.cpp
  LorentzFactor.cpp
  MassFlux.cpp
  SoundSpeedSquared.cpp
  SpecificEnthalpy.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  ErrorHandling
  Options
  )

if(BUILD_PYTHON_BINDINGS)
  add_subdirectory(EquationsOfState/Python)
endif()
