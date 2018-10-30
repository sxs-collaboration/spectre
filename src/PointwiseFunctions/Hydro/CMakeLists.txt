# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Hydro)

set(LIBRARY_SOURCES
  AlfvenSpeedSquared.cpp
  ComovingMagneticField.cpp
  EquationsOfState/DarkEnergyFluid.cpp
  EquationsOfState/IdealFluid.cpp
  EquationsOfState/PolytropicFluid.cpp
  LorentzFactor.cpp
  SpecificEnthalpy.cpp
  TransportVelocity.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )
