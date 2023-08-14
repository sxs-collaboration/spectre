# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyHydro")

spectre_python_add_module(
  Hydro
  LIBRARY_NAME ${LIBRARY}
  SOURCES
  Bindings.cpp
  ComovingMagneticField.cpp
  LorentzFactor.cpp
  LowerSpatialFourVelocity.cpp
  MassFlux.cpp
  MassWeightedFluidItems.cpp
  SoundSpeedSquared.cpp
  SpecificEnthalpy.cpp
  StressEnergy.cpp
  PYTHON_FILES
  __init__.py
  MODULE_PATH "PointwiseFunctions"
  )

spectre_python_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ComovingMagneticField.hpp
  LorentzFactor.hpp
  LowerSpatialFourVelocity.hpp
  MassFlux.hpp
  MassWeightedFluidItems.hpp
  SoundSpeedSquared.hpp
  SpecificEnthalpy.hpp
  StressEnergy.hpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Hydro
  pybind11::module
  Utilities
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PyTensor
  )
