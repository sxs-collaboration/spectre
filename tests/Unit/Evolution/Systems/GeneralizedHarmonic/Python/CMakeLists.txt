# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "Unit.Evolution.Systems.GeneralizedHarmonic.Python.SphericalShellPowerMonitor"
  Test_SphericalShellPowerMonitor.py
  "Unit;Evolution"
  PyEvolutionSystemsGeneralizedHarmonic
  PyGeneralRelativity
  PyGrSolutions
  TIMEOUT 10)

if(BUILD_PYTHON_BINDINGS)
  add_dependencies(unit-tests PyDomainCreators PyDomainTimeDependentOptions)
endif()
