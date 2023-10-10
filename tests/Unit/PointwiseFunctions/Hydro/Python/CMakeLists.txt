# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "Unit.Hydro.ComovingMagneticField.Python"
  Test_ComovingMagneticField.py
  "unit;hydro;python"
  PyHydro)

spectre_add_python_bindings_test(
  "Unit.Hydro.LorentzFactor.Python"
  Test_LorentzFactor.py
  "unit;hydro;python"
  PyHydro)

spectre_add_python_bindings_test(
  "Unit.Hydro.MassFlux.Python"
  Test_MassFlux.py
  "unit;hydro;python"
  PyHydro)

spectre_add_python_bindings_test(
  "Unit.Hydro.MassWeightedFluidItems.Python"
  Test_MassWeightedFluidItems.py
  "unit;hydro;python"
  PyHydro)

spectre_add_python_bindings_test(
  "Unit.Hydro.SoundSpeedSquared.Python"
  Test_SoundSpeedSquared.py
  "unit;hydro;python"
  PyHydro)

spectre_add_python_bindings_test(
  "Unit.Hydro.SpecificEnthalpy.Python"
  Test_SpecificEnthalpy.py
  "unit;hydro;python"
  PyHydro)
