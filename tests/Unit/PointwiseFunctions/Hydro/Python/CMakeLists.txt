# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "Unit.Hydro.ComovingMagneticField.Python"
  ComovingMagneticField.py
  "unit;hydro;python"
  PyHydro)

spectre_add_python_bindings_test(
  "Unit.Hydro.LorentzFactor.Python"
  LorentzFactor.py
  "unit;hydro;python"
  PyHydro)

spectre_add_python_bindings_test(
  "Unit.Hydro.MassFlux.Python"
  MassFlux.py
  "unit;hydro;python"
  PyHydro)

spectre_add_python_bindings_test(
  "Unit.Hydro.MassWeightedFluidItems.Python"
  MassWeightedFluidItems.py
  "unit;hydro;python"
  PyHydro)

spectre_add_python_bindings_test(
  "Unit.Hydro.SoundSpeedSquared.Python"
  SoundSpeedSquared.py
  "unit;hydro;python"
  PyHydro)

spectre_add_python_bindings_test(
  "Unit.Hydro.SpecificEnthalpy.Python"
  SpecificEnthalpy.py
  "unit;hydro;python"
  PyHydro)
