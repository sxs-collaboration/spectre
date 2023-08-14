// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <pybind11/pybind11.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/Python/ComovingMagneticField.hpp"
#include "PointwiseFunctions/Hydro/Python/LorentzFactor.hpp"
#include "PointwiseFunctions/Hydro/Python/LowerSpatialFourVelocity.hpp"
#include "PointwiseFunctions/Hydro/Python/MassFlux.hpp"
#include "PointwiseFunctions/Hydro/Python/MassWeightedFluidItems.hpp"
#include "PointwiseFunctions/Hydro/Python/SoundSpeedSquared.hpp"
#include "PointwiseFunctions/Hydro/Python/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Python/StressEnergy.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.Spectral");
  py_bindings::bind_comoving_magnetic_field(m);
  py_bindings::bind_lorentz(m);
  py_bindings::bind_lorentz_factor(m);
  py_bindings::bind_lower_spatial_four_velocity(m);
  py_bindings::bind_mass_flux(m);
  py_bindings::bind_mass_weighted(m);
  py_bindings::bind_sound_speed(m);
  py_bindings::bind_specific_enthalpy(m);
  py_bindings::bind_stress_energy(m);
}
