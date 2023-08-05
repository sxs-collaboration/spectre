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
  py_bindings::bind_comovingMF(m);
  py_bindings::bind_lorentz(m);
  py_bindings::bind_lorentz_factor(m);
  py_bindings::bind_lowerVel(m);
  py_bindings::bind_massFlux(m);
  py_bindings::bind_massWeighted(m);
  py_bindings::bind_soundSpeed<double, 1>(m);
  py_bindings::bind_soundSpeed<double, 2>(m);
  py_bindings::bind_soundSpeed<DataVector, 1>(m);
  py_bindings::bind_soundSpeed<DataVector, 2>(m);
  py_bindings::bind_specificEnthalpy(m);
  py_bindings::bind_stressEnergy(m);
}
