// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "Evolution/Systems/ScalarWave/Python/FilledSpherePowerMonitor.hpp"
#include "Evolution/Systems/ScalarWave/Python/SphericalShellPowerMonitor.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  py::module_::import("spectre.Domain");
  py::module_::import("spectre.Spectral");

  ScalarWave::power_monitor::py_bindings::bind_filled_sphere_power_monitor(m);
  ScalarWave::power_monitor::py_bindings::bind_spherical_shell_power_monitor(m);
}
