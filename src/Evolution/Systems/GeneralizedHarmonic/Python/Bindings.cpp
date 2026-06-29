// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "Evolution/Systems/GeneralizedHarmonic/Python/SphericalShellPowerMonitor.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  py::module_::import("spectre.Domain");
  py::module_::import("spectre.Spectral");

  gh::power_monitor::py_bindings::bind_spherical_shell_power_monitor(m);
}
