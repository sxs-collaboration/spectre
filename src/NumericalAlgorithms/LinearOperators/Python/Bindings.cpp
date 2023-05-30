// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "NumericalAlgorithms/LinearOperators/Python/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/LinearOperators/Python/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/Python/PowerMonitors.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  py::module_::import("spectre.Spectral");
  py_bindings::bind_definite_integral(m);
  py_bindings::bind_partial_derivatives(m);
  PowerMonitors::py_bindings::bind_power_monitors(m);
}
