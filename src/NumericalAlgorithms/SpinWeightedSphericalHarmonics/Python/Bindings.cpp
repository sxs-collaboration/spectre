// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/Python/SwshCoefficients.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/Python/SwshDerivatives.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py_bindings::bind_goldberg_to_nodal(m);
  py_bindings::bind_nodal_to_goldberg(m);
  py_bindings::bind_swsh_derivatives(m);
}
