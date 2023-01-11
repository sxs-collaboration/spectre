// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "NumericalAlgorithms/LinearOperators/Python/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/LinearOperators/Python/PartialDerivatives.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_PyLinearOperators, m) {  // NOLINT
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  py::module_::import("spectre.Spectral");
  py_bindings::bind_definite_integral(m);
  py_bindings::bind_partial_derivatives(m);
}
