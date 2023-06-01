// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "DataStructures/Tensor/EagerMath/Python/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Python/Magnitude.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  py_bindings::bind_determinant(m);
  py_bindings::bind_magnitude(m);
}
