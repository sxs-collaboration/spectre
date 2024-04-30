// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "NumericalAlgorithms/SphericalHarmonics/Python/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Python/StrahlkorperFunctions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  ylm::py_bindings::bind_strahlkorper(m);
  ylm::py_bindings::bind_strahlkorper_functions(m);
}
