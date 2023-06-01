// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/Python/Tov.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.Interpolation");
  py::module_::import("spectre.PointwiseFunctions.Hydro.EquationsOfState");
  RelativisticEuler::Solutions::py_bindings::bind_tov(m);
}
