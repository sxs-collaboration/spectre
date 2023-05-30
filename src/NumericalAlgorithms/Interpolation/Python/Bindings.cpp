// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "NumericalAlgorithms/Interpolation/Python/BarycentricRational.hpp"
#include "NumericalAlgorithms/Interpolation/Python/CubicSpline.hpp"
#include "NumericalAlgorithms/Interpolation/Python/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Interpolation/Python/RegularGridInterpolant.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.Spectral");
  intrp::py_bindings::bind_barycentric_rational(m);
  intrp::py_bindings::bind_cubic_spline(m);
  intrp::py_bindings::bind_irregular(m);
  intrp::py_bindings::bind_regular_grid(m);
}
