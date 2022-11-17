// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "NumericalAlgorithms/Interpolation/Python/RegularGridInterpolant.hpp"

PYBIND11_MODULE(_PyInterpolation, m) {  // NOLINT
  intrp::py_bindings::bind_regular_grid(m);
}
