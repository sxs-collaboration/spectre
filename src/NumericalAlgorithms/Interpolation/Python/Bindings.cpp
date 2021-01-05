// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace intrp::py_bindings {
void bind_regular_grid(py::module& m);  // NOLINT
}  // namespace intrp::py_bindings

PYBIND11_MODULE(_PyInterpolation, m) {  // NOLINT
  intrp::py_bindings::bind_regular_grid(m);
}
