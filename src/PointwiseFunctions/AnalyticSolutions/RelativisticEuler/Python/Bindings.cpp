// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace RelativisticEuler::Solutions::py_bindings {
void bind_tov(py::module& m);  // NOLINT
}  // namespace RelativisticEuler::Solutions::py_bindings

PYBIND11_MODULE(_PyRelativisticEulerSolutions, m) {  // NOLINT
  RelativisticEuler::Solutions::py_bindings::bind_tov(m);
}
