// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace py_bindings {
void bind_datavector(py::module& m);  // NOLINT
void bind_matrix(py::module& m);      // NOLINT
void bind_tensordata(py::module& m);  // NOLINT
}  // namespace py_bindings

PYBIND11_MODULE(_PyDataStructures, m) {  // NOLINT
  py_bindings::bind_datavector(m);
  py_bindings::bind_matrix(m);
  py_bindings::bind_tensordata(m);
}
