// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace py_bindings {
void bind_h5file(py::module& m);  // NOLINT
void bind_h5dat(py::module& m);   // NOLINT
void bind_h5vol(py::module& m);   // NOLINT
}  // namespace py_bindings

PYBIND11_MODULE(_PyH5, m) {  // NOLINT
  py_bindings::bind_h5file(m);
  py_bindings::bind_h5dat(m);
  py_bindings::bind_h5vol(m);
}
