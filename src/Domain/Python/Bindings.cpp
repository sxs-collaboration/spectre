// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace domain {

namespace py_bindings {
void bind_segment_id(py::module& m);  // NOLINT
void bind_element_id(py::module& m);  // NOLINT
}  // namespace py_bindings

PYBIND11_MODULE(_PyDomain, m) {  // NOLINT
  py_bindings::bind_segment_id(m);
  py_bindings::bind_element_id(m);
}

}  // namespace domain
