// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace domain {
namespace creators {

namespace py_bindings {
void bind_cylinder(py::module& m);  // NOLINT
void bind_domain_creator(py::module& m);  // NOLINT
}  // namespace py_bindings

PYBIND11_MODULE(_PyDomainCreators, m) {  // NOLINT
  py_bindings::bind_domain_creator(m);
  py_bindings::bind_cylinder(m);
}

}  // namespace creators
}  // namespace domain
