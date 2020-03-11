// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace domain {
namespace creators {

namespace py_bindings {
void bind_brick(py::module& m);           // NOLINT
void bind_cylinder(py::module& m);        // NOLINT
void bind_domain_creator(py::module& m);  // NOLINT
void bind_interval(py::module& m);        // NOLINT
void bind_rectangle(py::module& m);       // NOLINT
}  // namespace py_bindings

PYBIND11_MODULE(_PyDomainCreators, m) {  // NOLINT
  // Order is important: The base class `DomainCreator` needs to have its
  // bindings set up before the derived classes
  py_bindings::bind_domain_creator(m);
  py_bindings::bind_brick(m);
  py_bindings::bind_cylinder(m);
  py_bindings::bind_interval(m);
  py_bindings::bind_rectangle(m);
}

}  // namespace creators
}  // namespace domain
