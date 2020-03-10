// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>
#include <string>

#include "Domain/Creators/DomainCreator.hpp"

namespace py = pybind11;

namespace domain {
namespace creators {
namespace py_bindings {
void bind_domain_creator(py::module& m) {  // NOLINT
  py::class_<DomainCreator<1>>(m, "DomainCreator1D");  // NOLINT
  py::class_<DomainCreator<2>>(m, "DomainCreator2D");  // NOLINT
  py::class_<DomainCreator<3>>(m, "DomainCreator3D");  // NOLINT
}
}  // namespace py_bindings
}  // namespace creators
}  // namespace domain
