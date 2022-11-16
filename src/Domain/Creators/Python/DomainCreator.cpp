// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Python/DomainCreator.hpp"

#include <pybind11/pybind11.h>
#include <string>

#include "Domain/Creators/DomainCreator.hpp"

namespace py = pybind11;

namespace domain::creators::py_bindings {
void bind_domain_creator(py::module& m) {
  py::class_<DomainCreator<1>>(m, "DomainCreator1D");  // NOLINT
  py::class_<DomainCreator<2>>(m, "DomainCreator2D");  // NOLINT
  py::class_<DomainCreator<3>>(m, "DomainCreator3D");  // NOLINT
}
}  // namespace domain::creators::py_bindings
