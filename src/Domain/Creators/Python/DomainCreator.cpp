// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Python/DomainCreator.hpp"

#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace domain::creators::py_bindings {

namespace {
template <size_t Dim>
void bind_domain_creator_impl(py::module& m) {  // NOLINT
  py::class_<DomainCreator<Dim>>(
      m, ("DomainCreator" + get_output(Dim) + "D").c_str())
      .def("create_domain", &DomainCreator<Dim>::create_domain)
      .def("block_names", &DomainCreator<Dim>::block_names)
      .def("block_groups", &DomainCreator<Dim>::block_groups)
      .def("initial_extents", &DomainCreator<Dim>::initial_extents)
      .def("initial_refinement_levels",
           &DomainCreator<Dim>::initial_refinement_levels)
      .def("functions_of_time", &DomainCreator<Dim>::functions_of_time,
           py::arg("initial_expiration_times") =
               std::unordered_map<std::string, double>{});
}
}  // namespace

void bind_domain_creator(py::module& m) {  // NOLINT
  bind_domain_creator_impl<1>(m);
  bind_domain_creator_impl<2>(m);
  bind_domain_creator_impl<3>(m);
}
}  // namespace domain::creators::py_bindings
