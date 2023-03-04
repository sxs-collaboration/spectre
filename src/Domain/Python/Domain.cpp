// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Python/Domain.hpp"

#include <cstddef>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Parallel/Serialize.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace domain::py_bindings {

namespace {
template <size_t Dim>
void bind_domain_impl(py::module& m) {  // NOLINT
  py::class_<Domain<Dim>>(m, ("Domain" + get_output(Dim) + "D").c_str())
      .def_property_readonly_static(
          "dim", [](const py::object& /*self */) { return Dim; })
      .def("is_time_dependent", &Domain<Dim>::is_time_dependent)
      .def_property_readonly("blocks", &Domain<Dim>::blocks)
      .def_property_readonly("block_groups", &Domain<Dim>::block_groups)
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self == py::self)
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self != py::self);
  m.def("serialize_domain", &serialize<Domain<Dim>>);
  m.def(("deserialize_domain_" + get_output(Dim) + "d").c_str(),
        [](const std::vector<char>& serialized_domain) {
          return deserialize<Domain<Dim>>(serialized_domain.data());
        },
        py::arg("serialized_domain"));
}
}  // namespace

void bind_domain(py::module& m) {  // NOLINT
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  bind_domain_impl<1>(m);
  bind_domain_impl<2>(m);
  bind_domain_impl<3>(m);
}

}  // namespace domain::py_bindings
