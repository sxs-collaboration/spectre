// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Python/Direction.hpp"

#include <cstddef>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace domain::py_bindings {

namespace {
template <size_t Dim>
void bind_direction_impl(py::module& m) {  // NOLINT
  auto binding =
      py::class_<Direction<Dim>>(m,
                                 ("Direction" + get_output(Dim) + "D").c_str())
          .def(py::init<size_t, Side>(), py::arg("dimension"), py::arg("side"))
          .def_property_readonly("dimension", &Direction<Dim>::dimension)
          .def_property_readonly("side", &Direction<Dim>::side)
          .def_property_readonly("sign", &Direction<Dim>::sign)
          .def("opposite", &Direction<Dim>::opposite)
          .def("__repr__",
               [](const Direction<Dim>& direction) {
                 return get_output(direction);
               })
          // NOLINTNEXTLINE(misc-redundant-expression)
          .def(py::self == py::self)
          // NOLINTNEXTLINE(misc-redundant-expression)
          .def(py::self != py::self)
          // NOLINTNEXTLINE(misc-redundant-expression)
          .def(py::self < py::self)
          .def(hash(py::self));
  if constexpr (Dim >= 1) {
    binding.def_static("lower_xi", &Direction<Dim>::lower_xi)
        .def_static("upper_xi", &Direction<Dim>::upper_xi);
  }
  if constexpr (Dim >= 2) {
    binding.def_static("lower_eta", &Direction<Dim>::lower_eta)
        .def_static("upper_eta", &Direction<Dim>::upper_eta);
  }
  if constexpr (Dim >= 3) {
    binding.def_static("lower_zeta", &Direction<Dim>::lower_zeta)
        .def_static("upper_zeta", &Direction<Dim>::upper_zeta);
  }
}
}  // namespace

void bind_direction(py::module& m) {  // NOLINT
  py::enum_<Side>(m, "Side")
      .value("Lower", Side::Lower)
      .value("Upper", Side::Upper);
  bind_direction_impl<1>(m);
  bind_direction_impl<2>(m);
  bind_direction_impl<3>(m);
}

}  // namespace domain::py_bindings
