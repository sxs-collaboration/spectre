// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/Python/Strahlkorper.hpp"

#include <array>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"

namespace py = pybind11;

namespace ylm::py_bindings {
void bind_strahlkorper(pybind11::module& m) {  // NOLINT
  py::class_<ylm::Strahlkorper<Frame::Inertial>>(m, "Strahlkorper")
      .def(py::init<size_t, size_t, double, std::array<double, 3>>(),
           py::arg("l_max"), py::arg("m_max"), py::arg("radius"),
           py::arg("center"))
      .def(py::init<size_t, size_t, const DataVector&, std::array<double, 3>>(),
           py::arg("l_max"), py::arg("m_max"),
           py::arg("radius_at_collocation_points"), py::arg("center"))
      .def(
          py::init<size_t, size_t, const ylm::Strahlkorper<Frame::Inertial>&>(),
          py::arg("l_max"), py::arg("m_max"), py::arg("another_strahlkorper"))
      .def(
          py::init<size_t, size_t, const ModalVector&, std::array<double, 3>>(),
          py::arg("l_max"), py::arg("m_max"), py::arg("spectral_coefficients"),
          py::arg("center"))
      .def_property_readonly("l_max",
                             &ylm::Strahlkorper<Frame::Inertial>::l_max)
      .def_property_readonly("m_max",
                             &ylm::Strahlkorper<Frame::Inertial>::m_max)
      .def_property_readonly(
          "physical_extents",
          [](const ylm::Strahlkorper<Frame::Inertial>& strahlkorper) {
            return strahlkorper.ylm_spherepack().physical_extents();
          })
      .def_property_readonly(
          "expansion_center",
          &ylm::Strahlkorper<Frame::Inertial>::expansion_center)
      .def_property_readonly(
          "physical_center",
          &ylm::Strahlkorper<Frame::Inertial>::physical_center)
      .def_property_readonly(
          "average_radius", &ylm::Strahlkorper<Frame::Inertial>::average_radius)
      .def("radius", &ylm::Strahlkorper<Frame::Inertial>::radius,
           py::arg("theta"), py::arg("phi"))
      .def("point_is_contained",
           &ylm::Strahlkorper<Frame::Inertial>::point_is_contained,
           py::arg("x"))
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self == py::self)
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self != py::self);
}
}  // namespace ylm::py_bindings
