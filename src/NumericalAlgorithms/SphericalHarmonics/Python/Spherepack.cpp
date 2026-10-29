// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/Python/Spherepack.hpp"

#include <array>
#include <cstddef>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"

namespace py = pybind11;

namespace ylm::py_bindings {
void bind_spherepack(pybind11::module& m) {  // NOLINT
  py::class_<ylm::Spherepack>(m, "Spherepack")
      .def(py::init<size_t, size_t>(), py::arg("l_max"), py::arg("m_max"))
      .def_property_readonly("l_max", &ylm::Spherepack::l_max)
      .def_property_readonly("m_max", &ylm::Spherepack::m_max)
      .def_property_readonly("physical_size",
                             [](const ylm::Spherepack& spherepack) {
                               return spherepack.physical_size();
                             })
      .def_property_readonly("spectral_size",
                             [](const ylm::Spherepack& spherepack) {
                               return spherepack.spectral_size();
                             })
      .def_property_readonly("theta_phi_points",
                             &ylm::Spherepack::theta_phi_points)
      .def("phys_to_spec",
           static_cast<DataVector (ylm::Spherepack::*)(
               const DataVector& collocation_values,
               const size_t physical_stride, const size_t physical_offset)
                           const>(&ylm::Spherepack::phys_to_spec),
           py::arg("collocation_values"), py::arg("physical_stride") = 1,
           py::arg("physical_offset") = 0)
      .def("spec_to_phys",
           static_cast<DataVector (ylm::Spherepack::*)(
               const DataVector& spectral_coefs, const size_t spectral_stride,
               const size_t spectral_offset) const>(
               &ylm::Spherepack::spec_to_phys),
           py::arg("spectral_coefs"), py::arg("spectral_stride") = 1,
           py::arg("spectral_offset") = 0)
      .def("prolong_or_restrict",
           static_cast<DataVector (ylm::Spherepack::*)(
               const DataVector&, const ylm::Spherepack&) const>(
               &ylm::Spherepack::prolong_or_restrict),
           py::arg("spectral_coefs"), py::arg("target"))
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self == py::self)
      // NOLINTNEXTLINE(misc-redundant-expression)
      .def(py::self != py::self);

  py::class_<ylm::SpherepackIterator>(m, "SpherepackIterator")
      .def(py::init<size_t, size_t, size_t>(), py::arg("l_max"),
           py::arg("m_max"), py::arg("physical_stride") = 1)
      .def("__call__", &SpherepackIterator::operator())
      .def(
          "set",
          [](ylm::SpherepackIterator& spherepack, const size_t l_input,
             const int m_input) { spherepack.set(l_input, m_input); },
          py::arg("l"), py::arg("m"));
}
}  // namespace ylm::py_bindings
