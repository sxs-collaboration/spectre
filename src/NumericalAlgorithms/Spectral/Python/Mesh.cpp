// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace py = pybind11;

namespace py_bindings {

template <size_t Dim>
void bind_mesh_impl(py::module& m) {  // NOLINT
  // the bindings here are not complete
  py::class_<Mesh<Dim>>(m, ("Mesh" + std::to_string(Dim) + "D").c_str())
      .def_property_readonly_static(
          "dim", [](const py::object& /*self */) { return Dim; })
      .def(py::init<const size_t, const Spectral::Basis,
                    const Spectral::Quadrature>(),
           py::arg("isotropic_extents"), py::arg("basis"),
           py::arg("quadrature"))
      .def(py::init<std::array<size_t, Dim>, const Spectral::Basis,
                    const Spectral::Quadrature>(),
           py::arg("extents"), py::arg("basis"), py::arg("quadrature"))
      .def(py::init<std::array<size_t, Dim>, std::array<Spectral::Basis, Dim>,
                    std::array<Spectral::Quadrature, Dim>>(),
           py::arg("extents"), py::arg("bases"), py::arg("quadratures"))
      .def(
          "extents",
          [](const Mesh<Dim>& mesh) { return mesh.extents().indices(); },
          "The number of grid points in each dimension of the grid.")
      .def(
          "extents",
          static_cast<size_t (Mesh<Dim>::*)(size_t) const>(&Mesh<Dim>::extents),
          py::arg("d"),
          "The number of grid points in the requested dimension of the grid.")
      .def("number_of_grid_points", &Mesh<Dim>::number_of_grid_points,
           "The total number of grid points in all dimensions.")
      .def("basis",
           static_cast<const std::array<Spectral::Basis, Dim>& (Mesh<Dim>::*)()
                           const>(&Mesh<Dim>::basis),
           "The basis chosen in each dimension of the grid.")
      .def("basis",
           static_cast<Spectral::Basis (Mesh<Dim>::*)(const size_t) const>(
               &Mesh<Dim>::basis),
           py::arg("d"),
           "The basis chosen in the requested dimension of the grid.")
      .def("quadrature",
           static_cast<const std::array<Spectral::Quadrature, Dim>& (
               Mesh<Dim>::*)() const>(&Mesh<Dim>::quadrature),
           "The quadrature chosen in each dimension of the grid.")
      .def("quadrature",
           static_cast<Spectral::Quadrature (Mesh<Dim>::*)(const size_t) const>(
               &Mesh<Dim>::quadrature),
           py::arg("d"),
           "The quadrature chosen in the requested dimension of the grid.")
      .def("slices", &Mesh<Dim>::slices,
           "Returns the Meshes representing 1D slices of this Mesh.")
      .def(py::self == py::self)  // NOLINT
      .def(py::self != py::self)  // NOLINT
      .def(py::pickle(
          [](const Mesh<Dim>& mesh) {
            return py::make_tuple(mesh.extents().indices(), mesh.basis(),
                                  mesh.quadrature());
          },
          [](const py::tuple& state) {
            if (state.size() != 3) {
              throw std::runtime_error("Invalid state for mesh!");
            }
            return Mesh<Dim>(
                state[0].cast<std::array<size_t, Dim>>(),
                state[1].cast<std::array<Spectral::Basis, Dim>>(),
                state[2].cast<std::array<Spectral::Quadrature, Dim>>());
          }));
}

void bind_mesh(py::module& m) {  // NOLINT
  bind_mesh_impl<1>(m);
  bind_mesh_impl<2>(m);
  bind_mesh_impl<3>(m);
}

}  // namespace py_bindings
