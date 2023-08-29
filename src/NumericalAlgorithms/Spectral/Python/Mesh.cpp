// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Python/Mesh.hpp"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "NumericalAlgorithms/SpatialDiscretization/Basis.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Quadrature.hpp"

namespace py = pybind11;

namespace py_bindings {
namespace {
template <size_t Dim>
void bind_mesh_impl(py::module& m) {  // NOLINT
  // the bindings here are not complete
  py::class_<Mesh<Dim>>(m, ("Mesh" + std::to_string(Dim) + "D").c_str())
      .def_property_readonly_static(
          "dim", [](const py::object& /*self */) { return Dim; })
      .def(py::init<const size_t, const SpatialDiscretization::Basis,
                    const SpatialDiscretization::Quadrature>(),
           py::arg("isotropic_extents"), py::arg("basis"),
           py::arg("quadrature"))
      .def(py::init<std::array<size_t, Dim>, const SpatialDiscretization::Basis,
                    const SpatialDiscretization::Quadrature>(),
           py::arg("extents"), py::arg("basis"), py::arg("quadrature"))
      .def(py::init<std::array<size_t, Dim>,
                    std::array<SpatialDiscretization::Basis, Dim>,
                    std::array<SpatialDiscretization::Quadrature, Dim>>(),
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
           static_cast<const std::array<SpatialDiscretization::Basis, Dim>& (
               Mesh<Dim>::*)() const>(&Mesh<Dim>::basis),
           "The basis chosen in each dimension of the grid.")
      .def("basis",
           static_cast<SpatialDiscretization::Basis (Mesh<Dim>::*)(const size_t)
                           const>(&Mesh<Dim>::basis),
           py::arg("d"),
           "The basis chosen in the requested dimension of the grid.")
      .def("quadrature",
           static_cast<const std::array<SpatialDiscretization::Quadrature,
                                        Dim>& (Mesh<Dim>::*)() const>(
               &Mesh<Dim>::quadrature),
           "The quadrature chosen in each dimension of the grid.")
      .def("quadrature",
           static_cast<SpatialDiscretization::Quadrature (Mesh<Dim>::*)(
               const size_t) const>(&Mesh<Dim>::quadrature),
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
                state[1].cast<std::array<SpatialDiscretization::Basis, Dim>>(),
                state[2]
                    .cast<
                        std::array<SpatialDiscretization::Quadrature, Dim>>());
          }));
}
}  // namespace

void bind_mesh(py::module& m) {
  bind_mesh_impl<1>(m);
  bind_mesh_impl<2>(m);
  bind_mesh_impl<3>(m);
}

}  // namespace py_bindings
