// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace py = pybind11;

namespace intrp::py_bindings {

template <size_t Dim>
void bind_regular_grid_impl(py::module& m) {  // NOLINT
  // the bindings are not complete and can be extended
  py::class_<RegularGrid<Dim>>(
      m, ("RegularGrid" + std::to_string(Dim) + "D").c_str())
      .def(py::init<const Mesh<Dim>&, const Mesh<Dim>&,
                    const std::array<DataVector, Dim>&>(),
           py::arg("source_mesh"), py::arg("target_mesh"),
           py::arg("override_target_mesh_with_1d_logical_coords") =
               std::array<DataVector, Dim>{})
      .def("interpolate",
           static_cast<DataVector (RegularGrid<Dim>::*)(const DataVector&)
                           const>(&RegularGrid<Dim>::interpolate),
           py::arg("input"))
      .def("interpolation_matrices", &RegularGrid<Dim>::interpolation_matrices);
}

void bind_regular_grid(py::module& m) {  // NOLINT
  bind_regular_grid_impl<1>(m);
  bind_regular_grid_impl<2>(m);
  bind_regular_grid_impl<3>(m);
}

}  // namespace intrp::py_bindings
