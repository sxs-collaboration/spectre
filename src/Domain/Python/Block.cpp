// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Python/Block.hpp"

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "Domain/Block.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace domain::py_bindings {

namespace {
template <size_t Dim>
void bind_block_impl(py::module& m) {  // NOLINT
  py::class_<Block<Dim>>(m, ("Block" + get_output(Dim) + "D").c_str())
      .def("is_time_dependent", &Block<Dim>::is_time_dependent)
      .def("has_distorted_frame", &Block<Dim>::has_distorted_frame)
      .def_property_readonly("id", &Block<Dim>::id)
      .def_property_readonly("name", &Block<Dim>::name)
      .def_property_readonly("stationary_map", &Block<Dim>::stationary_map)
      .def_property_readonly("moving_mesh_logical_to_grid_map",
                             &Block<Dim>::moving_mesh_logical_to_grid_map)
      .def_property_readonly("moving_mesh_grid_to_inertial_map",
                             &Block<Dim>::moving_mesh_grid_to_inertial_map)
      .def_property_readonly("moving_mesh_grid_to_distorted_map",
                             &Block<Dim>::moving_mesh_grid_to_distorted_map)
      .def_property_readonly(
          "moving_mesh_distorted_to_inertial_map",
          &Block<Dim>::moving_mesh_distorted_to_inertial_map);
}
}  // namespace

void bind_block(py::module& m) {  // NOLINT
  bind_block_impl<1>(m);
  bind_block_impl<2>(m);
  bind_block_impl<3>(m);
}

}  // namespace domain::py_bindings
