// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Shell.hpp"

namespace py = pybind11;

namespace domain::creators::py_bindings {

void bind_shell(py::module& m) {  // NOLINT
  py::class_<Shell, DomainCreator<3>>(m, "Shell")
      .def(
          py::init([](const double inner_radius, const double outer_radius,
                      const size_t initial_refinement,
                      const std::array<size_t, 2> initial_grid_points,
                      const bool use_equiangular_map, const double aspect_ratio,
                      const size_t index_polar_axis) {
            return Shell{
                inner_radius,        outer_radius,
                initial_refinement,  initial_grid_points,
                use_equiangular_map, {{aspect_ratio, index_polar_axis}}};
          }),
          py::arg("inner_radius"), py::arg("outer_radius"),
          py::arg("initial_refinement"),
          py::arg("initial_number_of_grid_points"),
          py::arg("use_equiangular_map") = true, py::arg("aspect_ratio") = 1.,
          py::arg("index_polar_axis") = 2);
}
}  // namespace domain::creators::py_bindings
