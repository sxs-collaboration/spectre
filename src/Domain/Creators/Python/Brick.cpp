// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/DomainCreator.hpp"

namespace py = pybind11;

namespace domain {
namespace creators {
namespace py_bindings {
void bind_brick(py::module& m) {  // NOLINT
  py::class_<Brick, DomainCreator<3>>(m, "Brick")
      .def(py::init<std::array<double, 3>, std::array<double, 3>,
                    std::array<bool, 3>, std::array<size_t, 3>,
                    std::array<size_t, 3>>(),
           py::arg("lower_xyz"), py::arg("upper_xyz"),
           py::arg("is_periodic_in_xyz"),
           py::arg("initial_refinement_level_xyz"),
           py::arg("initial_number_of_grid_points_in_xyz"));
}
}  // namespace py_bindings
}  // namespace creators
}  // namespace domain
