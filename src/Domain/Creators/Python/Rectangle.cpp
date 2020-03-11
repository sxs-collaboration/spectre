// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Rectangle.hpp"

namespace py = pybind11;

namespace domain {
namespace creators {
namespace py_bindings {
void bind_rectangle(py::module& m) {  // NOLINT
  py::class_<Rectangle, DomainCreator<2>>(m, "Rectangle")
      .def(py::init<std::array<double, 2>, std::array<double, 2>,
                    std::array<bool, 2>, std::array<size_t, 2>,
                    std::array<size_t, 2>>(),
           py::arg("lower_xy"), py::arg("upper_xy"),
           py::arg("is_periodic_in_xy"), py::arg("initial_refinement_level_xy"),
           py::arg("initial_number_of_grid_points_in_xy"));
}
}  // namespace py_bindings
}  // namespace creators
}  // namespace domain
