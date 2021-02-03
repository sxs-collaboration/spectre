// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Interval.hpp"

namespace py = pybind11;

namespace domain::creators::py_bindings {
void bind_interval(py::module& m) {  // NOLINT
  py::class_<Interval, DomainCreator<1>>(m, "Interval")
      .def(
          py::init(
              [](const std::array<double, 1> lower_x,
                 const std::array<double, 1> upper_x,
                 const std::array<size_t, 1> initial_refinement_level_x,
                 const std::array<size_t, 1> initial_number_of_grid_points_in_x,
                 const std::array<bool, 1> is_periodic_in_x) {
                return domain::creators::Interval{
                    lower_x,
                    upper_x,
                    initial_refinement_level_x,
                    initial_number_of_grid_points_in_x,
                    is_periodic_in_x,
                    nullptr};
              }),
          py::arg("lower_x"), py::arg("upper_x"),
          py::arg("initial_refinement_level_x"),
          py::arg("initial_number_of_grid_points_in_x"),
          py::arg("is_periodic_in_x"));
}
}  // namespace domain::creators::py_bindings
