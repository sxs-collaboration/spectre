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

namespace domain {
namespace creators {
namespace py_bindings {
void bind_interval(py::module& m) {  // NOLINT
  py::class_<Interval, DomainCreator<1>>(m, "Interval")
      .def(py::init<std::array<double, 1>, std::array<double, 1>,
                    std::array<bool, 1>, std::array<size_t, 1>,
                    std::array<size_t, 1>>(),
           py::arg("lower_x"), py::arg("upper_x"), py::arg("is_periodic_in_x"),
           py::arg("initial_refinement_level_x"),
           py::arg("initial_number_of_grid_points_in_x"));
}
}  // namespace py_bindings
}  // namespace creators
}  // namespace domain
