// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Python/Interval.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Interval.hpp"

namespace py = pybind11;

namespace domain::creators::py_bindings {
void bind_interval(py::module& m) {
  py::class_<Interval, DomainCreator<1>>(m, "Interval")
      .def(py::init<std::array<double, 1>, std::array<double, 1>,
                    std::array<size_t, 1>, std::array<size_t, 1>,
                    std::array<bool, 1>, domain::CoordinateMaps::Distribution,
                    std::optional<double>>(),
           py::arg("lower_x"), py::arg("upper_x"),
           py::arg("initial_refinement_level_x"),
           py::arg("initial_number_of_grid_points_in_x"),
           py::arg("is_periodic_in_x") = std::array<bool, 1>{{false}},
           py::arg("distribution") =
               domain::CoordinateMaps::Distribution::Linear,
           py::arg("singularity") = std::nullopt);
}
}  // namespace domain::creators::py_bindings
