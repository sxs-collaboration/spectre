// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "Domain/Creators/Cylinder.hpp"
#include "Domain/Creators/DomainCreator.hpp"

namespace py = pybind11;

namespace domain {
namespace creators {
namespace py_bindings {
void bind_cylinder(py::module& m) {  // NOLINT
  py::class_<Cylinder, DomainCreator<3>>(m, "Cylinder")
      .def(py::init<double, double, double, double, bool, size_t,
                    std::array<size_t, 3>, bool>(),
           py::arg("inner_radius"), py::arg("outer_radius"),
           py::arg("lower_bound"), py::arg("upper_bound"),
           py::arg("is_periodic_in_z"), py::arg("initial_refinement"),
           py::arg("initial_number_of_grid_points"),
           py::arg("use_equiangular_map"));
}
}  // namespace py_bindings
}  // namespace creators
}  // namespace domain
