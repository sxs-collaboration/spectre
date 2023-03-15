// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Python/Sphere.hpp"

#include <array>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Sphere.hpp"

namespace py = pybind11;

namespace domain::creators::py_bindings {
void bind_sphere(py::module& m) {
  py::class_<Sphere, DomainCreator<3>>(m, "Sphere")
      .def(py::init([](const double inner_radius, const double outer_radius,
                       const size_t initial_refinement,
                       const size_t initial_num_points,
                       const bool use_equiangular_map,
                       const std::optional<bool>& excise,
                       const std::optional<double>& inner_cube_sphericity) {
             if (excise.has_value() == inner_cube_sphericity.has_value() or
                 (excise.has_value() and not excise.value())) {
               throw std::runtime_error(
                   "Specify either 'inner_cube_sphericity' or 'excise=True'.");
             }
             auto interior = [&inner_cube_sphericity]()
                 -> std::variant<Sphere::Excision, Sphere::InnerCube> {
               if (inner_cube_sphericity.has_value()) {
                 return Sphere::InnerCube{inner_cube_sphericity.value()};
               } else {
                 return Sphere::Excision{};
               }
             }();
             return Sphere{inner_radius,        outer_radius,
                           std::move(interior), initial_refinement,
                           initial_num_points,  use_equiangular_map};
           }),
           py::arg("inner_radius"), py::arg("outer_radius"),
           py::arg("initial_refinement"),
           py::arg("initial_number_of_grid_points"),
           py::arg("use_equiangular_map"), py::arg("excise") = std::nullopt,
           py::arg("inner_cube_sphericity") = std::nullopt);
}
}  // namespace domain::creators::py_bindings
