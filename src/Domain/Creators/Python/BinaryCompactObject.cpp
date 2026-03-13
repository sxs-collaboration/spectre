// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Python/BinaryCompactObject.hpp"

#include <array>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/DomainCreator.hpp"

namespace py = pybind11;

namespace domain::creators::py_bindings {
void bind_binary_compact_object(py::module& m) {
  py::class_<domain::creators::BinaryCompactObject<false>, DomainCreator<3>>(
      m, "BinaryCompactObject")
      .def(py::init(
               [](const double inner_radius_object_a,
                  const double outer_radius_object_a, const double x_coord_a,
                  const bool excise_a, const bool use_logarithmic_map_a,
                  const double inner_radius_object_b,
                  const double outer_radius_object_b, const double x_coord_b,
                  const bool excise_b, const bool use_logarithmic_map_b,
                  const std::array<double, 2> center_of_mass_offset,
                  const double envelope_radius, const double outer_radius,
                  const double cube_scale, const size_t initial_refinement,
                  const size_t initial_num_points,
                  const bool use_equiangular_map,
                  const std::vector<double>& radial_partitioning_outer_shell,
                  const double opening_angle_in_degrees,
                  const bool spherical_harmonics_in_wavezone,
                  std::optional<bco::TimeDependentMapOptions<false>>
                      time_dependent_options) {
                 return domain::creators::BinaryCompactObject<false>{
                     domain::creators::BinaryCompactObject<false>::Object{
                         inner_radius_object_a, outer_radius_object_a,
                         x_coord_a, excise_a, use_logarithmic_map_a},
                     domain::creators::BinaryCompactObject<false>::Object{
                         inner_radius_object_b, outer_radius_object_b,
                         x_coord_b, excise_b, use_logarithmic_map_b},
                     center_of_mass_offset,
                     envelope_radius,
                     outer_radius,
                     cube_scale,
                     initial_refinement,
                     initial_num_points,
                     use_equiangular_map,
                     domain::CoordinateMaps::Distribution::Logarithmic,
                     radial_partitioning_outer_shell,
                     domain::CoordinateMaps::Distribution::Linear,
                     opening_angle_in_degrees,
                     spherical_harmonics_in_wavezone,
                     std::move(time_dependent_options)};
               }),
           py::arg("inner_radius_a"), py::arg("outer_radius_a"),
           py::arg("x_coord_a"), py::arg("excise_a"),
           py::arg("use_logarithmic_map_a"), py::arg("inner_radius_b"),
           py::arg("outer_radius_b"), py::arg("x_coord_b"), py::arg("excise_b"),
           py::arg("use_logarithmic_map_b"), py::arg("center_of_mass_offset"),
           py::arg("envelope_radius"), py::arg("outer_radius"),
           py::arg("cube_scale"), py::arg("initial_refinement"),
           py::arg("initial_number_of_grid_points"),
           py::arg("use_equiangular_map"),
           py::arg("radial_partitioning_outer_shell") = std::vector<double>{},
           py::arg("opening_angle_in_degrees") = 120,
           py::arg("spherical_harmonics_in_wavezone") = false,
           py::arg("time_dependent_options"));
}
}  // namespace domain::creators::py_bindings
