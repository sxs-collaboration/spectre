// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Triggers/SeparationLessThan.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>

#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ExcisionSphere.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/ObjectLabel.hpp"

namespace Triggers {
SeparationLessThan::SeparationLessThan(const double separation)
    : separation_(separation) {}

bool SeparationLessThan::operator()(
    const double time, const ::Domain<3>& domain,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const tnsr::I<double, 3, Frame::Grid>& grid_object_center_a,
    const tnsr::I<double, 3, Frame::Grid>& grid_object_center_b) const {
  const std::unordered_map<std::string, ExcisionSphere<3>>& excision_spheres =
      domain.excision_spheres();

  const auto check_excision_sphere =
      [&excision_spheres](const std::string& object) {
        if (excision_spheres.count("ExcisionSphere" + object) != 1) {
          ERROR(
              "SeparationLessThan trigger expects an excision sphere named "
              "'ExcisionSphere"
              << object
              << "' in the domain, but there isn't one. Choose a "
                 "DomainCreator that has this excision sphere.");
        }
        if (not excision_spheres.at("ExcisionSphere" + object)
                    .is_time_dependent()) {
          ERROR("SeparationLessThan expects ExcisionSphere"
                << object << " to be time dependent, but it is not.");
        }
      };

  check_excision_sphere(get_output(domain::ObjectLabel::A));
  check_excision_sphere(get_output(domain::ObjectLabel::B));

  const auto& grid_to_inertial_map_a =
      excision_spheres.at("ExcisionSphere" + get_output(domain::ObjectLabel::A))
          .moving_mesh_grid_to_inertial_map();
  const auto& grid_to_inertial_map_b =
      excision_spheres.at("ExcisionSphere" + get_output(domain::ObjectLabel::B))
          .moving_mesh_grid_to_inertial_map();

  const tnsr::I<double, 3, Frame::Inertial> inertial_object_center_a =
      grid_to_inertial_map_a(grid_object_center_a, time, functions_of_time);
  const tnsr::I<double, 3, Frame::Inertial> inertial_object_center_b =
      grid_to_inertial_map_b(grid_object_center_b, time, functions_of_time);

  const tnsr::I<double, 3, Frame::Inertial> position_difference =
      tenex::evaluate<ti::I>(inertial_object_center_a(ti::I) -
                             inertial_object_center_b(ti::I));

  const double calculated_separation =
      sqrt(square(get<0>(position_difference)) +
           square(get<1>(position_difference)) +
           square(get<2>(position_difference)));

  return calculated_separation <= separation_;
}

void SeparationLessThan::pup(PUP::er& p) { p | separation_; }

PUP::able::PUP_ID SeparationLessThan::my_PUP_ID = 0;  // NOLINT
}  // namespace Triggers
