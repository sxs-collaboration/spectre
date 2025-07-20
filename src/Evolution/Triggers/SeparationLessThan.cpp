// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Triggers/SeparationLessThan.hpp"

#include <cmath>
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
#include "Domain/FunctionsOfTime/SettleToConstantQuaternion.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/ObjectLabel.hpp"

namespace Triggers {
template <bool UseGridCentersFunctionOfTime>
SeparationLessThan<UseGridCentersFunctionOfTime>::SeparationLessThan(
    const double separation)
    : separation_(separation) {}

template <bool UseGridCentersFunctionOfTime>
bool SeparationLessThan<UseGridCentersFunctionOfTime>::operator()(
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

  return calculated_separation < separation_;
}

template <bool UseGridCentersFunctionOfTime>
bool SeparationLessThan<UseGridCentersFunctionOfTime>::operator()(
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  if (dynamic_cast<const domain::FunctionsOfTime::SettleToConstantQuaternion*>(
          functions_of_time.at("Rotation").get()) != nullptr) {
    return false;
  }
  const DataVector fot = functions_of_time.at("GridCenters")->func(time)[0];
  const double separation =
      std::sqrt(square(fot[0] - fot[3]) + square(fot[1] - fot[4]) +
                square(fot[2] - fot[5]));
  return separation < separation_;
}

template <bool UseGridCentersFunctionOfTime>
void SeparationLessThan<UseGridCentersFunctionOfTime>::pup(PUP::er& p) {
  p | separation_;
}

#ifndef __CUDA_ARCH__
template <bool UseGridCentersFunctionOfTime>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PUP::able::PUP_ID SeparationLessThan<UseGridCentersFunctionOfTime>::my_PUP_ID =
    0;
#endif  // __CUDA_ARCH__

template class SeparationLessThan<true>;
template class SeparationLessThan<false>;
}  // namespace Triggers
