// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/RotatedBricks.hpp"

#include "DataStructures/Index.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/OrientationMap.hpp"

namespace domain::creators {
RotatedBricks::RotatedBricks(
    const typename LowerBound::type lower_xyz,
    const typename Midpoint::type midpoint_xyz,
    const typename UpperBound::type upper_xyz,
    const typename IsPeriodicIn::type is_periodic_in,
    const typename InitialRefinement::type initial_refinement_level_xyz,
    const typename InitialGridPoints::type
        initial_number_of_grid_points_in_xyz) noexcept
    // clang-tidy: trivially copyable
    : lower_xyz_(std::move(lower_xyz)),                        // NOLINT
      midpoint_xyz_(std::move(midpoint_xyz)),                  // NOLINT
      upper_xyz_(std::move(upper_xyz)),                        // NOLINT
      is_periodic_in_(std::move(is_periodic_in)),              // NOLINT
      initial_refinement_level_xyz_(                           // NOLINT
          std::move(initial_refinement_level_xyz)),            // NOLINT
      initial_number_of_grid_points_in_xyz_(                   // NOLINT
          std::move(initial_number_of_grid_points_in_xyz)) {}  // NOLINT

Domain<3> RotatedBricks::create_domain() const noexcept {
  return rectilinear_domain<3>(
      Index<3>{2, 2, 2},
      std::array<std::vector<double>, 3>{
          {{lower_xyz_[0], midpoint_xyz_[0], upper_xyz_[0]},
           {lower_xyz_[1], midpoint_xyz_[1], upper_xyz_[1]},
           {lower_xyz_[2], midpoint_xyz_[2], upper_xyz_[2]}}},
      {}, {},
      std::vector<OrientationMap<3>>{
          OrientationMap<3>{},
          OrientationMap<3>{std::array<Direction<3>, 3>{
              {Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
               Direction<3>::lower_xi()}}},
          OrientationMap<3>{std::array<Direction<3>, 3>{
              {Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
               Direction<3>::lower_eta()}}},
          OrientationMap<3>{std::array<Direction<3>, 3>{
              {Direction<3>::upper_zeta(), Direction<3>::lower_xi(),
               Direction<3>::lower_eta()}}},
          OrientationMap<3>{std::array<Direction<3>, 3>{
              {Direction<3>::upper_eta(), Direction<3>::lower_xi(),
               Direction<3>::upper_zeta()}}},
          OrientationMap<3>{std::array<Direction<3>, 3>{
              {Direction<3>::upper_eta(), Direction<3>::lower_zeta(),
               Direction<3>::lower_xi()}}},
          OrientationMap<3>{std::array<Direction<3>, 3>{
              {Direction<3>::upper_zeta(), Direction<3>::lower_xi(),
               Direction<3>::lower_eta()}}},
          OrientationMap<3>{std::array<Direction<3>, 3>{
              {Direction<3>::upper_xi(), Direction<3>::upper_eta(),
               Direction<3>::upper_zeta()}}}},
      is_periodic_in_);
}

std::vector<std::array<size_t, 3>> RotatedBricks::initial_extents() const
    noexcept {
  const size_t& x_0 = initial_number_of_grid_points_in_xyz_[0][0];
  const size_t& x_1 = initial_number_of_grid_points_in_xyz_[0][1];
  const size_t& y_0 = initial_number_of_grid_points_in_xyz_[1][0];
  const size_t& y_1 = initial_number_of_grid_points_in_xyz_[1][1];
  const size_t& z_0 = initial_number_of_grid_points_in_xyz_[2][0];
  const size_t& z_1 = initial_number_of_grid_points_in_xyz_[2][1];
  return {{{x_0, y_0, z_0}}, {{z_0, y_0, x_1}}, {{x_0, z_0, y_1}},
          {{y_1, z_0, x_1}}, {{y_0, x_0, z_1}}, {{z_1, x_1, y_0}},
          {{y_1, z_1, x_0}}, {{x_1, y_1, z_1}}};
}

std::vector<std::array<size_t, 3>> RotatedBricks::initial_refinement_levels()
    const noexcept {
  const size_t& x_0 = initial_refinement_level_xyz_[0];
  const size_t& y_0 = initial_refinement_level_xyz_[1];
  const size_t& z_0 = initial_refinement_level_xyz_[2];
  return {{{x_0, y_0, z_0}}, {{z_0, y_0, x_0}}, {{x_0, z_0, y_0}},
          {{y_0, z_0, x_0}}, {{y_0, x_0, z_0}}, {{z_0, x_0, y_0}},
          {{y_0, z_0, x_0}}, {{x_0, y_0, z_0}}};
}
}  // namespace domain::creators
