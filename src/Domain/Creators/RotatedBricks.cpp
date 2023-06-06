// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/RotatedBricks.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "DataStructures/Index.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Options/ParseError.hpp"

namespace domain::creators {
RotatedBricks::RotatedBricks(
    const typename LowerBound::type lower_xyz,
    const typename Midpoint::type midpoint_xyz,
    const typename UpperBound::type upper_xyz,
    const typename InitialRefinement::type initial_refinement_level_xyz,
    const typename InitialGridPoints::type initial_number_of_grid_points_in_xyz,
    const typename IsPeriodicIn::type is_periodic_in)
    // clang-tidy: trivially copyable
    : lower_xyz_(std::move(lower_xyz)),                      // NOLINT
      midpoint_xyz_(std::move(midpoint_xyz)),                // NOLINT
      upper_xyz_(std::move(upper_xyz)),                      // NOLINT
      is_periodic_in_(std::move(is_periodic_in)),            // NOLINT
      initial_refinement_level_xyz_(                         // NOLINT
          std::move(initial_refinement_level_xyz)),          // NOLINT
      initial_number_of_grid_points_in_xyz_(                 // NOLINT
          std::move(initial_number_of_grid_points_in_xyz)),  // NOLINT
      boundary_condition_(nullptr) {}

RotatedBricks::RotatedBricks(
    const typename LowerBound::type lower_xyz,
    const typename Midpoint::type midpoint_xyz,
    const typename UpperBound::type upper_xyz,
    const typename InitialRefinement::type initial_refinement_level_xyz,
    const typename InitialGridPoints::type initial_number_of_grid_points_in_xyz,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        boundary_condition,
    const Options::Context& context)
    // clang-tidy: trivially copyable
    : lower_xyz_(std::move(lower_xyz)),                      // NOLINT
      midpoint_xyz_(std::move(midpoint_xyz)),                // NOLINT
      upper_xyz_(std::move(upper_xyz)),                      // NOLINT
      is_periodic_in_{{false, false, false}},                // NOLINT
      initial_refinement_level_xyz_(                         // NOLINT
          std::move(initial_refinement_level_xyz)),          // NOLINT
      initial_number_of_grid_points_in_xyz_(                 // NOLINT
          std::move(initial_number_of_grid_points_in_xyz)),  // NOLINT
      boundary_condition_(std::move(boundary_condition)) {
  using domain::BoundaryConditions::is_none;
  if (is_none(boundary_condition_)) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow-type boundary condition, you must use that.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(boundary_condition_)) {
    is_periodic_in_[0] = true;
    is_periodic_in_[1] = true;
    is_periodic_in_[2] = true;
  }
}

Domain<3> RotatedBricks::create_domain() const {
  return rectilinear_domain<3>(
      Index<3>{2, 2, 2},
      std::array<std::vector<double>, 3>{
          {{lower_xyz_[0], midpoint_xyz_[0], upper_xyz_[0]},
           {lower_xyz_[1], midpoint_xyz_[1], upper_xyz_[1]},
           {lower_xyz_[2], midpoint_xyz_[2], upper_xyz_[2]}}},
      {},
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

std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
RotatedBricks::external_boundary_conditions() const {
  if (boundary_condition_ == nullptr) {
    return {};
  }
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{8};
  if (is_periodic_in_[0]) {
    return boundary_conditions;
  }
  std::vector<std::unordered_set<Direction<3>>>
      external_boundaries_in_each_block{
          {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
           Direction<3>::lower_zeta()},
          {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
           Direction<3>::upper_zeta()},
          {Direction<3>::lower_xi(), Direction<3>::upper_eta(),
           Direction<3>::upper_zeta()},
          {Direction<3>::lower_xi(), Direction<3>::upper_eta(),
           Direction<3>::upper_zeta()},
          {Direction<3>::upper_xi(), Direction<3>::lower_eta(),
           Direction<3>::upper_zeta()},
          {Direction<3>::lower_xi(), Direction<3>::upper_eta(),
           Direction<3>::upper_zeta()},
          {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
           Direction<3>::lower_zeta()},
          {Direction<3>::upper_xi(), Direction<3>::upper_eta(),
           Direction<3>::upper_zeta()}};
  for (size_t i = 0; i < 8; ++i) {
    for (const Direction<3>& direction : external_boundaries_in_each_block[i]) {
      boundary_conditions[i][direction] = boundary_condition_->get_clone();
    }
  }
  return boundary_conditions;
}

std::vector<std::array<size_t, 3>> RotatedBricks::initial_extents() const {
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
    const {
  const size_t& x_0 = initial_refinement_level_xyz_[0];
  const size_t& y_0 = initial_refinement_level_xyz_[1];
  const size_t& z_0 = initial_refinement_level_xyz_[2];
  return {{{x_0, y_0, z_0}}, {{z_0, y_0, x_0}}, {{x_0, z_0, y_0}},
          {{y_0, z_0, x_0}}, {{y_0, x_0, z_0}}, {{z_0, x_0, y_0}},
          {{y_0, z_0, x_0}}, {{x_0, y_0, z_0}}};
}
}  // namespace domain::creators
