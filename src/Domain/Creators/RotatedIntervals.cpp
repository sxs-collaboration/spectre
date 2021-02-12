// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/RotatedIntervals.hpp"

#include "DataStructures/Index.hpp"
#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/OrientationMap.hpp"

namespace domain::creators {
RotatedIntervals::RotatedIntervals(
    typename LowerBound::type lower_x, typename Midpoint::type midpoint_x,
    typename UpperBound::type upper_x,
    typename InitialRefinement::type initial_refinement_level_x,
    typename InitialGridPoints::type initial_number_of_grid_points_in_x,
    typename IsPeriodicIn::type is_periodic_in) noexcept
    // clang-tidy: trivially copyable
    : lower_x_(std::move(lower_x)),                // NOLINT
      midpoint_x_(std::move(midpoint_x)),          // NOLINT
      upper_x_(std::move(upper_x)),                // NOLINT
      is_periodic_in_(std::move(is_periodic_in)),  // NOLINT
      initial_refinement_level_x_(                 // NOLINT
          std::move(initial_refinement_level_x)),  // NOLINT
      initial_number_of_grid_points_in_x_(         // NOLINT
          initial_number_of_grid_points_in_x),
      lower_boundary_condition_(nullptr),
      upper_boundary_condition_(nullptr) {}

RotatedIntervals::RotatedIntervals(
    typename LowerBound::type lower_x, typename Midpoint::type midpoint_x,
    typename UpperBound::type upper_x,
    typename InitialRefinement::type initial_refinement_level_x,
    typename InitialGridPoints::type initial_number_of_grid_points_in_x,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        lower_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        upper_boundary_condition,
    const Options::Context& context)
    : lower_x_(lower_x),
      midpoint_x_(midpoint_x),
      upper_x_(upper_x),
      is_periodic_in_{{false}},
      initial_refinement_level_x_(initial_refinement_level_x),
      initial_number_of_grid_points_in_x_(initial_number_of_grid_points_in_x),
      lower_boundary_condition_(std::move(lower_boundary_condition)),
      upper_boundary_condition_(std::move(upper_boundary_condition)) {
  using domain::BoundaryConditions::is_none;
  if (is_none(lower_boundary_condition_) or
      is_none(upper_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow boundary condition, you must use that.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(lower_boundary_condition_) !=
      is_periodic(upper_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "Both the upper and lower boundary condition must be set to periodic "
        "if imposing periodic boundary conditions.");
  }
  if (is_periodic(lower_boundary_condition_)) {
    is_periodic_in_[0] = true;
    lower_boundary_condition_ = nullptr;
    upper_boundary_condition_ = nullptr;
  }
}

Domain<1> RotatedIntervals::create_domain() const noexcept {
  std::vector<DirectionMap<
      1, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions_all_blocks{};
  if (lower_boundary_condition_ != nullptr or
      upper_boundary_condition_ != nullptr) {
    ASSERT(lower_boundary_condition_ != nullptr and
               upper_boundary_condition_ != nullptr,
           "Both upper and lower boundary conditions must be specified, or "
           "neither.");
    ASSERT(not is_periodic_in_[0],
           "Can't specify both periodic and boundary conditions. Did you "
           "introduce a new constructor to reach this state?");
    boundary_conditions_all_blocks.resize(2);
    DirectionMap<1,
                 std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>
        boundary_conditions_block0{};
    boundary_conditions_block0[Direction<1>::lower_xi()] =
        lower_boundary_condition_->get_clone();
    boundary_conditions_all_blocks[0] = std::move(boundary_conditions_block0);
    DirectionMap<1,
                 std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>
        boundary_conditions_block1{};
    boundary_conditions_block1[Direction<1>::lower_xi()] =
        upper_boundary_condition_->get_clone();
    boundary_conditions_all_blocks[1] = std::move(boundary_conditions_block1);
  }

  return rectilinear_domain<1>(
      Index<1>{2}, {{{lower_x_[0], midpoint_x_[0], upper_x_[0]}}},
      std::move(boundary_conditions_all_blocks), {},
      {OrientationMap<1>{}, OrientationMap<1>{std::array<Direction<1>, 1>{
                                {Direction<1>::lower_xi()}}}},
      is_periodic_in_);
}

std::vector<std::array<size_t, 1>> RotatedIntervals::initial_extents() const
    noexcept {
  return {{{initial_number_of_grid_points_in_x_[0][0]}},
          {{initial_number_of_grid_points_in_x_[0][1]}}};
}

std::vector<std::array<size_t, 1>>
RotatedIntervals ::initial_refinement_levels() const noexcept {
  return {{{initial_refinement_level_x_[0]}},
          {{initial_refinement_level_x_[0]}}};
}
}  // namespace domain::creators
