// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Rectangle.hpp"

#include <array>
#include <memory>
#include <vector>

#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Utilities/ErrorHandling/Assert.hpp"

namespace Frame {
struct Inertial;
struct Logical;
}  // namespace Frame

namespace domain::creators {
Rectangle::Rectangle(
    typename LowerBound::type lower_xy, typename UpperBound::type upper_xy,
    typename InitialRefinement::type initial_refinement_level_xy,
    typename InitialGridPoints::type initial_number_of_grid_points_in_xy,
    typename IsPeriodicIn::type is_periodic_in_xy,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<2>>
        time_dependence) noexcept
    // clang-tidy: trivially copyable
    : lower_xy_(std::move(lower_xy)),                       // NOLINT
      upper_xy_(std::move(upper_xy)),                       // NOLINT
      is_periodic_in_xy_(std::move(is_periodic_in_xy)),     // NOLINT
      initial_refinement_level_xy_(                         // NOLINT
          std::move(initial_refinement_level_xy)),          // NOLINT
      initial_number_of_grid_points_in_xy_(                 // NOLINT
          std::move(initial_number_of_grid_points_in_xy)),  // NOLINT
      time_dependence_(std::move(time_dependence)),
      boundary_condition_(nullptr) {
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<2>>();
  }
}

Rectangle::Rectangle(
    typename LowerBound::type lower_xy, typename UpperBound::type upper_xy,
    typename InitialRefinement::type initial_refinement_level_xy,
    typename InitialGridPoints::type initial_number_of_grid_points_in_xy,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        boundary_condition,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<2>>
        time_dependence,
    const Options::Context& context)
    : lower_xy_(lower_xy),
      upper_xy_(upper_xy),
      is_periodic_in_xy_{{false, false}},
      initial_refinement_level_xy_(initial_refinement_level_xy),
      initial_number_of_grid_points_in_xy_(initial_number_of_grid_points_in_xy),
      time_dependence_(std::move(time_dependence)),
      boundary_condition_(std::move(boundary_condition)) {
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<2>>();
  }
  using domain::BoundaryConditions::is_none;
  if (is_none(boundary_condition_)) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow boundary condition, you must use that.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(boundary_condition_)) {
    is_periodic_in_xy_[0] = true;
    is_periodic_in_xy_[1] = true;
    boundary_condition_ = nullptr;
  }
}

Domain<2> Rectangle::create_domain() const noexcept {
  using Affine = CoordinateMaps::Affine;
  using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  std::vector<PairOfFaces> identifications{};
  if (is_periodic_in_xy_[0]) {
    identifications.push_back({{0, 2}, {1, 3}});
  }
  if (is_periodic_in_xy_[1]) {
    identifications.push_back({{0, 1}, {2, 3}});
  }

  std::vector<DirectionMap<
      2, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions_all_blocks{};
  if (boundary_condition_ != nullptr) {
    ASSERT(is_periodic_in_xy_[0] == false and is_periodic_in_xy_[1] == false,
           "Cannot have a boundary condition and periodic boundaries. Did you "
           "add a new constructor violating this constraint?");
    DirectionMap<2,
                 std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>
        boundary_conditions{};
    for (const auto& direction : Direction<2>::all_directions()) {
      boundary_conditions[direction] = boundary_condition_->get_clone();
    }
    boundary_conditions_all_blocks.push_back(std::move(boundary_conditions));
  }

  Domain<2> domain{
      make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine2D{Affine{-1., 1., lower_xy_[0], upper_xy_[0]},
                   Affine{-1., 1., lower_xy_[1], upper_xy_[1]}}),
      std::vector<std::array<size_t, 4>>{{{0, 1, 2, 3}}}, identifications,
      std::move(boundary_conditions_all_blocks)};

  if (not time_dependence_->is_none()) {
    domain.inject_time_dependent_map_for_block(
        0, std::move(time_dependence_->block_maps(1)[0]));
  }
  return domain;
}

std::vector<std::array<size_t, 2>> Rectangle::initial_extents() const noexcept {
  return {initial_number_of_grid_points_in_xy_};
}

std::vector<std::array<size_t, 2>> Rectangle::initial_refinement_levels() const
    noexcept {
  return {initial_refinement_level_xy_};
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
Rectangle::functions_of_time() const noexcept {
  if (time_dependence_->is_none()) {
    return {};
  } else {
    return time_dependence_->functions_of_time();
  }
}
}  // namespace domain::creators
