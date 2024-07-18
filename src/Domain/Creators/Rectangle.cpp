// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Rectangle.hpp"

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace Frame {
struct Inertial;
struct BlockLogical;
}  // namespace Frame

namespace domain::creators {
Rectangle::Rectangle(
    std::array<double, 2> lower_xy, std::array<double, 2> upper_xy,
    std::array<size_t, 2> initial_refinement_level_xy,
    std::array<size_t, 2> initial_number_of_grid_points_in_xy,
    std::array<bool, 2> is_periodic_in_xy,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<2>>
        time_dependence)
    : lower_xy_(lower_xy),
      upper_xy_(upper_xy),
      is_periodic_in_xy_(is_periodic_in_xy),
      initial_refinement_level_xy_(initial_refinement_level_xy),
      initial_number_of_grid_points_in_xy_(initial_number_of_grid_points_in_xy),
      time_dependence_(std::move(time_dependence)),
      boundary_condition_(nullptr) {
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<2>>();
  }
}

Rectangle::Rectangle(
    std::array<double, 2> lower_xy, std::array<double, 2> upper_xy,
    std::array<size_t, 2> initial_refinement_level_xy,
    std::array<size_t, 2> initial_number_of_grid_points_in_xy,
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
        "outflow-type boundary condition, you must use that.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(boundary_condition_)) {
    is_periodic_in_xy_[0] = true;
    is_periodic_in_xy_[1] = true;
  }
}

Domain<2> Rectangle::create_domain() const {
  using Affine = CoordinateMaps::Affine;
  using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  std::vector<PairOfFaces> identifications{};
  if (is_periodic_in_xy_[0]) {
    identifications.push_back({{0, 2}, {1, 3}});
  }
  if (is_periodic_in_xy_[1]) {
    identifications.push_back({{0, 1}, {2, 3}});
  }

  Domain<2> domain{
      make_vector_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          Affine2D{Affine{-1., 1., lower_xy_[0], upper_xy_[0]},
                   Affine{-1., 1., lower_xy_[1], upper_xy_[1]}}),
      std::vector<std::array<size_t, 4>>{{{0, 1, 2, 3}}},
      identifications,
      {},
      block_names_};

  if (not time_dependence_->is_none()) {
    domain.inject_time_dependent_map_for_block(
        0, std::move(time_dependence_->block_maps_grid_to_inertial(1)[0]),
        std::move(time_dependence_->block_maps_grid_to_distorted(1)[0]),
        std::move(time_dependence_->block_maps_distorted_to_inertial(1)[0]));
  }
  return domain;
}

std::vector<DirectionMap<
    2, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
Rectangle::external_boundary_conditions() const {
  if (boundary_condition_ == nullptr) {
    return {};
  }
  std::vector<DirectionMap<
      2, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{1};
  if (is_periodic_in_xy_[0]) {
    return boundary_conditions;
  }
  for (const auto& direction : Direction<2>::all_directions()) {
    boundary_conditions[0][direction] = boundary_condition_->get_clone();
  }
  return boundary_conditions;
}

std::vector<std::array<size_t, 2>> Rectangle::initial_extents() const {
  return {initial_number_of_grid_points_in_xy_};
}

std::vector<std::array<size_t, 2>> Rectangle::initial_refinement_levels()
    const {
  return {initial_refinement_level_xy_};
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
Rectangle::functions_of_time(const std::unordered_map<std::string, double>&
                                 initial_expiration_times) const {
  if (time_dependence_->is_none()) {
    return {};
  } else {
    return time_dependence_->functions_of_time(initial_expiration_times);
  }
}
}  // namespace domain::creators
