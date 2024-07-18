// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Brick.hpp"

#include <array>
#include <memory>
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
#include "Options/ParseError.hpp"

namespace Frame {
struct BlockLogical;
struct Inertial;
}  // namespace Frame

namespace domain::creators {
Brick::Brick(
    std::array<double, 3> lower_xyz, std::array<double, 3> upper_xyz,
    std::array<size_t, 3> initial_refinement_level_xyz,
    std::array<size_t, 3> initial_number_of_grid_points_in_xyz,
    std::array<bool, 3> is_periodic_in_xyz,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence)
    : lower_xyz_(lower_xyz),
      upper_xyz_(upper_xyz),
      is_periodic_in_xyz_(is_periodic_in_xyz),
      initial_refinement_level_xyz_(initial_refinement_level_xyz),
      initial_number_of_grid_points_in_xyz_(
          initial_number_of_grid_points_in_xyz),
      time_dependence_(std::move(time_dependence)),
      boundary_condition_in_lower_x_(nullptr),
      boundary_condition_in_upper_x_(nullptr),
      boundary_condition_in_lower_y_(nullptr),
      boundary_condition_in_upper_y_(nullptr),
      boundary_condition_in_lower_z_(nullptr),
      boundary_condition_in_upper_z_(nullptr) {
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<3>>();
  }
}

Brick::Brick(
    std::array<double, 3> lower_xyz, std::array<double, 3> upper_xyz,
    std::array<size_t, 3> initial_refinement_level_xyz,
    std::array<size_t, 3> initial_number_of_grid_points_in_xyz,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        boundary_condition_in_lower_x,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        boundary_condition_in_upper_x,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        boundary_condition_in_lower_y,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        boundary_condition_in_upper_y,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        boundary_condition_in_lower_z,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        boundary_condition_in_upper_z,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence,
    const Options::Context& context)
    : lower_xyz_(lower_xyz),
      upper_xyz_(upper_xyz),
      is_periodic_in_xyz_{{false, false, false}},
      initial_refinement_level_xyz_(initial_refinement_level_xyz),
      initial_number_of_grid_points_in_xyz_(
          initial_number_of_grid_points_in_xyz),
      time_dependence_(std::move(time_dependence)),
      boundary_condition_in_lower_x_(std::move(boundary_condition_in_lower_x)),
      boundary_condition_in_upper_x_(std::move(boundary_condition_in_upper_x)),
      boundary_condition_in_lower_y_(std::move(boundary_condition_in_lower_y)),
      boundary_condition_in_upper_y_(std::move(boundary_condition_in_upper_y)),
      boundary_condition_in_lower_z_(std::move(boundary_condition_in_lower_z)),
      boundary_condition_in_upper_z_(std::move(boundary_condition_in_upper_z)) {
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<3>>();
  }
  using domain::BoundaryConditions::is_none;
  ASSERT(boundary_condition_in_lower_x_ != nullptr and
             boundary_condition_in_upper_x_ != nullptr and
             boundary_condition_in_lower_y_ != nullptr and
             boundary_condition_in_upper_y_ != nullptr and
             boundary_condition_in_lower_z_ != nullptr and
             boundary_condition_in_upper_z_ != nullptr,
         "None of the boundary conditions can be nullptr.");
  if (is_none(boundary_condition_in_lower_x_) or
      is_none(boundary_condition_in_upper_x_) or
      is_none(boundary_condition_in_lower_y_) or
      is_none(boundary_condition_in_upper_y_) or
      is_none(boundary_condition_in_lower_z_) or
      is_none(boundary_condition_in_lower_z_)) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow-type boundary condition, you must use that.");
  }
  using domain::BoundaryConditions::is_periodic;

  if ((is_periodic(boundary_condition_in_lower_x_) !=
       is_periodic(boundary_condition_in_upper_x_)) or
      (is_periodic(boundary_condition_in_lower_y_) !=
       is_periodic(boundary_condition_in_upper_y_)) or
      (is_periodic(boundary_condition_in_lower_z_) !=
       is_periodic(boundary_condition_in_lower_z_))) {
    PARSE_ERROR(context,
                "Pierodic boundary condition must be applied for both "
                "upper and lower direction.");
  }

  if (is_periodic(boundary_condition_in_lower_x_) and
      is_periodic(boundary_condition_in_upper_x_)) {
    is_periodic_in_xyz_[0] = true;
  }
  if (is_periodic(boundary_condition_in_lower_y_) and
      is_periodic(boundary_condition_in_upper_y_)) {
    is_periodic_in_xyz_[1] = true;
  }
  if (is_periodic(boundary_condition_in_lower_z_) and
      is_periodic(boundary_condition_in_upper_z_)) {
    is_periodic_in_xyz_[2] = true;
  }
}

Domain<3> Brick::create_domain() const {
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  std::vector<PairOfFaces> identifications{};
  if (is_periodic_in_xyz_[0]) {
    identifications.push_back({{0, 4, 2, 6}, {1, 5, 3, 7}});
  }
  if (is_periodic_in_xyz_[1]) {
    identifications.push_back({{0, 1, 4, 5}, {2, 3, 6, 7}});
  }
  if (is_periodic_in_xyz_[2]) {
    identifications.push_back({{0, 1, 2, 3}, {4, 5, 6, 7}});
  }

  Domain<3> domain{
      make_vector_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          Affine3D{Affine{-1., 1., lower_xyz_[0], upper_xyz_[0]},
                   Affine{-1., 1., lower_xyz_[1], upper_xyz_[1]},
                   Affine{-1., 1., lower_xyz_[2], upper_xyz_[2]}}),
      std::vector<std::array<size_t, 8>>{{{0, 1, 2, 3, 4, 5, 6, 7}}},
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
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
Brick::external_boundary_conditions() const {
  if (boundary_condition_in_lower_x_ == nullptr) {
    ASSERT(boundary_condition_in_upper_x_ == nullptr and
               boundary_condition_in_lower_y_ == nullptr and
               boundary_condition_in_upper_y_ == nullptr and
               boundary_condition_in_lower_z_ == nullptr and
               boundary_condition_in_upper_z_ == nullptr,
           "Boundary conditions must be specified in all or no directions");
    return {};
  }
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{1};
  if (not is_periodic_in_xyz_[0]) {
    boundary_conditions[0][Direction<3>{0, Side::Lower}] =
        boundary_condition_in_lower_x_->get_clone();
    boundary_conditions[0][Direction<3>{0, Side::Upper}] =
        boundary_condition_in_upper_x_->get_clone();
  }
  if (not is_periodic_in_xyz_[1]) {
    boundary_conditions[0][Direction<3>{1, Side::Lower}] =
        boundary_condition_in_lower_y_->get_clone();
    boundary_conditions[0][Direction<3>{1, Side::Upper}] =
        boundary_condition_in_upper_y_->get_clone();
  }
  if (not is_periodic_in_xyz_[2]) {
    boundary_conditions[0][Direction<3>{2, Side::Lower}] =
        boundary_condition_in_lower_z_->get_clone();
    boundary_conditions[0][Direction<3>{2, Side::Upper}] =
        boundary_condition_in_upper_z_->get_clone();
  }
  return boundary_conditions;
}

std::vector<std::array<size_t, 3>> Brick::initial_extents() const {
  return {initial_number_of_grid_points_in_xyz_};
}

std::vector<std::array<size_t, 3>> Brick::initial_refinement_levels() const {
  return {initial_refinement_level_xyz_};
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
Brick::functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times) const {
  if (time_dependence_->is_none()) {
    return {};
  } else {
    return time_dependence_->functions_of_time(initial_expiration_times);
  }
}
}  // namespace domain::creators
