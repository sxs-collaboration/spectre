// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Sphere.hpp"

#include <cmath>
#include <memory>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/BulgedCube.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Inertial;
struct BlockLogical;
}  // namespace Frame

namespace domain::creators {
Sphere::Sphere(
    typename InnerRadius::type inner_radius,
    typename OuterRadius::type outer_radius, const double inner_cube_sphericity,
    typename InitialRefinement::type initial_refinement,
    typename InitialGridPoints::type initial_number_of_grid_points,
    typename UseEquiangularMap::type use_equiangular_map,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        boundary_condition,
    const Options::Context& context)
    // clang-tidy: trivially copyable
    : inner_radius_(std::move(inner_radius)),  // NOLINT
      outer_radius_(std::move(outer_radius)),  // NOLINT
      inner_cube_sphericity_(inner_cube_sphericity),
      initial_refinement_(                                   // NOLINT
          std::move(initial_refinement)),                    // NOLINT
      initial_number_of_grid_points_(                        // NOLINT
          std::move(initial_number_of_grid_points)),         // NOLINT
      use_equiangular_map_(std::move(use_equiangular_map)),  // NOLINT
      time_dependence_(std::move(time_dependence)),          // NOLINT
      boundary_condition_(std::move(boundary_condition)) {
  if (inner_cube_sphericity_ < 0.0 or inner_cube_sphericity_ >= 1.0) {
    PARSE_ERROR(
        context,
        "Inner cube sphericity must be >= 0.0 and strictly < 1.0, not " +
            get_output(inner_cube_sphericity_));
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
    PARSE_ERROR(context,
                "Cannot have periodic boundary conditions with a Sphere");
  }
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<3>>();
  }
}

Domain<3> Sphere::create_domain() const {
  std::vector<std::array<size_t, 8>> corners =
      corners_for_radially_layered_domains(1, true);

  auto coord_maps = domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                                            Frame::Inertial, 3>(
      sph_wedge_coordinate_maps(inner_radius_, outer_radius_,
                                inner_cube_sphericity_, 1.0,
                                use_equiangular_map_));
  if (inner_cube_sphericity_ == 0.0) {
    if (use_equiangular_map_) {
      coord_maps.emplace_back(
          make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
              Equiangular3D{
                  Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                              inner_radius_ / sqrt(3.0)),
                  Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                              inner_radius_ / sqrt(3.0)),
                  Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                              inner_radius_ / sqrt(3.0))}));
    } else {
      coord_maps.emplace_back(
          make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
              Affine3D{Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                              inner_radius_ / sqrt(3.0)),
                       Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                              inner_radius_ / sqrt(3.0)),
                       Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                              inner_radius_ / sqrt(3.0))}));
    }
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            BulgedCube{inner_radius_, inner_cube_sphericity_,
                       use_equiangular_map_}));
  }

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions_all_blocks{};
  if (boundary_condition_ != nullptr) {
    boundary_conditions_all_blocks.resize(7);
    ASSERT(coord_maps.size() == 7,
           "The number of blocks for which coordinate maps and boundary "
           "conditions are specified should be 7 but the coordinate maps is: "
               << coord_maps.size());
    for (size_t block_id = 0;
         block_id < boundary_conditions_all_blocks.size() - 1; ++block_id) {
      boundary_conditions_all_blocks[block_id][Direction<3>::upper_zeta()] =
          boundary_condition_->get_clone();
    }
  }

  Domain<3> domain{std::move(coord_maps),
                   corners,
                   {},
                   std::move(boundary_conditions_all_blocks)};

  if (not time_dependence_->is_none()) {
    const size_t number_of_blocks = domain.blocks().size();
    auto block_maps_grid_to_inertial =
        time_dependence_->block_maps_grid_to_inertial(number_of_blocks);
    auto block_maps_grid_to_distorted =
        time_dependence_->block_maps_grid_to_distorted(number_of_blocks);
    auto block_maps_distorted_to_inertial =
        time_dependence_->block_maps_distorted_to_inertial(number_of_blocks);
    for (size_t block_id = 0; block_id < number_of_blocks; ++block_id) {
      domain.inject_time_dependent_map_for_block(
          block_id, std::move(block_maps_grid_to_inertial[block_id]),
          std::move(block_maps_grid_to_distorted[block_id]),
          std::move(block_maps_distorted_to_inertial[block_id]));
    }
  }

  return domain;
}

std::vector<std::array<size_t, 3>> Sphere::initial_extents() const {
  std::vector<std::array<size_t, 3>> extents{
      6,
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1],
        initial_number_of_grid_points_[0]}}};
  extents.push_back(
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1],
        initial_number_of_grid_points_[1]}});
  return extents;
}

std::vector<std::array<size_t, 3>> Sphere::initial_refinement_levels() const {
  return {7, make_array<3>(initial_refinement_)};
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
Sphere::functions_of_time(const std::unordered_map<std::string, double>&
                              initial_expiration_times) const {
  if (time_dependence_->is_none()) {
    return {};
  } else {
    return time_dependence_->functions_of_time(initial_expiration_times);
  }
}
}  // namespace domain::creators
