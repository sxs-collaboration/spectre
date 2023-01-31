// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Sphere.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

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
#include "Domain/Creators/ExpandOverBlocks.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Inertial;
struct BlockLogical;
}  // namespace Frame

namespace domain::creators {
Sphere::Sphere(
    double inner_radius, double outer_radius, double inner_cube_sphericity,
    const typename InitialRefinement::type& initial_refinement,
    const typename InitialGridPoints::type& initial_number_of_grid_points,
    bool use_equiangular_map, std::vector<double> radial_partitioning,
    std::vector<domain::CoordinateMaps::Distribution> radial_distribution,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        boundary_condition,
    const Options::Context& context)
    : inner_radius_(inner_radius),
      outer_radius_(outer_radius),
      inner_cube_sphericity_(inner_cube_sphericity),
      use_equiangular_map_(use_equiangular_map),
      radial_partitioning_(std::move(radial_partitioning)),
      radial_distribution_(std::move(radial_distribution)),
      time_dependence_(std::move(time_dependence)),

      boundary_condition_(std::move(boundary_condition)) {
  if (inner_cube_sphericity_ < 0.0 or inner_cube_sphericity_ >= 1.0) {
    PARSE_ERROR(
        context,
        "Inner cube sphericity must be >= 0.0 and strictly < 1.0, not " +
            get_output(inner_cube_sphericity_));
  }

  if (inner_radius_ > outer_radius_) {
    PARSE_ERROR(context,
                "Inner radius must be smaller than outer radius, but inner "
                "radius is " +
                    std::to_string(inner_radius_) + " and outer radius is " +
                    std::to_string(outer_radius_) + ".");
  }
  if (not std::is_sorted(radial_partitioning_.begin(),
                         radial_partitioning_.end())) {
    PARSE_ERROR(context,
                "Specify radial partitioning in ascending order. Specified "
                "radial partitioning is: " +
                    get_output(radial_partitioning_));
  }
  if (not radial_partitioning_.empty()) {
    if (radial_partitioning_.front() <= inner_radius_) {
      PARSE_ERROR(context,
                  "First radial partition must be larger than inner "
                  "radius, but is: " +
                      std::to_string(inner_radius_));
    }
    if (radial_partitioning_.back() >= outer_radius_) {
      PARSE_ERROR(context,
                  "Last radial partition must be smaller than outer "
                  "radius, but is: " +
                      std::to_string(outer_radius_));
    }
  }

  const size_t num_shells = 1 + radial_partitioning_.size();
  if (radial_distribution_.size() != num_shells) {
    PARSE_ERROR(context,
                "Specify a 'RadialDistribution' for every spherical shell. You "
                "specified "
                    << radial_distribution_.size()
                    << " items, but the domain has " << num_shells
                    << " shells.");
  }
  if (radial_distribution_.front() !=
      domain::CoordinateMaps::Distribution::Linear) {
    PARSE_ERROR(context,
                "The 'RadialDistribution' must be 'Linear' for the innermost "
                "shell because it changes in sphericity. Add entries to "
                "'RadialPartitioning' to add outer shells for which you can "
                "select different radial distributions.");
  }

  // Create block names and groups
  static std::array<std::string, 6> wedge_directions{
      "UpperZ", "LowerZ", "UpperY", "LowerY", "UpperX", "LowerX"};
  for (size_t shell = 0; shell < num_shells; ++shell) {
    std::string shell_prefix = "Shell" + std::to_string(shell);
    for (size_t direction = 0; direction < 6; ++direction) {
      const std::string wedge_name =
          shell_prefix + gsl::at(wedge_directions, direction);
      block_names_.emplace_back(wedge_name);
      if (num_shells > 1) {
        block_groups_[shell_prefix].insert(wedge_name);
      }
      block_groups_["Wedges"].insert(wedge_name);
    }
  }
  block_names_.emplace_back("InnerCube");

  // Expand initial refinement and number of grid points over all blocks
  const ExpandOverBlocks<size_t, 3> expand_over_blocks{block_names_,
                                                       block_groups_};
  try {
    initial_refinement_ = std::visit(expand_over_blocks, initial_refinement);
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialRefinement': " << error.what());
  }
  try {
    initial_number_of_grid_points_ =
        std::visit(expand_over_blocks, initial_number_of_grid_points);
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialGridPoints': " << error.what());
  }

  // The central cube has no notion of a "radial" direction, so we set
  // refinement and number of grid points of the central cube z direction to
  // its y value, which corresponds to the azimuthal direction of the
  // wedges. This keeps the boundaries conforming when the radial direction
  // is chosen differently to the angular directions.
  auto& central_cube_refinement = initial_refinement_.back();
  auto& central_cube_grid_points = initial_number_of_grid_points_.back();
  central_cube_refinement[2] = central_cube_refinement[1];
  central_cube_grid_points[2] = central_cube_grid_points[1];

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
  const size_t num_shells = 1 + radial_partitioning_.size();
  std::vector<std::array<size_t, 8>> corners =
      corners_for_radially_layered_domains(num_shells, true);

  auto coord_maps = domain::make_vector_coordinate_map_base<
      Frame::BlockLogical, Frame::Inertial, 3>(sph_wedge_coordinate_maps(
      inner_radius_, outer_radius_, inner_cube_sphericity_, 1.0,
      use_equiangular_map_, false, radial_partitioning_, radial_distribution_));
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

  Domain<3> domain{std::move(coord_maps), corners,      {}, {},
                   block_names_,          block_groups_};

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

std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
Sphere::external_boundary_conditions() const {
  if (boundary_condition_ == nullptr) {
    return {};
  }

  // number of blocks = 1 inner_block + 6 * (number of shells)
  size_t number_of_blocks = 1 + 6 * (radial_partitioning_.size() + 1);

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{number_of_blocks};
  for (size_t i = number_of_blocks - 7; i < number_of_blocks - 1; ++i) {
    boundary_conditions[i][Direction<3>::upper_zeta()] =
        boundary_condition_->get_clone();
  }
  return boundary_conditions;
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
