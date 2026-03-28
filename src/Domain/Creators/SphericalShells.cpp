// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/SphericalShells.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/ShellDistribution.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"

namespace domain::creators {

SphericalShells::SphericalShells(
    const double inner_radius, const double outer_radius,
    const size_t initial_radial_refinement,
    const size_t initial_number_of_radial_grid_points,
    const size_t initial_spherical_harmonic_l,
    std::vector<double> radial_partitioning,
    const typename RadialDistribution::type& radial_distribution,
    std::optional<TimeDepOptionType> time_dependent_options,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        inner_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : inner_radius_(inner_radius),
      outer_radius_(outer_radius),
      initial_radial_refinement_(initial_radial_refinement),
      initial_number_of_radial_grid_points_(
          initial_number_of_radial_grid_points),
      initial_spherical_harmonic_l_(initial_spherical_harmonic_l),
      radial_partitioning_(std::move(radial_partitioning)),
      time_dependent_options_(std::move(time_dependent_options)),
      inner_boundary_condition_(std::move(inner_boundary_condition)),
      outer_boundary_condition_(std::move(outer_boundary_condition)),
      grid_anchors_{{{"Center", tnsr::I<double, 3, Frame::Grid>{
                                    std::array{0.0, 0.0, 0.0}}}}} {
  if (inner_radius_ > outer_radius_) {
    PARSE_ERROR(context,
                "Inner radius must be smaller than outer radius, but inner "
                "radius is " +
                    std::to_string(inner_radius_) + " and outer radius is " +
                    std::to_string(outer_radius_) + ".");
  }
  set_shell_distribution(
      make_not_null(&num_blocks_), make_not_null(&radial_distribution_),
      radial_partitioning_, radial_distribution, inner_radius_, outer_radius_,
      "inner", "outer", context);
  for (size_t shell = 0; shell < num_blocks_; ++shell) {
    const std::string shell_name = "Shell" + std::to_string(shell);
    block_names_.emplace_back(shell_name);
    // This makes consistent block groups with those created by Sphere
    block_groups_[shell_name].insert(shell_name);
  }

  // Validate boundary conditions
  using domain::BoundaryConditions::is_none;
  if (is_none(inner_boundary_condition_) or
      is_none(outer_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow-type boundary condition, you must use that.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(inner_boundary_condition_) or
      is_periodic(outer_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "Cannot have periodic boundary conditions with SphericalShells");
  }
  // Validate consistency of inner and outer boundary condition
  if ((inner_boundary_condition_ == nullptr) !=
      (outer_boundary_condition_ == nullptr)) {
    PARSE_ERROR(context,
                "Must specify either both inner and outer boundary conditions "
                "or neither.");
  }

  if (time_dependent_options_.has_value()) {
    use_hard_coded_maps_ =
        std::holds_alternative<sphere::TimeDependentMapOptions>(
            time_dependent_options_.value());
    if (use_hard_coded_maps_) {
      std::get<sphere::TimeDependentMapOptions>(time_dependent_options_.value())
          .build_maps(std::array{0.0, 0.0, 0.0}, false, inner_radius_,
                      radial_partitioning_, outer_radius_);
    }
  }
  domain_ = build_domain(context);
}

Domain<3> SphericalShells::create_domain(
    const Options::Context& /*context*/) const {
  std::vector<Block<3>> blocks;
  blocks.reserve(num_blocks_);
  const auto aligned = OrientationMap<3>::create_aligned();
  for (size_t i = 0; i < num_blocks_; ++i) {
    CoordinateMaps::Interval radial_map{
        -1.0,
        1.0,
        i == 0 ? inner_radius_ : radial_partitioning_[i - 1],
        i == num_blocks_ - 1 ? outer_radius_ : radial_partitioning_[i],
        radial_distribution_[i],
        0.0};
    auto stationary_map =
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            CoordinateMaps::ProductOf2Maps<CoordinateMaps::Interval,
                                           CoordinateMaps::Identity<2>>{
                std::move(radial_map), CoordinateMaps::Identity<2>{}},
            CoordinateMaps::SphericalToCartesianPfaffian{});
    DirectionMap<3, BlockNeighbors<3>> neighbors;
    if (i > 0) {
      neighbors.emplace(std::pair{Direction<3>::lower_xi(),
                                  BlockNeighbors<3>{i - 1, aligned}});
    }
    if (i < num_blocks_ - 1) {
      neighbors.emplace(std::pair{Direction<3>::upper_xi(),
                                  BlockNeighbors<3>{i + 1, aligned}});
    }
    blocks.emplace_back(std::move(stationary_map), i, std::move(neighbors),
                        block_names_.at(i),
                        domain::topologies::spherical_shell);
  }

  std::unordered_map<std::string, ExcisionSphere<3>> excision_spheres{};
  excision_spheres.emplace(
      "ExcisionSphere", ExcisionSphere<3>{inner_radius_,
                                          tnsr::I<double, 3, Frame::Grid>{0.0},
                                          {{0, Direction<3>::lower_xi()}}});

  Domain<3> domain(std::move(blocks), std::move(excision_spheres),
                   block_groups_);

  if (time_dependent_options_.has_value()) {
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
        block_maps_grid_to_inertial{num_blocks_};
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, 3>>>
        block_maps_grid_to_distorted{num_blocks_};
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, 3>>>
        block_maps_distorted_to_inertial{num_blocks_};

    if (use_hard_coded_maps_) {
      const auto& hard_coded_options =
          std::get<sphere::TimeDependentMapOptions>(
              time_dependent_options_.value());

      for (size_t block_id = 0; block_id < num_blocks_; block_id++) {
        const bool is_outer_shell = block_id == num_blocks_ - 1;
        block_maps_grid_to_distorted[block_id] =
            hard_coded_options.grid_to_distorted_map(block_id, false, 1);
        block_maps_distorted_to_inertial[block_id] =
            hard_coded_options.distorted_to_inertial_map(block_id, false, 1);
        block_maps_grid_to_inertial[block_id] =
            hard_coded_options.grid_to_inertial_map(block_id, is_outer_shell,
                                                    false, 1);
      }

      domain.inject_time_dependent_map_for_excision_sphere(
          "ExcisionSphere",
          hard_coded_options.grid_to_inertial_map(0, false, true, 1));
    } else {
      const auto& time_dependence = std::get<std::unique_ptr<
          domain::creators::time_dependence::TimeDependence<3>>>(
          time_dependent_options_.value());

      block_maps_grid_to_inertial =
          time_dependence->block_maps_grid_to_inertial(num_blocks_);
      block_maps_grid_to_distorted =
          time_dependence->block_maps_grid_to_distorted(num_blocks_);
      block_maps_distorted_to_inertial =
          time_dependence->block_maps_distorted_to_inertial(num_blocks_);
    }

    for (size_t block_id = 0; block_id < num_blocks_; ++block_id) {
      domain.inject_time_dependent_map_for_block(
          block_id, std::move(block_maps_grid_to_inertial[block_id]),
          std::move(block_maps_grid_to_distorted[block_id]),
          std::move(block_maps_distorted_to_inertial[block_id]));
    }
  }

  return domain;
}

std::unordered_map<std::string, tnsr::I<double, 3, Frame::Grid>>
SphericalShells::grid_anchors() const {
  return grid_anchors_;
}

std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
SphericalShells::external_boundary_conditions() const {
  if (outer_boundary_condition_ == nullptr) {
    return {};
  }
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{num_blocks_};
  boundary_conditions[0][Direction<3>::lower_xi()] =
      inner_boundary_condition_->get_clone();
  boundary_conditions[num_blocks_ - 1][Direction<3>::upper_xi()] =
      outer_boundary_condition_->get_clone();
  return boundary_conditions;
}

std::vector<std::array<size_t, 3>> SphericalShells::initial_extents() const {
  return std::vector{num_blocks_,
                     std::array{initial_number_of_radial_grid_points_,
                                initial_spherical_harmonic_l_ + 1,
                                2 * initial_spherical_harmonic_l_ + 1}};
}

Domain<3> SphericalShells::create_domain() const { return domain_; }

std::vector<std::array<size_t, 3>> SphericalShells::initial_refinement_levels()
    const {
  return std::vector{num_blocks_,
                     std::array{initial_radial_refinement_, 0_st, 0_st}};
}

std::vector<std::string> SphericalShells::block_names() const {
  return block_names_;
}

std::unordered_map<std::string, std::unordered_set<std::string>>
SphericalShells::block_groups() const {
  return block_groups_;
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
SphericalShells::functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  if (time_dependent_options_.has_value()) {
    if (use_hard_coded_maps_) {
      return std::get<sphere::TimeDependentMapOptions>(
                 time_dependent_options_.value())
          .create_functions_of_time(initial_expiration_times);
    } else {
      return std::get<std::unique_ptr<
          domain::creators::time_dependence::TimeDependence<3>>>(
                 time_dependent_options_.value())
          ->functions_of_time(initial_expiration_times);
    }
  } else {
    return {};
  }
}
}  // namespace domain::creators
