// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/AngularCartoonSphere2D.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/PolarToCartesian.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Options/ParseError.hpp"

namespace Frame {
struct Inertial;
struct BlockLogical;
}  // namespace Frame

namespace domain::creators {
AngularCartoonSphere2D::AngularCartoonSphere2D(
    const typename InnerRadius::type inner_radius,
    const typename OuterRadius::type outer_radius,
    typename RadialPartitioning::type radial_partitioning,
    typename RadialDistribution::type radial_distribution,
    typename InitialGridPoints::type initial_grid_points,
    typename InitialRefinementInR::type initial_refinement_in_r,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        inner_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : inner_radius_(inner_radius),
      outer_radius_(outer_radius),
      radial_partitioning_(std::move(radial_partitioning)),
      radial_distribution_(std::move(radial_distribution)),
      initial_refinement_in_r_(std::move(initial_refinement_in_r)),
      time_dependence_(std::move(time_dependence)) {
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<3>>();
  }

  inner_boundary_condition_ = std::move(inner_boundary_condition);
  outer_boundary_condition_ = std::move(outer_boundary_condition);

  if ((inner_boundary_condition_ == nullptr) !=
      (outer_boundary_condition_ == nullptr)) {
    PARSE_ERROR(context,
                "Either both InnerBoundary and OuterBoundary boundary "
                "conditions must be specified, or neither.");
  }
  if (inner_boundary_condition_ != nullptr) {
    if (domain::BoundaryConditions::is_none(inner_boundary_condition_) or
        domain::BoundaryConditions::is_none(outer_boundary_condition_)) {
      PARSE_ERROR(context,
                  "None boundary condition is not supported. If you "
                  "would like an outflow-type boundary "
                  "condition, you must use that.");
    }
    if (domain::BoundaryConditions::is_periodic(inner_boundary_condition_) or
        domain::BoundaryConditions::is_periodic(outer_boundary_condition_)) {
      PARSE_ERROR(context,
                  "Cannot have periodic boundary conditions on a 2D sphere.");
    }
  }

  if (inner_radius_ <= 0.0) {
    PARSE_ERROR(context,
                "InnerRadius must be positive because the origin must be "
                "excised, but is "
                    << inner_radius_);
  }
  if (inner_radius_ >= outer_radius_) {
    PARSE_ERROR(context,
                "InnerRadius must be smaller than OuterRadius, but "
                "InnerRadius is "
                    << inner_radius_ << " and OuterRadius is " << outer_radius_
                    << ".");
  }

  num_blocks_ = 1 + radial_partitioning_.size();

  if (initial_refinement_in_r_.size() != num_blocks_) {
    PARSE_ERROR(context,
                "InitialRefinementInR must have one entry per radial block "
                "(num_blocks = "
                    << num_blocks_ << "), but has size "
                    << initial_refinement_in_r_.size() << ".");
  }

  if (not radial_partitioning_.empty()) {
    if (not std::ranges::is_sorted(radial_partitioning_)) {
      PARSE_ERROR(context,
                  "You must specify RadialPartitioning in ascending order.");
    }
    const auto duplicate = std::ranges::adjacent_find(radial_partitioning_);
    if (duplicate != radial_partitioning_.end()) {
      PARSE_ERROR(context, "RadialPartitioning contains duplicate element: "
                               << *duplicate);
    }
    if (radial_partitioning_.front() <= inner_radius_) {
      PARSE_ERROR(context, "Radial partitions must be larger than InnerRadius ("
                               << inner_radius_ << "), but the first one is "
                               << radial_partitioning_.front() << ".");
    }
    if (radial_partitioning_.back() >= outer_radius_) {
      PARSE_ERROR(context,
                  "Radial partitions must be smaller than OuterRadius ("
                      << outer_radius_ << "), but the last one is "
                      << radial_partitioning_.back() << ".");
    }
  }

  if (radial_distribution_.size() != num_blocks_) {
    PARSE_ERROR(context,
                "RadialDistribution must have one entry per radial block "
                "(num_blocks = "
                    << num_blocks_ << "), but has size "
                    << radial_distribution_.size() << ".");
  }

  // Expand the grid points over the radial blocks.
  if (std::holds_alternative<std::array<size_t, 2>>(initial_grid_points)) {
    initial_grid_points_ = std::vector<std::array<size_t, 2>>(
        num_blocks_, std::get<std::array<size_t, 2>>(initial_grid_points));
  } else {
    initial_grid_points_ =
        std::get<std::vector<std::array<size_t, 2>>>(initial_grid_points);
    if (initial_grid_points_.size() != num_blocks_) {
      PARSE_ERROR(context,
                  "InitialGridPoints must have one entry per radial block "
                  "(num_blocks = "
                      << num_blocks_ << "), but has size "
                      << initial_grid_points_.size() << ".");
    }
  }

  // Build block names and groups.
  block_names_.reserve(num_blocks_);
  for (size_t shell = 0; shell < num_blocks_; ++shell) {
    const std::string shell_name = "Shell" + std::to_string(shell);
    block_names_.emplace_back(shell_name);
    block_groups_["Shells"].insert(shell_name);
  }
}

Domain<3> AngularCartoonSphere2D::create_domain() const {
  using Identity1D = CoordinateMaps::Identity<1>;
  using Interval = CoordinateMaps::Interval;
  const auto aligned = OrientationMap<3>::create_aligned();

  // Maps (x, y) -> (y, -x), so that after PolarToCartesian the azimuthal
  // coordinate phi in (0, pi) sweeps from the -y axis through +x to the +y
  // axis, keeping x = r sin(phi) >= 0.
  const CoordinateMaps::DiscreteRotation<2> rotate_cw{
      OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}}};

  std::vector<Block<3>> blocks;
  blocks.reserve(num_blocks_);

  for (size_t radial = 0; radial < num_blocks_; ++radial) {
    const double inner_r =
        radial == 0 ? inner_radius_ : radial_partitioning_[radial - 1];
    const double outer_r = radial == num_blocks_ - 1
                               ? outer_radius_
                               : radial_partitioning_[radial];
    auto coord_map =
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            CoordinateMaps::ProductOf3Maps<Interval, Identity1D, Identity1D>{
                Interval{-1.0, 1.0, inner_r, outer_r,
                         radial_distribution_[radial], 0.0},
                Identity1D{}, Identity1D{}},
            CoordinateMaps::ProductOf2Maps<CoordinateMaps::PolarToCartesian,
                                           Identity1D>{
                CoordinateMaps::PolarToCartesian{}, Identity1D{}},
            CoordinateMaps::ProductOf2Maps<CoordinateMaps::DiscreteRotation<2>,
                                           Identity1D>{rotate_cw,
                                                       Identity1D{}});

    DirectionMap<3, BlockNeighbors<3>> neighbors{};

    if (radial > 0) {
      neighbors.emplace(Direction<3>::lower_xi(),
                        BlockNeighbors<3>(radial - 1, aligned));
    }
    if (radial < num_blocks_ - 1) {
      neighbors.emplace(Direction<3>::upper_xi(),
                        BlockNeighbors<3>(radial + 1, aligned));
    }

    blocks.emplace_back(std::move(coord_map), radial, std::move(neighbors),
                        block_names_.at(radial),
                        domain::topologies::cartoon_cylinder_annulus);
  }

  Domain<3> domain(std::move(blocks), {}, block_groups_);

  if (not time_dependence_->is_none()) {
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
        block_maps_grid_to_inertial =
            time_dependence_->block_maps_grid_to_inertial(num_blocks_);
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, 3>>>
        block_maps_grid_to_distorted =
            time_dependence_->block_maps_grid_to_distorted(num_blocks_);
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, 3>>>
        block_maps_distorted_to_inertial =
            time_dependence_->block_maps_distorted_to_inertial(num_blocks_);
    for (size_t block_id = 0; block_id < num_blocks_; ++block_id) {
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
AngularCartoonSphere2D::external_boundary_conditions() const {
  if (inner_boundary_condition_ == nullptr) {
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

std::vector<std::array<size_t, 3>> AngularCartoonSphere2D::initial_extents()
    const {
  std::vector<std::array<size_t, 3>> extents;
  extents.reserve(num_blocks_);
  for (const auto& grid_points : initial_grid_points_) {
    // The Cartoon direction always has a single grid point.
    extents.emplace_back(
        std::array<size_t, 3>{grid_points[0], grid_points[1], 1});
  }
  return extents;
}

std::vector<std::array<size_t, 3>>
AngularCartoonSphere2D::initial_refinement_levels() const {
  std::vector<std::array<size_t, 3>> refinement_levels;
  refinement_levels.reserve(num_blocks_);
  for (const size_t refinement_in_r : initial_refinement_in_r_) {
    // Neither the azimuthal nor the Cartoon direction can be h-refined.
    refinement_levels.push_back({refinement_in_r, 0, 0});
  }
  return refinement_levels;
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
AngularCartoonSphere2D::functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  return time_dependence_->functions_of_time(initial_expiration_times);
}
}  // namespace domain::creators
