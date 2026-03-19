// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/AngularDisk.hpp"

#include <memory>
#include <utility>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/PolarToCartesian.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
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
AngularDisk::AngularDisk(
    typename OuterRadius::type outer_radius,
    typename RadialPartitioning::type radial_partitioning,
    typename InitialDiskThetaGridPoints::type initial_disk_grid_points,
    typename InitialAnnulusGridPoints::type initial_annulus_grid_points,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<2>>
        time_dependence,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        boundary_condition,
    const Options::Context& context)
    : outer_radius_(std::move(outer_radius)),
      radial_partitioning_(std::move(radial_partitioning)),
      initial_disk_grid_points_(std::move(initial_disk_grid_points)),
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
  if (boundary_condition_ != nullptr and is_periodic(boundary_condition_)) {
    PARSE_ERROR(context, "Cannot have periodic boundary conditions on a disk.");
  }

  if (outer_radius_ <= 0) {
    PARSE_ERROR(context,
                "Must have a positive outer radius, got " << outer_radius);
  }
  if (not radial_partitioning_.empty()) {
    if (not std::is_sorted(radial_partitioning_.begin(),
                           radial_partitioning_.end())) {
      PARSE_ERROR(context, "Specify radial partitioning in ascending order");
    }
    if (radial_partitioning_.front() <= 0) {
      PARSE_ERROR(context, "Radial partitions must be positive, got "
                               << radial_partitioning_.front());
    }
    if (radial_partitioning_.back() >= outer_radius_) {
      PARSE_ERROR(context,
                  "Last radial partition must be smaller than the outer "
                  "radius, but is "
                      << radial_partitioning_.back() << " with outer radius "
                      << outer_radius_);
    }
    const auto duplicate = std::adjacent_find(radial_partitioning_.begin(),
                                              radial_partitioning_.end());
    if (duplicate != radial_partitioning_.end()) {
      PARSE_ERROR(context, "Radial partitioning contains duplicate element: "
                               << *duplicate);
    }
  }
  if (initial_disk_grid_points_ % 2 != 1) {
    PARSE_ERROR(context,
                "The number of angular grid points must be odd (this helps "
                "with numerical stability), for inner disk got "
                    << initial_disk_grid_points);
  }
  num_blocks_ = 1 + radial_partitioning_.size();

  if (std::holds_alternative<std::array<size_t, 2>>(
          initial_annulus_grid_points)) {
    const auto input_value =
        std::get<std::array<size_t, 2>>(initial_annulus_grid_points);
    if (input_value[1] % 2 != 1) {
      PARSE_ERROR(context,
                  "The number of angular grid points must be odd (this helps "
                  "with numerical stability), for shell(s) got "
                      << input_value[1]);
    }
    initial_annulus_grid_points_ =
        std::vector<std::array<size_t, 2>>(num_blocks_ - 1, input_value);
  } else {
    initial_annulus_grid_points_ = std::get<std::vector<std::array<size_t, 2>>>(
        initial_annulus_grid_points);
    if (initial_annulus_grid_points_.size() != num_blocks_ - 1) {
      PARSE_ERROR(context,
                  "InitialAnnulusGridPoints must be one larger than "
                  "RadialPartitioning (size="
                      << radial_partitioning_.size() << "), but has size "
                      << initial_annulus_grid_points_.size() << ".");
    }
    for (size_t i = 0; i < initial_annulus_grid_points_.size(); ++i) {
      if (initial_annulus_grid_points_[i][1] % 2 != 1) {
        PARSE_ERROR(context,
                    "The number of angular grid points must be odd (this helps "
                    "with numerical stability), for shell "
                        << i << " got " << initial_annulus_grid_points_[i][1]);
      }
    }
  }

  block_names_.reserve(num_blocks_);
  block_names_.emplace_back("InnerDisk");
  block_groups_["InnerDisk"];
  block_groups_["InnerDisk"].insert("InnerDisk");
  if (num_blocks_ > 1) {
    block_groups_["Shells"];
    for (size_t i = 1; i < num_blocks_; ++i) {
      const std::string shell = "Shell" + std::to_string(i - 1);
      block_names_.emplace_back(shell);
      block_groups_["Shells"].insert(shell);
    }
  }
}

Domain<2> AngularDisk::create_domain() const {
  using Identity = CoordinateMaps::Identity<1>;
  using Affine = CoordinateMaps::Affine;
  const auto aligned = OrientationMap<2>::create_aligned();

  std::vector<Block<2>> blocks;
  blocks.reserve(num_blocks_);

  for (size_t i = 0; i < num_blocks_; ++i) {
    const double inner_radius = i == 0 ? 0.0 : radial_partitioning_[i - 1];
    const double outer_radius =
        i == num_blocks_ - 1 ? outer_radius_ : radial_partitioning_[i];
    auto coord_map =
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            CoordinateMaps::ProductOf2Maps<Affine, Identity>{
                Affine{-1.0, 1.0, inner_radius, outer_radius}, Identity{}},
            CoordinateMaps::PolarToCartesian{});
    DirectionMap<2, BlockNeighbors<2>> neighbors{};
    if (num_blocks_ > 1) {
      if (i == 0) {
        neighbors.emplace(std::pair(Direction<2>::upper_xi(),
                                    BlockNeighbors<2>(i + 1, aligned)));
      } else {
        neighbors.emplace(std::pair(Direction<2>::lower_xi(),
                                    BlockNeighbors<2>(i - 1, aligned)));
        if (i != num_blocks_ - 1) {
          neighbors.emplace(std::pair(Direction<2>::upper_xi(),
                                      BlockNeighbors<2>(i + 1, aligned)));
        }
      }
    }

    blocks.emplace_back(
        std::move(coord_map), i, std::move(neighbors), block_names_.at(i),
        i == 0 ? domain::topologies::disk : domain::topologies::annulus);
  }

  Domain<2> domain(std::move(blocks), {}, block_groups_);

  if (not time_dependence_->is_none()) {
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 2>>>
        block_maps_grid_to_inertial =
            time_dependence_->block_maps_grid_to_inertial(num_blocks_);
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, 2>>>
        block_maps_grid_to_distorted =
            time_dependence_->block_maps_grid_to_distorted(num_blocks_);
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, 2>>>
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
    2, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
AngularDisk::external_boundary_conditions() const {
  if (boundary_condition_ == nullptr) {
    return {};
  }
  std::vector<DirectionMap<
      2, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{num_blocks_};
  boundary_conditions[num_blocks_ - 1][Direction<2>::upper_xi()] =
      boundary_condition_->get_clone();
  return boundary_conditions;
}

std::vector<std::array<size_t, 2>> AngularDisk::initial_extents() const {
  // Throughout the code, we require n_phi be odd for numerical stability
  // We also require the angular modal space of the angular dimension to be <=
  // the angular modal space of the radial dimension. Here we set them to be
  // equal
  // \Phi angular modal max M = N_\Phi / 2 (integer division)
  // r angular modal max M = 2 * N_r - 2
  const size_t disk_M = initial_disk_grid_points_ / 2;
  const size_t disk_n_r = disk_M / 2 + 1 + disk_M % 2;

  std::vector<std::array<size_t, 2>> extents;
  extents.reserve(num_blocks_);
  extents.emplace_back(
      std::array<size_t, 2>{disk_n_r, initial_disk_grid_points_});
  for (const auto& elem : initial_annulus_grid_points_) {
    extents.emplace_back(elem);
  }
  return extents;
}

std::vector<std::array<size_t, 2>> AngularDisk::initial_refinement_levels()
    const {
  // ZernikeB2 and Fourier should never be refined. The shell I1 refinement is
  // not implemented
  return {num_blocks_, {0, 0}};
}
}  // namespace domain::creators
