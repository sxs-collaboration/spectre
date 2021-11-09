// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Shell.hpp"

#include <algorithm>
#include <memory>
#include <utility>

#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/Structure/ExcisionSphere.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Inertial;
struct BlockLogical;
}  // namespace Frame

namespace domain::creators {
Shell::Shell(
    double inner_radius, double outer_radius, size_t initial_refinement,
    std::array<size_t, 2> initial_number_of_grid_points,
    bool use_equiangular_map,
    std::optional<domain::creators::Shell::EquatorialCompressionOptions>
        equatorial_compression,
    std::vector<double> radial_partitioning,
    std::vector<domain::CoordinateMaps::Distribution> radial_distribution,
    ShellWedges which_wedges,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        inner_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : inner_radius_(inner_radius),
      outer_radius_(outer_radius),
      initial_refinement_(initial_refinement),
      initial_number_of_grid_points_(initial_number_of_grid_points),
      use_equiangular_map_(use_equiangular_map),
      radial_partitioning_(std::move(radial_partitioning)),
      radial_distribution_(std::move(radial_distribution)),
      which_wedges_(which_wedges),
      time_dependence_(std::move(time_dependence)),
      inner_boundary_condition_(std::move(inner_boundary_condition)),
      outer_boundary_condition_(std::move(outer_boundary_condition)) {
  number_of_layers_ = radial_partitioning_.size() + 1;
  blocks_per_layer_ =
      which_wedges_ == ShellWedges::All             ?  6
    : which_wedges_ == ShellWedges::AllAndHalvesXY  ? 10
    : which_wedges_ == ShellWedges::AllAndHalvesXZ  ? 10
    : which_wedges_ == ShellWedges::AllAndHalvesYZ  ? 10
    : which_wedges_ == ShellWedges::FourOnEquatorXY ?  4
    : which_wedges_ == ShellWedges::FourOnEquatorXZ ?  4
    : which_wedges_ == ShellWedges::FourOnEquatorYZ ?  4
    : 1;

  if (equatorial_compression.has_value()) {
    aspect_ratio_ = equatorial_compression.value().aspect_ratio;
    index_polar_axis_ = equatorial_compression.value().index_polar_axis;
  } else {
    aspect_ratio_ = 1.0;
    index_polar_axis_ = 2;
  }
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<3>>();
  }
  if ((inner_boundary_condition_ != nullptr and
       outer_boundary_condition_ == nullptr) or
      (inner_boundary_condition_ == nullptr and
       outer_boundary_condition_ != nullptr)) {
    PARSE_ERROR(context,
                "Must specify either both inner and outer boundary conditions "
                "or neither.");
  }
  if (inner_boundary_condition_ != nullptr and
      which_wedges_ != ShellWedges::All) {
    PARSE_ERROR(context,
                "Can only apply boundary conditions when using the full shell. "
                "Additional cases can be supported by adding them to the Shell "
                "domain creator's create_domain function.");
  }
  using domain::BoundaryConditions::is_none;
  if (is_none(inner_boundary_condition_) or
      is_none(outer_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow boundary condition, you must use that.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(inner_boundary_condition_) or
      is_periodic(outer_boundary_condition_)) {
    PARSE_ERROR(context,
                "Cannot have periodic boundary conditions with a shell");
  }
  if (not radial_partitioning_.empty()) {
    if (not std::is_sorted(radial_partitioning_.begin(),
                           radial_partitioning_.end())) {
      PARSE_ERROR(context,
                  "Specify radial partitioning in ascending order. Specified "
                  "radial partitioning is: "
                      << get_output(radial_partitioning_));
    }
    if (radial_partitioning_.front() <= inner_radius_) {
      PARSE_ERROR(
          context,
          "First radial partition must be larger than inner radius, but is: "
              << inner_radius_);
    }
    if (radial_partitioning_.back() >= outer_radius_) {
      PARSE_ERROR(
          context,
          "Last radial partition must be smaller than outer radius, but is: "
              << outer_radius_);
    }
    const auto duplicate = std::adjacent_find(radial_partitioning_.begin(),
                                              radial_partitioning_.end());
    if (duplicate != radial_partitioning_.end()) {
      PARSE_ERROR(context, "Radial partitioning contains duplicate element: "
                               << *duplicate);
    }
  }
  if (radial_distribution_.size() != number_of_layers_) {
    PARSE_ERROR(context,
                "Specify a 'RadialDistribution' for every spherical shell. You "
                "specified "
                    << radial_distribution_.size()
                    << " items, but the domain has " << number_of_layers_
                    << " shells.");
  }
}

Domain<3> Shell::create_domain() const {
  std::vector<std::unique_ptr<
      CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>
      coord_maps = sph_wedge_coordinate_maps<Frame::Inertial>(
          inner_radius_, outer_radius_, 1.0, 1.0, use_equiangular_map_, 0.0,
          aspect_ratio_, index_polar_axis_, radial_partitioning_,
          radial_distribution_, which_wedges_);

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions_all_blocks{};

  if (inner_boundary_condition_ != nullptr) {
    // This assumes 6 wedges making up the shell. If you need to support the
    // configurations other than the full Shell with six full wedges,
    // the below code needs to be updated. This would require adding more
    // boundary condition options to the domain creator.
    boundary_conditions_all_blocks.resize(blocks_per_layer_ *
                                          number_of_layers_);
    for (size_t block_id = 0; block_id < blocks_per_layer_; ++block_id) {
      boundary_conditions_all_blocks[block_id][Direction<3>::lower_zeta()] =
          inner_boundary_condition_->get_clone();
      boundary_conditions_all_blocks[boundary_conditions_all_blocks.size() -
                                     block_id - 1][Direction<3>::upper_zeta()] =
          outer_boundary_condition_->get_clone();
    }
  }

  Domain<3> domain{
      std::move(coord_maps),
      corners_for_radially_layered_domains(
          number_of_layers_, false, {{1, 2, 3, 4, 5, 6, 7, 8}}, which_wedges_),
      {},
      std::move(boundary_conditions_all_blocks),
      {{"CentralExcisionSphere",
        ExcisionSphere<3>{inner_radius_, {{0.0, 0.0, 0.0}}}}}};

  if (not time_dependence_->is_none()) {
    const size_t number_of_blocks = domain.blocks().size();
    auto block_maps = time_dependence_->block_maps(number_of_blocks);
    for (size_t block_id = 0; block_id < number_of_blocks; ++block_id) {
      domain.inject_time_dependent_map_for_block(
          block_id, std::move(block_maps[block_id]));
    }
  }
  return domain;
}

std::vector<std::array<size_t, 3>> Shell::initial_extents() const {
  return {
      blocks_per_layer_ * number_of_layers_,
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1],
        initial_number_of_grid_points_[0]}}};
}

std::vector<std::array<size_t, 3>> Shell::initial_refinement_levels() const {
  return {blocks_per_layer_ * number_of_layers_,
          make_array<3>(initial_refinement_)};
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
Shell::functions_of_time() const {
  if (time_dependence_->is_none()) {
    return {};
  } else {
    return time_dependence_->functions_of_time();
  }
}
}  // namespace domain::creators
