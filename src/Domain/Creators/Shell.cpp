// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Shell.hpp"

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
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Inertial;
struct Logical;
}  // namespace Frame

namespace domain::creators {
Shell::Shell(
    typename InnerRadius::type inner_radius,
    typename OuterRadius::type outer_radius,
    typename InitialRefinement::type initial_refinement,
    typename InitialGridPoints::type initial_number_of_grid_points,
    typename UseEquiangularMap::type use_equiangular_map,
    typename AspectRatio::type aspect_ratio,
    typename UseLogarithmicMap::type use_logarithmic_map,
    typename WhichWedges::type which_wedges,
    typename RadialBlockLayers::type number_of_layers,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        inner_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    // clang-tidy: trivially copyable
    : inner_radius_(std::move(inner_radius)),                // NOLINT
      outer_radius_(std::move(outer_radius)),                // NOLINT
      initial_refinement_(                                   // NOLINT
          std::move(initial_refinement)),                    // NOLINT
      initial_number_of_grid_points_(                        // NOLINT
          std::move(initial_number_of_grid_points)),         // NOLINT
      use_equiangular_map_(std::move(use_equiangular_map)),  // NOLINT
      aspect_ratio_(std::move(aspect_ratio)),                // NOLINT
      use_logarithmic_map_(std::move(use_logarithmic_map)),  // NOLINT
      which_wedges_(std::move(which_wedges)),                // NOLINT
      number_of_layers_(std::move(number_of_layers)),        // NOLINT
      time_dependence_(std::move(time_dependence)),
      inner_boundary_condition_(std::move(inner_boundary_condition)),
      outer_boundary_condition_(std::move(outer_boundary_condition)) {
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
}

Domain<3> Shell::create_domain() const noexcept {
  std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Inertial, 3>>>
      coord_maps = sph_wedge_coordinate_maps<Frame::Inertial>(
          inner_radius_, outer_radius_, 1.0, 1.0, use_equiangular_map_, 0.0,
          false, aspect_ratio_, use_logarithmic_map_, which_wedges_,
          number_of_layers_);

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions_all_blocks{};

  if (inner_boundary_condition_ != nullptr) {
    // This assumes 6 wedges making up the shell. If you need to support the
    // FourOnEquator or OneAlongMinusX configurations the below code needs to be
    // updated. This would require adding more boundary condition options to the
    // domain creator.
    const size_t blocks_per_layer =
        which_wedges_ == ShellWedges::All                 ? 6
            : which_wedges_ == ShellWedges::FourOnEquator ? 4
                                                          : 1;

    boundary_conditions_all_blocks.resize(blocks_per_layer * number_of_layers_);
    for (size_t block_id = 0; block_id < blocks_per_layer; ++block_id) {
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
      std::move(boundary_conditions_all_blocks)};

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

std::vector<std::array<size_t, 3>> Shell::initial_extents() const noexcept {
  std::vector<std::array<size_t, 3>>::size_type num_wedges =
      6 * number_of_layers_;
  if (UNLIKELY(which_wedges_ == ShellWedges::FourOnEquator)) {
    num_wedges = 4 * number_of_layers_;
  } else if (UNLIKELY(which_wedges_ == ShellWedges::OneAlongMinusX)) {
    num_wedges = number_of_layers_;
  }
  return {
      num_wedges,
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1],
        initial_number_of_grid_points_[0]}}};
}

std::vector<std::array<size_t, 3>> Shell::initial_refinement_levels() const
    noexcept {
  std::vector<std::array<size_t, 3>>::size_type num_wedges =
      6 * number_of_layers_;
  if (UNLIKELY(which_wedges_ == ShellWedges::FourOnEquator)) {
    num_wedges = 4 * number_of_layers_;
  } else if (UNLIKELY(which_wedges_ == ShellWedges::OneAlongMinusX)) {
    num_wedges = number_of_layers_;
  }
  return {num_wedges, make_array<3>(initial_refinement_)};
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
Shell::functions_of_time() const noexcept {
  if (time_dependence_->is_none()) {
    return {};
  } else {
    return time_dependence_->functions_of_time();
  }
}
}  // namespace domain::creators
