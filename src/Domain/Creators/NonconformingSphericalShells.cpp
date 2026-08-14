// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/NonconformingSphericalShells.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/BulgedCube.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/ExpandOverBlocks.hpp"
#include "Domain/Creators/ShellDistribution.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace Frame {
struct Inertial;
struct BlockLogical;
}  // namespace Frame

namespace domain::creators {

NonconformingSphericalShells::NonconformingSphericalShells(
    const double inner_radius, const double interface_radius,
    const double outer_radius, std::variant<Excision, InnerCube> interior,
    const typename InitialCubeRefinement::type& initial_cube_refinement,
    const typename InitialSHRefinement::type& initial_sh_refinement,
    const typename InitialCubeGridPoints::type& initial_cube_grid_points,
    const typename InitialSHGridPoints::type& initial_sh_grid_points,
    std::array<std::vector<double>, 2> radial_partitioning,
    std::array<std::vector<domain::CoordinateMaps::Distribution>, 2>
        radial_distribution,
    const bool use_equiangular_map,
    std::optional<TimeDepOptionType> time_dependent_options,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : inner_radius_(inner_radius),
      interface_radius_(interface_radius),
      outer_radius_(outer_radius),
      interior_(std::move(interior)),
      fill_interior_(std::holds_alternative<InnerCube>(interior_)),
      radial_partitioning_(std::move(radial_partitioning)),
      radial_distribution_{},
      use_equiangular_map_(use_equiangular_map),
      time_dependent_options_(std::move(time_dependent_options)),
      outer_boundary_condition_(std::move(outer_boundary_condition)),
      grid_anchors_{{{"Center", tnsr::I<double, 3, Frame::Grid>{
                                    std::array{0.0, 0.0, 0.0}}}}} {
  if (inner_radius_ > interface_radius_) {
    PARSE_ERROR(context,
                "Inner radius must be smaller than interface radius, but inner "
                "radius is " +
                    std::to_string(inner_radius_) +
                    " and interface radius is " +
                    std::to_string(interface_radius_) + ".");
  }
  if (interface_radius_ > outer_radius_) {
    PARSE_ERROR(
        context,
        "Interface radius must be smaller than outer radius, but interface "
        "radius is " +
            std::to_string(interface_radius_) + " and outer radius is " +
            std::to_string(outer_radius_) + ".");
  }

  // Validate boundary conditions
  using domain::BoundaryConditions::is_none;
  using domain::BoundaryConditions::is_periodic;
  if (not fill_interior_) {
    const auto& inner_bc = std::get<Excision>(interior_).boundary_condition;
    if (is_none(inner_bc)) {
      PARSE_ERROR(context,
                  "None boundary condition for the inner boundary is not "
                  "supported when the center is excised. If you would like an "
                  "outflow-type boundary condition, you must use that.");
    }
    if (is_periodic(inner_bc)) {
      PARSE_ERROR(context,
                  "Cannot have periodic boundary conditions with "
                  "NonconformingSphericalShells");
    }
    if ((inner_bc == nullptr) != (outer_boundary_condition_ == nullptr)) {
      PARSE_ERROR(
          context,
          "Must specify either both inner and outer boundary conditions "
          "or neither.");
    }
  }
  if (is_none(outer_boundary_condition_)) {
    PARSE_ERROR(context,
                "None boundary condition is not supported for the outer "
                "boundary. If you would like an outflow-type boundary "
                "condition, you must use that.");
  }
  if (is_periodic(outer_boundary_condition_)) {
    PARSE_ERROR(context,
                "Cannot have periodic boundary conditions with "
                "NonconformingSphericalShells");
  }

  // Validate inner radial partitions
  set_shell_distribution(make_not_null(&num_cube_shells_),
                         make_not_null(radial_distribution_.data()),
                         radial_partitioning_[0], radial_distribution[0],
                         inner_radius_, interface_radius_, "inner", "interface",
                         context);
  if (fill_interior_ and radial_distribution_[0].front() !=
                             domain::CoordinateMaps::Distribution::Linear) {
    PARSE_ERROR(context,
                "The 'RadialDistribution' must be 'Linear' for the innermost "
                "shell filled with a cube because it changes in sphericity. "
                "Add entries to 'RadialPartitioning' to add outer shells for "
                "which you can select different radial distributions.");
  }
  // Validate outer radial partitions
  set_shell_distribution(
      make_not_null(&num_sh_shells_), make_not_null(&radial_distribution_[1]),
      radial_partitioning_[1], radial_distribution[1], interface_radius_,
      outer_radius_, "interface", "outer", context);

  num_blocks_ =
      6 * num_cube_shells_ + num_sh_shells_ + (fill_interior_ ? 1 : 0);

  // Build block names and groups
  static const std::array<std::string, 6> wedge_directions{
      "UpperZ", "LowerZ", "UpperY", "LowerY", "UpperX", "LowerX"};

  std::vector<std::string> cube_block_names;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      cube_block_groups;
  if (fill_interior_) {
    block_names_.emplace_back("InnerCube");
    cube_block_names.emplace_back("InnerCube");
    block_groups_["InnerRegion"].insert("InnerCube");
    cube_block_groups["InnerRegion"].insert("InnerCube");
  }
  for (size_t shell = 0; shell < num_cube_shells_; ++shell) {
    const std::string shell_prefix = "InnerShell" + std::to_string(shell);
    for (const auto& dir : wedge_directions) {
      const std::string name = shell_prefix + dir;
      block_names_.emplace_back(name);
      cube_block_names.emplace_back(name);
      block_groups_[shell_prefix].insert(name);
      block_groups_["Wedges"].insert(name);
      block_groups_["InnerRegion"].insert(name);
      cube_block_groups[shell_prefix].insert(name);
      cube_block_groups["Wedges"].insert(name);
      cube_block_groups["InnerRegion"].insert(name);
    }
  }

  std::vector<std::string> sh_block_names;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      sh_block_groups;
  for (size_t sh = 0; sh < num_sh_shells_; ++sh) {
    const std::string name = "OuterShell" + std::to_string(sh);
    block_names_.emplace_back(name);
    sh_block_names.emplace_back(name);
    block_groups_["OuterShells"].insert(name);
    sh_block_groups["OuterShells"].insert(name);
  }

  ASSERT(block_names_.size() == num_blocks_,
         "Invalid number of block names. Should be "
             << num_blocks_ << " but is " << block_names_.size() << ".");

  // Expand initial refinement and grid points over cube blocks
  const ExpandOverBlocks<std::array<size_t, 2>> expand_cube{cube_block_names,
                                                            cube_block_groups};
  try {
    initial_cube_refinement_ = std::visit(expand_cube, initial_cube_refinement);
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialCubeRefinement': " << error.what());
  }
  try {
    initial_cube_grid_points_ =
        std::visit(expand_cube, initial_cube_grid_points);
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialCubeGridPoints': " << error.what());
  }

  if (fill_interior_) {
    if (initial_cube_refinement_.front()[0] !=
        initial_cube_refinement_.front()[1]) {
      PARSE_ERROR(
          context,
          "The inner cube has different refinement for angular and radial "
          "input. This block should only ever have the same values (it is not "
          "worth getting the creator to only take one, hence an error). Got ["
              << initial_cube_refinement_.front()[0] << ", "
              << initial_cube_refinement_.front()[1] << "].");
    }
    if (initial_cube_grid_points_.front()[0] !=
        initial_cube_grid_points_.front()[1]) {
      PARSE_ERROR(
          context,
          "The inner cube has a different number of grid points for angular "
          "and radial input. This block should only ever have the same values "
          "(it is not worth getting the creator to only take one, hence an "
          "error). Got ["
              << initial_cube_grid_points_.front()[0] << ", "
              << initial_cube_grid_points_.front()[1] << "].");
    }
  }

  // Expand initial refinement and grid points over SH blocks
  const ExpandOverBlocks<size_t> expand_sh_ref{sh_block_names, sh_block_groups};
  try {
    initial_sh_refinement_ = std::visit(expand_sh_ref, initial_sh_refinement);
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialSHRefinement': " << error.what());
  }

  const ExpandOverBlocks<std::array<size_t, 2>> expand_sh_gp{sh_block_names,
                                                             sh_block_groups};
  try {
    initial_sh_grid_points_ = std::visit(expand_sh_gp, initial_sh_grid_points);
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialSHGridPoints': " << error.what());
  }

  if (time_dependent_options_.has_value()) {
    use_hard_coded_maps_ =
        std::holds_alternative<sphere::TimeDependentMapOptions>(
            time_dependent_options_.value());
    if (use_hard_coded_maps_) {
      // Build combined radial partitioning for TimeDependentMapOptions:
      // inner partitions + interface_radius + outer partitions
      std::vector<double> combined_partitioning = radial_partitioning_[0];
      combined_partitioning.push_back(interface_radius_);
      for (const double r : radial_partitioning_[1]) {
        combined_partitioning.push_back(r);
      }
      std::get<sphere::TimeDependentMapOptions>(time_dependent_options_.value())
          .build_maps(std::array{0.0, 0.0, 0.0}, fill_interior_, inner_radius_,
                      combined_partitioning, outer_radius_);
    }
  }
}

Domain<3> NonconformingSphericalShells::create_domain() const {
  const size_t num_cube_blocks =
      6 * num_cube_shells_ + (fill_interior_ ? 1 : 0);

  // Build inner block coordinate maps (wedges)
  auto cube_coord_maps =
      make_vector_coordinate_map_base<Frame::BlockLogical, Frame::Inertial, 3>(
          sph_wedge_coordinate_maps(
              inner_radius_, interface_radius_,
              fill_interior_ ? std::get<InnerCube>(interior_).sphericity : 1.0,
              1.0, use_equiangular_map_, std::nullopt, false,
              radial_partitioning_[0], radial_distribution_[0]));

  // Build inner cube coordinate map and insert it before the wedges
  if (fill_interior_) {
    const double sphericity = std::get<InnerCube>(interior_).sphericity;
    std::unique_ptr<
        domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>
        cube_map;
    if (sphericity == 0.0) {
      if (use_equiangular_map_) {
        cube_map =
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                Equiangular3D{
                    Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                                inner_radius_ / sqrt(3.0)),
                    Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                                inner_radius_ / sqrt(3.0)),
                    Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                                inner_radius_ / sqrt(3.0))});
      } else {
        cube_map =
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                Affine3D{Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                                inner_radius_ / sqrt(3.0)),
                         Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                                inner_radius_ / sqrt(3.0)),
                         Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                                inner_radius_ / sqrt(3.0))});
      }
    } else {
      cube_map = make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          BulgedCube{inner_radius_, sphericity, use_equiangular_map_});
    }
    cube_coord_maps.insert(cube_coord_maps.begin(), std::move(cube_map));
  }

  // Build neighbor maps for inner blocks using corner numbering.
  // When the interior is filled, the cube block is placed first (index 0),
  // so rotate the corners vector so the central block entry comes first.
  auto corners =
      corners_for_radially_layered_domains(num_cube_shells_, fill_interior_);
  if (fill_interior_) {
    std::rotate(corners.begin(), corners.end() - 1, corners.end());
  }
  std::vector<DirectionMap<3, BlockNeighbors<3>>> inner_neighbors{};
  set_internal_boundaries<3>(make_not_null(&inner_neighbors), corners);

  // Non-conforming interface: xi of the first SH shell maps to zeta of the
  // wedges (no discrete rotation in the angular directions).
  const OrientationMap<3> shell_to_wedge{
      {{Direction<3>::upper_zeta(), Direction<3>::self(),
        Direction<3>::self()}}};
  // Wedge block indices are offset by 1 when the cube occupies block 0.
  const size_t cube_block_offset = fill_interior_ ? 1 : 0;
  for (size_t i = cube_block_offset + 6 * (num_cube_shells_ - 1);
       i < cube_block_offset + 6 * num_cube_shells_; ++i) {
    inner_neighbors[i].emplace(
        Direction<3>::upper_zeta(),
        BlockNeighbors<3>{{num_cube_blocks},
                          {{num_cube_blocks, shell_to_wedge.inverse_map()}},
                          false});
  }

  // Excision sphere for non-filled interior
  std::unordered_map<std::string, ExcisionSphere<3>> excision_spheres{};
  if (not fill_interior_) {
    excision_spheres.emplace(
        "ExcisionSphere",
        ExcisionSphere<3>{inner_radius_,
                          tnsr::I<double, 3, Frame::Grid>{0.0},
                          {{0, Direction<3>::lower_zeta()},
                           {1, Direction<3>::lower_zeta()},
                           {2, Direction<3>::lower_zeta()},
                           {3, Direction<3>::lower_zeta()},
                           {4, Direction<3>::lower_zeta()},
                           {5, Direction<3>::lower_zeta()}}});
  }

  std::vector<Block<3>> blocks;
  blocks.reserve(num_blocks_);

  // Inner cube blocks (wedges + optional cube)
  for (size_t i = 0; i < num_cube_blocks; ++i) {
    blocks.emplace_back(std::move(cube_coord_maps[i]), i,
                        std::move(inner_neighbors[i]), block_names_[i],
                        domain::topologies::hypercube<3>);
  }

  // SH shell blocks
  const auto aligned = OrientationMap<3>::create_aligned();
  for (size_t sh_i = 0; sh_i < num_sh_shells_; ++sh_i) {
    const size_t block_id = num_cube_blocks + sh_i;
    const double sh_inner =
        (sh_i == 0) ? interface_radius_ : radial_partitioning_[1][sh_i - 1];
    const double sh_outer = (sh_i == num_sh_shells_ - 1)
                                ? outer_radius_
                                : radial_partitioning_[1][sh_i];
    auto sh_map =
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            CoordinateMaps::ProductOf2Maps<CoordinateMaps::Interval,
                                           CoordinateMaps::Identity<2>>{
                CoordinateMaps::Interval{-1.0, 1.0, sh_inner, sh_outer,
                                         radial_distribution_[1][sh_i], 0.0},
                CoordinateMaps::Identity<2>{}},
            CoordinateMaps::SphericalToCartesianPfaffian{});

    DirectionMap<3, BlockNeighbors<3>> sh_neighbors;
    if (sh_i == 0) {
      // Non-conforming connection: lower_xi of the first SH shell abuts the
      // upper_zeta faces of the 6 outermost inner wedge blocks.
      const size_t w = cube_block_offset + 6 * (num_cube_shells_ - 1);
      sh_neighbors.emplace(
          Direction<3>::lower_xi(),
          BlockNeighbors<3>{{w + 0, w + 1, w + 2, w + 3, w + 4, w + 5},
                            {{w + 0, shell_to_wedge},
                             {w + 1, shell_to_wedge},
                             {w + 2, shell_to_wedge},
                             {w + 3, shell_to_wedge},
                             {w + 4, shell_to_wedge},
                             {w + 5, shell_to_wedge}},
                            false});
    } else {
      sh_neighbors.emplace(Direction<3>::lower_xi(),
                           BlockNeighbors<3>{block_id - 1, aligned});
    }
    if (sh_i < num_sh_shells_ - 1) {
      sh_neighbors.emplace(Direction<3>::upper_xi(),
                           BlockNeighbors<3>{block_id + 1, aligned});
    }
    blocks.emplace_back(std::move(sh_map), block_id, std::move(sh_neighbors),
                        block_names_[block_id],
                        domain::topologies::spherical_shell);
  }

  ASSERT(blocks.size() == num_blocks_, "Unexpected number of blocks. Expected "
                                           << num_blocks_ << " but created "
                                           << blocks.size() << ".");

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

      // Inner cube block (if present): block 0, shell = num_cube_shells_
      if (fill_interior_) {
        block_maps_grid_to_distorted[0] =
            hard_coded_options.grid_to_distorted_map(num_cube_shells_, 0, true);
        block_maps_distorted_to_inertial[0] =
            hard_coded_options.distorted_to_inertial_map(num_cube_shells_,
                                                         true);
        block_maps_grid_to_inertial[0] =
            hard_coded_options.grid_to_inertial_map(num_cube_shells_, 0, false,
                                                    true);
      }

      // Wedge blocks: block_id = cube_block_offset + shell*6 + shape_map_index
      for (size_t block_id = cube_block_offset;
           block_id < cube_block_offset + 6 * num_cube_shells_; ++block_id) {
        const size_t shell = (block_id - cube_block_offset) / 6;
        const size_t shape_map_index = (block_id - cube_block_offset) % 6;
        block_maps_grid_to_distorted[block_id] =
            hard_coded_options.grid_to_distorted_map(shell, shape_map_index,
                                                     false);
        block_maps_distorted_to_inertial[block_id] =
            hard_coded_options.distorted_to_inertial_map(shell, false);
        block_maps_grid_to_inertial[block_id] =
            hard_coded_options.grid_to_inertial_map(shell, shape_map_index,
                                                    false, false);
      }

      // SH shell blocks: shell = num_cube_shells_ + sh_i
      for (size_t sh_i = 0; sh_i < num_sh_shells_; ++sh_i) {
        const size_t block_id = num_cube_blocks + sh_i;
        const size_t shell = num_cube_shells_ + sh_i;
        const bool is_outer_shell = (sh_i == num_sh_shells_ - 1);
        block_maps_grid_to_distorted[block_id] =
            hard_coded_options.grid_to_distorted_map(shell, 0, false);
        block_maps_distorted_to_inertial[block_id] =
            hard_coded_options.distorted_to_inertial_map(shell, false);
        block_maps_grid_to_inertial[block_id] =
            hard_coded_options.grid_to_inertial_map(shell, 0, is_outer_shell,
                                                    false);
      }

      if (not fill_interior_) {
        domain.inject_time_dependent_map_for_excision_sphere(
            "ExcisionSphere",
            hard_coded_options.grid_to_inertial_map(0, 0, false, true));
      }
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
NonconformingSphericalShells::grid_anchors() const {
  return grid_anchors_;
}

std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
NonconformingSphericalShells::external_boundary_conditions() const {
  if (outer_boundary_condition_ == nullptr) {
    return {};
  }
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{num_blocks_};

  // Outer boundary: upper_xi of the last SH shell block
  boundary_conditions[num_blocks_ - 1][Direction<3>::upper_xi()] =
      outer_boundary_condition_->get_clone();

  // Inner boundary: lower_zeta of the innermost wedge blocks (excision case)
  if (not fill_interior_) {
    const auto& inner_bc = std::get<Excision>(interior_).boundary_condition;
    for (size_t i = 0; i < 6; ++i) {
      boundary_conditions[i][Direction<3>::lower_zeta()] =
          inner_bc->get_clone();
    }
  }

  return boundary_conditions;
}

std::vector<std::array<size_t, 3>>
NonconformingSphericalShells::initial_extents() const {
  std::vector<std::array<size_t, 3>> extents;
  extents.reserve(num_blocks_);

  // Cube blocks: use stored cube grid points directly
  for (const auto& gp : initial_cube_grid_points_) {
    extents.emplace_back(std::array{gp[0], gp[0], gp[1]});
  }

  // SH blocks: stored as [l_max, r], convert to {r, l_max+1, 2*l_max+1}
  for (const auto& sh_gp : initial_sh_grid_points_) {
    const size_t l_max = sh_gp[0];
    const size_t n_r = sh_gp[1];
    extents.push_back({n_r, l_max + 1, 2 * l_max + 1});
  }

  return extents;
}

std::vector<std::array<size_t, 3>>
NonconformingSphericalShells::initial_refinement_levels() const {
  std::vector<std::array<size_t, 3>> refinement;
  refinement.reserve(num_blocks_);

  // Cube blocks: use stored cube refinement directly
  for (const auto& ref : initial_cube_refinement_) {
    refinement.emplace_back(std::array{ref[0], ref[0], ref[1]});
  }

  // SH blocks: radial refinement only (angular directions are not AMR-refined)
  for (const size_t sh_ref : initial_sh_refinement_) {
    refinement.push_back({sh_ref, 0_st, 0_st});
  }

  return refinement;
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
NonconformingSphericalShells::functions_of_time(
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
