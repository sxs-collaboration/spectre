// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/NonconformingSphericalShells.hpp"

#include <array>
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
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/Gsl.hpp"

namespace domain::creators {

NonconformingSphericalShells::NonconformingSphericalShells(
    const double inner_radius, const double interface_radius,
    const double outer_radius, const size_t initial_radial_refinement,
    const size_t initial_angular_refinement,
    const size_t initial_number_of_radial_grid_points,
    const size_t initial_spherical_harmonic_l,
    const size_t initial_number_of_angular_grid_points_of_wedges,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        inner_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : inner_radius_(inner_radius),
      interface_radius_(interface_radius),
      outer_radius_(outer_radius),
      initial_radial_refinement_(initial_radial_refinement),
      initial_angular_refinement_(initial_angular_refinement),
      initial_number_of_radial_grid_points_(
          initial_number_of_radial_grid_points),
      initial_spherical_harmonic_l_(initial_spherical_harmonic_l),
      initial_number_of_angular_grid_points_of_wedges_(
          initial_number_of_angular_grid_points_of_wedges),
      initial_refinement_levels_{
          7,
          {{initial_angular_refinement_, initial_angular_refinement_,
            initial_radial_refinement_}}},
      initial_number_of_grid_points_{
          7,
          {{initial_number_of_angular_grid_points_of_wedges_,
            initial_number_of_angular_grid_points_of_wedges_,
            initial_number_of_radial_grid_points_}}},
      inner_boundary_condition_(std::move(inner_boundary_condition)),
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
    PARSE_ERROR(context,
                "Cannot have periodic boundary conditions with "
                "NonconformingSphericalShells");
  }
  // Validate consistency of inner and outer boundary condition
  if ((inner_boundary_condition_ == nullptr) !=
      (outer_boundary_condition_ == nullptr)) {
    PARSE_ERROR(context,
                "Must specify either both inner and outer boundary conditions "
                "or neither.");
  }

  // correct for outer spherical shell
  initial_number_of_grid_points_[6] = {{initial_number_of_radial_grid_points_,
                                        initial_spherical_harmonic_l_ + 1,
                                        2 * initial_spherical_harmonic_l_ + 1}};
  initial_refinement_levels_[6] = {{initial_radial_refinement_, 0_st, 0_st}};
  if (context != Options::Context{}) {
    // Run create_domain for non-default contexts to validate the constructed
    // domain.
    (void)create_domain(context);
  }
}

Domain<3> NonconformingSphericalShells::create_domain(
    const Options::Context& /*context*/) const {
  std::vector<Block<3>> blocks;
  blocks.reserve(7);
  const std::vector<std::array<size_t, 8>> corners_of_wedges =
      corners_for_radially_layered_domains(1, false);
  std::vector<DirectionMap<3, BlockNeighbors<3>>> neighbors_of_wedges{};
  set_internal_boundaries<3>(make_not_null(&neighbors_of_wedges),
                             corners_of_wedges);
  const OrientationMap<3> shell_to_wedge{
      {{Direction<3>::upper_zeta(), Direction<3>::self(),
        Direction<3>::self()}}};
  DirectionMap<3, BlockNeighbors<3>> neighbors_of_shell{};
  neighbors_of_shell.emplace(std::pair{Direction<3>::lower_xi(),
                                       BlockNeighbors<3>{{0, 1, 2, 3, 4, 5},
                                                         {{0, shell_to_wedge},
                                                          {1, shell_to_wedge},
                                                          {2, shell_to_wedge},
                                                          {3, shell_to_wedge},
                                                          {4, shell_to_wedge},
                                                          {5, shell_to_wedge}},
                                                         false}});
  for (size_t i = 0; i < 6; ++i) {
    neighbors_of_wedges[i].emplace(std::pair{
        Direction<3>::upper_zeta(),
        BlockNeighbors<3>{{6}, {{6, shell_to_wedge.inverse_map()}}, false}});
  }

  auto wedge_coord_maps =
      make_vector_coordinate_map_base<Frame::BlockLogical, Frame::Inertial, 3>(
          sph_wedge_coordinate_maps(inner_radius_, interface_radius_, 1.0, 1.0,
                                    true));
  auto sphere_map =
      make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Identity<2>>{
              CoordinateMaps::Affine{-1.0, 1.0, interface_radius_,
                                     outer_radius_},
              CoordinateMaps::Identity<2>{}},
          CoordinateMaps::SphericalToCartesianPfaffian{});
  for (size_t i = 0; i < 6; ++i) {
    blocks.emplace_back(
        std::move(wedge_coord_maps[i]), i, std::move(neighbors_of_wedges[i]),
        "Wedge" + std::to_string(i), domain::topologies::hypercube<3>);
  }
  blocks.emplace_back(std::move(sphere_map), 6_st,
                      std::move(neighbors_of_shell), "Shell",
                      domain::topologies::spherical_shell);
  return Domain(std::move(blocks));
}

Domain<3> NonconformingSphericalShells::create_domain() const {
  return create_domain(Options::Context{});
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
      boundary_conditions{7};
  for (size_t i = 0; i < 6; ++i) {
    boundary_conditions[i][Direction<3>::lower_zeta()] =
        inner_boundary_condition_->get_clone();
  }
  boundary_conditions[6][Direction<3>::upper_xi()] =
      outer_boundary_condition_->get_clone();
  return boundary_conditions;
}

std::vector<std::string> NonconformingSphericalShells::block_names() const {
  std::vector<std::string> names{};
  names.reserve(7);
  for (size_t i = 0; i < 6; ++i) {
    names.emplace_back("Wedge" + std::to_string(i));
  }
  names.emplace_back("Shell");
  return names;
}

std::vector<std::array<size_t, 3>>
NonconformingSphericalShells::initial_extents() const {
  return initial_number_of_grid_points_;
}

std::vector<std::array<size_t, 3>>
NonconformingSphericalShells::initial_refinement_levels() const {
  return initial_refinement_levels_;
}
}  // namespace domain::creators
