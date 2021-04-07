// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Sphere.hpp"

#include <cmath>
#include <memory>

#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Inertial;  // IWYU pragma: keep
struct Logical;   // IWYU pragma: keep
}  // namespace Frame

namespace domain::creators {
Sphere::Sphere(typename InnerRadius::type inner_radius,
               typename OuterRadius::type outer_radius,
               typename InitialRefinement::type initial_refinement,
               typename InitialGridPoints::type initial_number_of_grid_points,
               typename UseEquiangularMap::type use_equiangular_map,
               std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
                   boundary_condition,
               const Options::Context& context)
    // clang-tidy: trivially copyable
    : inner_radius_(std::move(inner_radius)),                // NOLINT
      outer_radius_(std::move(outer_radius)),                // NOLINT
      initial_refinement_(                                   // NOLINT
          std::move(initial_refinement)),                    // NOLINT
      initial_number_of_grid_points_(                        // NOLINT
          std::move(initial_number_of_grid_points)),         // NOLINT
      use_equiangular_map_(std::move(use_equiangular_map)),  // NOLINT
      boundary_condition_(std::move(boundary_condition)) {
  using domain::BoundaryConditions::is_none;
  if (is_none(boundary_condition_)) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow boundary condition, you must use that.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(boundary_condition_)) {
    PARSE_ERROR(context,
                "Cannot have periodic boundary conditions with a Sphere");
  }
}

Domain<3> Sphere::create_domain() const noexcept {
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3D =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>;
  std::vector<std::array<size_t, 8>> corners =
      corners_for_radially_layered_domains(1, true);

  std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Inertial, 3>>>
      coord_maps = sph_wedge_coordinate_maps<Frame::Inertial>(
          inner_radius_, outer_radius_, 0.0, 1.0, use_equiangular_map_);
  if (use_equiangular_map_) {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, Frame::Inertial>(Equiangular3D{
            Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                        inner_radius_ / sqrt(3.0)),
            Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                        inner_radius_ / sqrt(3.0)),
            Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                        inner_radius_ / sqrt(3.0))}));
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            Affine3D{Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                            inner_radius_ / sqrt(3.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                            inner_radius_ / sqrt(3.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                            inner_radius_ / sqrt(3.0))}));
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

  return Domain<3>(std::move(coord_maps), corners, {},
                   std::move(boundary_conditions_all_blocks));
}

std::vector<std::array<size_t, 3>> Sphere::initial_extents() const noexcept {
  std::vector<std::array<size_t, 3>> extents{
      6,
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1],
        initial_number_of_grid_points_[0]}}};
  extents.push_back(
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1],
        initial_number_of_grid_points_[1]}});
  return extents;
}

std::vector<std::array<size_t, 3>> Sphere::initial_refinement_levels() const
    noexcept {
  return {7, make_array<3>(initial_refinement_)};
}
}  // namespace domain::creators
