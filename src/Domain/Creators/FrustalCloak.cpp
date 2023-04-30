// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/FrustalCloak.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Inertial;
struct BlockLogical;
}  // namespace Frame
namespace domain {
template <typename, typename, size_t>
class CoordinateMapBase;
}  // namespace domain

namespace domain::creators {
FrustalCloak::FrustalCloak(
    typename InitialRefinement::type initial_refinement_level,
    typename InitialGridPoints::type initial_number_of_grid_points,
    typename UseEquiangularMap::type use_equiangular_map,
    typename ProjectionFactor::type projection_factor,
    typename LengthInnerCube::type length_inner_cube,
    typename LengthOuterCube::type length_outer_cube,
    typename OriginPreimage::type origin_preimage,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        boundary_condition,
    const Options::Context& context)
    // clang-tidy: trivially copyable
    : initial_refinement_level_(                      // NOLINT
          std::move(initial_refinement_level)),       // NOLINT
      initial_number_of_grid_points_(                 // NOLINT
          std::move(initial_number_of_grid_points)),  // NOLINT
      use_equiangular_map_(use_equiangular_map),      // NOLINT
      projection_factor_(projection_factor),          // NOLINT
      length_inner_cube_(length_inner_cube),          // NOLINT
      length_outer_cube_(length_outer_cube),          // NOLINT
      origin_preimage_(origin_preimage),              // NOLINT
      boundary_condition_(std::move(boundary_condition)) {
  using domain::BoundaryConditions::is_none;
  if (is_none(boundary_condition_)) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow-type boundary condition, you must use that.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (boundary_condition_ != nullptr and is_periodic(boundary_condition_)) {
    PARSE_ERROR(
        context,
        "Cannot have periodic boundary conditions with a frustal cloak");
  }
}

Domain<3> FrustalCloak::create_domain() const {
  std::vector<std::unique_ptr<
      CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>
      coord_maps = domain::make_vector_coordinate_map_base<
          Frame::BlockLogical, Frame::Inertial, 3>(frustum_coordinate_maps(
          length_inner_cube_, length_outer_cube_, use_equiangular_map_,
          origin_preimage_, domain::CoordinateMaps::Distribution::Projective,
          projection_factor_));
  return Domain<3>{std::move(coord_maps),
                   corners_for_biradially_layered_domains(
                       0, 1, false, false, {{1, 2, 3, 4, 5, 6, 7, 8}})};
}

std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
FrustalCloak::external_boundary_conditions() const {
  if (boundary_condition_ == nullptr) {
    return {};
  }
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{10};
  for (size_t i = 0; i < 10; ++i) {
    boundary_conditions[i][Direction<3>::lower_zeta()] =
        boundary_condition_->get_clone();
    boundary_conditions[i][Direction<3>::upper_zeta()] =
        boundary_condition_->get_clone();
  }
  return boundary_conditions;
}

std::vector<std::array<size_t, 3>> FrustalCloak::initial_extents() const {
  return {
      10,
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1],
        initial_number_of_grid_points_[0]}}};
}

std::vector<std::array<size_t, 3>> FrustalCloak::initial_refinement_levels()
    const {
  return {10, make_array<3>(initial_refinement_level_)};
}
}  // namespace domain::creators
