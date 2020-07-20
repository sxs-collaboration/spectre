// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/FrustalCloak.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include "Domain/Block.hpp"                   // IWYU pragma: keep
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Utilities/MakeArray.hpp"

/// \cond
namespace Frame {
struct Inertial;
struct Logical;
}  // namespace Frame
namespace domain {
template <typename, typename, size_t>
class CoordinateMapBase;
}  // namespace domain
/// \endcond

namespace domain::creators {
FrustalCloak::FrustalCloak(
    typename InitialRefinement::type initial_refinement_level,
    typename InitialGridPoints::type initial_number_of_grid_points,
    typename UseEquiangularMap::type use_equiangular_map,
    typename ProjectionFactor::type projection_factor,
    typename LengthInnerCube::type length_inner_cube,
    typename LengthOuterCube::type length_outer_cube,
    typename OriginPreimage::type origin_preimage,
    const OptionContext& /*context*/) noexcept
    // clang-tidy: trivially copyable
    : initial_refinement_level_(                      // NOLINT
          std::move(initial_refinement_level)),       // NOLINT
      initial_number_of_grid_points_(                 // NOLINT
          std::move(initial_number_of_grid_points)),  // NOLINT
      use_equiangular_map_(use_equiangular_map),      // NOLINT
      projection_factor_(projection_factor),          // NOLINT
      length_inner_cube_(length_inner_cube),          // NOLINT
      length_outer_cube_(length_outer_cube),          // NOLINT
      origin_preimage_(origin_preimage) {}            // NOLINT

Domain<3> FrustalCloak::create_domain() const noexcept {
  std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Inertial, 3>>>
      coord_maps = frustum_coordinate_maps<Frame::Inertial>(
          length_inner_cube_, length_outer_cube_, use_equiangular_map_,
          origin_preimage_, projection_factor_);
  return Domain<3>{std::move(coord_maps),
                   corners_for_biradially_layered_domains(
                       0, 1, false, false, {{1, 2, 3, 4, 5, 6, 7, 8}})};
}

std::vector<std::array<size_t, 3>> FrustalCloak::initial_extents() const
    noexcept {
  return {
      10,
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1],
        initial_number_of_grid_points_[0]}}};
}

std::vector<std::array<size_t, 3>> FrustalCloak::initial_refinement_levels()
    const noexcept {
  return {10, make_array<3>(initial_refinement_level_)};
}
}  // namespace domain::creators
