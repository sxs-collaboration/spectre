// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/DomainCreators/Shell.hpp"

#include <memory>
#include <utility>

#include "Domain/Block.hpp"                         // IWYU pragma: keep
#include "Domain/BlockNeighbor.hpp"                 // IWYU pragma: keep
#include "Domain/CoordinateMaps/CoordinateMap.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/DomainHelpers.hpp"
#include "Utilities/MakeArray.hpp"

/// \cond
namespace Frame {
struct Grid;
struct Inertial;
struct Logical;
}  // namespace Frame
/// \endcond

namespace DomainCreators {

template <typename TargetFrame>
Shell<TargetFrame>::Shell(
    typename InnerRadius::type inner_radius,
    typename OuterRadius::type outer_radius,
    typename InitialRefinement::type initial_refinement,
    typename InitialGridPoints::type initial_number_of_grid_points,
    typename UseEquiangularMap::type use_equiangular_map) noexcept
    // clang-tidy: trivially copyable
    : inner_radius_(std::move(inner_radius)),                  // NOLINT
      outer_radius_(std::move(outer_radius)),                  // NOLINT
      initial_refinement_(                                     // NOLINT
          std::move(initial_refinement)),                      // NOLINT
      initial_number_of_grid_points_(                          // NOLINT
          std::move(initial_number_of_grid_points)),           // NOLINT
      use_equiangular_map_(std::move(use_equiangular_map)) {}  // NOLINT

template <typename TargetFrame>
Domain<3, TargetFrame> Shell<TargetFrame>::create_domain() const noexcept {
  std::vector<std::array<size_t, 8>> corners{
      {{7, 5, 8, 6, 15, 13, 16, 14}},   // Upper z
      {{4, 2, 3, 1, 12, 10, 11, 9}},    // Lower z
      {{8, 6, 4, 2, 16, 14, 12, 10}},   // Upper y
      {{3, 1, 7, 5, 11, 9, 15, 13}},    // Lower y
      {{1, 2, 5, 6, 9, 10, 13, 14}},    // Upper x
      {{4, 3, 8, 7, 12, 11, 16, 15}}};  // Lower x
  std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, TargetFrame, 3>>>
      coord_maps = wedge_coordinate_maps<TargetFrame>(
          inner_radius_, outer_radius_, 1.0, 1.0, use_equiangular_map_);
  return Domain<3, TargetFrame>{std::move(coord_maps), corners};
}

template <typename TargetFrame>
std::vector<std::array<size_t, 3>> Shell<TargetFrame>::initial_extents() const
    noexcept {
  return {
      6,
      {{initial_number_of_grid_points_[1], initial_number_of_grid_points_[1],
        initial_number_of_grid_points_[0]}}};
}
template <typename TargetFrame>
std::vector<std::array<size_t, 3>>
Shell<TargetFrame>::initial_refinement_levels() const noexcept {
  return {6, make_array<3>(initial_refinement_)};
}
}  // namespace DomainCreators

template class DomainCreators::Shell<Frame::Grid>;
template class DomainCreators::Shell<Frame::Inertial>;
