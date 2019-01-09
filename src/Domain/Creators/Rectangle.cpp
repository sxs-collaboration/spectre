// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Rectangle.hpp"

#include <array>
#include <vector>

#include "Domain/Block.hpp"          // IWYU pragma: keep
#include "Domain/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"

/// \cond
namespace Frame {
struct Grid;
struct Inertial;
struct Logical;
}  // namespace Frame
/// \endcond

namespace domain {
namespace creators {

template <typename TargetFrame>
Rectangle<TargetFrame>::Rectangle(
    typename LowerBound::type lower_xy, typename UpperBound::type upper_xy,
    typename IsPeriodicIn::type is_periodic_in_xy,
    typename InitialRefinement::type initial_refinement_level_xy,
    typename InitialGridPoints::type initial_number_of_grid_points_in_xy,
    const OptionContext& /*context*/) noexcept
    // clang-tidy: trivially copyable
    : lower_xy_(std::move(lower_xy)),                         // NOLINT
      upper_xy_(std::move(upper_xy)),                         // NOLINT
      is_periodic_in_xy_(std::move(is_periodic_in_xy)),       // NOLINT
      initial_refinement_level_xy_(                           // NOLINT
          std::move(initial_refinement_level_xy)),            // NOLINT
      initial_number_of_grid_points_in_xy_(                   // NOLINT
          std::move(initial_number_of_grid_points_in_xy)) {}  // NOLINT

template <typename TargetFrame>
Domain<2, TargetFrame> Rectangle<TargetFrame>::create_domain() const noexcept {
  using Affine = CoordinateMaps::Affine;
  using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  std::vector<PairOfFaces> identifications{};
  if (is_periodic_in_xy_[0]) {
    identifications.push_back({{0, 2}, {1, 3}});
  }
  if (is_periodic_in_xy_[1]) {
    identifications.push_back({{0, 1}, {2, 3}});
  }

  return Domain<2, TargetFrame>{
      make_vector_coordinate_map_base<Frame::Logical, TargetFrame>(
          Affine2D{Affine{-1., 1., lower_xy_[0], upper_xy_[0]},
                   Affine{-1., 1., lower_xy_[1], upper_xy_[1]}}),
      std::vector<std::array<size_t, 4>>{{{0, 1, 2, 3}}}, identifications};
}

template <typename TargetFrame>
std::vector<std::array<size_t, 2>> Rectangle<TargetFrame>::initial_extents()
    const noexcept {
  return {initial_number_of_grid_points_in_xy_};
}

template <typename TargetFrame>
std::vector<std::array<size_t, 2>>
Rectangle<TargetFrame>::initial_refinement_levels() const noexcept {
  return {initial_refinement_level_xy_};
}

template class Rectangle<Frame::Inertial>;
template class Rectangle<Frame::Grid>;
}  // namespace creators
}  // namespace domain
