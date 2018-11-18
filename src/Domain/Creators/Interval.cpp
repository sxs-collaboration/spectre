// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Interval.hpp"

#include <array>
#include <vector>

#include "Domain/Block.hpp"          // IWYU pragma: keep
#include "Domain/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
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
Interval<TargetFrame>::Interval(
    typename LowerBound::type lower_x, typename UpperBound::type upper_x,
    typename IsPeriodicIn::type is_periodic_in_x,
    typename InitialRefinement::type initial_refinement_level_x,
    typename InitialGridPoints::type
        initial_number_of_grid_points_in_x) noexcept
    // clang-tidy: trivially copyable
    : lower_x_(std::move(lower_x)),                          // NOLINT
      upper_x_(std::move(upper_x)),                          // NOLINT
      is_periodic_in_x_(std::move(is_periodic_in_x)),        // NOLINT
      initial_refinement_level_x_(                           // NOLINT
          std::move(initial_refinement_level_x)),            // NOLINT
      initial_number_of_grid_points_in_x_(                   // NOLINT
          std::move(initial_number_of_grid_points_in_x)) {}  // NOLINT

template <typename TargetFrame>
Domain<1, TargetFrame> Interval<TargetFrame>::create_domain() const noexcept {
  return Domain<1, TargetFrame>{
      make_vector_coordinate_map_base<Frame::Logical, TargetFrame>(
          CoordinateMaps::Affine{-1., 1., lower_x_[0], upper_x_[0]}),
      std::vector<std::array<size_t, 2>>{{{1, 2}}},
      is_periodic_in_x_[0] ? std::vector<PairOfFaces>{{{1}, {2}}}
                           : std::vector<PairOfFaces>{}};
}

template <typename TargetFrame>
std::vector<std::array<size_t, 1>> Interval<TargetFrame>::initial_extents()
    const noexcept {
  return {{{initial_number_of_grid_points_in_x_}}};
}

template <typename TargetFrame>
std::vector<std::array<size_t, 1>>
Interval<TargetFrame>::initial_refinement_levels() const noexcept {
  return {{{initial_refinement_level_x_}}};
}

template class Interval<Frame::Inertial>;
template class Interval<Frame::Grid>;
}  // namespace creators
}  // namespace domain
