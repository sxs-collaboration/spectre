// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/DomainCreators/RotatedIntervals.hpp"

#include "Domain/Block.hpp"          // IWYU pragma: keep
#include "Domain/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/DomainHelpers.hpp"

/// \cond
namespace Frame {
struct Grid;
struct Inertial;
struct Logical;
}  // namespace Frame
/// \endcond

namespace DomainCreators {
template <typename TargetFrame>
RotatedIntervals<TargetFrame>::RotatedIntervals(
    typename LowerBound::type lower_x, typename Midpoint::type midpoint_x,
    typename UpperBound::type upper_x,
    typename IsPeriodicIn::type is_periodic_in,
    typename InitialRefinement::type initial_refinement_level_x,
    typename InitialGridPoints::type
        initial_number_of_grid_points_in_x) noexcept
    // clang-tidy: trivially copyable
    : lower_x_(std::move(lower_x)),                          // NOLINT
      midpoint_x_(std::move(midpoint_x)),                    // NOLINT
      upper_x_(std::move(upper_x)),                          // NOLINT
      is_periodic_in_(std::move(is_periodic_in)),            // NOLINT
      initial_refinement_level_x_(                           // NOLINT
          std::move(initial_refinement_level_x)),            // NOLINT
      initial_number_of_grid_points_in_x_(                   // NOLINT
          std::move(initial_number_of_grid_points_in_x)) {}  // NOLINT

template <typename TargetFrame>
Domain<1, TargetFrame> RotatedIntervals<TargetFrame>::create_domain() const
    noexcept {
  return Domain<1, TargetFrame>{
      make_vector_coordinate_map_base<Frame::Logical, TargetFrame>(
          CoordinateMaps::Affine{-1.0, 1.0, lower_x_[0], midpoint_x_[0]},
          CoordinateMaps::Affine{-1.0, 1.0, upper_x_[0], midpoint_x_[0]}),
      std::vector<std::array<size_t, 2>>{{{1, 2}}, {{3, 2}}},
      is_periodic_in_[0] ? std::vector<PairOfFaces>{{{1}, {3}}}
                         : std::vector<PairOfFaces>{}};
}

template <typename TargetFrame>
std::vector<std::array<size_t, 1>>
RotatedIntervals<TargetFrame>::initial_extents() const noexcept {
  return {{{initial_number_of_grid_points_in_x_[0][0]}},
          {{initial_number_of_grid_points_in_x_[0][1]}}};
}

template <typename TargetFrame>
std::vector<std::array<size_t, 1>>
RotatedIntervals<TargetFrame>::initial_refinement_levels() const noexcept {
  return {{{initial_refinement_level_x_[0]}},
          {{initial_refinement_level_x_[0]}}};
}
}  // namespace DomainCreators

template class DomainCreators::RotatedIntervals<Frame::Inertial>;
template class DomainCreators::RotatedIntervals<Frame::Grid>;
