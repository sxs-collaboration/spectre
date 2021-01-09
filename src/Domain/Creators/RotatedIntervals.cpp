// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/RotatedIntervals.hpp"

#include "DataStructures/Index.hpp"
#include "Domain/Block.hpp"                   // IWYU pragma: keep
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/OrientationMap.hpp"

namespace domain::creators {
RotatedIntervals::RotatedIntervals(
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

Domain<1> RotatedIntervals::create_domain() const noexcept {
  return rectilinear_domain<1>(
      Index<1>{2}, {{{lower_x_[0], midpoint_x_[0], upper_x_[0]}}}, {}, {},
      {OrientationMap<1>{}, OrientationMap<1>{std::array<Direction<1>, 1>{
                                {Direction<1>::lower_xi()}}}},
      is_periodic_in_);
}

std::vector<std::array<size_t, 1>> RotatedIntervals::initial_extents() const
    noexcept {
  return {{{initial_number_of_grid_points_in_x_[0][0]}},
          {{initial_number_of_grid_points_in_x_[0][1]}}};
}

std::vector<std::array<size_t, 1>>
RotatedIntervals ::initial_refinement_levels() const noexcept {
  return {{{initial_refinement_level_x_[0]}},
          {{initial_refinement_level_x_[0]}}};
}
}  // namespace domain::creators
