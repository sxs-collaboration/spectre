// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/DomainCreators/Interval.hpp"

#include <array>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BlockNeighbor.hpp"
#include "Domain/CoordinateMaps/AffineMap.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Options/Options.hpp"
#include "Utilities/MakeVector.hpp"

namespace DomainCreators {

Interval::Interval(
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

Domain<1, Frame::Inertial> Interval::create_domain() const noexcept {
  return Domain<1, Frame::Inertial>{
      make_vector<std::unique_ptr<
          CoordinateMapBase<Frame::Logical, Frame::Inertial, 1>>>(
          std::make_unique<CoordinateMap<Frame::Logical, Frame::Inertial,
                                         CoordinateMaps::AffineMap>>(
              CoordinateMaps::AffineMap{-1., 1., lower_x_[0], upper_x_[0]})),
      std::vector<std::array<size_t, 2>>{{{1, 2}}},
      is_periodic_in_x_[0] ? std::vector<PairOfFaces>{{{1}, {2}}}
                           : std::vector<PairOfFaces>{}};
}

std::array<size_t, 1> Interval::initial_extents(const size_t block_index) const
    noexcept {
  ASSERT(0 == block_index, "index = " << block_index);
  return initial_number_of_grid_points_in_x_;
}

std::array<size_t, 1> Interval::initial_refinement_levels(
    const size_t block_index) const noexcept {
  ASSERT(0 == block_index, "index = " << block_index);
  return initial_refinement_level_x_;
}
}  // namespace DomainCreators
