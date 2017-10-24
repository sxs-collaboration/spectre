// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/DomainCreators/Rectangle.hpp"

#include <array>
#include <unordered_set>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BlockNeighbor.hpp"
#include "Domain/CoordinateMaps/AffineMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Options/Options.hpp"
#include "Utilities/MakeVector.hpp"

namespace DomainCreators {

Rectangle::Rectangle(
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

Domain<2, Frame::Inertial> Rectangle::create_domain() const noexcept {
  using AffineMap = CoordinateMaps::AffineMap;
  using AffineMap2D = CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
  std::vector<PairOfFaces> identifications{};
  if (is_periodic_in_xy_[0]) {
    identifications.push_back({{0, 2}, {1, 3}});
  }
  if (is_periodic_in_xy_[1]) {
    identifications.push_back({{0, 1}, {2, 3}});
  }

  return Domain<2, Frame::Inertial>{
      make_vector<std::unique_ptr<
          CoordinateMapBase<Frame::Logical, Frame::Inertial, 2>>>(
          std::make_unique<
              CoordinateMap<Frame::Logical, Frame::Inertial, AffineMap2D>>(
              AffineMap2D{AffineMap{-1., 1., lower_xy_[0], upper_xy_[0]},
                          AffineMap{-1., 1., lower_xy_[1], upper_xy_[1]}})),
      std::vector<std::array<size_t, 4>>{{{0, 1, 2, 3}}}, identifications};
}

std::array<size_t, 2> Rectangle::initial_extents(const size_t block_index) const
    noexcept {
  ASSERT(0 == block_index, "index = " << block_index);
  return initial_number_of_grid_points_in_xy_;
}

std::array<size_t, 2> Rectangle::initial_refinement_levels(
    const size_t block_index) const noexcept {
  ASSERT(0 == block_index, "index = " << block_index);
  return initial_refinement_level_xy_;
}
}  // namespace DomainCreators
