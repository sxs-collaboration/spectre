// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Brick.hpp"

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
Brick<TargetFrame>::Brick(
    typename LowerBound::type lower_xyz, typename UpperBound::type upper_xyz,
    typename IsPeriodicIn::type is_periodic_in_xyz,
    typename InitialRefinement::type initial_refinement_level_xyz,
    typename InitialGridPoints::type initial_number_of_grid_points_in_xyz,
    const OptionContext& /*context*/) noexcept
    // clang-tidy: trivially copyable
    : lower_xyz_(std::move(lower_xyz)),                        // NOLINT
      upper_xyz_(std::move(upper_xyz)),                        // NOLINT
      is_periodic_in_xyz_(std::move(is_periodic_in_xyz)),      // NOLINT
      initial_refinement_level_xyz_(                           // NOLINT
          std::move(initial_refinement_level_xyz)),            // NOLINT
      initial_number_of_grid_points_in_xyz_(                   // NOLINT
          std::move(initial_number_of_grid_points_in_xyz)) {}  // NOLINT

template <typename TargetFrame>
Domain<3, TargetFrame> Brick<TargetFrame>::create_domain() const noexcept {
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  std::vector<PairOfFaces> identifications{};
  if (is_periodic_in_xyz_[0]) {
    identifications.push_back({{0, 4, 2, 6}, {1, 5, 3, 7}});
  }
  if (is_periodic_in_xyz_[1]) {
    identifications.push_back({{0, 1, 4, 5}, {2, 3, 6, 7}});
  }
  if (is_periodic_in_xyz_[2]) {
    identifications.push_back({{0, 1, 2, 3}, {4, 5, 6, 7}});
  }

  return Domain<3, TargetFrame>{
      make_vector_coordinate_map_base<Frame::Logical, TargetFrame>(
          Affine3D{Affine{-1., 1., lower_xyz_[0], upper_xyz_[0]},
                   Affine{-1., 1., lower_xyz_[1], upper_xyz_[1]},
                   Affine{-1., 1., lower_xyz_[2], upper_xyz_[2]}}),
      std::vector<std::array<size_t, 8>>{{{0, 1, 2, 3, 4, 5, 6, 7}}},
      identifications};
}

template <typename TargetFrame>
std::vector<std::array<size_t, 3>> Brick<TargetFrame>::initial_extents() const
    noexcept {
  return {initial_number_of_grid_points_in_xyz_};
}

template <typename TargetFrame>
std::vector<std::array<size_t, 3>>
Brick<TargetFrame>::initial_refinement_levels() const noexcept {
  return {initial_refinement_level_xyz_};
}

template class Brick<Frame::Inertial>;
template class Brick<Frame::Grid>;
}  // namespace creators
}  // namespace domain
