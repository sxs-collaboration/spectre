// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/DomainCreators/RotatedRectangles.hpp"

#include <algorithm>

#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Domain/OrientationMap.hpp"

namespace DomainCreators {

template <typename TargetFrame>
RotatedRectangles<TargetFrame>::RotatedRectangles(
    const typename LowerBound::type lower_xy,
    const typename Midpoint::type midpoint_xy,
    const typename UpperBound::type upper_xy,
    const typename InitialRefinement::type initial_refinement_level_xy,
    const typename InitialGridPoints::type
        initial_number_of_grid_points_in_xy) noexcept
    // clang-tidy: trivially copyable
    : lower_xy_(std::move(lower_xy)),                         // NOLINT
      midpoint_xy_(std::move(midpoint_xy)),                   // NOLINT
      upper_xy_(std::move(upper_xy)),                         // NOLINT
      initial_refinement_level_xy_(                           // NOLINT
          std::move(initial_refinement_level_xy)),            // NOLINT
      initial_number_of_grid_points_in_xy_(                   // NOLINT
          std::move(initial_number_of_grid_points_in_xy)) {}  // NOLINT

template <typename TargetFrame>
Domain<2, TargetFrame> RotatedRectangles<TargetFrame>::create_domain() const
    noexcept {
  using Affine = CoordinateMaps::Affine;
  using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using DiscreteRotation2D = CoordinateMaps::DiscreteRotation<2>;

  const Affine lower_x_map(-1.0, 1.0, lower_xy_[0], midpoint_xy_[0]);
  const Affine upper_x_map(-1.0, 1.0, midpoint_xy_[0], upper_xy_[0]);
  const Affine lower_y_map(-1.0, 1.0, lower_xy_[1], midpoint_xy_[1]);
  const Affine upper_y_map(-1.0, 1.0, midpoint_xy_[1], upper_xy_[1]);
  std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, TargetFrame, 2>>>
      coord_maps;
  coord_maps.emplace_back(make_coordinate_map_base<Frame::Logical, TargetFrame>(
      Affine2D(lower_x_map, lower_y_map)));
  coord_maps.emplace_back(make_coordinate_map_base<Frame::Logical, TargetFrame>(
      Affine2D(lower_x_map, upper_y_map),
      DiscreteRotation2D{OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}}}));
  coord_maps.emplace_back(make_coordinate_map_base<Frame::Logical, TargetFrame>(
      Affine2D(upper_x_map, lower_y_map),
      DiscreteRotation2D{OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}}}));
  coord_maps.emplace_back(make_coordinate_map_base<Frame::Logical, TargetFrame>(
      Affine2D(upper_x_map, upper_y_map),
      DiscreteRotation2D{OrientationMap<2>{std::array<Direction<2>, 2>{
          {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}}}));
  std::vector<std::array<size_t, 4>> corners{
      {{0, 1, 3, 4}}, {{7, 6, 4, 3}}, {{2, 5, 1, 4}}, {{7, 4, 8, 5}}};
  return Domain<2, TargetFrame>(std::move(coord_maps), std::move(corners));
}

template <typename TargetFrame>
std::vector<std::array<size_t, 2>>
RotatedRectangles<TargetFrame>::initial_extents() const noexcept {
  const size_t& x_0 = initial_number_of_grid_points_in_xy_[0][0];
  const size_t& x_1 = initial_number_of_grid_points_in_xy_[0][1];
  const size_t& y_0 = initial_number_of_grid_points_in_xy_[1][0];
  const size_t& y_1 = initial_number_of_grid_points_in_xy_[1][1];
  return {{{x_0, y_0}}, {{x_0, y_1}}, {{y_0, x_1}}, {{y_1, x_1}}};
}

template <typename TargetFrame>
std::vector<std::array<size_t, 2>>
RotatedRectangles<TargetFrame>::initial_refinement_levels() const noexcept {
  const size_t& x_0 = initial_refinement_level_xy_[0];
  const size_t& y_0 = initial_refinement_level_xy_[1];
  return {{{x_0, y_0}}, {{x_0, y_0}}, {{y_0, x_0}}, {{y_0, x_0}}};
}
}  // namespace DomainCreators

template class DomainCreators::RotatedRectangles<Frame::Inertial>;
template class DomainCreators::RotatedRectangles<Frame::Grid>;
