// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class Affine;
template <size_t VolumeDim>
class DiscreteRotation;
template <typename Map1, typename Map2>
class ProductOf2Maps;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain {
namespace creators {
/// Create a 2D Domain consisting of four rotated Blocks.
/// - The lower left block has its logical \f$\xi\f$-axis aligned with
/// the grid x-axis.
///
/// - The lower right block is rotated a half-turn (180 degrees) relative to the
/// lower left block.
///
/// - The upper left block is rotated a quarter-turn counterclockwise
/// (+90 degrees) relative to the lower left block.
//
/// - The upper right block is rotated a quarter-turn clockwise
/// (-90 degrees) relative to the lower left block.
///
/// This DomainCreator is useful for testing code that deals with
/// unaligned blocks.
class RotatedRectangles : public DomainCreator<2> {
 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<
          Frame::Logical, Frame::Inertial,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Affine>>,
      domain::CoordinateMap<
          Frame::Logical, Frame::Inertial, CoordinateMaps::DiscreteRotation<2>,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Affine>>>;

  struct LowerBound {
    using type = std::array<double, 2>;
    static constexpr Options::String help = {
        "Sequence of [x,y] for lower bounds in the target frame."};
  };

  struct Midpoint {
    using type = std::array<double, 2>;
    static constexpr Options::String help = {
        "Sequence of [x,y] for midpoints in the target frame."};
  };

  struct UpperBound {
    using type = std::array<double, 2>;
    static constexpr Options::String help = {
        "Sequence of [x,y] for upper bounds in the target frame."};
  };

  struct IsPeriodicIn {
    using type = std::array<bool, 2>;
    static constexpr Options::String help = {
        "Sequence for [x], true if periodic."};
    static type default_value() noexcept { return {{false, false}}; }
  };

  struct InitialRefinement {
    using type = std::array<size_t, 2>;
    static constexpr Options::String help = {
        "Initial refinement level in [x, y]."};
  };

  struct InitialGridPoints {
    using type = std::array<std::array<size_t, 2>, 2>;
    static constexpr Options::String help = {
        "Initial number of grid points in [[x], [y]]."};
  };

  using options = tmpl::list<LowerBound, Midpoint, UpperBound, IsPeriodicIn,
                             InitialRefinement, InitialGridPoints>;

  static constexpr Options::String help = {
      "A DomainCreator useful for testing purposes.\n"
      "RotatedRectangles uses four rotated Blocks to create the rectangle\n"
      "[LowerX,UpperX] x [LowerY,UpperY]. The outermost index to\n"
      "InitialGridPoints is the dimension index, and the innermost index is\n"
      "the block index along that dimension."};

  RotatedRectangles(
      typename LowerBound::type lower_xy, typename Midpoint::type midpoint_xy,
      typename UpperBound::type upper_xy,
      typename IsPeriodicIn::type is_periodic_in,
      typename InitialRefinement::type initial_refinement_level_xy,
      typename InitialGridPoints::type
          initial_number_of_grid_points_in_xy) noexcept;

  RotatedRectangles() = default;
  RotatedRectangles(const RotatedRectangles&) = delete;
  RotatedRectangles(RotatedRectangles&&) noexcept = default;
  RotatedRectangles& operator=(const RotatedRectangles&) = delete;
  RotatedRectangles& operator=(RotatedRectangles&&) noexcept = default;
  ~RotatedRectangles() override = default;

  Domain<2> create_domain() const noexcept override;

  std::vector<std::array<size_t, 2>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 2>> initial_refinement_levels() const
      noexcept override;

 private:
  typename LowerBound::type lower_xy_{
      {std::numeric_limits<double>::signaling_NaN()}};
  typename Midpoint::type midpoint_xy_{
      {std::numeric_limits<double>::signaling_NaN()}};
  typename UpperBound::type upper_xy_{
      {std::numeric_limits<double>::signaling_NaN()}};
  typename IsPeriodicIn::type is_periodic_in_{{false, false}};
  typename InitialRefinement::type initial_refinement_level_xy_{
      {std::numeric_limits<size_t>::max()}};
  typename InitialGridPoints::type initial_number_of_grid_points_in_xy_{
      {{{std::numeric_limits<size_t>::max()}}}};
};
}  // namespace creators
}  // namespace domain
