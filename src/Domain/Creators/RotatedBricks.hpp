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

namespace domain {
namespace creators {

/// \ingroup DomainCreatorsGroup
/// Create a 3D Domain consisting of eight rotated Blocks.
///
/// \image html eightcubes_rotated_exploded.png
///
/// The orientations of the blocks are described in two different ways:
///
///  - 1 - As orientations of blocks relative to the physical cartesian axes.
///
///  - 2 - As rotations relative to the physical cartesian axes written using
///     Rubik's cube rotation notation, for the sake of shorter variable names.
///
/// For reference, this is the notation used:
///
///  - U - A clockwise 90 degree rotation about the +z axis (The "up" side)
///
///  - R - A clockwise 90 degree rotation about the +x axis (The "right" side)
///
///  - F - A clockwise 90 degree rotation about the -y axis (The "front" side)
///
/// For reference, D, L, and B ("down", "left", and "back") are the inverse
/// rotation to the aforementioned ones, respectively.
/// Note: Whereas Rubik's cube rotations rotate a layer of the 3x3 puzzle
/// cube, we are adopting the notation to apply to rotations of the
/// cube itself.
///
/// - The -x, -y, -z block has the aligned orientation, that is,
///   xi is aligned with x, eta is aligned with y, and zeta with z.
///
/// - The +x, -y, -z block has the orientation (zeta, eta, -xi).
///   It corresponds to the orientation obtained by the rotation F.
///
/// - The -x, +y, -z block has the orientation (xi, zeta, -eta).
///   It corresponds to the orientation obtained by the rotation R.
///
/// - The +x, +y, -z block has the orientation (zeta, -xi, -eta).
///   It corresponds to the orientation obtained by the rotation F
///   followed by the rotation R.
///
/// - The -x, -y, +z block has the orientation (eta, -xi, zeta).
///   It corresponds to the orientation obtained by the rotation U.
///
/// - The +x, -y, +z block has the orientation (eta, -zeta, -xi).
///   It corresponds to the orientation obtained by the rotation F
///   followed by the rotation U.
///
/// - The -x, +y, +z block has the orientation (zeta, -xi, -eta).
///   It corresponds to the orientation obtained by the rotation R
///   followed by the rotation U (equivalently, F followed by R).
///
/// - The +x, +y, +z block also has the aligned orientation
///   (xi, eta, zeta), relative to the edifice. It is not aligned
///   relative to its neighbors.
///
/// This DomainCreator is useful for testing code that deals with
/// unaligned blocks.
template <typename TargetFrame>
class RotatedBricks : public DomainCreator<3, TargetFrame> {
 public:
  struct LowerBound {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {
        "Sequence [x,y,z] for lower bound in the target frame."};
  };

  struct Midpoint {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {
        "Sequence [x,y,z] for midpoint in the target frame."};
  };

  struct UpperBound {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {
        "Sequence [x,y,z] for upper bound in the target frame."};
  };

  struct IsPeriodicIn {
    using type = std::array<bool, 3>;
    static constexpr OptionString help = {
        "Sequence in [x,y,z], true if periodic."};
    static type default_value() noexcept { return {{false, false, false}}; }
  };

  struct InitialRefinement {
    using type = std::array<size_t, 3>;
    static constexpr OptionString help = {
        "Initial refinement level in [x, y, z]."};
  };

  struct InitialGridPoints {
    using type = std::array<std::array<size_t, 2>, 3>;
    static constexpr OptionString help = {
        "Initial number of grid points in [[x], [y], [z]]."};
  };

  using options = tmpl::list<LowerBound, Midpoint, UpperBound, IsPeriodicIn,
                             InitialRefinement, InitialGridPoints>;

  static constexpr OptionString help = {
      "A DomainCreator useful for testing purposes.\n"
      "RotatedBricks uses eight rotated Blocks to create the rectangular\n"
      "prism [LowerX,UpperX] x [LowerY,UpperY] x [LowerZ,UpperZ]. The\n"
      "outermost index to InitialGridPoints is the dimension index, and\n"
      "the innermost index is the block index along that dimension."};

  RotatedBricks(typename LowerBound::type lower_xyz,
                typename Midpoint::type midpoint_xyz,
                typename UpperBound::type upper_xyz,
                typename IsPeriodicIn::type is_periodic_in,
                typename InitialRefinement::type initial_refinement_level_xyz,
                typename InitialGridPoints::type
                    initial_number_of_grid_points_in_xyz) noexcept;

  RotatedBricks() = default;
  RotatedBricks(const RotatedBricks&) = delete;
  RotatedBricks(RotatedBricks&&) noexcept = default;
  RotatedBricks& operator=(const RotatedBricks&) = delete;
  RotatedBricks& operator=(RotatedBricks&&) noexcept = default;
  ~RotatedBricks() override = default;

  Domain<3, TargetFrame> create_domain() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const
      noexcept override;

 private:
  typename LowerBound::type lower_xyz_{
      {std::numeric_limits<double>::signaling_NaN()}};
  typename Midpoint::type midpoint_xyz_{
      {std::numeric_limits<double>::signaling_NaN()}};
  typename UpperBound::type upper_xyz_{
      {std::numeric_limits<double>::signaling_NaN()}};
  typename IsPeriodicIn::type is_periodic_in_{{false, false, false}};
  typename InitialRefinement::type initial_refinement_level_xyz_{
      {std::numeric_limits<size_t>::max()}};
  typename InitialGridPoints::type initial_number_of_grid_points_in_xyz_{
      {{{std::numeric_limits<size_t>::max()}}}};
};
}  // namespace creators
}  // namespace domain
