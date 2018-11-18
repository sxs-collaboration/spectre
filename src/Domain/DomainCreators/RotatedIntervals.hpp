// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class RotatedIntervals.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

#include "Domain/Domain.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim, typename Frame>
class DomainCreator;  // IWYU pragma: keep
/// \endcond

namespace domain {
namespace creators {

/// \ingroup DomainCreatorsGroup
/// Create a 1D Domain consisting of two rotated Blocks.
/// The left block has its logical \f$\xi\f$-axis aligned with the grid x-axis.
/// The right block has its logical \f$\xi\f$-axis opposite to the grid x-axis.
/// This is useful for testing code that deals with unaligned blocks.
template <typename TargetFrame>
class RotatedIntervals : public DomainCreator<1, TargetFrame> {
 public:
  struct LowerBound {
    using type = std::array<double, 1>;
    static constexpr OptionString help = {
        "Sequence of [x], the lower bound in the target frame."};
  };

  struct Midpoint {
    using type = std::array<double, 1>;
    static constexpr OptionString help = {
        "Sequence of [x], the midpoint in the target frame."};
  };

  struct UpperBound {
    using type = std::array<double, 1>;
    static constexpr OptionString help = {
        "Sequence of [x], the upper bound in the target frame."};
  };

  struct IsPeriodicIn {
    using type = std::array<bool, 1>;
    static constexpr OptionString help = {
        "Sequence for [x], true if periodic."};
    static type default_value() noexcept { return {{false}}; }
  };
  struct InitialRefinement {
    using type = std::array<size_t, 1>;
    static constexpr OptionString help = {"Initial refinement level in [x]."};
  };

  struct InitialGridPoints {
    using type = std::array<std::array<size_t, 2>, 1>;
    static constexpr OptionString help = {
        "Initial number of grid points in [[x]]."};
  };

  using options = tmpl::list<LowerBound, Midpoint, UpperBound, IsPeriodicIn,
                             InitialRefinement, InitialGridPoints>;

  static constexpr OptionString help = {
      "A DomainCreator useful for testing purposes.\n"
      "RotatedIntervals creates the interval [LowerX,UpperX] from two\n"
      "rotated Blocks. The outermost index to InitialGridPoints is the\n"
      "dimension index (of which there is only one in the case of\n"
      "RotatedIntervals), and the innermost index is the block index\n"
      "along that dimension."};

  RotatedIntervals(typename LowerBound::type lower_x,
                   typename Midpoint::type midpoint_x,
                   typename UpperBound::type upper_x,
                   typename IsPeriodicIn::type is_periodic_in,
                   typename InitialRefinement::type initial_refinement_level_x,
                   typename InitialGridPoints::type
                       initial_number_of_grid_points_in_x) noexcept;

  RotatedIntervals() = default;
  RotatedIntervals(const RotatedIntervals&) = delete;
  RotatedIntervals(RotatedIntervals&&) noexcept = default;
  RotatedIntervals& operator=(const RotatedIntervals&) = delete;
  RotatedIntervals& operator=(RotatedIntervals&&) noexcept = default;
  ~RotatedIntervals() override = default;

  Domain<1, TargetFrame> create_domain() const noexcept override;

  std::vector<std::array<size_t, 1>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 1>> initial_refinement_levels() const
      noexcept override;

 private:
  typename LowerBound::type lower_x_{
      {std::numeric_limits<double>::signaling_NaN()}};
  typename Midpoint::type midpoint_x_{
      {std::numeric_limits<double>::signaling_NaN()}};
  typename UpperBound::type upper_x_{
      {std::numeric_limits<double>::signaling_NaN()}};
  typename IsPeriodicIn::type is_periodic_in_{{false}};
  typename InitialRefinement::type initial_refinement_level_x_{
      {std::numeric_limits<size_t>::max()}};
  typename InitialGridPoints::type initial_number_of_grid_points_in_x_{
      {{{std::numeric_limits<size_t>::max()}}}};
};
}  // namespace creators
}  // namespace domain
