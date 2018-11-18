// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Rectangle.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "Domain/Domain.hpp"
#include "Options/Options.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim, typename Frame>
class DomainCreator;  // IWYU pragma: keep
/// \endcond

namespace domain {
namespace creators {

/// \ingroup DomainCreatorsGroup
/// Create a 2D Domain consisting of a single Block.
template <typename TargetFrame>
class Rectangle : public DomainCreator<2, TargetFrame> {
 public:
  struct LowerBound {
    using type = std::array<double, 2>;
    static constexpr OptionString help = {
        "Sequence of [x,y] for lower bounds."};
  };

  struct UpperBound {
    using type = std::array<double, 2>;
    static constexpr OptionString help = {
        "Sequence of [x,y] for upper bounds."};
  };
  struct IsPeriodicIn {
    using type = std::array<bool, 2>;
    static constexpr OptionString help = {
        "Sequence for [x,y], true if periodic."};
    static type default_value() noexcept { return make_array<2>(false); }
  };

  struct InitialRefinement {
    using type = std::array<size_t, 2>;
    static constexpr OptionString help = {"Initial refinement level in [x,y]."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, 2>;
    static constexpr OptionString help = {
        "Initial number of grid points in [x,y]."};
  };
  using options = tmpl::list<LowerBound, UpperBound, IsPeriodicIn,
                             InitialRefinement, InitialGridPoints>;

  static constexpr OptionString help{"Creates a 2D rectangle."};

  Rectangle(
      typename LowerBound::type lower_xy, typename UpperBound::type upper_xy,
      typename IsPeriodicIn::type is_periodic_in_xy,
      typename InitialRefinement::type initial_refinement_level_xy,
      typename InitialGridPoints::type initial_number_of_grid_points_in_xy,
      const OptionContext& context = {}) noexcept;

  Rectangle() = default;
  Rectangle(const Rectangle&) = delete;
  Rectangle(Rectangle&&) noexcept = default;
  Rectangle& operator=(const Rectangle&) = delete;
  Rectangle& operator=(Rectangle&&) noexcept = default;
  ~Rectangle() noexcept override = default;

  Domain<2, TargetFrame> create_domain() const noexcept override;

  std::vector<std::array<size_t, 2>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 2>> initial_refinement_levels() const
      noexcept override;

 private:
  typename LowerBound::type lower_xy_{};
  typename UpperBound::type upper_xy_{};
  typename IsPeriodicIn::type is_periodic_in_xy_{};
  typename InitialRefinement::type initial_refinement_level_xy_{};
  typename InitialGridPoints::type initial_number_of_grid_points_in_xy_{};
};
}  // namespace creators
}  // namespace domain
