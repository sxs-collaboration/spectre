// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Interval.

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
/// Create a 1D Domain consisting of a single Block.
template <typename TargetFrame>
class Interval : public DomainCreator<1, TargetFrame> {
 public:
  struct LowerBound {
    using type = std::array<double, 1>;
    static constexpr OptionString help = {"Sequence of [x] for lower bounds."};
  };
  struct UpperBound {
    using type = std::array<double, 1>;
    static constexpr OptionString help = {"Sequence of [x] for upper bounds."};
  };
  struct IsPeriodicIn {
    using type = std::array<bool, 1>;
    static constexpr OptionString help = {
        "Sequence for [x], true if periodic."};
    static type default_value() noexcept { return make_array<1>(false); }
  };
  struct InitialRefinement {
    using type = std::array<size_t, 1>;
    static constexpr OptionString help = {"Initial refinement level in [x]."};
  };
  struct InitialGridPoints {
    using type = std::array<size_t, 1>;
    static constexpr OptionString help = {
        "Initial number of grid points in [x]."};
  };

  using options = tmpl::list<LowerBound, UpperBound, IsPeriodicIn,
                             InitialRefinement, InitialGridPoints>;

  static constexpr OptionString help = {"Creates a 1D interval."};

  Interval(typename LowerBound::type lower_x, typename UpperBound::type upper_x,
           typename IsPeriodicIn::type is_periodic_in_x,
           typename InitialRefinement::type initial_refinement_level_x,
           typename InitialGridPoints::type
               initial_number_of_grid_points_in_x) noexcept;

  Interval() = default;
  Interval(const Interval&) = delete;
  Interval(Interval&&) noexcept = default;
  Interval& operator=(const Interval&) = delete;
  Interval& operator=(Interval&&) noexcept = default;
  ~Interval() override = default;

  Domain<1, TargetFrame> create_domain() const noexcept override;

  std::vector<std::array<size_t, 1>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 1>> initial_refinement_levels() const
      noexcept override;

 private:
  typename LowerBound::type lower_x_{};
  typename UpperBound::type upper_x_{};
  typename IsPeriodicIn::type is_periodic_in_x_{};
  typename InitialRefinement::type initial_refinement_level_x_{};
  typename InitialGridPoints::type initial_number_of_grid_points_in_x_{};
};
}  // namespace creators
}  // namespace domain
