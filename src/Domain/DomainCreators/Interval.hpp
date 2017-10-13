// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Interval.

#pragma once

#include "Domain/DomainCreators/DomainCreator.hpp"
#include "Options/Options.hpp"

namespace DomainCreators {
/// Create a 1D Domain consisting of a single Block.
class Interval : public DomainCreator<1, Frame::Inertial> {
 public:
  struct LowerBound {
    using type = std::array<double, 1>;
    static constexpr OptionString_t help = {
        "Sequence of [x] for lower bounds."};
  };
  struct UpperBound {
    using type = std::array<double, 1>;
    static constexpr OptionString_t help = {
        "Sequence of [x] for upper bounds."};
  };
  struct IsPeriodicIn {
    using type = std::array<bool, 1>;
    static constexpr OptionString_t help = {
        "Sequence for [x], true if periodic."};
    static type default_value() { return make_array<1>(false); }
  };
  struct InitialRefinement {
    using type = std::array<size_t, 1>;
    static constexpr OptionString_t help = {"Initial refinement level in [x]."};
  };
  struct InitialGridPoints {
    using type = std::array<size_t, 1>;
    static constexpr OptionString_t help = {
        "Initial number of grid points in [x]."};
  };

  using options = tmpl::list<LowerBound, UpperBound, IsPeriodicIn,
                             InitialRefinement, InitialGridPoints>;

  static constexpr OptionString_t help = {"Creates a 1D interval."};

  Interval(typename LowerBound::type lower_x, typename UpperBound::type upper_x,
           typename IsPeriodicIn::type is_periodic_in_x,
           typename InitialRefinement::type initial_refinement_level_x,
           typename InitialGridPoints::type initial_number_of_grid_points_in_x,
           const OptionContext& context = {});

  Interval() = default;
  Interval(const Interval&) = delete;
  Interval(Interval&&) noexcept = default;
  Interval& operator=(const Interval&) = delete;
  Interval& operator=(Interval&&) noexcept = default;
  ~Interval() override = default;

  explicit Interval(CkMigrateMessage* /*unused*/) noexcept {}

  Domain<1, Frame::Inertial> create_domain() const override;

  std::array<size_t, 1> initial_extents(size_t block_index) const override;

  std::array<size_t, 1> initial_refinement_levels(
      size_t block_index) const override;

 private:
  typename LowerBound::type lower_x_{};
  typename UpperBound::type upper_x_{};
  typename IsPeriodicIn::type is_periodic_in_x_{};
  typename InitialRefinement::type initial_refinement_level_x_{};
  typename InitialGridPoints::type initial_number_of_grid_points_in_x_{};
};
}  // namespace DomainCreators
