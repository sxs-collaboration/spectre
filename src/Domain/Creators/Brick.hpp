// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Brick.

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
/// Create a 3D Domain consisting of a single Block.
template <typename TargetFrame>
class Brick : public DomainCreator<3, TargetFrame> {
 public:
  struct LowerBound {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {
        "Sequence of [x,y,z] for lower bounds."};
  };

  struct UpperBound {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {
        "Sequence of [x,y,z] for upper bounds."};
  };
  struct IsPeriodicIn {
    using type = std::array<bool, 3>;
    static constexpr OptionString help = {
        "Sequence for [x,y,z], true if periodic."};
    static type default_value() noexcept { return make_array<3>(false); }
  };

  struct InitialRefinement {
    using type = std::array<size_t, 3>;
    static constexpr OptionString help = {
        "Initial refinement level in [x,y,z]."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, 3>;
    static constexpr OptionString help = {
        "Initial number of grid points in [x,y,z]."};
  };
  using options = tmpl::list<LowerBound, UpperBound, IsPeriodicIn,
                             InitialRefinement, InitialGridPoints>;

  static constexpr OptionString help{"Creates a 3D brick."};

  Brick(typename LowerBound::type lower_xyz,
        typename UpperBound::type upper_xyz,
        typename IsPeriodicIn::type is_periodic_in_xyz,
        typename InitialRefinement::type initial_refinement_level_xyz,
        typename InitialGridPoints::type initial_number_of_grid_points_in_xyz,
        const OptionContext& context = {}) noexcept;

  Brick() = default;
  Brick(const Brick&) = delete;
  Brick(Brick&&) noexcept = default;
  Brick& operator=(const Brick&) = delete;
  Brick& operator=(Brick&&) noexcept = default;
  ~Brick() noexcept override = default;

  Domain<3, TargetFrame> create_domain() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const
      noexcept override;

 private:
  typename LowerBound::type lower_xyz_{};
  typename UpperBound::type upper_xyz_{};
  typename IsPeriodicIn::type is_periodic_in_xyz_{};
  typename InitialRefinement::type initial_refinement_level_xyz_{};
  typename InitialGridPoints::type initial_number_of_grid_points_in_xyz_{};
};
}  // namespace creators
}  // namespace domain
