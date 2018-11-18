// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t, class>
class DomainCreator;  // IWYU pragma: keep
/// \endcond

namespace domain {
namespace creators {

/*!
 * \ingroup DomainCreatorsGroup
 *
 * \brief Creates a 3D Domain in the shape of a hollow spherical shell
 * consisting of six wedges.
 *
 * \image html WedgeOrientations.png "The orientation of each wedge in Shell."
 */

template <typename TargetFrame>
class Shell : public DomainCreator<3, TargetFrame> {
 public:
  struct InnerRadius {
    using type = double;
    static constexpr OptionString help = {"Inner radius of the Shell."};
  };

  struct OuterRadius {
    using type = double;
    static constexpr OptionString help = {"Outer radius of the Shell."};
  };

  struct InitialRefinement {
    using type = size_t;
    static constexpr OptionString help = {
        "Initial refinement level in each dimension."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, 2>;
    static constexpr OptionString help = {
        "Initial number of grid points in [r,angular]."};
  };

  struct UseEquiangularMap {
    using type = bool;
    static constexpr OptionString help = {
        "Use equiangular instead of equidistant coordinates."};
    static constexpr type default_value() noexcept { return true; }
  };

  struct AspectRatio {
    using type = double;
    static constexpr OptionString help = {"The equatorial compression factor."};
    static constexpr type default_value() noexcept { return 1.0; }
  };

  struct UseLogarithmicMap {
    using type = bool;
    static constexpr OptionString help = {
        "Use a logarithmically spaced radial grid."};
    static constexpr type default_value() noexcept { return false; }
  };

  struct WhichWedges {
    using type = ShellWedges;
    static constexpr OptionString help = {
        "Which wedges to include in the shell."};
    static constexpr type default_value() noexcept { return ShellWedges::All; }
  };

  struct RadialBlockLayers {
    using type = size_t;
    static constexpr OptionString help = {
        "The number of concentric layers of Blocks to have."};
    static constexpr type default_value() noexcept { return 1; }
    static type lower_bound() { return 1; }
  };

  using options = tmpl::list<InnerRadius, OuterRadius, InitialRefinement,
                             InitialGridPoints, UseEquiangularMap, AspectRatio,
                             UseLogarithmicMap, WhichWedges, RadialBlockLayers>;

  static constexpr OptionString help{
      "Creates a 3D spherical shell with 6 Blocks. `UseEquiangularMap` has\n"
      "a default value of `true` because there is no central Block in this\n"
      "domain. Equidistant coordinates are best suited to Blocks with\n"
      "Cartesian grids. However, the option is allowed for testing "
      "purposes. The `aspect_ratio` moves grid points on the shell towards\n"
      "the equator for values greater than 1.0, and towards the poles for\n"
      "positive values less than 1.0. If `UseLogarithmicMap` is set to true,\n"
      "the radial gridpoints will be spaced uniformly in \f$log(r)\f$. The\n"
      "user may also choose to use only a single wedge (along the -x\n"
      "direction), or four wedges along the x-y plane using the `WhichWedges`\n"
      "option. Using the RadialBlockLayers option, a user may increase the\n"
      "number of Blocks in the radial direction."};

  Shell(typename InnerRadius::type inner_radius,
        typename OuterRadius::type outer_radius,
        typename InitialRefinement::type initial_refinement,
        typename InitialGridPoints::type initial_number_of_grid_points,
        typename UseEquiangularMap::type use_equiangular_map,
        typename AspectRatio::type aspect_ratio = 1.0,
        typename UseLogarithmicMap::type use_logarithmic_map = false,
        typename WhichWedges::type which_wedges = ShellWedges::All,
        typename RadialBlockLayers::type number_of_layers = 1) noexcept;

  Shell() = default;
  Shell(const Shell&) = delete;
  Shell(Shell&&) noexcept = default;
  Shell& operator=(const Shell&) = delete;
  Shell& operator=(Shell&&) noexcept = default;
  ~Shell() noexcept override = default;

  Domain<3, TargetFrame> create_domain() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const
      noexcept override;

 private:
  typename InnerRadius::type inner_radius_{};
  typename OuterRadius::type outer_radius_{};
  typename InitialRefinement::type initial_refinement_{};
  typename InitialGridPoints::type initial_number_of_grid_points_{};
  typename UseEquiangularMap::type use_equiangular_map_ = true;
  typename AspectRatio::type aspect_ratio_ = 1.0;
  typename UseLogarithmicMap::type use_logarithmic_map_ = false;
  typename WhichWedges::type which_wedges_ = ShellWedges::All;
  typename RadialBlockLayers::type number_of_layers_ = 1;
};
}  // namespace creators
}  // namespace domain
