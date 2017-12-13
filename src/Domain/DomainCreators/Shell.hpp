// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <vector>

#include "Domain/DomainCreators/DomainCreator.hpp"
#include "Options/Options.hpp"

namespace DomainCreators {

/// \ingroup DomainCreatorsGroup
/// Create a 3D Domain in the shape of a hollow spherical shell consisting of
/// six wedges.
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
    static type default_value() { return true; }
  };
  using options = tmpl::list<InnerRadius, OuterRadius, InitialRefinement,
                             InitialGridPoints, UseEquiangularMap>;

  static constexpr OptionString help{
      "Creates a 3D spherical shell with 6 Blocks. `UseEquiangularMap` has\n"
      "a default value of `true` because there is no central Block in this\n"
      "domain. Equidistant coordinates are best suited to Blocks with\n"
      "Cartesian grids. However, the option is allowed for testing "
      "purposes.\n"};

  Shell(typename InnerRadius::type inner_radius,
        typename OuterRadius::type outer_radius,
        typename InitialRefinement::type initial_refinement,
        typename InitialGridPoints::type initial_number_of_grid_points,
        typename UseEquiangularMap::type use_equiangular_map) noexcept;

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
};
}  // namespace DomainCreators
