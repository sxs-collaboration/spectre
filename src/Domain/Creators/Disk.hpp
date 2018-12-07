// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
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
/// Create a 2D Domain in the shape of a disk from a square surrounded by four
/// wedges.
template <typename TargetFrame>
class Disk : public DomainCreator<2, TargetFrame> {
 public:
  struct InnerRadius {
    using type = double;
    static constexpr OptionString help = {
        "Radius of the circle circumscribing the inner square."};
  };

  struct OuterRadius {
    using type = double;
    static constexpr OptionString help = {"Radius of the Disk."};
  };

  struct InitialRefinement {
    using type = size_t;
    static constexpr OptionString help = {
        "Initial refinement level in each dimension."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, 2>;
    static constexpr OptionString help = {
        "Initial number of grid points in [r,theta]."};
  };

  struct UseEquiangularMap {
    using type = bool;
    static constexpr OptionString help = {
        "Use equiangular instead of equidistant coordinates."};
  };

  using options = tmpl::list<InnerRadius, OuterRadius, InitialRefinement,
                             InitialGridPoints, UseEquiangularMap>;

  static constexpr OptionString help{
      "Creates a 2D Disk with five Blocks.\n"
      "Only one refinement level for both dimensions is currently supported.\n"
      "The number of gridpoints in each dimension can be set independently.\n"
      "The number of gridpoints along the dimensions of the square is equal\n"
      "to the number of gridpoints along the angular dimension of the wedges.\n"
      "Equiangular coordinates give better gridpoint spacings in the angular\n"
      "direction, while equidistant coordinates give better gridpoint\n"
      "spacings in the center block."};

  Disk(typename InnerRadius::type inner_radius,
       typename OuterRadius::type outer_radius,
       typename InitialRefinement::type initial_refinement,
       typename InitialGridPoints::type initial_number_of_grid_points,
       typename UseEquiangularMap::type use_equiangular_map) noexcept;

  Disk() = default;
  Disk(const Disk&) = delete;
  Disk(Disk&&) noexcept = default;
  Disk& operator=(const Disk&) = delete;
  Disk& operator=(Disk&&) noexcept = default;
  ~Disk() noexcept override = default;

  Domain<2, TargetFrame> create_domain() const noexcept override;

  std::vector<std::array<size_t, 2>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 2>> initial_refinement_levels() const
      noexcept override;

 private:
  typename InnerRadius::type inner_radius_{};
  typename OuterRadius::type outer_radius_{};
  typename InitialRefinement::type initial_refinement_{};
  typename InitialGridPoints::type initial_number_of_grid_points_{};
  typename UseEquiangularMap::type use_equiangular_map_{false};
};
}  // namespace creators
}  // namespace domain
