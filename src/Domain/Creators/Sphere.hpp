// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "Domain/Domain.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

// IWYU wants to include things we definitely don't need...
// IWYU pragma: no_include <memory> // Needed in cpp file
// IWYU pragma: no_include <pup.h>  // Not needed

// IWYU pragma: no_include "DataStructures/Tensor/Tensor.hpp" // Not needed

/// \cond
template <size_t Dim, typename Frame>
class DomainCreator;  // IWYU pragma: keep
/// \endcond

namespace domain {
namespace creators {

/// \ingroup DomainCreatorsGroup
/// Create a 3D Domain in the shape of a sphere consisting of six wedges
/// and a central cube. For an image showing how the wedges are aligned in
/// this Domain, see the documentation for Shell.
template <typename TargetFrame>
class Sphere : public DomainCreator<3, TargetFrame> {
 public:
  struct InnerRadius {
    using type = double;
    static constexpr OptionString help = {
        "Radius of the sphere circumscribing the inner cube."};
  };

  struct OuterRadius {
    using type = double;
    static constexpr OptionString help = {"Radius of the Sphere."};
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
  };
  using options = tmpl::list<InnerRadius, OuterRadius, InitialRefinement,
                             InitialGridPoints, UseEquiangularMap>;

  static constexpr OptionString help{
      "Creates a 3D Sphere with seven Blocks.\n"
      "Only one refinement level for all dimensions is currently supported.\n"
      "The number of gridpoints in the radial direction can be set\n"
      "independently of the number of gridpoints in the angular directions.\n"
      "The number of gridpoints along the dimensions of the cube is equal\n"
      "to the number of gridpoints along the angular dimensions of the "
      "wedges.\n"
      "Equiangular coordinates give better gridpoint spacings in the angular\n"
      "directions, while equidistant coordinates give better gridpoint\n"
      "spacings in the center block."};

  Sphere(typename InnerRadius::type inner_radius,
         typename OuterRadius::type outer_radius,
         typename InitialRefinement::type initial_refinement,
         typename InitialGridPoints::type initial_number_of_grid_points,
         typename UseEquiangularMap::type use_equiangular_map) noexcept;

  Sphere() = default;
  Sphere(const Sphere&) = delete;
  Sphere(Sphere&&) noexcept = default;
  Sphere& operator=(const Sphere&) = delete;
  Sphere& operator=(Sphere&&) noexcept = default;
  ~Sphere() noexcept override = default;

  Domain<3, TargetFrame> create_domain() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const
      noexcept override;

 private:
  typename InnerRadius::type inner_radius_{};
  typename OuterRadius::type outer_radius_{};
  typename InitialRefinement::type initial_refinement_{};
  typename InitialGridPoints::type initial_number_of_grid_points_{};
  typename UseEquiangularMap::type use_equiangular_map_ = false;
};
}  // namespace creators
}  // namespace domain
