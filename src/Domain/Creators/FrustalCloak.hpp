// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class FrustalCloak.

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
/// Create a 3D cubical domain with two equal-sized abutting excised cubes in
/// the center. This is done by combining ten frusta.
template <typename TargetFrame>
class FrustalCloak : public DomainCreator<3, TargetFrame> {
 public:
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
    static constexpr type default_value() noexcept { return false; }
  };

  struct ProjectionFactor {
    using type = double;
    static constexpr OptionString help = {"Grid compression factor."};
    static constexpr type default_value() noexcept { return 1.0; }
  };

  struct LengthInnerCube {
    using type = double;
    static constexpr OptionString help = {"Side length of each inner cube."};
    static constexpr type default_value() noexcept { return 0.0; }
  };

  struct LengthOuterCube {
    using type = double;
    static constexpr OptionString help = {"Side length of the outer cube."};
    static constexpr type default_value() noexcept { return 0.0; }
  };

  struct OriginPreimage {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {"The origin preimage in [x,y,z]."};
  };

  using options = tmpl::list<InitialRefinement, InitialGridPoints,
                             UseEquiangularMap, ProjectionFactor,
                             LengthInnerCube, LengthOuterCube, OriginPreimage>;

  static constexpr OptionString help{
      "Creates a cubical domain with two equal-sized abutting excised cubes\n"
      "in the center. This is done by combining ten frusta. The parameter\n"
      "`UseEquiangularMap` can be used to apply a tangent mapping to the xi\n"
      "and eta logical coordinates of each frustum, while the parameter\n"
      "`ProjectionFactor` can be used to apply a projective map to the zeta\n"
      "logical coordinate of each frustum. Increasing the\n"
      "`ProjectionFactor` value can give better gridpoint spacings in the\n"
      "z direction. The user also specifies values for `LengthInnerCube` and\n"
      "`LengthOuterCube`. This will create a cubical Domain of side"
      "length `LengthOuterCube` with the center excised. The size of the\n"
      "excised region is determined by the value set for `LengthInnerCube`.\n"
      "`OriginPreimage` moves the blocks such that the origin preimage is\n"
      "mapped to the origin. Note that the abutting excised cubes share a\n"
      "face in the x-direction. This Domain is primarily for testing the\n"
      "frustal cloak in the BinaryCompactObject Domain."};

  FrustalCloak(typename InitialRefinement::type initial_refinement_level,
               typename InitialGridPoints::type initial_number_of_grid_points,
               typename UseEquiangularMap::type use_equiangular_map = false,
               typename ProjectionFactor::type projection_factor = 1.0,
               typename LengthInnerCube::type length_inner_cube = 0.0,
               typename LengthOuterCube::type length_outer_cube = 0.0,
               typename OriginPreimage::type origin_preimage = {{0.0, 0.0,
                                                                 0.0}},
               const OptionContext& context = {}) noexcept;

  FrustalCloak() = default;
  FrustalCloak(const FrustalCloak&) = delete;
  FrustalCloak(FrustalCloak&&) noexcept = default;
  FrustalCloak& operator=(const FrustalCloak&) = delete;
  FrustalCloak& operator=(FrustalCloak&&) noexcept = default;
  ~FrustalCloak() noexcept override = default;

  Domain<3, TargetFrame> create_domain() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const
      noexcept override;

 private:
  typename InitialRefinement::type initial_refinement_level_{};
  typename InitialGridPoints::type initial_number_of_grid_points_{};
  typename UseEquiangularMap::type use_equiangular_map_ = false;
  typename ProjectionFactor::type projection_factor_{};
  typename LengthInnerCube::type length_inner_cube_{0.0};
  typename LengthOuterCube::type length_outer_cube_{0.0};
  typename OriginPreimage::type origin_preimage_{{0.0, 0.0, 0.0}};
};
}  // namespace creators
}  // namespace domain
