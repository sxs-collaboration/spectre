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
/// Create a 3D Domain in the shape of a cylinder where the cross-section
/// is a square surrounded by four two-dimensional wedges (see Wedge2D).
///
/// \image html Cylinder.png "The Cylinder Domain."
template <typename TargetFrame>
class Cylinder : public DomainCreator<3, TargetFrame> {
 public:
  struct InnerRadius {
    using type = double;
    static constexpr OptionString help = {
        "Radius of the circle circumscribing the inner square."};
  };

  struct OuterRadius {
    using type = double;
    static constexpr OptionString help = {"Radius of the cylinder."};
  };

  struct LowerBound {
    using type = double;
    static constexpr OptionString help = {
        "z-coordinate of the base of the cylinder."};
  };

  struct UpperBound {
    using type = double;
    static constexpr OptionString help = {
        "z-coordinate of the top of the cylinder."};
  };

  struct IsPeriodicInZ {
    using type = bool;
    static constexpr OptionString help = {
        "True if periodic in the cylindrical z direction."};
    static type default_value() noexcept { return true; }
  };

  struct InitialRefinement {
    using type = size_t;
    static constexpr OptionString help = {
        "Initial refinement level in each dimension."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, 3>;
    static constexpr OptionString help = {
        "Initial number of grid points in [r, theta, z]."};
  };

  struct UseEquiangularMap {
    using type = bool;
    static constexpr OptionString help = {
        "Use equiangular instead of equidistant coordinates."};
    static type default_value() noexcept { return false; }
  };

  using options = tmpl::list<InnerRadius, OuterRadius, LowerBound, UpperBound,
                             IsPeriodicInZ, InitialRefinement,
                             InitialGridPoints, UseEquiangularMap>;

  static constexpr OptionString help{
      "Creates a 3D Cylinder with five Blocks.\n"
      "Only one refinement level for all dimensions is currently supported.\n"
      "The number of gridpoints in each dimension can be set independently.\n"
      "The number of gridpoints along the dimensions of the square is equal\n"
      "to the number of gridpoints along the angular dimension of the wedges.\n"
      "Equiangular coordinates give better gridpoint spacings in the angular\n"
      "direction, while equidistant coordinates give better gridpoint\n"
      "spacings in the center block. This Domain uses equidistant coordinates\n"
      "by default. The boundary conditions are set to be periodic along the\n"
      "cylindrical z-axis by default."};

  Cylinder(typename InnerRadius::type inner_radius,
           typename OuterRadius::type outer_radius,
           typename LowerBound::type lower_bound,
           typename UpperBound::type upper_bound,
           typename IsPeriodicInZ::type is_periodic_in_z,
           typename InitialRefinement::type initial_refinement,
           typename InitialGridPoints::type initial_number_of_grid_points,
           typename UseEquiangularMap::type use_equiangular_map) noexcept;

  Cylinder() = default;
  Cylinder(const Cylinder&) = delete;
  Cylinder(Cylinder&&) noexcept = default;
  Cylinder& operator=(const Cylinder&) = delete;
  Cylinder& operator=(Cylinder&&) noexcept = default;
  ~Cylinder() noexcept override = default;

  Domain<3, TargetFrame> create_domain() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const
      noexcept override;

 private:
  typename InnerRadius::type inner_radius_{};
  typename OuterRadius::type outer_radius_{};
  typename LowerBound::type lower_bound_{};
  typename UpperBound::type upper_bound_{};
  typename IsPeriodicInZ::type is_periodic_in_z_{true};
  typename InitialRefinement::type initial_refinement_{};
  typename InitialGridPoints::type initial_number_of_grid_points_{};
  typename UseEquiangularMap::type use_equiangular_map_{false};
};
}  // namespace creators
}  // namespace domain
