// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim, typename T>
class DirectionMap;
template <size_t Dim>
class Domain;
namespace domain {
namespace CoordinateMaps {
class Affine;
template <size_t Dim>
class Identity;
class Interval;
template <typename Map1, typename Map2>
class ProductOf2Maps;
class SphericalToCartesianPfaffian;
template <size_t Dim>
class Wedge;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain::creators {
/*!
 * \brief A set of non-conforming concentric spherical shells
 *
 * \details The inner spherical shells are decomposed into six wedges
 * surrounding an excised interior region.  The outer spherical shells will use
 * a spherical harmonic basis which cannot be used with subcell.
 *
 * This domain creator offers one grid anchor "Center" at the origin.
 *
 */
class NonconformingSphericalShells final : public DomainCreator<3> {
 public:
  using maps_list =
      tmpl::list<domain::CoordinateMap<
                     Frame::BlockLogical, Frame::Inertial,
                     domain::CoordinateMaps::ProductOf2Maps<
                         domain::CoordinateMaps::Affine,
                         domain::CoordinateMaps::Identity<2>>,
                     domain::CoordinateMaps::SphericalToCartesianPfaffian>,
                 domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                       CoordinateMaps::Wedge<3>>>;

  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "Inner radius of the inner wedges."};
  };

  struct InterfaceRadius {
    using type = double;
    static constexpr Options::String help = {
        "Radius of interface between the inner wedges and the outer spherical "
        "shells."};
  };

  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {
        "Outer radius of the outer spherical shell."};
  };

  struct InitialRadialRefinement {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial radial refinement level for both the inner wedges and the "
        "outer spherical shells."};
  };

  struct InitialAngularRefinementOfWedges {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial angular refinement levels of inner wedges."};
  };

  struct InitialNumberOfRadialGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of radial grid points for both the inner wedges and "
        "the outer spherical shells."};
  };

  struct InitialSphericalHarmonicL {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial spherical harmonic resolution specified as the highest "
        "spherical harmonic represented on the grid."};
  };

  struct InitialNumberOfAngularGridPointsOfWedges {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial angular refinement levels of inner wedges."};
  };

  template <typename BoundaryConditionsBase>
  struct InnerBoundaryCondition {
    static constexpr Options::String help =
        "Options for the boundary conditions at the inner radius.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  template <typename BoundaryConditionsBase>
  struct OuterBoundaryCondition {
    static constexpr Options::String help =
        "Options for the boundary conditions at the outer radius.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  using basic_options =
      tmpl::list<InnerRadius, InterfaceRadius, OuterRadius,
                 InitialRadialRefinement, InitialAngularRefinementOfWedges,
                 InitialNumberOfRadialGridPoints, InitialSphericalHarmonicL,
                 InitialNumberOfAngularGridPointsOfWedges>;

  template <typename Metavariables>
  using options = tmpl::conditional_t<
      domain::BoundaryConditions::has_boundary_conditions_base_v<
          typename Metavariables::system>,
      tmpl::push_back<
          basic_options,
          InnerBoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>,
          OuterBoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>>,
      basic_options>;

  static constexpr Options::String help{
      "A set of concentric spherical shells centered at the origin."};

  NonconformingSphericalShells(
      double inner_radius, double interface_radius, double outer_radius,
      size_t initial_radial_refinement,
      size_t initial_angular_refinement,
      size_t initial_number_of_radial_grid_points,
      size_t initial_spherical_harmonic_l,
      size_t initial_number_of_angular_grid_points_of_wedges,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          inner_boundary_condition = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          outer_boundary_condition = nullptr,
      const Options::Context& context = {});

  NonconformingSphericalShells() = default;
  NonconformingSphericalShells(const NonconformingSphericalShells&) = delete;
  NonconformingSphericalShells(NonconformingSphericalShells&&) = default;
  NonconformingSphericalShells& operator=(const NonconformingSphericalShells&) =
      delete;
  NonconformingSphericalShells& operator=(NonconformingSphericalShells&&) =
      default;
  ~NonconformingSphericalShells() override = default;

  Domain<3> create_domain() const override;

  std::unordered_map<std::string, tnsr::I<double, 3, Frame::Grid>>
  grid_anchors() const override;

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override;

  std::vector<std::string> block_names() const override;

  std::vector<std::array<size_t, 3>> initial_extents() const override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const override;
 private:
  Domain<3> build_domain(const Options::Context& context) const;
  double inner_radius_{};
  double interface_radius_{};
  double outer_radius_{};
  size_t initial_radial_refinement_{};
  size_t initial_angular_refinement_{};
  size_t initial_number_of_radial_grid_points_{};
  size_t initial_spherical_harmonic_l_{};
  size_t initial_number_of_angular_grid_points_of_wedges_{};
  std::vector<std::array<size_t, 3>> initial_refinement_levels_{};
  std::vector<std::array<size_t, 3>> initial_number_of_grid_points_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      inner_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_{};
  std::unordered_map<std::string, tnsr::I<double, 3, Frame::Grid>>
      grid_anchors_{};
};
}  // namespace domain::creators
