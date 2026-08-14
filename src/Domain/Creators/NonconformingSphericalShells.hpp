// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/CoordinateMaps/BulgedCube.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Creators/TimeDependentOptions/Sphere.hpp"
#include "Options/Auto.hpp"
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
 * surrounding an optionally excised interior region.  The outer spherical
 * shells will use a spherical harmonic basis which cannot be used with subcell.
 *
 * This domain creator offers one grid anchor "Center" at the origin.
 *
 */
class NonconformingSphericalShells : public DomainCreator<3> {
 private:
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3D =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>;
  using BulgedCube = CoordinateMaps::BulgedCube;

 public:
  using maps_list = tmpl::list<
      // Inner cube
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial, BulgedCube>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial, Affine3D>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            Equiangular3D>,
      // Wedges
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::Wedge<3>>,
      // Spherical shells
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial,
          domain::CoordinateMaps::ProductOf2Maps<
              domain::CoordinateMaps::Interval,
              domain::CoordinateMaps::Identity<2>>,
          domain::CoordinateMaps::SphericalToCartesianPfaffian>,
      typename sphere::TimeDependentMapOptions::maps_list>;

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

  using Excision = detail::Excision;
  using InnerCube = detail::InnerCube;

  struct Interior {
    using type = std::variant<Excision, InnerCube>;
    static constexpr Options::String help = {
        "Specify 'ExciseWithBoundaryCondition' and a boundary condition to "
        "excise the interior of the sphere, leaving a spherical shell "
        "(or just 'Excise' if boundary conditions are disabled). "
        "Or specify 'FillWithSphericity' to fill the interior."};
  };

  struct InitialCubeRefinement {
    using type =
        std::variant<std::array<size_t, 2>, std::vector<std::array<size_t, 2>>,
                     std::unordered_map<std::string, std::array<size_t, 2>>>;
    static constexpr Options::String help = {
        "Initial cube refinement level. Specify one of: a "
        "list representing [angular, r], or such a list for every block "
        "in the domain. The central cube always uses the angular value for all "
        "directions."};
  };

  struct InitialSHRefinement {
    using type = std::variant<size_t, std::vector<size_t>,
                              std::unordered_map<std::string, size_t>>;
    static constexpr Options::String help = {
        "Initial spherical harmonic shell radial refinement level. Specify one "
        "of: a single number, or such a number for every block in the domain."};
  };

  struct InitialCubeGridPoints {
    using type =
        std::variant<std::array<size_t, 2>, std::vector<std::array<size_t, 2>>,
                     std::unordered_map<std::string, std::array<size_t, 2>>>;
    static constexpr Options::String help = {
        "Initial number of grid points for the cube region. Specify one of: a "
        "list representing [angular, r], or such a list for every block "
        "in the domain. The central cube always uses the angular value for all "
        "directions."};
  };

  struct InitialSHGridPoints {
    using type =
        std::variant<std::array<size_t, 2>, std::vector<std::array<size_t, 2>>,
                     std::unordered_map<std::string, std::array<size_t, 2>>>;
    static constexpr Options::String help = {
        "Initial number of grid points for the spherical harmonic shells. "
        "Specify one of: a list representing [l_max, r], or such a list for "
        "every block in the domain."};
  };

  struct RadialPartitioning {
    using type = std::array<std::vector<double>, 2>;
    static constexpr Options::String help = {
        "Radial coordinates of the boundaries splitting the spherical shell "
        "between InnerRadius and InterfaceRadius and then the InterfaceRadius "
        "and OuterRadius. They must be given in ascending order."};
  };

  struct RadialDistribution {
    using type =
        std::array<std::vector<domain::CoordinateMaps::Distribution>, 2>;
    static constexpr Options::String help = {
        "Select the radial distribution of grid points in each spherical "
        "shell. There must be N+1 radial distributions specified for N radial "
        "partitions for both the wedges and spherical shells. If the interior "
        "of the sphere is filled with a cube, the innermost shell must have a "
        "'Linear' distribution because it changes in sphericity."};
  };

  struct UseEquiangularMap {
    using type = bool;
    static constexpr Options::String help = {
        "Use equiangular instead of equidistant coordinates. Equiangular "
        "coordinates give better gridpoint spacings in the angular "
        "directions, while equidistant coordinates give better gridpoint "
        "spacings in the inner cube."};
  };

  using TimeDepOptionType = std::variant<
      sphere::TimeDependentMapOptions,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>>;

  struct TimeDependentMaps {
    using type = Options::Auto<TimeDepOptionType, Options::AutoLabel::None>;
    static constexpr Options::String help = {
        "The options for time dependent maps. This can either be a "
        "TimeDependence or hard coded time dependent options. Specify `None` "
        "for no time dependent maps."};
  };

  template <typename BoundaryConditionsBase>
  struct OuterBoundaryCondition {
    static constexpr Options::String help =
        "Options for the boundary conditions at the outer radius.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  using basic_options =
      tmpl::list<InnerRadius, InterfaceRadius, OuterRadius, Interior,
                 InitialCubeRefinement, InitialSHRefinement,
                 InitialCubeGridPoints, InitialSHGridPoints, RadialPartitioning,
                 RadialDistribution, UseEquiangularMap, TimeDependentMaps>;

  template <typename Metavariables>
  using options = tmpl::conditional_t<
      domain::BoundaryConditions::has_boundary_conditions_base_v<
          typename Metavariables::system>,
      tmpl::push_back<
          basic_options,
          OuterBoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>>,
      basic_options>;

  static constexpr Options::String help{
      "A set of concentric spherical shells centered at the origin."};

  NonconformingSphericalShells(
      double inner_radius, double interface_radius, double outer_radius,
      std::variant<Excision, InnerCube> interior,
      const typename InitialCubeRefinement::type& initial_cube_refinement,
      const typename InitialSHRefinement::type& initial_sh_refinement,
      const typename InitialCubeGridPoints::type& initial_cube_grid_points,
      const typename InitialSHGridPoints::type& initial_sh_grid_points,
      std::array<std::vector<double>, 2> radial_partitioning = {},
      std::array<std::vector<domain::CoordinateMaps::Distribution>, 2>
          radial_distribution =
              {std::vector<domain::CoordinateMaps::Distribution>{
                   domain::CoordinateMaps::Distribution::Linear},
               std::vector<domain::CoordinateMaps::Distribution>{
                   domain::CoordinateMaps::Distribution::Linear}},
      bool use_equiangular_map = true,
      std::optional<TimeDepOptionType> time_dependent_options = std::nullopt,
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

  std::vector<std::string> block_names() const override { return block_names_; }

  std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const override {
    return block_groups_;
  }

  std::vector<std::array<size_t, 3>> initial_extents() const override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const override;

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  double inner_radius_{};
  double interface_radius_{};
  double outer_radius_{};
  std::variant<Excision, InnerCube> interior_{};
  bool fill_interior_ = false;
  std::vector<std::array<size_t, 2>> initial_cube_refinement_{};
  std::vector<size_t> initial_sh_refinement_{};
  std::vector<std::array<size_t, 2>> initial_cube_grid_points_{};
  std::vector<std::array<size_t, 2>> initial_sh_grid_points_{};
  std::array<std::vector<double>, 2> radial_partitioning_;
  std::array<std::vector<domain::CoordinateMaps::Distribution>, 2>
      radial_distribution_;
  bool use_equiangular_map_ = false;
  std::optional<TimeDepOptionType> time_dependent_options_{};
  bool use_hard_coded_maps_{false};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_{};
  std::vector<std::string> block_names_;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_;
  size_t num_blocks_{};
  size_t num_cube_shells_{};
  size_t num_sh_shells_{};
  std::unordered_map<std::string, tnsr::I<double, 3, Frame::Grid>>
      grid_anchors_{};
};
}  // namespace domain::creators
