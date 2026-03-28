// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Creators/TimeDependentOptions/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim, typename T>
class DirectionMap;
namespace domain {
namespace CoordinateMaps {
template <size_t Dim>
class Identity;
class Interval;
template <typename Map1, typename Map2>
class ProductOf2Maps;
class SphericalToCartesianPfaffian;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain::creators {
/// \brief A set of concentric spherical shells
///
/// \note This domain will use a spherical harmonic basis in the angular
/// directions.  It cannot be used with subcell.
///
/// \see Sphere for a spherical domain compatible with subcell
///
/// This domain creator offers one grid anchor "Center" at the origin.
///
/// #### Time dependent maps
/// There are two ways to add time dependent maps to the SphericalShells domain
/// creator. In the input file, these are specified under the
/// `TimeDependentMaps:` block.
///
/// ##### TimeDependence
/// You can use a simple TimeDependence (e.g.
/// `domain::creators::time_dependence::UniformTranslation` or
/// `domain::creators::time_dependence::RotationAboutZAxis`) to add time
/// dependent maps. This method will add the same maps to all blocks in the
/// domain.
///
/// ##### Hard-coded time dependent maps
/// The SphericalShells domain creator also has the option to use some hard
/// coded time dependent maps that may be useful in certain scenarios. This
/// method adds the maps in `domain::creators::sphere::TimeDependentMapOptions`
/// to the domain. Currently, the first (inner-most) shell has maps between
/// `Frame::Grid`, `Frame::Distorted`, and `Frame::Inertial` while all
/// subsequent shells only have maps between `Frame::Grid` and
/// `Frame::Inertial`.
///
/// ##### None
/// To not have any time dependent maps, pass a `std::nullopt` as the
/// appropriate argument in the constructor. In the input file, simply have
/// `TimeDependentMaps: None`.
class SphericalShells final : public DomainCreator<3> {
 public:
  using maps_list =
      tmpl::append<tmpl::list<domain::CoordinateMap<
                       Frame::BlockLogical, Frame::Inertial,
                       domain::CoordinateMaps::ProductOf2Maps<
                           domain::CoordinateMaps::Interval,
                           domain::CoordinateMaps::Identity<2>>,
                       domain::CoordinateMaps::SphericalToCartesianPfaffian>>,
                   typename sphere::TimeDependentMapOptions::maps_list>;

  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "Inner radius of the spherical shells."};
  };

  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {
        "Outer radius of the spherical shells."};
  };

  struct InitialRadialRefinement {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial radial refinement level."};
  };

  struct InitialNumberOfRadialGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of radial grid points."};
  };

  struct InitialSphericalHarmonicL {
    using type = size_t;
    static size_t lower_bound() { return 6; }
    static constexpr Options::String help = {
        "Initial spherical harmonic resolution specified as the highest "
        "spherical harmonic represented on the grid.  Minimum value is 6."};
  };

  struct RadialPartitioning {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "Radial coordinates of the boundaries splitting the spherical shell "
        "between InnerRadius and OuterRadius. They must be given in ascending "
        "order. This should be used if boundaries need to be set at specific "
        "radii. If the number but not the specific locations of the boundaries "
        "are important, use InitialRefinement instead."};
  };

  struct RadialDistribution {
    using type =
        std::variant<domain::CoordinateMaps::Distribution,
                     std::vector<domain::CoordinateMaps::Distribution>>;
    static constexpr Options::String help = {
        "Select the radial distribution of grid points in each spherical "
        "shell. There must be N+1 radial distributions specified for N radial "
        "partitions. You can also specify just a single radial distribution "
        "(not in a vector) which will use the same distribution for all "
        "partitions."};
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
      tmpl::list<InnerRadius, OuterRadius, InitialRadialRefinement,
                 InitialNumberOfRadialGridPoints, InitialSphericalHarmonicL,
                 RadialPartitioning, RadialDistribution, TimeDependentMaps>;

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

  SphericalShells(
      double inner_radius, double outer_radius,
      size_t initial_radial_refinement,
      size_t initial_number_of_radial_grid_points,
      size_t initial_spherical_harmonic_l,
      std::vector<double> radial_partitioning = {},
      const typename RadialDistribution::type& radial_distribution =
          domain::CoordinateMaps::Distribution::Linear,
      std::optional<TimeDepOptionType> time_dependent_options = std::nullopt,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          inner_boundary_condition = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          outer_boundary_condition = nullptr,
      const Options::Context& context = {});

  SphericalShells() = default;
  SphericalShells(const SphericalShells&) = delete;
  SphericalShells(SphericalShells&&) = default;
  SphericalShells& operator=(const SphericalShells&) = delete;
  SphericalShells& operator=(SphericalShells&&) = default;
  ~SphericalShells() override = default;

  Domain<3> create_domain() const override;

  /// A single grid anchor "Center" at the origin.
  std::unordered_map<std::string, tnsr::I<double, 3, Frame::Grid>>
  grid_anchors() const override;

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override;

  std::vector<std::array<size_t, 3>> initial_extents() const override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const override;

  /// The block names are Shell0, Shell1, ..., starting with the innermost
  /// Block.
  std::vector<std::string> block_names() const override;

  /// The block groups are Shell0, Shell1, ..., starting with the innermost
  /// Block.
  std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const override;

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  Domain<3> build_domain(const Options::Context& context) const;
  Domain<3> domain_{};
  double inner_radius_{};
  double outer_radius_{};
  size_t initial_radial_refinement_{};
  size_t initial_number_of_radial_grid_points_{};
  size_t initial_spherical_harmonic_l_{};
  std::vector<double> radial_partitioning_{};
  std::vector<domain::CoordinateMaps::Distribution> radial_distribution_{};
  std::optional<TimeDepOptionType> time_dependent_options_{};
  bool use_hard_coded_maps_{false};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      inner_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_{};
  size_t num_blocks_{};
  std::vector<std::string> block_names_{};
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_{};
  std::unordered_map<std::string, tnsr::I<double, 3, Frame::Grid>>
      grid_anchors_{};
};
}  // namespace domain::creators
