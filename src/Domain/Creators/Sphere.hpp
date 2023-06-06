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
#include "Domain/Creators/SphereTimeDependentMaps.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/Options.hpp"
#include "Options/ParseError.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class Affine;
class BulgedCube;
class EquatorialCompression;
class Equiangular;
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
template <size_t Dim>
class Wedge;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain::creators::detail {

/// Options for excising the interior of the sphere. This class parses as the
/// `ExcisionFromOptions` subclass if boundary conditions are enabled, and as a
/// plain string if boundary conditions are disabled.
struct Excision {
  Excision() = default;
  Excision(std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
               boundary_condition);
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition = nullptr;
};

struct ExcisionFromOptions : Excision {
  static constexpr Options::String help = {
      "Excise the interior of the sphere, leaving a spherical shell."};
  template <typename BoundaryConditionsBase>
  struct BoundaryCondition {
    static std::string name() { return "ExciseWithBoundaryCondition"; }
    using type = std::unique_ptr<BoundaryConditionsBase>;
    static constexpr Options::String help = {
        "The boundary condition to impose on the excision surface."};
  };
  template <typename Metavariables>
  using options = tmpl::list<BoundaryCondition<
      domain::BoundaryConditions::get_boundary_conditions_base<
          typename Metavariables::system>>>;
  using Excision::Excision;
};

/// Options for filling the interior of the sphere with a cube
struct InnerCube {
  static constexpr Options::String help = {
      "Fill the interior of the sphere with a cube."};
  struct Sphericity {
    static std::string name() { return "FillWithSphericity"; }
    using type = double;
    static constexpr Options::String help = {
        "Sphericity of the inner cube. A sphericity of 0 uses a product "
        "of 1D maps as the map in the center. A sphericity > 0 uses a "
        "BulgedCube. A sphericity of exactly 1 is not allowed. See "
        "BulgedCube docs for why."};
    static double lower_bound() { return 0.0; }
    static double upper_bound() { return 1.0; }
  };
  using options = tmpl::list<Sphericity>;
  double sphericity = std::numeric_limits<double>::signaling_NaN();
};

}  // namespace domain::creators::detail

template <>
struct Options::create_from_yaml<domain::creators::detail::Excision> {
  template <typename Metavariables>
  static domain::creators::detail::Excision create(
      const Options::Option& options) {
    if constexpr (domain::BoundaryConditions::has_boundary_conditions_base_v<
                      typename Metavariables::system>) {
      // Boundary conditions are enabled. Parse with a nested option.
      return options.parse_as<domain::creators::detail::ExcisionFromOptions,
                              Metavariables>();
    } else {
      // Boundary conditions are disabled. Parse as a plain string.
      if (options.parse_as<std::string>() == "Excise") {
        return domain::creators::detail::Excision{};
      } else {
        PARSE_ERROR(options.context(), "Parse error");
      }
    }
  }
};

namespace domain::creators {

/*!
 * \brief A 3D cubed sphere.
 *
 * Six wedges surround an interior region, which is either excised or filled in
 * with a seventh block. The interior region is a (possibly deformed) sphere
 * when excised, or a (possibly deformed) cube when filled in. Additional
 * spherical shells, each composed of six wedges, can be added with the
 * 'RadialPartitioning' option.
 *
 * \image html WedgeOrientations.png "The orientation of each wedge in a cubed
 * sphere."
 *
 * #### Inner cube sphericity
 * The inner cube is a BulgedCube except if the inner cube sphericity is
 * exactly 0. Then an Equiangular or Affine map is used (depending on if it's
 * equiangular or not) to avoid a root find in the BulgedCube map.
 *
 * #### Time dependent maps
 * There are two ways to add time dependent maps to the Sphere domain
 * creator. In the input file, these are specified under the
 * `TimeDependentMaps:` block.
 *
 * ##### TimeDependence
 * You can use a simple TimeDependence (e.g.
 * `domain::creators::time_dependence::UniformTranslation` or
 * `domain::creators::time_dependence::RotationAboutZAxis`) to add time
 * dependent maps. This method will add the same maps to all blocks in the
 * domain. This method can be used with an inner cube or with an excision
 * surface.
 *
 * ##### Hard-coded time dependent maps
 * The Sphere domain creator also has the option to use some hard coded time
 * dependent maps that may be useful in certain scenarios. This method adds the
 * maps in `domain::creators::sphere::TimeDependentMapOptions` to the domain.
 * Currently, the first (inner-most) shell has maps between `Frame::Grid`,
 * `Frame::Distorted`, and `Frame::Inertial` while all subsequent shells only
 * have maps between `Frame::Grid` and `Frame::Inertial`.
 *
 * \note You can only use hard-coded time dependent maps if you have an excision
 * surface. You cannot have a inner cube.
 *
 * ##### None
 * To not have any time dependent maps, pass a `std::nullopt` to appropriate
 * argument in the constructor. In the input file, simple have
 * `TimeDependentMaps: None`.
 *
 */
class Sphere : public DomainCreator<3> {
 private:
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3D =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>;
  using BulgedCube = CoordinateMaps::BulgedCube;

 public:
  using maps_list = tmpl::append<
      tmpl::list<
          // Inner cube
          domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                BulgedCube>,
          domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial, Affine3D>,
          domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                Equiangular3D>,
          // Wedges
          domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                CoordinateMaps::Wedge<3>>,
          domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                CoordinateMaps::Wedge<3>,
                                CoordinateMaps::EquatorialCompression>>,
      typename sphere::TimeDependentMapOptions::maps_list>;

  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "Radius circumscribing the inner cube or the excision."};
  };

  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {"Radius of the sphere."};
  };

  using Excision = detail::Excision;
  using InnerCube = detail::InnerCube;

  struct Interior {
    using type = std::variant<Excision, InnerCube>;
    static constexpr Options::String help = {
        "Specify 'ExciseWithBoundaryCondition' and a boundary condition to "
        "excise the interior of the sphere, leaving a spherical shell "
        "(or just 'Excise' if boundary conditions are disabled). "
        "Or specify 'CubeWithSphericity' to fill the interior."};
  };

  struct InitialRefinement {
    using type =
        std::variant<size_t, std::array<size_t, 3>,
                     std::vector<std::array<size_t, 3>>,
                     std::unordered_map<std::string, std::array<size_t, 3>>>;
    static constexpr Options::String help = {
        "Initial refinement level. Specify one of: a single number, a "
        "list representing [phi, theta, r], or such a list for every block "
        "in the domain. The central cube always uses the value for 'theta' "
        "in both y- and z-direction."};
  };

  struct InitialGridPoints {
    using type =
        std::variant<size_t, std::array<size_t, 3>,
                     std::vector<std::array<size_t, 3>>,
                     std::unordered_map<std::string, std::array<size_t, 3>>>;
    static constexpr Options::String help = {
        "Initial number of grid points. Specify one of: a single number, a "
        "list representing [phi, theta, r], or such a list for every block "
        "in the domain. The central cube always uses the value for 'theta' "
        "in both y- and z-direction."};
  };

  struct UseEquiangularMap {
    using type = bool;
    static constexpr Options::String help = {
        "Use equiangular instead of equidistant coordinates. Equiangular "
        "coordinates give better gridpoint spacings in the angular "
        "directions, while equidistant coordinates give better gridpoint "
        "spacings in the inner cube."};
  };

  /// Options for the EquatorialCompression map
  struct EquatorialCompressionOptions {
    static constexpr Options::String help = {
        "Options for the EquatorialCompression map."};
    struct AspectRatio {
      using type = double;
      static constexpr Options::String help = {
          "An aspect ratio greater than 1 moves grid points toward the "
          "equator, and an aspect ratio smaller than 1 moves grid points "
          "toward the poles."};
      static double lower_bound() { return 0.0; }
    };
    struct IndexPolarAxis {
      using type = size_t;
      static constexpr Options::String help = {
          "The index (0, 1, or 2) of the axis along which equatorial "
          "compression is applied, where 0 is x, 1 is y, and 2 is z."};
      static size_t upper_bound() { return 2; }
    };
    using options = tmpl::list<AspectRatio, IndexPolarAxis>;

    double aspect_ratio;
    size_t index_polar_axis;
  };

  struct EquatorialCompression {
    using type =
        Options::Auto<EquatorialCompressionOptions, Options::AutoLabel::None>;
    static constexpr Options::String help = {
        "Apply an equatorial compression map to focus resolution on the "
        "equator or on the poles. The equatorial compression is an angular "
        "redistribution of grid points and will preserve the spherical shape "
        "of the inner and outer boundaries."};
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
        "partitions. If the interior of the sphere is filled with a cube, the "
        "innermost shell must have a 'Linear' distribution because it changes "
        "in sphericity. You can also specify just a single radial distribution "
        "(not in a vector) which will use the same distribution for all "
        "partitions."};
  };

  struct WhichWedges {
    using type = ShellWedges;
    static constexpr Options::String help = {
        "Which wedges to include in the shell."};
    static constexpr type suggested_value() { return ShellWedges::All; }
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
      tmpl::list<InnerRadius, OuterRadius, Interior, InitialRefinement,
                 InitialGridPoints, UseEquiangularMap, EquatorialCompression,
                 RadialPartitioning, RadialDistribution, WhichWedges,
                 TimeDependentMaps>;

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
      "A 3D cubed sphere. Six wedges surround an interior region, which is "
      "either excised or filled in with a seventh block. The interior region "
      "is a (possibly deformed) sphere when excised, or a (possibly deformed) "
      "cube when filled in. Additional spherical shells, each composed of six "
      "wedges, can be added with the 'RadialPartitioning' option."};

  Sphere(
      double inner_radius, double outer_radius,
      std::variant<Excision, InnerCube> interior,
      const typename InitialRefinement::type& initial_refinement,
      const typename InitialGridPoints::type& initial_number_of_grid_points,
      bool use_equiangular_map,
      std::optional<EquatorialCompressionOptions> equatorial_compression = {},
      std::vector<double> radial_partitioning = {},
      const typename RadialDistribution::type& radial_distribution =
          domain::CoordinateMaps::Distribution::Linear,
      ShellWedges which_wedges = ShellWedges::All,
      std::optional<TimeDepOptionType> time_dependent_options = std::nullopt,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          outer_boundary_condition = nullptr,
      const Options::Context& context = {});

  Sphere() = default;
  Sphere(const Sphere&) = delete;
  Sphere(Sphere&&) = default;
  Sphere& operator=(const Sphere&) = delete;
  Sphere& operator=(Sphere&&) = default;
  ~Sphere() override = default;

  Domain<3> create_domain() const override;

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override;

  std::vector<std::array<size_t, 3>> initial_extents() const override {
    return initial_number_of_grid_points_;
  }

  std::vector<std::array<size_t, 3>> initial_refinement_levels()
      const override {
    return initial_refinement_;
  }

  std::vector<std::string> block_names() const override { return block_names_; }

  std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const override {
    return block_groups_;
  }

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  double inner_radius_{};
  double outer_radius_{};
  std::variant<Excision, InnerCube> interior_{};
  bool fill_interior_ = false;
  std::vector<std::array<size_t, 3>> initial_refinement_{};
  std::vector<std::array<size_t, 3>> initial_number_of_grid_points_{};
  bool use_equiangular_map_ = false;
  std::optional<EquatorialCompressionOptions> equatorial_compression_{};
  std::vector<double> radial_partitioning_{};
  std::vector<domain::CoordinateMaps::Distribution> radial_distribution_{};
  ShellWedges which_wedges_ = ShellWedges::All;
  std::optional<TimeDepOptionType> time_dependent_options_{};
  bool use_hard_coded_maps_{false};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_;
  size_t num_shells_;
  size_t num_blocks_;
  size_t num_blocks_per_shell_;
  std::vector<std::string> block_names_{};
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_{};
};

}  // namespace domain::creators
