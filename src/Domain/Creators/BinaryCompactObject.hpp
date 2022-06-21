// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class Affine;
class Equiangular;
template <size_t VolumeDim>
class Identity;
template <typename Map1, typename Map2>
class ProductOf2Maps;
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
template <size_t Dim>
class Wedge;
template <size_t VolumeDim>
class DiscreteRotation;
class Frustum;
namespace TimeDependent {
template <size_t VolumeDim>
class CubicScale;
template <size_t VolumeDim>
class Rotation;
template <bool InteriorMap>
class SphericalCompression;
}  // namespace TimeDependent
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;

namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
}  // namespace domain

namespace Frame {
struct Inertial;
struct BlockLogical;
}  // namespace Frame

namespace BinaryCompactObject_detail {
// If `Metavariables` has a `domain_parameters` member struct and
// `domain_parameters::enable_time_dependent_maps` is `true`, then
// inherit from `std::true_type`; otherwise, inherit from `std::false_type`.
template <typename Metavariables, typename = std::void_t<>>
struct enable_time_dependent_maps : std::false_type {};

template <typename Metavariables>
struct enable_time_dependent_maps<Metavariables,
                                  std::void_t<typename Metavariables::domain>>
    : std::bool_constant<Metavariables::domain::enable_time_dependent_maps> {};

template <typename Metavariables>
constexpr bool enable_time_dependent_maps_v =
    enable_time_dependent_maps<Metavariables>::value;
}  // namespace BinaryCompactObject_detail
/// \endcond

namespace domain {
namespace creators {

/*!
 * \ingroup ComputationalDomainGroup
 *
 * \brief A general domain for two compact objects.
 *
 * \image html binary_compact_object_domain.png "A BHNS domain."
 *
 * Creates a 3D Domain that represents a binary compact object solution. The
 * Domain consists of 4, 5, or 6 nested layers of blocks; these layers are,
 * working from the interior toward the exterior:
 * - 0: (optionally) The block at the center of each compact object, if not
 *      excised. If present, this block is a cube. If excised, the hole left
 *      by its absence is spherical.
 * - 1: The blocks that resolve each individual compact object. This layer has
 *      a spherical outer boundary - if the corresponding layer-0 block exists,
 *      then the layer is a cube-to-sphere transition; if the layer-0 block is
 *      excised, then the layer is a spherical shell.
 * - 2: The blocks that surround each object with a cube. Around each compact
 *      object, this layer transitions from a sphere to a cube.
 * - 3: The blocks that surround each cube with a half-cube. At this layer, the
 *      two compact objects are enclosed in a single cube-shaped grid. This
 *      layer can have a spherical outer shape by setting the "frustum
 *      sphericity" to one.
 * - 4: The 10 blocks that form the first outer shell. This layer transitions
 *      back to spherical. The gridpoints are distributed linearly with respect
 *      to radius. This layer can be omitted if the "frustum sphericity" is one,
 *      so layer 3 is already spherical.
 * - 5: The 10 blocks that form a second outer shell. This layer is
 *      spherical, so a logarithmic map can optionally be used in this layer.
 *      This allows the domain to extend to large radial distances from the
 *      compact objects. This layer can be h-refined radially,
 *      creating a layer of multiple concentric spherical shells.
 *
 * In the code and options below, `ObjectA` and `ObjectB` refer to the two
 * compact objects, and by extension, also refer to the layers that immediately
 * surround each compact object. Note that `ObjectA` is located to the left of
 * the origin (along the negative x-axis) and `ObjectB` is located to the right
 * of the origin. `enveloping cube` refers to the outer surface of Layer 3.
 * `outer sphere` is the radius of the spherical outer boundary, which is
 * the outer boundary of Layer 5. The `enveloping cube` and `outer sphere`
 * are both centered at the origin. `cutting plane` refers to the plane along
 * which the domain divides into two hemispheres. In the final coordinates, the
 * cutting plane always intersects the x-axis at the origin.
 *
 * \note The x-coordinate locations of the `ObjectA` and `ObjectB` should be
 * chosen such that the center of mass is located at x=0.
 *
 * \note When using this domain, the
 * metavariables struct can contain a struct named `domain`
 * that conforms to domain::protocols::Metavariables. If
 * domain::enable_time_dependent_maps is either set to `false`
 * or not specified in the metavariables, then this domain will be
 * time-independent. If domain::enable_time_dependent_maps is set
 * to `true`, then this domain also includes a time-dependent map, along with
 * additional options (and a corresponding constructor) for initializing the
 * time-dependent map. These options include the `InitialTime` which specifies
 * the initial time for the FunctionsOfTime controlling the map. The
 * time-dependent map itself consists of a composition of a CubicScale expansion
 * map and a Rotation map everywhere except possibly in layer 1; in that case,
 * if `ObjectA` or `ObjectB` is excised, then the time-dependent map in the
 * corresponding blocks in layer 1 is a composition of a SphericalCompression
 * size map, a CubicScale expansion map, and a Rotation map.
 */
class BinaryCompactObject : public DomainCreator<3> {
 private:
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Identity2D = CoordinateMaps::Identity<2>;
  using Translation = CoordinateMaps::ProductOf2Maps<Affine, Identity2D>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3D =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>;

 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial, Affine3D>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            Equiangular3D>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial, Affine3D,
                            Translation>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::DiscreteRotation<3>, Affine3D>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            Equiangular3D>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial, Equiangular3D,
                            Translation>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::Frustum>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::Wedge<3>>,
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::Wedge<3>, Translation>,
      domain::CoordinateMap<
          Frame::Grid, Frame::Inertial,
          domain::CoordinateMaps::TimeDependent::CubicScale<3>,
          domain::CoordinateMaps::TimeDependent::Rotation<3>>,
      domain::CoordinateMap<
          Frame::Grid, Frame::Inertial,
          domain::CoordinateMaps::TimeDependent::SphericalCompression<false>,
          domain::CoordinateMaps::TimeDependent::CubicScale<3>,
          domain::CoordinateMaps::TimeDependent::Rotation<3>>>;

  /// Options for an excision region in the domain
  struct Excision {
    static constexpr Options::String help = {
        "Excise the interior of the object, leaving a spherical hole in its "
        "absence."};
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
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        boundary_condition;
  };

  /// Options for one of the two objects in the binary domain
  struct Object {
    static constexpr Options::String help = {
        "Options for an object in a binary domain."};
    struct InnerRadius {
      using type = double;
      static constexpr Options::String help = {
          "Inner coordinate radius of Layer 1."};
      static double lower_bound() { return 0.; }
    };
    struct OuterRadius {
      using type = double;
      static constexpr Options::String help = {
          "Outer coordinate radius of Layer 1"};
      static double lower_bound() { return 0.; }
    };
    struct XCoord {
      using type = double;
      static constexpr Options::String help = {"x-coordinate of center."};
    };
    struct Interior {
      using type = Options::Auto<Excision>;
      static constexpr Options::String help = {
          "Specify 'ExciseWithBoundaryCondition' and a boundary condition to "
          "excise Layer 0, leaving a spherical hole in its absence, or set to "
          "'Auto' to fill the interior."};
    };
    struct ExciseInterior {
      using type = bool;
      static constexpr Options::String help = {
          "Excise Layer 0, leaving a spherical hole in its absence."};
    };
    struct UseLogarithmicMap {
      using type = bool;
      static constexpr Options::String help = {
          "Use a logarithmically spaced radial grid in the part of Layer 1 "
          "enveloping the object (requires the interior is excised)"};
    };
    template <typename Metavariables>
    using options = tmpl::list<
        InnerRadius, OuterRadius, XCoord,
        tmpl::conditional_t<
            domain::BoundaryConditions::has_boundary_conditions_base_v<
                typename Metavariables::system>,
            Interior, ExciseInterior>,
        UseLogarithmicMap>;
    Object() = default;
    Object(double local_inner_radius, double local_outer_radius,
           double local_x_coord, std::optional<Excision> interior,
           bool local_use_logarithmic_map)
        : inner_radius(local_inner_radius),
          outer_radius(local_outer_radius),
          x_coord(local_x_coord),
          inner_boundary_condition(
              interior.has_value()
                  ? std::make_optional(std::move(interior->boundary_condition))
                  : std::nullopt),
          use_logarithmic_map(local_use_logarithmic_map) {}
    Object(double local_inner_radius, double local_outer_radius,
           double local_x_coord, bool local_excise_interior,
           bool local_use_logarithmic_map)
        : inner_radius(local_inner_radius),
          outer_radius(local_outer_radius),
          x_coord(local_x_coord),
          inner_boundary_condition(
              local_excise_interior
                  ? std::optional<std::unique_ptr<
                        domain::BoundaryConditions::BoundaryCondition>>{nullptr}
                  : std::nullopt),
          use_logarithmic_map(local_use_logarithmic_map) {}

    /// Whether or not the object should be excised from the domain, leaving a
    /// spherical hole. When this is true, `inner_boundary_condition` is
    /// guaranteed to hold a value (though it might be a `nullptr` if we are not
    /// working with boundary conditions).
    bool is_excised() const;

    double inner_radius;
    double outer_radius;
    double x_coord;
    std::optional<
        std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>
        inner_boundary_condition;
    bool use_logarithmic_map;
  };

  struct ObjectA {
    using type = Object;
    static constexpr Options::String help = {
        "Options for the object to the left of the origin (along the negative "
        "x-axis)."};
  };

  struct ObjectB {
    using type = Object;
    static constexpr Options::String help = {
        "Options for the object to the right of the origin (along the positive "
        "x-axis)."};
  };

  struct EnvelopingCube {
    static constexpr Options::String help = {
        "Options for the cube enveloping the two objects."};
  };

  struct RadiusEnvelopingCube {
    using group = EnvelopingCube;
    static std::string name() { return "Radius"; }
    using type = double;
    static constexpr Options::String help = {
        "Radius of Layer 3 which circumscribes the Frustums."};
  };

  struct OuterShell {
    static constexpr Options::String help = {
        "Options for the outer spherical shell."};
  };

  struct OuterRadius {
    using group = OuterShell;
    using type = double;
    static constexpr Options::String help = {"Radius of the entire domain."};
  };

  struct RadiusEnvelopingSphere {
    using group = OuterShell;
    static std::string name() { return "InnerRadius"; }
    using type = Options::Auto<double>;
    static constexpr Options::String help = {
        "Inner radius of the outer spherical shell. Set to 'Auto' to compute a "
        "reasonable value automatically based on the "
        "'OuterShell.RadialDistribution', or to omit the layer of blocks "
        "altogether when EnvelopingCube.Sphericity is 1 and hence the "
        "cube-to-sphere transition is not needed."};
  };

  struct InitialRefinement {
    using type =
        std::variant<size_t, std::array<size_t, 3>,
                     std::vector<std::array<size_t, 3>>,
                     std::unordered_map<std::string, std::array<size_t, 3>>>;
    static constexpr Options::String help = {
        "Initial refinement level in each block of the domain. See main help "
        "text for details."};
  };

  struct InitialGridPoints {
    using type =
        std::variant<size_t, std::array<size_t, 3>,
                     std::vector<std::array<size_t, 3>>,
                     std::unordered_map<std::string, std::array<size_t, 3>>>;
    static constexpr Options::String help = {
        "Initial number of grid points in the elements of each block of the "
        "domain. See main help text for details."};
  };

  struct UseProjectiveMap {
    using group = EnvelopingCube;
    using type = bool;
    static constexpr Options::String help = {
        "Use projective scaling on the frustal cloak."};
  };

  struct FrustumSphericity {
    using group = EnvelopingCube;
    static std::string name() { return "Sphericity"; }
    using type = double;
    static constexpr Options::String help = {
        "Sphericity of the enveloping cube. The value 0.0 corresponds "
        "to a cubical envelope of frustums, the value 1.0 corresponds "
        "to a spherical envelope of frustums."};
    static double lower_bound() { return 0.; }
    static double upper_bound() { return 1.; }
    // Suggest spherical frustums to encourage upgrading, but keep supporting
    // cubical frustums until spherical frustums are sufficiently battle-tested
    static double suggested_value() { return 1.; }
  };

  struct RadialDistributionOuterShell {
    using group = OuterShell;
    static std::string name() { return "RadialDistribution"; }
    using type = CoordinateMaps::Distribution;
    static constexpr Options::String help = {
        "The distribution of radial grid points in Layer 5, the outer "
        "spherical shell that covers the wave zone."};
  };

  struct RadiusAdditionalOuterShell {
    using group = OuterShell;
    static std::string name() { return "RadiusAdditionalOuterShell"; }
    using type = double;
    static constexpr Options::String help = {
        "Radius of an additional layer of Blocks beyond Layer 5."};
  };

  struct RadialDistributionAdditionalOuterShell {
    using group = OuterShell;
    static std::string name() {
      return "RadialDistributionAdditionalOuterShell";
    }
    using type = CoordinateMaps::Distribution;
    static constexpr Options::String help = {
        "The distribution of radial grid points in Layer 6, the additional "
        "outer spherical shell that covers the wave zone."};
  };

  template <typename BoundaryConditionsBase>
  struct OuterBoundaryCondition {
    using group = OuterShell;
    static std::string name() { return "BoundaryCondition"; }
    static constexpr Options::String help =
        "Options for the outer boundary conditions.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  // The following options are for optional time dependent maps
  struct TimeDependentMaps {
    static constexpr Options::String help = {"Options for time-dependent maps"};
  };

  /// \brief The initial time of the functions of time.
  struct InitialTime {
    using type = double;
    static constexpr Options::String help = {
        "The initial time of the functions of time"};
    using group = TimeDependentMaps;
  };

  struct ExpansionMap {
    static constexpr Options::String help = {
        "Options for a time-dependent expansion map (specifically, a "
        "CubicScale map)"};
    using group = TimeDependentMaps;
  };

  /// \brief The outer boundary or pivot point of the
  /// `domain::CoordinateMaps::TimeDependent::CubicScale` map
  struct ExpansionMapOuterBoundary {
    using type = double;
    static constexpr Options::String help = {
        "Outer boundary or pivot point of the map"};
    using group = ExpansionMap;
    static std::string name() { return "OuterBoundary"; }
  };
  /// \brief The initial value of the expansion factor.
  struct InitialExpansion {
    using type = double;
    static constexpr Options::String help = {
        "Expansion value at initial time."};
    using group = ExpansionMap;
  };
  /// \brief The velocity of the expansion factor.
  struct InitialExpansionVelocity {
    using type = double;
    static constexpr Options::String help = {"The rate of expansion."};
    using group = ExpansionMap;
  };
  /// \brief The asymptotic radial velocity of the outer boundary.
  struct AsymptoticVelocityOuterBoundary {
    using type = double;
    static constexpr Options::String help = {
        "The asymptotic velocity of the outer boundary."};
    using group = ExpansionMap;
  };
  /// \brief The timescale for how fast the outer boundary velocity approaches
  /// its asymptotic value.
  struct DecayTimescaleOuterBoundaryVelocity {
    using type = double;
    static constexpr Options::String help = {
        "The timescale for how fast the outer boundary velocity approaches its "
        "asymptotic value."};
    using group = ExpansionMap;
  };

  struct RotationMap {
    static constexpr Options::String help = {
        "Options for a time-dependent rotation map about an arbitrary axis."};
    using group = TimeDependentMaps;
  };
  /// \brief The angular velocity of the rotation.
  struct InitialAngularVelocity {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {"The angular velocity."};
    using group = RotationMap;
  };

  struct SizeMap {
    static constexpr Options::String help = {
        "Options for a time-dependent size maps."};
    using group = TimeDependentMaps;
  };

  /// \brief Initial values for functions of time for size maps for objects A,B.
  ///
  /// \details If object A is not excised, no size map is applied for object A,
  /// and this option is ignored for object A. If object B is not excised, no
  /// size map is applied for object B, and this option is ignored for object B.
  /// If neither object A nor object B are excised, this option is completely
  /// ignored.
  struct InitialSizeMapValues {
    using type = std::array<double, 2>;
    static constexpr Options::String help = {
        "SizeMapA, SizeMapB values at initial time."};
    using group = SizeMap;
    static std::string name() { return "InitialValues"; }
  };
  /// \brief Initial velocities for functions of time for size maps for objects
  /// A,B.
  ///
  /// \details If object A is not excised, no size map is applied for object A,
  /// and this option is ignored for object A. If object B is not excised, no
  /// size map is applied for object B, and this option is ignored for object B.
  /// If neither object A nor object B are excised, this option is completely
  /// ignored.
  struct InitialSizeMapVelocities {
    using type = std::array<double, 2>;
    static constexpr Options::String help = {
        "SizeMapA, SizeMapB initial velocities."};
    using group = SizeMap;
    static std::string name() { return "InitialVelocities"; }
  };
  /// \brief Initial accelerations for functions of time for size maps for
  /// objects A,B
  ///
  /// \details If object A is not excised, no size map is applied for object A,
  /// and this option is ignored for object A. If object B is not excised, no
  /// size map is applied for object B, and this option is ignored for object B.
  /// If neither object A nor object B are excised, this option is completely
  /// ignored.
  struct InitialSizeMapAccelerations {
    using type = std::array<double, 2>;
    static constexpr Options::String help = {
        "SizeMapA, SizeMapB initial accelerations."};
    using group = SizeMap;
    static std::string name() { return "InitialAccelerations"; }
  };

  template <typename Metavariables>
  using time_independent_options = tmpl::append<
      tmpl::list<ObjectA, ObjectB, RadiusEnvelopingCube, OuterRadius,
                 InitialRefinement, InitialGridPoints, UseProjectiveMap,
                 FrustumSphericity, RadiusEnvelopingSphere,
                 RadialDistributionOuterShell, RadiusAdditionalOuterShell,
                 RadialDistributionAdditionalOuterShell>,
      tmpl::conditional_t<
          domain::BoundaryConditions::has_boundary_conditions_base_v<
              typename Metavariables::system>,
          tmpl::list<OuterBoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>>,
          tmpl::list<>>>;

  using time_dependent_options =
      tmpl::list<InitialTime, ExpansionMapOuterBoundary, InitialExpansion,
                 InitialExpansionVelocity, AsymptoticVelocityOuterBoundary,
                 DecayTimescaleOuterBoundaryVelocity, InitialAngularVelocity,
                 InitialSizeMapValues, InitialSizeMapVelocities,
                 InitialSizeMapAccelerations>;

  template <typename Metavariables>
  using options = tmpl::conditional_t<
      BinaryCompactObject_detail::enable_time_dependent_maps_v<Metavariables>,
      tmpl::append<time_dependent_options,
                   time_independent_options<Metavariables>>,
      time_independent_options<Metavariables>>;

  static constexpr Options::String help{
      "The BinaryCompactObject domain is a general domain for two compact "
      "objects. The user must provide the inner and outer radii of the "
      "spherical shells surrounding each of the two compact objects A and B "
      "(\"ObjectAShell\" and \"ObjectBShell\"). Each object is enveloped in "
      "a cube (\"ObjectACube\" and \"ObjectBCube\")."
      "The user must also provide the radius of the sphere that circumscribes "
      "the cube containing both compact objects (\"EnvelopingCube\"). "
      "A radial layer transitions from the enveloping cube to a sphere "
      "(\"CubedShell\"). A final radial layer transitions to the outer "
      "boundary (\"OuterShell\"). The options Object{A,B}.Interior (or "
      "Object{A,B}.ExciseInterior if we're not working with boundary "
      "conditions) determine whether blocks are present inside each compact "
      "object (\"ObjectAInterior\" and \"ObjectBInterior\"). If set to a "
      "boundary condition or 'false', the region will be excised. The user "
      "specifies Object{A,B}.XCoord, the x-coordinates of the locations of the "
      "centers of each compact object. In these coordinates, the location for "
      "the axis of rotation is x=0. ObjectA is located on the left and ObjectB "
      "is located on the right. Please make sure that your choices of "
      "x-coordinate locations are such that the resulting center of mass "
      "is located at zero.\n"
      "\n"
      "Both the InitialRefinement and the InitialGridPoints can be one of "
      "the following:\n"
      "  - A single number: Uniform refinement in all blocks and "
      "dimensions\n"
      "  - Three numbers: Refinement in [polar, azimuthal, radial] direction "
      "in all blocks\n"
      "  - A map from block names or groups to three numbers: Per-block "
      "refinement in [polar, azimuthal, radial] direction\n"
      "  - A list, with [polar, azimuthal, radial] refinement for each block\n"
      "\n"
      "The domain optionally includes time-dependent maps. Enabling "
      "the time-dependent maps requires adding a "
      "struct named domain to the Metavariables, with this "
      "struct conforming to domain::protocols::Metavariables. To enable the "
      "time-dependent maps, set "
      "Metavariables::domain::enable_time_dependent_maps to "
      "true."};

  // Constructor for time-independent version of the domain
  // (i.e., for when
  // Metavariables::domain::enable_time_dependent_maps == false or
  // when the metavariables do not define
  // Metavariables::domain::enable_time_dependent_maps)
  BinaryCompactObject(
      Object object_A, Object object_B, double radius_enveloping_cube,
      double outer_radius_domain,
      const typename InitialRefinement::type& initial_refinement,
      const typename InitialGridPoints::type& initial_number_of_grid_points,
      bool use_projective_map = true, double frustum_sphericity = 0.0,
      const std::optional<double>& radius_enveloping_sphere = std::nullopt,
      CoordinateMaps::Distribution radial_distribution_outer_shell =
          CoordinateMaps::Distribution::Linear,
      double radius_add_outer_shell = 0.0,
      CoordinateMaps::Distribution radial_distribution_additional_outer_shell =
          CoordinateMaps::Distribution::Inverse,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          outer_boundary_condition = nullptr,
      const Options::Context& context = {});

  // Constructor for time-dependent version of the domain
  // (i.e., for when
  // Metavariables::domain::enable_time_dependent_maps == true),
  // with parameters corresponding to the additional options
  BinaryCompactObject(
      double initial_time, double expansion_map_outer_boundary,
      double initial_expansion, double initial_expansion_velocity,
      double asymptotic_velocity_outer_boundary,
      double decay_timescale_outer_boundary_velocity,
      std::array<double, 3> initial_angular_velocity,
      std::array<double, 2> initial_size_map_values,
      std::array<double, 2> initial_size_map_velocities,
      std::array<double, 2> initial_size_map_accelerations, Object object_A,
      Object object_B, double radius_enveloping_cube,
      double outer_radius_domain,
      const typename InitialRefinement::type& initial_refinement,
      const typename InitialGridPoints::type& initial_number_of_grid_points,
      bool use_projective_map = true, double frustum_sphericity = 0.0,
      const std::optional<double>& radius_enveloping_sphere = std::nullopt,
      CoordinateMaps::Distribution radial_distribution_outer_shell =
          CoordinateMaps::Distribution::Linear,
      double radius_add_outer_shell = 0.0,
      CoordinateMaps::Distribution radial_distribution_additional_outer_shell =
          CoordinateMaps::Distribution::Inverse,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          outer_boundary_condition = nullptr,
      const Options::Context& context = {});

  BinaryCompactObject() = default;
  BinaryCompactObject(const BinaryCompactObject&) = delete;
  BinaryCompactObject(BinaryCompactObject&&) = default;
  BinaryCompactObject& operator=(const BinaryCompactObject&) = delete;
  BinaryCompactObject& operator=(BinaryCompactObject&&) = default;
  ~BinaryCompactObject() override = default;

  Domain<3> create_domain() const override;

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
  Object object_A_{};
  Object object_B_{};
  bool need_cube_to_sphere_transition_{};
  double radius_enveloping_cube_{};
  double radius_enveloping_sphere_{};
  double outer_radius_domain_{};
  std::vector<std::array<size_t, 3>> initial_refinement_{};
  std::vector<std::array<size_t, 3>> initial_number_of_grid_points_{};
  static constexpr bool use_equiangular_map_ =
      false;  // Doesn't work properly yet
  bool use_projective_map_ = true;
  double frustum_sphericity_{};
  CoordinateMaps::Distribution radial_distribution_outer_shell_ =
      CoordinateMaps::Distribution::Linear;
  double projective_scale_factor_{};
  double translation_{};
  double length_inner_cube_{};
  double length_outer_cube_{};
  size_t number_of_blocks_{};
  double radius_add_outer_shell_ = 0.0;
  CoordinateMaps::Distribution radial_distribution_additional_outer_shell_ =
      CoordinateMaps::Distribution::Inverse;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_;
  std::vector<std::string> block_names_{};
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_{};

  // Variables for FunctionsOfTime options
  bool enable_time_dependence_{false};
  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  double expansion_map_outer_boundary_{
      std::numeric_limits<double>::signaling_NaN()};
  double initial_expansion_{std::numeric_limits<double>::signaling_NaN()};
  double initial_expansion_velocity_{
      std::numeric_limits<double>::signaling_NaN()};
  inline static const std::string expansion_function_of_time_name_{"Expansion"};
  double asymptotic_velocity_outer_boundary_{
      std::numeric_limits<double>::signaling_NaN()};
  double decay_timescale_outer_boundary_velocity_{
      std::numeric_limits<double>::signaling_NaN()};
  DataVector initial_angular_velocity_{3, 0.0};
  DataVector initial_quaternion_{4, 0.0};
  inline static const std::string rotation_function_of_time_name_{"Rotation"};
  std::array<double, 2> initial_size_map_values_{
      std::numeric_limits<double>::signaling_NaN(),
      std::numeric_limits<double>::signaling_NaN()};
  std::array<double, 2> initial_size_map_velocities_{
      std::numeric_limits<double>::signaling_NaN(),
      std::numeric_limits<double>::signaling_NaN()};
  std::array<double, 2> initial_size_map_accelerations_{
      std::numeric_limits<double>::signaling_NaN(),
      std::numeric_limits<double>::signaling_NaN()};
  inline static const std::array<std::string, 2>
      size_map_function_of_time_names_{{"SizeA", "SizeB"}};
};
}  // namespace creators
}  // namespace domain
