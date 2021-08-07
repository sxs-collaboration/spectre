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
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
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
template <typename Map1, typename Map2>
class ProductOf2Maps;
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
struct Logical;
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
 *      two compact objects are enclosed in a single cube-shaped grid.
 * - 4: The 10 blocks that form the first outer shell. This layer transitions
 *      back to spherical. The gridpoints are distributed linearly with respect
 *      to radius.
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
 * time-dependent map. These options include `InitialTime` and
 * `InitialExpirationDeltaT`, which specify the initial time and the
 * initial updating time interval, respectively, for the FunctionsOfTime
 * controlling the map. The time-dependent map itself consists of a composition
 * of a CubicScale expansion map and a Rotation map about the z axis everywhere
 * except possibly in layer 1; in that case, if `ObjectA` or `ObjectB` is
 * excised, then the time-dependent map in the corresponding blocks in
 * layer 1 is a composition of a SphericalCompression size map, a CubicScale
 * expansion map, and a Rotation map about the z axis.
 */
class BinaryCompactObject : public DomainCreator<3> {
 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                            CoordinateMaps::ProductOf3Maps<
                                CoordinateMaps::Affine, CoordinateMaps::Affine,
                                CoordinateMaps::Affine>>,
      domain::CoordinateMap<
          Frame::Logical, Frame::Inertial,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Equiangular,
                                         CoordinateMaps::Equiangular,
                                         CoordinateMaps::Equiangular>>,
      domain::CoordinateMap<
          Frame::Logical, Frame::Inertial,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Affine,
                                         CoordinateMaps::Affine>,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Identity<2>>>,
      domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                            CoordinateMaps::DiscreteRotation<3>,
                            CoordinateMaps::ProductOf3Maps<
                                CoordinateMaps::Affine, CoordinateMaps::Affine,
                                CoordinateMaps::Affine>>,
      domain::CoordinateMap<
          Frame::Logical, Frame::Inertial,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Equiangular,
                                         CoordinateMaps::Equiangular,
                                         CoordinateMaps::Equiangular>>,
      domain::CoordinateMap<
          Frame::Logical, Frame::Inertial,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Equiangular,
                                         CoordinateMaps::Equiangular,
                                         CoordinateMaps::Equiangular>,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Identity<2>>>,
      domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                            CoordinateMaps::Frustum>,
      domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                            CoordinateMaps::Wedge<3>>,
      domain::CoordinateMap<
          Frame::Grid, Frame::Inertial,
          domain::CoordinateMaps::TimeDependent::CubicScale<3>,
          domain::CoordinateMaps::TimeDependent::ProductOf2Maps<
              domain::CoordinateMaps::TimeDependent::Rotation<2>,
              domain::CoordinateMaps::Identity<1>>>,
      domain::CoordinateMap<
          Frame::Grid, Frame::Inertial,
          domain::CoordinateMaps::TimeDependent::SphericalCompression<false>,
          domain::CoordinateMaps::TimeDependent::CubicScale<3>,
          domain::CoordinateMaps::TimeDependent::ProductOf2Maps<
              domain::CoordinateMaps::TimeDependent::Rotation<2>,
              domain::CoordinateMaps::Identity<1>>>>;

  /// Options for an excision region in the domain
  struct Excision {
    static constexpr Options::String help = {
        "Excise the interior of the object, leaving a spherical hole in its "
        "absence."};
    template <typename BoundaryConditionsBase>
    struct BoundaryCondition {
      static std::string name() noexcept {
        return "ExciseWithBoundaryCondition";
      }
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
      static double lower_bound() noexcept { return 0.; }
    };
    struct OuterRadius {
      using type = double;
      static constexpr Options::String help = {
          "Outer coordinate radius of Layer 1"};
      static double lower_bound() noexcept { return 0.; }
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
    struct AdditionToRadialRefinementLevel {
      using type = size_t;
      static constexpr Options::String help = {
          "Addition to radial refinement level in the part of Layer 1 "
          "enveloping the object, beyond the refinement level set by "
          "InitialRefinement."};
    };
    template <typename Metavariables>
    using options = tmpl::list<
        InnerRadius, OuterRadius, XCoord,
        tmpl::conditional_t<
            domain::BoundaryConditions::has_boundary_conditions_base_v<
                typename Metavariables::system>,
            Interior, ExciseInterior>,
        UseLogarithmicMap, AdditionToRadialRefinementLevel>;
    Object() = default;
    Object(double local_inner_radius, double local_outer_radius,
           double local_x_coord, std::optional<Excision> interior,
           bool local_use_logarithmic_map,
           size_t local_addition_to_radial_refinement_level) noexcept
        : inner_radius(local_inner_radius),
          outer_radius(local_outer_radius),
          x_coord(local_x_coord),
          inner_boundary_condition(
              interior.has_value()
                  ? std::make_optional(std::move(interior->boundary_condition))
                  : std::nullopt),
          use_logarithmic_map(local_use_logarithmic_map),
          addition_to_radial_refinement_level(
              local_addition_to_radial_refinement_level) {}
    Object(double local_inner_radius, double local_outer_radius,
           double local_x_coord, bool local_excise_interior,
           bool local_use_logarithmic_map,
           size_t local_addition_to_radial_refinement_level) noexcept
        : inner_radius(local_inner_radius),
          outer_radius(local_outer_radius),
          x_coord(local_x_coord),
          inner_boundary_condition(
              local_excise_interior
                  ? std::optional<std::unique_ptr<
                        domain::BoundaryConditions::BoundaryCondition>>{nullptr}
                  : std::nullopt),
          use_logarithmic_map(local_use_logarithmic_map),
          addition_to_radial_refinement_level(
              local_addition_to_radial_refinement_level) {}

    /// Whether or not the object should be excised from the domain, leaving a
    /// spherical hole. When this is true, `inner_boundary_condition` is
    /// guaranteed to hold a value (though it might be a `nullptr` if we are not
    /// working with boundary conditions).
    bool is_excised() const noexcept;

    double inner_radius;
    double outer_radius;
    double x_coord;
    std::optional<
        std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>
        inner_boundary_condition;
    bool use_logarithmic_map;
    size_t addition_to_radial_refinement_level;
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
    static std::string name() noexcept { return "Radius"; }
    using type = double;
    static constexpr Options::String help = {
        "Radius of Layer 3 which circumscribes the Frustums."};
  };

  struct OuterSphere {
    static constexpr Options::String help = {
        "Options for the outer spherical shell."};
  };

  struct RadiusOuterSphere {
    using group = OuterSphere;
    static std::string name() noexcept { return "Radius"; }
    using type = double;
    static constexpr Options::String help = {"Radius of the entire domain."};
  };

  struct InitialRefinement {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial refinement level. Applied to each dimension."};
  };

  struct InitialGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of grid points in each dim per element."};
  };

  struct UseProjectiveMap {
    using type = bool;
    static constexpr Options::String help = {
        "Use projective scaling on the frustal cloak."};
  };

  struct UseLogarithmicMapOuterSphericalShell {
    using group = OuterSphere;
    static std::string name() noexcept { return "UseLogarithmicMap"; }
    using type = bool;
    static constexpr Options::String help = {
        "Use a logarithmically spaced radial grid in Layer 5, the outer "
        "spherical shell that covers the wave zone."};
  };

  struct AdditionToOuterLayerRadialRefinementLevel {
    using group = OuterSphere;
    static std::string name() noexcept {
      return "AdditionToRadialRefinementLevel";
    }
    using type = size_t;
    static constexpr Options::String help = {
        "Addition to radial refinement level in Layer 5 (the outer spherical "
        "shell that covers that wave zone), beyond the refinement "
        "level set by InitialRefinement."};
  };

  template <typename BoundaryConditionsBase>
  struct OuterBoundaryCondition {
    using group = OuterSphere;
    static std::string name() noexcept { return "BoundaryCondition"; }
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
  /// \brief The initial time interval for updates of the functions of time.
  struct InitialExpirationDeltaT {
    using type = Options::Auto<double>;
    static constexpr Options::String help = {
        "The initial time interval for updates of the functions of time. If "
        "Auto, then the functions of time do not expire, nor can they be "
        "updated."};
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
    static std::string name() noexcept { return "OuterBoundary"; }
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
  /// \brief The name of the function of time to be added to the DataBox for
  /// the expansion map.
  struct ExpansionFunctionOfTimeName {
    using type = std::string;
    static constexpr Options::String help = {"Names of the functions of time."};
    using group = ExpansionMap;
    static std::string name() noexcept { return "FunctionOfTimeName"; }
  };

  struct RotationAboutZAxisMap {
    static constexpr Options::String help = {
        "Options for a time-dependent rotation map about the z axis"};
    using group = TimeDependentMaps;
  };
  /// \brief The initial value of the rotation angle.
  struct InitialRotationAngle {
    using type = double;
    static constexpr Options::String help = {"Rotation angle at initial time."};
    using group = RotationAboutZAxisMap;
  };
  /// \brief The angular velocity of the rotation.
  struct InitialAngularVelocity {
    using type = double;
    static constexpr Options::String help = {"The angular velocity."};
    using group = RotationAboutZAxisMap;
  };
  /// \brief The name of the function of time to be added to the added to the
  /// DataBox for the rotation-about-the-z-axis map.
  struct RotationAboutZAxisFunctionOfTimeName {
    using type = std::string;
    static constexpr Options::String help = {"Name of the function of time."};
    using group = RotationAboutZAxisMap;
    static std::string name() noexcept { return "FunctionOfTimeName"; }
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
    static std::string name() noexcept { return "InitialValues"; }
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
    static std::string name() noexcept { return "InitialVelocities"; }
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
    static std::string name() noexcept { return "InitialAccelerations"; }
  };
  /// \brief The names of the functions of times to be added to the added to the
  /// DataBox for the size map.
  ///
  /// \details If object A is not excised, no size map is applied for object A,
  /// and this option is ignored for object A. If object B is not excised, no
  /// size map is applied for object B, and this option is ignored for object B.
  /// If neither object A nor object B are excised, this option is completely
  /// ignored.
  struct SizeMapFunctionOfTimeNames {
    using type = std::array<std::string, 2>;
    static constexpr Options::String help = {
        "Names of SizeMapA, SizeMapB functions of time."};
    using group = SizeMap;
    static std::string name() noexcept { return "FunctionOfTimeNames"; }
  };

  template <typename Metavariables>
  using time_independent_options = tmpl::append<
      tmpl::list<ObjectA, ObjectB, RadiusEnvelopingCube, RadiusOuterSphere,
                 InitialRefinement, InitialGridPoints, UseProjectiveMap,
                 UseLogarithmicMapOuterSphericalShell,
                 AdditionToOuterLayerRadialRefinementLevel>,
      tmpl::conditional_t<
          domain::BoundaryConditions::has_boundary_conditions_base_v<
              typename Metavariables::system>,
          tmpl::list<OuterBoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>>,
          tmpl::list<>>>;

  using time_dependent_options =
      tmpl::list<InitialTime, InitialExpirationDeltaT,
                 ExpansionMapOuterBoundary, InitialExpansion,
                 InitialExpansionVelocity, ExpansionFunctionOfTimeName,
                 AsymptoticVelocityOuterBoundary,
                 DecayTimescaleOuterBoundaryVelocity, InitialRotationAngle,
                 InitialAngularVelocity, RotationAboutZAxisFunctionOfTimeName,
                 InitialSizeMapValues, InitialSizeMapVelocities,
                 InitialSizeMapAccelerations, SizeMapFunctionOfTimeNames>;

  template <typename Metavariables>
  using options = tmpl::conditional_t<
      BinaryCompactObject_detail::enable_time_dependent_maps_v<Metavariables>,
      tmpl::append<time_dependent_options,
                   time_independent_options<Metavariables>>,
      time_independent_options<Metavariables>>;

  static constexpr Options::String help{
      "The BinaryCompactObject domain is a general domain for two compact "
      "objects. The user must provide the inner and outer radii of the "
      "spherical shells surrounding each of the two compact objects A and "
      "B. The radial refinement levels for these shells are (InitialRefinement "
      "+ Object{A,B}.AdditionToRadialRefinementLevel).\n\n"
      "The user must also provide the radius of the sphere that "
      "circumscribes the cube containing both compact objects, and the "
      "radius of the outer boundary. The options Object{A,B}.Interior (or "
      "Object{A,B}.ExciseInterior if we're not working with boundary "
      "conditions) determine whether the layer-zero blocks are present "
      "inside each compact object. If set to a boundary condition or 'false', "
      "the domain will not contain layer zero for that object. The user "
      "specifies Object{A,B}.XCoord, the x-coordinates of the locations of the "
      "centers of each compact object. In these coordinates, the location for "
      "the axis of rotation is x=0. ObjectA is located on the left and ObjectB "
      "is located on the right. Please make sure that your choices of "
      "x-coordinate locations are such that the resulting center of mass "
      "is located at zero.\n\n"
      "Two radial layers join the enveloping cube to the spherical outer "
      "boundary. The first of these layers transitions from sphericity == 0.0 "
      "on the inner boundary to sphericity == 1.0 on the outer boundary. The "
      "second has sphericity == 1 (so either linear or logarithmic mapping can "
      "be used in the radial direction), extends to the spherical outer "
      "boundary of the domain, and has a radial refinement level of "
      "(InitialRefinement + OuterSphere.AdditionToRadialRefinementLevel)."
      "Note that the domain optionally includes time-dependent maps; enabling "
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
      double radius_enveloping_sphere, size_t initial_refinement,
      size_t initial_grid_points_per_dim, bool use_projective_map = true,
      bool use_logarithmic_map_outer_spherical_shell = false,
      size_t addition_to_outer_layer_radial_refinement_level = 0,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          outer_boundary_condition = nullptr,
      const Options::Context& context = {});

  // Constructor for time-dependent version of the domain
  // (i.e., for when
  // Metavariables::domain::enable_time_dependent_maps == true),
  // with parameters corresponding to the additional options
  BinaryCompactObject(
      double initial_time, std::optional<double> initial_expiration_delta_t,
      double expansion_map_outer_boundary, double initial_expansion,
      double initial_expansion_velocity,
      std::string expansion_function_of_time_name,
      double asymptotic_velocity_outer_boundary,
      double decay_timescale_outer_boundary_velocity,
      double initial_rotation_angle, double initial_angular_velocity,
      std::string rotation_about_z_axis_function_of_time_name,
      std::array<double, 2> initial_size_map_values,
      std::array<double, 2> initial_size_map_velocities,
      std::array<double, 2> initial_size_map_accelerations,
      std::array<std::string, 2> size_map_function_of_time_names,
      Object object_A, Object object_B, double radius_enveloping_cube,
      double radius_enveloping_sphere, size_t initial_refinement,
      size_t initial_grid_points_per_dim, bool use_projective_map = true,
      bool use_logarithmic_map_outer_spherical_shell = false,
      size_t addition_to_outer_layer_radial_refinement_level = 0,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          outer_boundary_condition = nullptr,
      const Options::Context& context = {});

  BinaryCompactObject() = default;
  BinaryCompactObject(const BinaryCompactObject&) = delete;
  BinaryCompactObject(BinaryCompactObject&&) noexcept = default;
  BinaryCompactObject& operator=(const BinaryCompactObject&) = delete;
  BinaryCompactObject& operator=(BinaryCompactObject&&) noexcept = default;
  ~BinaryCompactObject() noexcept override = default;

  Domain<3> create_domain() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const
      noexcept override;

  auto functions_of_time() const noexcept -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  void check_for_parse_errors(const Options::Context& context) const;
  void initialize_calculated_member_variables() noexcept;

  Object object_A_{};
  Object object_B_{};
  double radius_enveloping_cube_{};
  double radius_enveloping_sphere_{};
  size_t initial_refinement_{};
  size_t initial_grid_points_per_dim_{};
  static constexpr bool use_equiangular_map_ =
      false;  // Doesn't work properly yet
  bool use_projective_map_ = true;
  bool use_logarithmic_map_outer_spherical_shell_ = false;
  size_t addition_to_outer_layer_radial_refinement_level_{};
  double projective_scale_factor_{};
  double translation_{};
  double length_inner_cube_{};
  double length_outer_cube_{};
  size_t number_of_blocks_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_;

  // Variables for FunctionsOfTime options
  bool enable_time_dependence_;
  double initial_time_;
  std::optional<double> initial_expiration_delta_t_;
  double expansion_map_outer_boundary_;
  double initial_expansion_;
  double initial_expansion_velocity_;
  std::string expansion_function_of_time_name_;
  double asymptotic_velocity_outer_boundary_;
  double decay_timescale_outer_boundary_velocity_;
  double initial_rotation_angle_;
  double initial_angular_velocity_;
  std::string rotation_about_z_axis_function_of_time_name_;
  std::array<double, 2> initial_size_map_values_;
  std::array<double, 2> initial_size_map_velocities_;
  std::array<double, 2> initial_size_map_accelerations_;
  std::array<std::string, 2> size_map_function_of_time_names_;
};
}  // namespace creators
}  // namespace domain
