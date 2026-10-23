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
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CylindricalFlatEndcapInterior.hpp"
#include "Domain/CoordinateMaps/CylindricalSphericalShell.hpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/FlatOffsetWedge.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/PolarToCartesian.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"
#include "Domain/CoordinateMaps/UniformCylindricalSide.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime

namespace Frame {
struct Grid;
struct Distorted;
struct Inertial;
struct BlockLogical;
}  // namespace Frame
/// \endcond

namespace domain::creators {

/*!
 * \ingroup ComputationalDomainGroup
 *
 * \brief A domain for a binary of two filled (non-excised) objects, such as
 * two neutron stars.
 *
 * \details The `XAxisPoints` option gives four \f$x\f$-coordinates
 * [\f$x_{\rm left}\f$, \f$x_B\f$, \f$x_A\f$, \f$x_{\rm right}\f$]. If doing
 * subcell, ideally these would be equally spaced, with the object centers lying
 * on the \f$x\f$-axis at \f$x_A\f$ and \f$x_B\f$, with \f$0 < x_A \le
 * |x_B|\f$ and \f$x_B < 0\f$, and the cubed cylinders extend from \f$x_{\rm
 * left} < x_B\f$ to \f$x_{\rm right} > x_A\f$. The domain has three layers,
 * from the inside out:
 *
 * 1. **Cubed cylinders.** Four groups of five blocks each surround the
 *    \f$x\f$-axis out to `WedgeOuterRadius` (see `Bulge` below for the shape
 *    of their outer faces). From left to right, the groups (`BLeft`,
 *    `BRight`, `ALeft`, `ARight`) span \f$x\f$ in
 *    [\f$x_{\rm left}\f$, \f$x_B\f$], [\f$x_B\f$, 0], [0, \f$x_A\f$], and
 *    [\f$x_A\f$, \f$x_{\rm right}\f$]. Each group
 *    has a central cube with half-width `WedgeInnerRadius`\f$/\sqrt{2}\f$ in
 *    \f$y\f$ and \f$z\f$ (so its edges lie at radius `WedgeInnerRadius` from
 *    the \f$x\f$-axis) and four deformed-cube wedges around it.
 * 2. **Cylinders.** A hollow cylinder surrounds each of the four cubed-cylinder
 *    groups, and a filled cylinder (with a ZernikeB2 cross-section) caps each
 *    end, at \f$x = x_{\rm left}\f$ and \f$x = x_{\rm right}\f$. The outer
 *    faces of these six blocks together form a sphere of radius
 *    `CylinderOuterRadius` centered at the origin.
 * 3. **Spherical shells.** One or more spherical shells extend from
 *    `CylinderOuterRadius` to `OuterRadius`. The number of shells is set by
 *    `SphericalShellsRadialPartitioning`.
 *
 * The interfaces between the cubed cylinders and the cylinders, and between
 * the cylinders and the innermost spherical shell, are non-conforming.
 *
 * The `Bulge` option selects the maps for the deformed-cube wedges and the
 * hollow cylinders. When `false`, the wedges are extruded 2D wedges, so their
 * outer faces lie on a circular cylinder of radius `WedgeOuterRadius`. When
 * `true`, the wedges use FlatOffsetWedge maps whose outer faces lie on spheres
 * centered on each object. The sphere around object B has radius
 * `WedgeOuterRadius`, and the radius of the sphere around object A is chosen
 * so that the two spheres meet the \f$x = 0\f$ plane in the same circle.
 *
 * Time-dependent maps (expansion, rotation, and translation) are supported
 * through `TimeDependentMaps`. Shape maps are not supported, because there are
 * no excision surfaces. The time-dependent maps are a `RotScaleTrans` map
 * centered at the origin. All blocks inside `CylinderOuterRadius` are mapped
 * rigidly. In the spherical shells the expansion transitions to
 * `ExpansionOuterBoundary` and the translation falls off to zero at
 * `OuterRadius`, while the rotation is applied everywhere.
 */
class BinaryNeutronStars : public DomainCreator<3> {
 public:
  using maps_list = tmpl::flatten<tmpl::list<
      // Central cubes (blocks 0, 5, 10, 15): Affine3D
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::ProductOf3Maps<
                                CoordinateMaps::Affine, CoordinateMaps::Affine,
                                CoordinateMaps::Affine>>,
      // Bulge=false: Wedge prisms (blocks 1-4, 6-9, 11-14, 16-19):
      // Wedge<2> x Affine + rotation to final orientation
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Wedge<2>,
                                         CoordinateMaps::Affine>,
          CoordinateMaps::DiscreteRotation<3>>,
      // Bulge=true: ALeft/BLeft FOW blocks: FOW + ShiftX + Rotation
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial, CoordinateMaps::FlatOffsetWedge,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Identity<1>,
                                         CoordinateMaps::Identity<1>>,
          CoordinateMaps::DiscreteRotation<3>>,
      // Bulge=true: BRight FOW blocks: FOW + Rotation
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::FlatOffsetWedge,
                            CoordinateMaps::DiscreteRotation<3>>,
      // Bulge=true: ARight FOW blocks:
      // FOW + Rotation + ShiftX + Rotation
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial, CoordinateMaps::FlatOffsetWedge,
          CoordinateMaps::DiscreteRotation<3>,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Identity<1>,
                                         CoordinateMaps::Identity<1>>,
          CoordinateMaps::DiscreteRotation<3>>,
      // Filled cylinders (blocks 20, 25): radial Affine +
      // polar-to-Cartesian + half-turn + interior endcap + rotation to
      // x-axis + rotation to align angle
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Identity<1>,
                                         CoordinateMaps::Identity<1>>,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::PolarToCartesian,
                                         CoordinateMaps::Identity<1>>,
          CoordinateMaps::DiscreteRotation<3>,
          CoordinateMaps::CylindricalFlatEndcapInterior,
          CoordinateMaps::DiscreteRotation<3>>,
      // Hollow cylinders (blocks 21-24): bulge = false
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::CylindricalSphericalShell,
                            CoordinateMaps::DiscreteRotation<3>>,
      // Hollow cylinders (blocks 21-24): bulge = true
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Identity<1>,
                                         CoordinateMaps::Interval>,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::PolarToCartesian,
                                         CoordinateMaps::Identity<1>>,
          CoordinateMaps::DiscreteRotation<3>,
          CoordinateMaps::UniformCylindricalSide,
          CoordinateMaps::DiscreteRotation<3>>,
      // Spherical shells (blocks 26 and up): radial Interval + S2
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Interval,
                                         CoordinateMaps::Identity<2>>,
          CoordinateMaps::SphericalToCartesianPfaffian>,
      bco::TimeDependentMapOptions<true>::maps_list>>;

  /*!
   * \brief Grid \f$x\f$-coordinates of the ends of the cubed cylinders and the
   * centers of the objects
   *
   * Given in ascending order as [\f$x_{\rm left}\f$, \f$x_B\f$, \f$x_A\f$,
   * \f$x_{\rm right}\f$], where \f$x_{\rm left}\f$ and \f$x_{\rm right}\f$
   * are the left and right faces of the cubed cylinders and \f$x_B\f$ and
   * \f$x_A\f$ are the centers of Objects B and A. Must satisfy
   * \f$x_{\rm left} < x_B < 0 < x_A < x_{\rm right}\f$ and
   * \f$x_A \le |x_B|\f$.
   */
  struct XAxisPoints {
    using type = std::array<double, 4>;
    static constexpr Options::String help = {
        "Grid x-coordinates [LeftEnd, CenterB, CenterA, RightEnd] in ascending "
        "order: the left face of the cubed cylinders, the centers of Objects B "
        "and A, and the right face of the cubed cylinders. Must satisfy "
        "LeftEnd < CenterB < 0 < CenterA < RightEnd and CenterA <= |CenterB|."};
  };
  /*!
   * \brief Grid-coordinate distance from the \f$x\f$-axis to the edges of the
   * central cubes, which is also the inner radius of the deformed-cube wedges
   *
   * The central cubes have half-width WedgeInnerRadius\f$/\sqrt{2}\f$ in
   * \f$y\f$ and \f$z\f$.
   */
  struct WedgeInnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "Grid-coordinate distance from the x-axis to the edges of the central "
        "cubes. The cubes have half-width WedgeInnerRadius/sqrt(2) in y and "
        "z."};
  };
  /*!
   * \brief Grid-coordinate radius of the outer faces of the deformed-cube
   * wedges
   *
   * With `Bulge: false` this is the radius of the cylinder about the
   * \f$x\f$-axis. With `Bulge: true` this is the radius of the sphere
   * centered on Object B; the radius of the sphere centered on Object A is
   * adjusted so that the two spheres meet at \f$x = 0\f$.
   */
  struct WedgeOuterRadius {
    using type = double;
    static constexpr Options::String help = {
        "Grid-coordinate radius of the outer faces of the deformed-cube wedges "
        "around the central cubes. With Bulge=false this is a cylinder about "
        "the x-axis. With Bulge=true this is a sphere centered on Object B "
        "(the sphere around Object A is adjusted to meet it at x=0)."};
  };
  /*!
   * \brief Grid-coordinate radius of the sphere, centered at the origin,
   * formed by the outer faces of the cylinders
   *
   * This is also the inner radius of the innermost spherical shell.
   */
  struct CylinderOuterRadius {
    using type = double;
    static constexpr Options::String help = {
        "Grid-coordinate radius of the sphere, centered at the origin, formed "
        "by the outer faces of the filled and hollow cylinders. This is also "
        "the inner radius of the spherical shells."};
  };
  /*!
   * \brief Grid-coordinate radius of the outer boundary of the domain
   */
  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {
        "Grid-coordinate radius for outer boundary of spherical shells."};
  };

  /*!
   * \brief Initial number of grid points for each cubed-cylinder group
   *
   * One entry per group, ordered [BLeft, BRight, ALeft, ARight], or a single
   * value used for all groups. This sets the grid points in all three
   * directions of the central cube, and in the directions tangent to the outer
   * face of the four surrounding deformed-cube wedges.
   */
  struct CubeInitialGridPoints {
    using type = std::variant<size_t, std::array<size_t, 4>>;
    static constexpr Options::String help = {
        "Initial number of grid points for each cubed-cylinder group, ordered "
        "[BLeft, BRight, ALeft, ARight], or a single value used for all "
        "groups. Sets all directions of the central cube and the directions "
        "tangent to the outer face of the surrounding deformed-cube wedges."};
  };
  /*!
   * \brief Initial \f$x\f$ refinement for each cubed-cylinder group
   *
   * One entry per group, ordered [BLeft, BRight, ALeft, ARight], or a single
   * value used for all groups. This sets the \f$x\f$ refinement of both the
   * central cube and the four surrounding deformed-cube wedges.
   */
  struct CubeInitialXRefinement {
    using type = std::variant<size_t, std::array<size_t, 4>>;
    static constexpr Options::String help = {
        "Initial x refinement for each cubed-cylinder group, ordered [BLeft, "
        "BRight, ALeft, ARight], or a single value used for all groups. "
        "Applies to both the central cube and the surrounding deformed-cube "
        "wedges."};
  };
  /*!
   * \brief Initial \f$y\f$ and \f$z\f$ refinement for each cubed-cylinder
   * group
   *
   * One entry per group, ordered [BLeft, BRight, ALeft, ARight], or a single
   * value used for all groups. This sets the \f$y\f$ and \f$z\f$ refinement
   * of the central cube and the angular refinement of the four surrounding
   * deformed-cube wedges.
   */
  struct CubeInitialYZRefinement {
    using type = std::variant<size_t, std::array<size_t, 4>>;
    static constexpr Options::String help = {
        "Initial y/z refinement for each cubed-cylinder group, ordered [BLeft, "
        "BRight, ALeft, ARight], or a single value used for all groups. "
        "Applies to the central cube and to the angular direction of the "
        "surrounding deformed-cube wedges."};
  };
  /*!
   * \brief Initial number of radial grid points for the deformed-cube wedges
   */
  struct WedgePrismInitialRadialGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of radial grid points for the deformed-cube wedges "
        "surrounding the central cubes."};
  };
  /*!
   * \brief Initial radial refinement for the deformed-cube wedges
   */
  struct WedgePrismInitialRadialRefinement {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial radial refinement for the deformed-cube wedges surrounding "
        "the central cubes."};
  };
  /*!
   * \brief Initial number of grid points for the filled and hollow cylinders
   * in the direction from the cubed cylinders out to `CylinderOuterRadius`
   *
   * For the hollow cylinders this is the radial direction. For the filled
   * cylinders this is the axial direction, from the flat face at
   * \f$x_{\rm left}\f$/\f$x_{\rm right}\f$ (see `XAxisPoints`) to the outer
   * sphere.
   */
  struct CylinderInitialRadialGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of grid points for the cylinders in the direction from "
        "the cubed cylinders out to CylinderOuterRadius (radial for the hollow "
        "cylinders, axial for the filled cylinders)."};
  };
  /*!
   * \brief Initial refinement for the filled and hollow cylinders in the
   * direction from the cubed cylinders out to `CylinderOuterRadius`
   *
   * See `CylinderInitialRadialGridPoints` for which logical direction this is.
   */
  struct CylinderInitialRadialRefinement {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial refinement for the cylinders in the direction from the cubed "
        "cylinders out to CylinderOuterRadius (radial for the hollow "
        "cylinders, axial for the filled cylinders)."};
  };
  /*!
   * \brief Initial number of angular grid points for the ZernikeB2
   * cross-section of the filled cylinders
   *
   * Must be odd. The number of radial grid points in the cross-section is
   * derived from this so that the radial and angular modal spaces match. The
   * cross-section cannot be h-refined.
   */
  struct B2InitialAngularGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of angular grid points for the ZernikeB2 cross-section "
        "of the filled cylinders. Must be odd. The radial grid points of the "
        "cross-section are derived from this."};
  };
  /*!
   * \brief Initial number of grid points for the hollow cylinders in the
   * angular and axial directions
   *
   * Passed as logical [eta, zeta], where eta is the angle about the
   * \f$x\f$-axis (must be odd) and zeta is the axial direction. Neither
   * direction is h-refined.
   */
  struct HollowCylinderInitialAngularGridPoints {
    using type = std::array<size_t, 2>;
    static constexpr Options::String help = {
        "Initial number of grid points for the hollow cylinders, passed as "
        "logical [eta, zeta]: eta is the angle about the x-axis (must be odd) "
        "and zeta is the axial direction."};
  };
  /*!
   * \brief Initial radial refinement of spherical shells
   */
  struct SphericalShellsInitialRadialRefinement {
    using type = std::variant<size_t, std::vector<size_t>>;
    static constexpr Options::String help = {
        "Initial radial refinement for spherical shells. There must be N+1 "
        "refinement levels specified for N radial partitions. You can also "
        "specify just a single refinement level (not in a vector) which will "
        "be used for all shells."};
  };
  /*!
   * \brief Initial number of radial grid points for spherical shells
   */
  struct SphericalShellsInitialRadialGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of radial grid points for spherical shells."};
  };
  /*!
   * \brief Initial spherical harmonic resolution for spherical shells
   */
  struct InitialSphericalHarmonicL {
    using type = size_t;
    static size_t lower_bound() { return 6; }
    static constexpr Options::String help = {
        "Initial spherical harmonic resolution specified as the highest "
        "spherical harmonic represented on the grid.  Minimum value is 6."};
  };
  /*!
   * \brief Radial partitioning of the spherical shell region
   */
  struct SphericalShellsRadialPartitioning {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "Radial coordinates of the boundaries splitting the spherical-shell "
        "region between CylinderOuterRadius and OuterRadius into multiple "
        "shells. Must be given in ascending order, strictly between "
        "CylinderOuterRadius and OuterRadius."};
  };
  /*!
   * \brief Radial distribution for spherical shells
   */
  struct SphericalShellsRadialDistribution {
    using type =
        std::variant<domain::CoordinateMaps::Distribution,
                     std::vector<domain::CoordinateMaps::Distribution>>;
    static constexpr Options::String help = {
        "Select the radial distribution of grid points in each spherical "
        "shell. There must be N+1 radial distributions specified for N "
        "radial partitions. You can also specify just a single radial "
        "distribution (not in a vector) which will use the same "
        "distribution for all shells."};
  };

  /*!
   * \brief Whether to use bulging coordinate maps for the deformed-cube wedges
   * and the hollow cylinders
   *
   * When `true`, uses `FlatOffsetWedge` for the deformed-cube wedges, whose
   * outer faces lie on spheres centered on each object, and
   * `UniformCylindricalSide` for the hollow cylinders. When `false`, uses
   * extruded 2D `Wedge`s, whose outer faces lie on a cylinder about the
   * \f$x\f$-axis, and `CylindricalSphericalShell` for the hollow cylinders.
   */
  struct Bulge {
    using type = bool;
    static constexpr Options::String help = {
        "If true, the outer faces of the deformed-cube wedges lie on spheres "
        "centered on each object (FlatOffsetWedge). If false, they lie on a "
        "cylinder about the x-axis (extruded 2D Wedge)."};
  };

  /*!
   * \brief Outer boundary condition
   */
  template <typename BoundaryConditionsBase>
  struct OuterBoundaryCondition {
    static std::string name() { return "OuterBoundary"; }
    static constexpr Options::String help =
        "Options for the outer boundary conditions.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  /*!
   * \brief Time dependent maps
   */
  struct TimeDependentMaps {
    using type = Options::Auto<bco::TimeDependentMapOptions<true>,
                               Options::AutoLabel::None>;
    static constexpr Options::String help =
        bco::TimeDependentMapOptions<true>::help;
  };

  template <typename Metavariables>
  using options = tmpl::append<
      tmpl::list<
          XAxisPoints, WedgeInnerRadius, WedgeOuterRadius, CylinderOuterRadius,
          OuterRadius, CubeInitialGridPoints, CubeInitialXRefinement,
          CubeInitialYZRefinement, WedgePrismInitialRadialGridPoints,
          WedgePrismInitialRadialRefinement, CylinderInitialRadialGridPoints,
          CylinderInitialRadialRefinement, B2InitialAngularGridPoints,
          HollowCylinderInitialAngularGridPoints,
          SphericalShellsInitialRadialRefinement,
          SphericalShellsInitialRadialGridPoints, InitialSphericalHarmonicL,
          SphericalShellsRadialPartitioning, SphericalShellsRadialDistribution,
          Bulge, TimeDependentMaps>,
      tmpl::conditional_t<
          domain::BoundaryConditions::has_boundary_conditions_base_v<
              typename Metavariables::system>,
          tmpl::list<OuterBoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>>,
          tmpl::list<>>>;

  static constexpr Options::String help{
      "A domain for a binary of two filled (non-excised) objects, such as two "
      "neutron stars, centered on the x-axis. Four cubed cylinders (a central "
      "cube surrounded by four deformed-cube wedges) span the x-axis between "
      "the points in XAxisPoints. These are surrounded by four hollow "
      "cylinders and capped at each end by a filled cylinder, whose outer "
      "faces together form a sphere of radius CylinderOuterRadius. Spherical "
      "shells extend from there to OuterRadius. The Bulge option makes the "
      "outer faces of the deformed-cube wedges spherical instead of "
      "cylindrical."};

  BinaryNeutronStars(
      const std::array<double, 4>& x_axis_points, double wedge_inner_radius,
      double wedge_outer_radius, double cylinder_outer_radius,
      double outer_radius, const CubeInitialGridPoints::type& cube_grid_points,
      const CubeInitialXRefinement::type& cube_x_refinement,
      const CubeInitialYZRefinement::type& cube_yz_refinement,
      size_t wedge_prism_radial_grid_points,
      size_t wedge_prism_radial_refinement, size_t cylinder_radial_grid_points,
      size_t cylinder_radial_refinement, size_t b2_angular_grid_points,
      std::array<size_t, 2> hollow_cylinder_angular_grid_points,
      const SphericalShellsInitialRadialRefinement::type&
          spherical_shells_radial_refinement,
      size_t spherical_shells_radial_grid_points, size_t spherical_harmonic_l,
      std::vector<double> spherical_shells_radial_partitioning,
      const SphericalShellsRadialDistribution::type&
          spherical_shells_radial_distribution,
      bool bulge = false,
      std::optional<bco::TimeDependentMapOptions<true>> time_dependent_options =
          std::nullopt,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          outer_boundary_condition = nullptr,
      const Options::Context& context = {});

  BinaryNeutronStars() = default;
  BinaryNeutronStars(const BinaryNeutronStars&) = delete;
  BinaryNeutronStars(BinaryNeutronStars&&) = default;
  BinaryNeutronStars& operator=(const BinaryNeutronStars&) = delete;
  BinaryNeutronStars& operator=(BinaryNeutronStars&&) = default;
  ~BinaryNeutronStars() override = default;

  Domain<3> create_domain() const override;

  std::unordered_map<std::string, tnsr::I<double, 3, Frame::Grid>>
  grid_anchors() const override {
    return grid_anchors_;
  }

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override;

  std::vector<std::array<size_t, 3>> initial_extents() const override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const override;

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

  /// \brief The block names, which use left/right directionality as from a
  /// viewer looking down the \f$y\f$-axis.
  ///
  /// \details The block names start with the central cubes:
  ///  - `BLeftCubedCylinderCenter`, `BLeftCubedCylinderFront`,
  /// `BLeftCubedCylinderTop`, `BLeftCubedCylinderBack`,
  /// `BLeftCubedCylinderBottom`
  ///  - `BRightCubedCylinderCenter`, `BRightCubedCylinderFront`, ...
  ///  - `ALeftCubedCylinderCenter`, ...
  ///  - `ARightCubedCylinderCenter`, ...
  ///
  /// Then there are the cylinders:
  ///  - `BFilledCylinder`
  ///  - `BLeftHollowCylinder`
  ///  - `BRightHollowCylinder`
  ///  - `ALeftHollowCylinder`
  ///  - `ARightHollowCylinder`
  ///  - `AFilledCylinder`
  ///
  /// Finally, there are the spherical shells: `SphericalShell0`,
  /// `SphericalShell1`, etc., one for each radial partition of the region
  /// between `CylinderOuterRadius` and `OuterRadius`.
  std::vector<std::string> block_names() const override { return block_names_; }

  /// \brief The block groups, which are `CubedCylinders`, `Wedges`,
  /// `Cylinders`, and `SphericalShells`. The `Wedges` group is the subset of
  /// `CubedCylinders` consisting of the deformed cube wedges surrounding the
  /// central cubes.
  std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const override {
    return block_groups_;
  }

 private:
  // Block indices. The 20 cubed-cylinder blocks come first, then the
  // cylinders, then the spherical shells.
  static constexpr size_t b_filled_cylinder_block = 20;
  static constexpr size_t first_hollow_cylinder_block = 21;
  static constexpr size_t a_filled_cylinder_block = 25;
  static constexpr size_t first_spherical_shell_block = 26;

  double leftmost_x_{};
  double center_B_{};
  double center_A_{};
  double rightmost_x_{};
  double wedge_inner_radius_{};
  double wedge_outer_radius_{};
  double cylinder_outer_radius_{};
  double outer_radius_{};
  std::array<size_t, 4> cube_grid_points_{};
  std::array<size_t, 4> cube_x_refinement_{};
  std::array<size_t, 4> cube_yz_refinement_{};
  size_t wedge_prism_radial_grid_points_{};
  size_t wedge_prism_radial_refinement_{};
  size_t cylinder_radial_grid_points_{};
  size_t cylinder_radial_refinement_{};
  size_t b2_angular_grid_points_{};
  std::array<size_t, 2> hollow_cylinder_angular_grid_points_{};
  std::vector<size_t> spherical_shells_radial_refinement_{};
  size_t spherical_shells_radial_grid_points_{};
  size_t spherical_harmonic_l_{};
  std::vector<double> spherical_shells_radial_partitioning_{};
  std::vector<domain::CoordinateMaps::Distribution>
      spherical_shells_radial_distribution_{};
  size_t number_of_spherical_shells_{1};
  bool bulge_{};
  // Radii of the spheres around objects A and B on which the outer faces of
  // the deformed-cube wedges lie with Bulge=true (both are WedgeOuterRadius
  // with Bulge=false)
  double bulge_radius_A_{};
  double bulge_radius_B_{};
  size_t number_of_blocks_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_;
  std::optional<bco::TimeDependentMapOptions<true>> time_dependent_options_{};
  std::vector<std::string> block_names_{};
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_{};
  std::unordered_map<std::string, tnsr::I<double, 3, Frame::Grid>>
      grid_anchors_{};
  std::vector<std::array<size_t, 3>> initial_refinement_{};
  std::vector<std::array<size_t, 3>> initial_grid_points_{};
};
}  // namespace domain::creators
