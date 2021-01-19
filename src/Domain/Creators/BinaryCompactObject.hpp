// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
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
class Wedge3D;
template <size_t VolumeDim>
class DiscreteRotation;
class Frustum;
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
 * `enveloping sphere` is the radius of the spherical outer boundary, which is
 * the outer boundary of Layer 5. The `enveloping cube` and `enveloping sphere`
 * are both centered at the origin. `cutting plane` refers to the plane along
 * which the domain divides into two hemispheres. In the final coordinates, the
 * cutting plane always intersects the x-axis at the origin.
 *
 * \note The x-coordinate locations of the `ObjectA` and `ObjectB` should be
 * chosen such that the center of mass is located at x=0.
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
      domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                            CoordinateMaps::ProductOf3Maps<
                                CoordinateMaps::Equiangular,
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
                            CoordinateMaps::Wedge3D>>;

  struct InnerRadiusObjectA {
    using type = double;
    static constexpr Options::String help = {
        "Inner coordinate radius of Layer 1 for Object A."};
  };

  struct OuterRadiusObjectA {
    using type = double;
    static constexpr Options::String help = {
        "Outer coordinate radius of Layer 1 for Object A."};
  };

  struct XCoordObjectA {
    using type = double;
    static constexpr Options::String help = {
        "x-coordinate of center for Object A."};
  };

  struct ExciseInteriorA {
    using type = bool;
    static constexpr Options::String help = {
        "Exclude Layer 0 for ObjectA. Set to `true` for a BH."};
  };

  struct InnerRadiusObjectB {
    using type = double;
    static constexpr Options::String help = {
        "Inner coordinate radius of Layer 1 for Object B."};
  };

  struct OuterRadiusObjectB {
    using type = double;
    static constexpr Options::String help = {
        "Outer coordinate radius of Layer 1 for Object B."};
  };

  struct XCoordObjectB {
    using type = double;
    static constexpr Options::String help = {
        "x-coordinate of the center for Object B."};
  };

  struct ExciseInteriorB {
    using type = bool;
    static constexpr Options::String help = {
        "Exclude Layer 0 for ObjectB. Set to `true` for a BH."};
  };

  struct RadiusOuterCube {
    using type = double;
    static constexpr Options::String help = {
        "Radius of Layer 3 which circumscribes the Frustums."};
  };

  struct RadiusOuterSphere {
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
    using type = bool;
    static constexpr Options::String help = {
        "Use a logarithmically spaced radial grid in Layer 5, the outer "
        "spherical shell that covers the wave zone."};
  };

  struct AdditionToOuterLayerRadialRefinementLevel {
    using type = size_t;
    static constexpr Options::String help = {
        "Addition to radial refinement level in Layer 5 (the outer spherical "
        "shell that covers that wave zone), beyond the refinement "
        "level set by InitialRefinement."};
  };

  struct UseLogarithmicMapObjectA {
    using type = bool;
    static constexpr Options::String help = {
        "Use a logarithmically spaced radial grid in the part of Layer 1 "
        "enveloping Object A (requires ExciseInteriorA == true)"};
  };

  struct AdditionToObjectARadialRefinementLevel {
    using type = size_t;
    static constexpr Options::String help = {
        "Addition to radial refinement level in the part of Layer 1 enveloping "
        "Object A, beyond the refinement level set by InitialRefinement."};
  };

  struct UseLogarithmicMapObjectB {
    using type = bool;
    static constexpr Options::String help = {
        "Use a logarithmically spaced radial grid in the part of Layer 1 "
        "enveloping Object B (requires ExciseInteriorB == true)"};
  };

  struct AdditionToObjectBRadialRefinementLevel {
    using type = size_t;
    static constexpr Options::String help = {
        "Addition to radial refinement level in the part of Layer 1 enveloping "
        "Object B, beyond the refinement level set by InitialRefinement."};
  };

  struct TimeDependence {
    using type =
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>;
    static constexpr Options::String help = {
        "The time dependence of the moving mesh domain."};
  };

  using options = tmpl::list<
      InnerRadiusObjectA, OuterRadiusObjectA, XCoordObjectA, ExciseInteriorA,
      InnerRadiusObjectB, OuterRadiusObjectB, XCoordObjectB, ExciseInteriorB,
      RadiusOuterCube, RadiusOuterSphere, InitialRefinement, InitialGridPoints,
      UseProjectiveMap, UseLogarithmicMapOuterSphericalShell,
      AdditionToOuterLayerRadialRefinementLevel, UseLogarithmicMapObjectA,
      AdditionToObjectARadialRefinementLevel, UseLogarithmicMapObjectB,
      AdditionToObjectBRadialRefinementLevel, TimeDependence>;

  static constexpr Options::String help{
      "The BinaryCompactObject domain is a general domain for two compact "
      "objects. The user must provide the inner and outer radii of the "
      "spherical shells surrounding each of the two compact objects A and "
      "B. The radial refinement levels for these shells are (InitialRefinement "
      "+ AdditionToObjectARadialRefinementLevel) and (InitialRefinement + "
      "AdditionToObjectBRadialRefinementLevel), respectively.\n\n"
      "The user must also provide the radius of the sphere that "
      "circumscribes the cube containing both compact objects, and the "
      "radius of the outer boundary. The options ExciseInteriorA and "
      "ExciseInteriorB determine whether the layer-zero blocks are present "
      "inside each compact object. If set to `true`, the domain will not "
      "contain layer zero for that object. The user specifies XCoordObjectA "
      "and XCoordObjectB, the x-coordinates of the locations of the centers "
      "of each compact object. In these coordinates, the location for the "
      "axis of rotation is x=0. ObjectA is located on the left and ObjectB "
      "is located on the right. Please make sure that your choices of "
      "x-coordinate locations are such that the resulting center of mass "
      "is located at zero.\n\n"
      "Two radial layers join the outer cube to the spherical outer boundary. "
      "The first of these layers transitions from sphericity == 0.0 on the "
      "inner boundary to sphericity == 1.0 on the outer boundary. The second "
      "has sphericity == 1 (so either linear or logarithmic mapping can be "
      "used in the radial direction), extends to the spherical outer boundary "
      "of the domain, and has a radial refinement level of (InitialRefinement "
      "+ AdditionToOuterLayerRadialRefinementLevel)."};

  BinaryCompactObject(
      typename InnerRadiusObjectA::type inner_radius_object_A,
      typename OuterRadiusObjectA::type outer_radius_object_A,
      typename XCoordObjectA::type xcoord_object_A,
      typename ExciseInteriorA::type excise_interior_A,
      typename InnerRadiusObjectB::type inner_radius_object_B,
      typename OuterRadiusObjectB::type outer_radius_object_B,
      typename XCoordObjectB::type xcoord_object_B,
      typename ExciseInteriorB::type excise_interior_B,
      typename RadiusOuterCube::type radius_enveloping_cube,
      typename RadiusOuterSphere::type radius_enveloping_sphere,
      typename InitialRefinement::type initial_refinement,
      typename InitialGridPoints::type initial_grid_points_per_dim,
      typename UseProjectiveMap::type use_projective_map = true,
      typename UseLogarithmicMapOuterSphericalShell::type
          use_logarithmic_map_outer_spherical_shell = false,
      typename AdditionToOuterLayerRadialRefinementLevel::type
          addition_to_outer_layer_radial_refinement_level = 0,
      typename UseLogarithmicMapObjectA::type use_logarithmic_map_object_A =
          false,
      typename AdditionToObjectARadialRefinementLevel::type
          addition_to_object_A_radial_refinement_level = 0,
      typename UseLogarithmicMapObjectB::type use_logarithmic_map_object_B =
          false,
      typename AdditionToObjectBRadialRefinementLevel::type
          addition_to_object_B_radial_refinement_level = 0,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
          time_dependence = nullptr,
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
  typename InnerRadiusObjectA::type inner_radius_object_A_{};
  typename OuterRadiusObjectA::type outer_radius_object_A_{};
  typename XCoordObjectA::type xcoord_object_A_{};
  typename ExciseInteriorA::type excise_interior_A_{};
  typename InnerRadiusObjectB::type inner_radius_object_B_{};
  typename OuterRadiusObjectB::type outer_radius_object_B_{};
  typename XCoordObjectB::type xcoord_object_B_{};
  typename ExciseInteriorB::type excise_interior_B_{};
  typename RadiusOuterCube::type radius_enveloping_cube_{};
  typename RadiusOuterSphere::type radius_enveloping_sphere_{};
  typename InitialRefinement::type initial_refinement_{};
  typename InitialGridPoints::type initial_grid_points_per_dim_{};
  static constexpr bool use_equiangular_map_ =
      false;  // Doesn't work properly yet
  typename UseProjectiveMap::type use_projective_map_ = true;
  typename UseLogarithmicMapOuterSphericalShell::type
      use_logarithmic_map_outer_spherical_shell_ = false;
  typename AdditionToOuterLayerRadialRefinementLevel::type
      addition_to_outer_layer_radial_refinement_level_{};
  typename UseLogarithmicMapObjectA::type use_logarithmic_map_object_A_ = false;
  typename AdditionToObjectARadialRefinementLevel::type
      addition_to_object_A_radial_refinement_level_{};
  typename UseLogarithmicMapObjectB::type use_logarithmic_map_object_B_ = false;
  typename AdditionToObjectBRadialRefinementLevel::type
      addition_to_object_B_radial_refinement_level_{};
  double projective_scale_factor_{};
  double translation_{};
  double length_inner_cube_{};
  double length_outer_cube_{};
  size_t number_of_blocks_{};
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dependence_;
};
}  // namespace creators
}  // namespace domain
