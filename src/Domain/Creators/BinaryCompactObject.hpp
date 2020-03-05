// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Options/Options.hpp"
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
}  // namespace domain
/// \endcond

namespace domain {
namespace creators {

/*!
 * \ingroup DomainCreatorsGroup
 *
 * \brief A general domain for two compact objects.
 *
 * \image html binary_compact_object_domain.png "A BHNS domain."
 *
 * Creates a 3D Domain that represents a binary compact object solution. The
 * Domain consists of four/five nested layers of blocks; these layers are,
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
 * - 4: The blocks that form the outer sphere. This layer transitions back to
 *      spherical and can extend to large radial distances from the compact
 *      objects.
 * In the code and options below, `ObjectA` and `ObjectB` refer to the two
 * compact objects, and by extension, also refer to the layers that immediately
 * surround each compact object. Note that `ObjectA` is located to the left of
 * the origin (along the negative x-axis) and `ObjectB` is located to the right
 * of the origin.
 * `enveloping cube` and `enveloping sphere` refer to the outer surfaces of
 * layers 3 and 4 respectively. Both of these surfaces are centered at the
 * origin.
 * `cutting plane` refers to the plane along which the domain divides into two
 * hemispheres. In the final coordinates, the cutting plane always intersects
 * the x-axis at the origin.
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
    static constexpr OptionString help = {
        "Inner coordinate radius of Layer 1 for Object A."};
  };

  struct OuterRadiusObjectA {
    using type = double;
    static constexpr OptionString help = {
        "Outer coordinate radius of Layer 1 for Object A."};
  };

  struct XCoordObjectA {
    using type = double;
    static constexpr OptionString help = {
        "x-coordinate of center for Object A."};
  };

  struct ExciseInteriorA {
    using type = bool;
    static constexpr OptionString help = {
        "Exclude Layer 0 for ObjectA. Set to `true` for a BH."};
  };

  struct InnerRadiusObjectB {
    using type = double;
    static constexpr OptionString help = {
        "Inner coordinate radius of Layer 1 for Object B."};
  };

  struct OuterRadiusObjectB {
    using type = double;
    static constexpr OptionString help = {
        "Outer coordinate radius of Layer 1 for Object B."};
  };

  struct XCoordObjectB {
    using type = double;
    static constexpr OptionString help = {
        "x-coordinate of the center for Object B."};
  };

  struct ExciseInteriorB {
    using type = bool;
    static constexpr OptionString help = {
        "Exclude Layer 0 for ObjectB. Set to `true` for a BH."};
  };

  struct RadiusOuterCube {
    using type = double;
    static constexpr OptionString help = {
        "Radius of Layer 3 which circumscribes the Frustums."};
  };

  struct RadiusOuterSphere {
    using type = double;
    static constexpr OptionString help = {"Radius of the entire domain."};
  };

  struct InitialRefinement {
    using type = size_t;
    static constexpr OptionString help = {
        "Initial refinement level. Applied to each dimension."};
  };

  struct InitialGridPoints {
    using type = size_t;
    static constexpr OptionString help = {
        "Initial number of grid points in each dim per element."};
  };

  struct UseEquiangularMap {
    using type = bool;
    static constexpr OptionString help = {
        "Use equiangular instead of equidistant coordinates."};
    static type default_value() { return false; }
  };

  struct UseProjectiveMap {
    using type = bool;
    static constexpr OptionString help = {
        "Use projective scaling on the frustal cloak."};
    static type default_value() { return true; }
  };

  using options =
      tmpl::list<InnerRadiusObjectA, OuterRadiusObjectA, XCoordObjectA,
                 ExciseInteriorA, InnerRadiusObjectB, OuterRadiusObjectB,
                 XCoordObjectB, ExciseInteriorB, RadiusOuterCube,
                 RadiusOuterSphere, InitialRefinement, InitialGridPoints,
                 UseEquiangularMap, UseProjectiveMap>;

  static constexpr OptionString help{
      "The BinaryCompactObject domain is a general domain for two compact \n"
      "objects. The user must provide the inner and outer radii of the \n"
      "spherical shells surrounding each of the two compact objects A and B. \n"
      "The user must also provide the radius of the sphere that \n"
      "circumscribes the cube containing both compact objects, and the \n"
      "radius of the outer boundary. The options ExciseInteriorA and \n"
      "ExciseInteriorB determine whether the layer-zero blocks are present \n"
      "inside each compact object. If set to `true`, the domain will not \n"
      "contain layer zero for that object. The user specifies XCoordObjectA \n"
      "and XCoordObjectB, the x-coordinates of the locations of the centers \n"
      "of each compact object. In these coordinates, the location for the \n"
      "axis of rotation is x=0. ObjectA is located on the left and ObjectB \n"
      "is located on the right. Please make sure that your choices of \n"
      "x-coordinate locations are such that the resulting center of mass\n"
      "is located at zero."};

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
      typename UseEquiangularMap::type use_equiangular_map,
      typename UseProjectiveMap::type use_projective_map = true,
      const OptionContext& context = {});

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
  typename UseEquiangularMap::type use_equiangular_map_ = true;
  typename UseProjectiveMap::type use_projective_map_ = true;
  double projective_scale_factor_{};
  double translation_{};
  double length_inner_cube_{};
  double length_outer_cube_{};
};
}  // namespace creators
}  // namespace domain
