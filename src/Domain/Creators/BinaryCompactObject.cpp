// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/BinaryCompactObject.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Block.hpp"          // IWYU pragma: keep
#include "Domain/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Utilities/MakeArray.hpp"

/// \cond
namespace Frame {
struct Logical;
}  // namespace Frame
/// \endcond

namespace domain {
namespace creators {

BinaryCompactObject::BinaryCompactObject(
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
    typename UseProjectiveMap::type use_projective_map,
    const OptionContext& context)
    // clang-tidy: trivially copyable
    : inner_radius_object_A_(std::move(inner_radius_object_A)),        // NOLINT
      outer_radius_object_A_(std::move(outer_radius_object_A)),        // NOLINT
      xcoord_object_A_(std::move(xcoord_object_A)),                    // NOLINT
      excise_interior_A_(std::move(excise_interior_A)),                // NOLINT
      inner_radius_object_B_(std::move(inner_radius_object_B)),        // NOLINT
      outer_radius_object_B_(std::move(outer_radius_object_B)),        // NOLINT
      xcoord_object_B_(std::move(xcoord_object_B)),                    // NOLINT
      excise_interior_B_(std::move(excise_interior_B)),                // NOLINT
      radius_enveloping_cube_(std::move(radius_enveloping_cube)),      // NOLINT
      radius_enveloping_sphere_(std::move(radius_enveloping_sphere)),  // NOLINT
      initial_refinement_(                                             // NOLINT
          std::move(initial_refinement)),                              // NOLINT
      initial_grid_points_per_dim_(                                    // NOLINT
          std::move(initial_grid_points_per_dim)),                     // NOLINT
      use_equiangular_map_(std::move(use_equiangular_map)),            // NOLINT
      use_projective_map_(std::move(use_projective_map))               // NOLINT
{
  // Determination of parameters for domain construction:
  translation_ = 0.5 * (xcoord_object_B_ + xcoord_object_A_);
  length_inner_cube_ = abs(xcoord_object_A_ - xcoord_object_B_);
  length_outer_cube_ = 2.0 * radius_enveloping_cube_ / sqrt(3.0);
  if (xcoord_object_A_ >= 0.0) {
    PARSE_ERROR(
        context,
        "The x-coordinate of ObjectA's center is expected to be negative.");
  }
  if (xcoord_object_B <= 0.0) {
    PARSE_ERROR(
        context,
        "The x-coordinate of ObjectB's center is expected to be positive.");
  }
  if (length_outer_cube_ <= 2.0 * length_inner_cube_) {
    const double suggested_value = 2.0 * length_inner_cube_ * sqrt(3.0);
    PARSE_ERROR(
        context,
        "The radius for the enveloping cube is too small! The Frustums will be "
        "malformed. A recommended radius is:\n"
            << suggested_value);
  }
  if (outer_radius_object_A < inner_radius_object_A) {
    PARSE_ERROR(context,
                "ObjectA's inner radius must be less than its outer radius.");
  }
  if (outer_radius_object_B < inner_radius_object_B) {
    PARSE_ERROR(context,
                "ObjectB's inner radius must be less than its outer radius.");
  }
  if (use_projective_map_) {
    projective_scale_factor_ = length_inner_cube_ / length_outer_cube_;
  } else {
    projective_scale_factor_ = 1.0;
  }
}

Domain<3> BinaryCompactObject::create_domain() const noexcept {
  const double inner_sphericity_A = excise_interior_A_ ? 1.0 : 0.0;
  const double inner_sphericity_B = excise_interior_B_ ? 1.0 : 0.0;

  using Maps = std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Inertial, 3>>>;

  Maps maps;
  // ObjectA/B is on the left/right, respectively.
  Maps maps_center_A = wedge_coordinate_maps<Frame::Inertial>(
      inner_radius_object_A_, outer_radius_object_A_, inner_sphericity_A, 1.0,
      use_equiangular_map_, xcoord_object_A_, false);
  Maps maps_cube_A = wedge_coordinate_maps<Frame::Inertial>(
      outer_radius_object_A_, sqrt(3.0) * 0.5 * length_inner_cube_, 1.0, 0.0,
      use_equiangular_map_, xcoord_object_A_, false);
  Maps maps_center_B = wedge_coordinate_maps<Frame::Inertial>(
      inner_radius_object_B_, outer_radius_object_B_, inner_sphericity_B, 1.0,
      use_equiangular_map_, xcoord_object_B_, false);
  Maps maps_cube_B = wedge_coordinate_maps<Frame::Inertial>(
      outer_radius_object_B_, sqrt(3.0) * 0.5 * length_inner_cube_, 1.0, 0.0,
      use_equiangular_map_, xcoord_object_B_, false);
  Maps maps_frustums = frustum_coordinate_maps<Frame::Inertial>(
      length_inner_cube_, length_outer_cube_, use_equiangular_map_,
      {{-translation_, 0.0, 0.0}}, projective_scale_factor_);
  Maps maps_outer_shell = wedge_coordinate_maps<Frame::Inertial>(
      sqrt(3.0) * 0.5 * length_outer_cube_, radius_enveloping_sphere_, 0.0, 1.0,
      use_equiangular_map_, 0.0, true);

  std::move(maps_center_A.begin(), maps_center_A.end(),
            std::back_inserter(maps));
  std::move(maps_cube_A.begin(), maps_cube_A.end(), std::back_inserter(maps));
  std::move(maps_center_B.begin(), maps_center_B.end(),
            std::back_inserter(maps));
  std::move(maps_cube_B.begin(), maps_cube_B.end(), std::back_inserter(maps));
  std::move(maps_frustums.begin(), maps_frustums.end(),
            std::back_inserter(maps));
  std::move(maps_outer_shell.begin(), maps_outer_shell.end(),
            std::back_inserter(maps));

  // Set up the maps for the central cubes, if any exist.
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3D =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>;
  using Identity2D = CoordinateMaps::Identity<2>;
  auto shift_1d_A =
      Affine{-1.0, 1.0, -1.0 + xcoord_object_A_, 1.0 + xcoord_object_A_};
  auto shift_1d_B =
      Affine{-1.0, 1.0, -1.0 + xcoord_object_B_, 1.0 + xcoord_object_B_};

  // clang-tidy: trivially copyable
  const auto translation_A = CoordinateMaps::ProductOf2Maps<Affine, Identity2D>(
      std::move(shift_1d_A), Identity2D{});  // NOLINT
  const auto translation_B = CoordinateMaps::ProductOf2Maps<Affine, Identity2D>(
      std::move(shift_1d_B), Identity2D{});  // NOLINT
  if (not excise_interior_A_) {
    const double scaled_r_inner_A = inner_radius_object_A_ / sqrt(3.0);
    if (use_equiangular_map_) {
      maps.emplace_back(
          make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
              Equiangular3D{Equiangular(-1.0, 1.0, -1.0 * scaled_r_inner_A,
                                        scaled_r_inner_A),
                            Equiangular(-1.0, 1.0, -1.0 * scaled_r_inner_A,
                                        scaled_r_inner_A),
                            Equiangular(-1.0, 1.0, -1.0 * scaled_r_inner_A,
                                        scaled_r_inner_A)},
              translation_A));
    } else {
      maps.emplace_back(
          make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
              Affine3D{
                  Affine(-1.0, 1.0, -1.0 * scaled_r_inner_A, scaled_r_inner_A),
                  Affine(-1.0, 1.0, -1.0 * scaled_r_inner_A, scaled_r_inner_A),
                  Affine(-1.0, 1.0, -1.0 * scaled_r_inner_A, scaled_r_inner_A)},
              translation_A));
    }
  }
  if (not excise_interior_B_) {
    const double scaled_r_inner_B = inner_radius_object_B_ / sqrt(3.0);
    if (use_equiangular_map_) {
      maps.emplace_back(
          make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
              Equiangular3D{Equiangular(-1.0, 1.0, -1.0 * scaled_r_inner_B,
                                        scaled_r_inner_B),
                            Equiangular(-1.0, 1.0, -1.0 * scaled_r_inner_B,
                                        scaled_r_inner_B),
                            Equiangular(-1.0, 1.0, -1.0 * scaled_r_inner_B,
                                        scaled_r_inner_B)},
              translation_B));
    } else {
      maps.emplace_back(
          make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
              Affine3D{
                  Affine(-1.0, 1.0, -1.0 * scaled_r_inner_B, scaled_r_inner_B),
                  Affine(-1.0, 1.0, -1.0 * scaled_r_inner_B, scaled_r_inner_B),
                  Affine(-1.0, 1.0, -1.0 * scaled_r_inner_B, scaled_r_inner_B)},
              translation_B));
    }
  }

  return Domain<3>{std::move(maps),
                   corners_for_biradially_layered_domains(
                       2, 2, not excise_interior_A_, not excise_interior_B_)};
}

std::vector<std::array<size_t, 3>> BinaryCompactObject::initial_extents() const
    noexcept {
  size_t number_of_blocks = 44;
  if (not excise_interior_A_) {
    number_of_blocks++;
  }
  if (not excise_interior_B_) {
    number_of_blocks++;
  }
  return {number_of_blocks, make_array<3>(initial_grid_points_per_dim_)};
}

std::vector<std::array<size_t, 3>>
BinaryCompactObject::initial_refinement_levels() const noexcept {
  size_t number_of_blocks = 44;
  if (not excise_interior_A_) {
    number_of_blocks++;
  }
  if (not excise_interior_B_) {
    number_of_blocks++;
  }
  return {number_of_blocks, make_array<3>(initial_refinement_)};
}

}  // namespace creators
}  // namespace domain
