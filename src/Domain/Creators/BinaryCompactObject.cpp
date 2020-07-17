// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/BinaryCompactObject.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Utilities/MakeArray.hpp"

/// \cond
namespace Frame {
struct Logical;
}  // namespace Frame
/// \endcond

namespace domain::creators {
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
    typename UseLogarithmicMapOuterSphericalShell::type
        use_logarithmic_map_outer_spherical_shell,
    typename AdditionToOuterLayerRadialRefinementLevel::type
        addition_to_outer_layer_radial_refinement_level,
    typename UseLogarithmicMapObjectA::type use_logarithmic_map_object_A,
    typename AdditionToObjectARadialRefinementLevel::type
        addition_to_object_A_radial_refinement_level,
    typename UseLogarithmicMapObjectB::type use_logarithmic_map_object_B,
    typename AdditionToObjectBRadialRefinementLevel::type
        addition_to_object_B_radial_refinement_level,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence,
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
      use_projective_map_(std::move(use_projective_map)),              // NOLINT
      use_logarithmic_map_outer_spherical_shell_(
          std::move(use_logarithmic_map_outer_spherical_shell)),  // NOLINT
      addition_to_outer_layer_radial_refinement_level_(
          addition_to_outer_layer_radial_refinement_level),  // NOLINT
      use_logarithmic_map_object_A_(
          std::move(use_logarithmic_map_object_A)),  // NOLINT
      addition_to_object_A_radial_refinement_level_(
          addition_to_object_A_radial_refinement_level),  // NOLINT
      use_logarithmic_map_object_B_(
          std::move(use_logarithmic_map_object_B)),  // NOLINT
      addition_to_object_B_radial_refinement_level_(
          addition_to_object_B_radial_refinement_level),  // NOLINT
      time_dependence_(std::move(time_dependence)) {
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
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<3>>();
  }
  if (use_logarithmic_map_object_A_ and not excise_interior_A_) {
    PARSE_ERROR(
        context,
        "excise_interior_A must be true if use_logarithmic_map_object_A is "
        "true; that is, using a logarithmically spaced radial grid in the part "
        "of Layer 1 enveloping Object A requires excising the interior of "
        "Object A");
  }
  if (use_logarithmic_map_object_B_ and not excise_interior_B_) {
    PARSE_ERROR(
        context,
        "excise_interior_B must be true if use_logarithmic_map_object_B is "
        "true; that is, using a logarithmically spaced radial grid in the part "
        "of Layer 1 enveloping Object B requires excising the interior of "
        "Object B");
  }

  // Calculate number of blocks
  // Layers 1, 2, 3, 4, and 5 have 12, 12, 10, 10, and 10 blocks, respectively,
  // for 54 total.
  number_of_blocks_ = 54;

  // For each object whose interior is not excised, add 1 block
  if (not excise_interior_A_) {
    number_of_blocks_++;
  }
  if (not excise_interior_B_) {
    number_of_blocks_++;
  }
}

Domain<3> BinaryCompactObject::create_domain() const noexcept {
  const double inner_sphericity_A = excise_interior_A_ ? 1.0 : 0.0;
  const double inner_sphericity_B = excise_interior_B_ ? 1.0 : 0.0;

  using Maps = std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Inertial, 3>>>;

  Maps maps;
  // ObjectA/B is on the left/right, respectively.
  Maps maps_center_A = sph_wedge_coordinate_maps<Frame::Inertial>(
      inner_radius_object_A_, outer_radius_object_A_, inner_sphericity_A, 1.0,
      use_equiangular_map_, xcoord_object_A_, false, 1.0,
      use_logarithmic_map_object_A_);
  Maps maps_cube_A = sph_wedge_coordinate_maps<Frame::Inertial>(
      outer_radius_object_A_, sqrt(3.0) * 0.5 * length_inner_cube_, 1.0, 0.0,
      use_equiangular_map_, xcoord_object_A_, false);
  Maps maps_center_B = sph_wedge_coordinate_maps<Frame::Inertial>(
      inner_radius_object_B_, outer_radius_object_B_, inner_sphericity_B, 1.0,
      use_equiangular_map_, xcoord_object_B_, false, 1.0,
      use_logarithmic_map_object_B_);
  Maps maps_cube_B = sph_wedge_coordinate_maps<Frame::Inertial>(
      outer_radius_object_B_, sqrt(3.0) * 0.5 * length_inner_cube_, 1.0, 0.0,
      use_equiangular_map_, xcoord_object_B_, false);
  Maps maps_frustums = frustum_coordinate_maps<Frame::Inertial>(
      length_inner_cube_, length_outer_cube_, use_equiangular_map_,
      {{-translation_, 0.0, 0.0}}, projective_scale_factor_);

  std::move(maps_center_A.begin(), maps_center_A.end(),
            std::back_inserter(maps));
  std::move(maps_cube_A.begin(), maps_cube_A.end(), std::back_inserter(maps));
  std::move(maps_center_B.begin(), maps_center_B.end(),
            std::back_inserter(maps));
  std::move(maps_cube_B.begin(), maps_cube_B.end(), std::back_inserter(maps));
  std::move(maps_frustums.begin(), maps_frustums.end(),
            std::back_inserter(maps));

  // The first shell (Layer 4) goes from sphericity == 0.0 to sphericity
  // == 1.0. This shell is surrounded by a shell with sphericity == 1.0
  // throughout (Layer 5) that will have a radial refinement level of
  // (addition_to_outer_layer_radial_refinement_level_ + initial_refinement_).
  const double inner_radius_first_outer_shell =
      sqrt(3.0) * 0.5 * length_outer_cube_;

  // Adjust the outer boundary of the cubed sphere to conform to the
  // spacing of the spherical shells after refinement, so the cubed sphere is
  // the same size as the first radial division of the spherical shell
  // (for linear mapping) or smaller by the same factor as adjacent radial
  // divisions in the spherical shell (for logarithmic mapping)
  const double radial_divisions_in_outer_layers =
      pow(2, addition_to_outer_layer_radial_refinement_level_) + 1;
  const double outer_radius_first_outer_shell =
      use_logarithmic_map_outer_spherical_shell_
          ? inner_radius_first_outer_shell *
                pow(radius_enveloping_sphere_ / inner_radius_first_outer_shell,
                    1.0 / static_cast<double>(radial_divisions_in_outer_layers))
          : inner_radius_first_outer_shell +
                (radius_enveloping_sphere_ - inner_radius_first_outer_shell) /
                    static_cast<double>(radial_divisions_in_outer_layers);
  Maps maps_first_outer_shell = sph_wedge_coordinate_maps<Frame::Inertial>(
      inner_radius_first_outer_shell, outer_radius_first_outer_shell, 0.0, 1.0,
      use_equiangular_map_, 0.0, true, 1.0, false, ShellWedges::All, 1);
  Maps maps_second_outer_shell = sph_wedge_coordinate_maps<Frame::Inertial>(
      outer_radius_first_outer_shell, radius_enveloping_sphere_, 1.0, 1.0,
      use_equiangular_map_, 0.0, true, 1.0,
      use_logarithmic_map_outer_spherical_shell_, ShellWedges::All, 1);
  std::move(maps_first_outer_shell.begin(), maps_first_outer_shell.end(),
            std::back_inserter(maps));
  std::move(maps_second_outer_shell.begin(), maps_second_outer_shell.end(),
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
  Domain<3> domain{std::move(maps),
                   corners_for_biradially_layered_domains(
                       2, 3, not excise_interior_A_, not excise_interior_B_)};
  if (not time_dependence_->is_none()) {
    for (size_t block = 0; block < number_of_blocks_; ++block) {
      domain.inject_time_dependent_map_for_block(
          block,
          std::move(time_dependence_->block_maps(number_of_blocks_)[block]));
    }
  }
  return domain;
}

std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
BinaryCompactObject::TimeDependence::default_value() noexcept {
  return std::make_unique<domain::creators::time_dependence::None<3>>();
}

std::vector<std::array<size_t, 3>> BinaryCompactObject::initial_extents() const
    noexcept {
  return {number_of_blocks_, make_array<3>(initial_grid_points_per_dim_)};
}

std::vector<std::array<size_t, 3>>
BinaryCompactObject::initial_refinement_levels() const noexcept {
  std::vector<std::array<size_t, 3>> initial_levels{
      number_of_blocks_, make_array<3>(initial_refinement_)};
  // Increase the radial refinement level of the blocks corresponding to the
  // part of Layer 1 enveloping object A (block 0 through block 5, inclusive)
  if (addition_to_object_A_radial_refinement_level_ > 0) {
    for (size_t block = 0; block < 6; ++block) {
      // Refine in the radial direction, which is direction 2
      // (i.e. the zeta direction)
      gsl::at(initial_levels[block], 2) +=
          addition_to_object_A_radial_refinement_level_;
    }
  }

  // Increase the radial refinement level of the blocks corresponding to the
  // part of Layer 1 enveloping object B (block 12 through block 17, inclusive).
  if (addition_to_object_B_radial_refinement_level_ > 0) {
    for (size_t block = 12; block < 18; ++block) {
      // Refine in the radial direction, which is direction 2
      // (i.e. the zeta direction)
      gsl::at(initial_levels[block], 2) +=
          addition_to_object_B_radial_refinement_level_;
    }
  }

  // Increase the radial refinement of the blocks corresponding to the outer
  // spherical shell (with sphericity == 1 throughout) to achieve the desired
  // number of radial refinements. The outer layer consists of 10 blocks,
  // created via sph_wedge_coordinate_maps() with use_half_wedges == true.
  // Because this outer layer of blocks is added last to the CoordinateMaps in
  // create_domain(), the 10 blocks to refine are the last 10 blocks in the
  // domain--unless one or both of the interiors are not excised. (For each
  // interior not excised, there is one extra block appended to the list of
  // CoordinateMaps.)
  if (addition_to_outer_layer_radial_refinement_level_ > 0) {
    for (size_t block = 44; block < 54; ++block) {
      // Refine in the radial direction, which is direction 2
      // (i.e. the zeta direction)
      gsl::at(initial_levels[block], 2) +=
          addition_to_outer_layer_radial_refinement_level_;
    }
  }
  return initial_levels;
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
BinaryCompactObject::functions_of_time() const noexcept {
  if (time_dependence_->is_none()) {
    return {};
  } else {
    return time_dependence_->functions_of_time();
  }
}
}  // namespace domain::creators
