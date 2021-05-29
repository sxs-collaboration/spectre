// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/BinaryCompactObject.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"
#include "Domain/CoordinateMaps/TimeDependent/SphericalCompression.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Logical;
}  // namespace Frame

namespace domain::creators {

bool BinaryCompactObject::Object::is_excised() const noexcept {
  return inner_boundary_condition.has_value();
}

void BinaryCompactObject::check_for_parse_errors(
    const Options::Context& context) const {
  if (object_A_.x_coord >= 0.0) {
    PARSE_ERROR(
        context,
        "The x-coordinate of ObjectA's center is expected to be negative.");
  }
  if (object_B_.x_coord <= 0.0) {
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
  if (object_A_.outer_radius < object_A_.inner_radius) {
    PARSE_ERROR(context,
                "ObjectA's inner radius must be less than its outer radius.");
  }
  if (object_B_.outer_radius < object_B_.inner_radius) {
    PARSE_ERROR(context,
                "ObjectB's inner radius must be less than its outer radius.");
  }
  if (object_A_.use_logarithmic_map and not object_A_.is_excised()) {
    PARSE_ERROR(
        context,
        "Using a logarithmically spaced radial grid in the part "
        "of Layer 1 enveloping Object A requires excising the interior of "
        "Object A");
  }
  if (object_B_.use_logarithmic_map and not object_B_.is_excised()) {
    PARSE_ERROR(
        context,
        "Using a logarithmically spaced radial grid in the part "
        "of Layer 1 enveloping Object B requires excising the interior of "
        "Object B");
  }
  if (object_A_.is_excised() and
      ((*object_A_.inner_boundary_condition == nullptr) !=
       (outer_boundary_condition_ == nullptr))) {
    PARSE_ERROR(context,
                "Must specify either both inner and outer boundary conditions "
                "or neither.");
  }
  if (object_B_.is_excised() and
      ((*object_B_.inner_boundary_condition == nullptr) !=
       (outer_boundary_condition_ == nullptr))) {
    PARSE_ERROR(context,
                "Must specify either both inner and outer boundary conditions "
                "or neither.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(outer_boundary_condition_) or
      (object_A_.is_excised() and
       is_periodic(*object_A_.inner_boundary_condition)) or
      (object_B_.is_excised() and
       is_periodic(*object_B_.inner_boundary_condition))) {
    PARSE_ERROR(
        context,
        "Cannot have periodic boundary conditions with a binary domain");
  }
}

void BinaryCompactObject::initialize_calculated_member_variables() noexcept {
  // Determination of parameters for domain construction:
  translation_ = 0.5 * (object_B_.x_coord + object_A_.x_coord);
  length_inner_cube_ = abs(object_A_.x_coord - object_B_.x_coord);
  length_outer_cube_ = 2.0 * radius_enveloping_cube_ / sqrt(3.0);
  if (use_projective_map_) {
    projective_scale_factor_ = length_inner_cube_ / length_outer_cube_;
  } else {
    projective_scale_factor_ = 1.0;
  }

  // Calculate number of blocks
  // Layers 1, 2, 3, 4, and 5 have 12, 12, 10, 10, and 10 blocks, respectively,
  // for 54 total.
  number_of_blocks_ = 54;

  // For each object whose interior is not excised, add 1 block
  if (not object_A_.is_excised()) {
    number_of_blocks_++;
  }
  if (not object_B_.is_excised()) {
    number_of_blocks_++;
  }
}

// Time-independent constructor
BinaryCompactObject::BinaryCompactObject(
    Object object_A, Object object_B, double radius_enveloping_cube,
    double radius_enveloping_sphere, size_t initial_refinement,
    size_t initial_grid_points_per_dim, bool use_projective_map,
    bool use_logarithmic_map_outer_spherical_shell,
    size_t addition_to_outer_layer_radial_refinement_level,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : object_A_(std::move(object_A)),
      object_B_(std::move(object_B)),
      radius_enveloping_cube_(radius_enveloping_cube),
      radius_enveloping_sphere_(radius_enveloping_sphere),
      initial_refinement_(initial_refinement),
      initial_grid_points_per_dim_(initial_grid_points_per_dim),
      use_projective_map_(use_projective_map),
      use_logarithmic_map_outer_spherical_shell_(
          use_logarithmic_map_outer_spherical_shell),
      addition_to_outer_layer_radial_refinement_level_(
          addition_to_outer_layer_radial_refinement_level),
      outer_boundary_condition_(std::move(outer_boundary_condition)),
      enable_time_dependence_(false),
      initial_time_(std::numeric_limits<double>::signaling_NaN()),
      initial_expiration_delta_t_(std::numeric_limits<double>::signaling_NaN()),
      expansion_map_outer_boundary_(
          std::numeric_limits<double>::signaling_NaN()),
      initial_expansion_({}),
      initial_expansion_velocity_({}),
      expansion_function_of_time_names_({}),
      initial_rotation_angle_(std::numeric_limits<double>::signaling_NaN()),
      initial_angular_velocity_(std::numeric_limits<double>::signaling_NaN()),
      rotation_about_z_axis_function_of_time_name_({}),
      initial_size_map_values_({}),
      initial_size_map_velocities_({}),
      initial_size_map_accelerations_({}),
      size_map_function_of_time_names_({}) {
  initialize_calculated_member_variables();
  check_for_parse_errors(context);
}

// Time-dependent constructor, with additional options for specifying
// the time-dependent maps
BinaryCompactObject::BinaryCompactObject(
    double initial_time, std::optional<double> initial_expiration_delta_t,
    double expansion_map_outer_boundary,
    std::array<double, 2> initial_expansion,
    std::array<double, 2> initial_expansion_velocity,
    std::array<std::string, 2> expansion_function_of_time_names,
    double initial_rotation_angle, double initial_angular_velocity,
    std::string rotation_about_z_axis_function_of_time_name,
    std::array<double, 2> initial_size_map_values,
    std::array<double, 2> initial_size_map_velocities,
    std::array<double, 2> initial_size_map_accelerations,
    std::array<std::string, 2> size_map_function_of_time_names, Object object_A,
    Object object_B, double radius_enveloping_cube,
    double radius_enveloping_sphere, size_t initial_refinement,
    size_t initial_grid_points_per_dim, bool use_projective_map,
    bool use_logarithmic_map_outer_spherical_shell,
    size_t addition_to_outer_layer_radial_refinement_level,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : object_A_(std::move(object_A)),
      object_B_(std::move(object_B)),
      radius_enveloping_cube_(radius_enveloping_cube),
      radius_enveloping_sphere_(radius_enveloping_sphere),
      initial_refinement_(initial_refinement),
      initial_grid_points_per_dim_(initial_grid_points_per_dim),
      use_projective_map_(use_projective_map),
      use_logarithmic_map_outer_spherical_shell_(
          use_logarithmic_map_outer_spherical_shell),
      addition_to_outer_layer_radial_refinement_level_(
          addition_to_outer_layer_radial_refinement_level),
      outer_boundary_condition_(std::move(outer_boundary_condition)),
      enable_time_dependence_(true),
      initial_time_(initial_time),
      initial_expiration_delta_t_(initial_expiration_delta_t),
      expansion_map_outer_boundary_(expansion_map_outer_boundary),
      initial_expansion_(initial_expansion),
      initial_expansion_velocity_(initial_expansion_velocity),
      expansion_function_of_time_names_(
          std::move(expansion_function_of_time_names)),
      initial_rotation_angle_(initial_rotation_angle),
      initial_angular_velocity_(initial_angular_velocity),
      rotation_about_z_axis_function_of_time_name_(
          std::move(rotation_about_z_axis_function_of_time_name)),
      initial_size_map_values_(initial_size_map_values),
      initial_size_map_velocities_(initial_size_map_velocities),
      initial_size_map_accelerations_(initial_size_map_accelerations),
      size_map_function_of_time_names_(
          std::move(size_map_function_of_time_names)) {
  initialize_calculated_member_variables();
  check_for_parse_errors(context);
}

Domain<3> BinaryCompactObject::create_domain() const noexcept {
  const double inner_sphericity_A = object_A_.is_excised() ? 1.0 : 0.0;
  const double inner_sphericity_B = object_B_.is_excised() ? 1.0 : 0.0;

  using Maps = std::vector<
      std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Inertial, 3>>>;
  using BcMap = DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>;

  std::vector<BcMap> boundary_conditions_all_blocks{};

  const std::vector<domain::CoordinateMaps::Distribution>
      object_A_radial_distribution{
          object_A_.use_logarithmic_map
              ? domain::CoordinateMaps::Distribution::Logarithmic
              : domain::CoordinateMaps::Distribution::Linear};

  const std::vector<domain::CoordinateMaps::Distribution>
      object_B_radial_distribution{
          object_B_.use_logarithmic_map
              ? domain::CoordinateMaps::Distribution::Logarithmic
              : domain::CoordinateMaps::Distribution::Linear};

  Maps maps{};
  // ObjectA/B is on the left/right, respectively.
  Maps maps_center_A = sph_wedge_coordinate_maps<Frame::Inertial>(
      object_A_.inner_radius, object_A_.outer_radius, inner_sphericity_A, 1.0,
      use_equiangular_map_, object_A_.x_coord, false, 1.0, {},
      object_A_radial_distribution);
  Maps maps_cube_A = sph_wedge_coordinate_maps<Frame::Inertial>(
      object_A_.outer_radius, sqrt(3.0) * 0.5 * length_inner_cube_, 1.0, 0.0,
      use_equiangular_map_, object_A_.x_coord, false);
  Maps maps_center_B = sph_wedge_coordinate_maps<Frame::Inertial>(
      object_B_.inner_radius, object_B_.outer_radius, inner_sphericity_B, 1.0,
      use_equiangular_map_, object_B_.x_coord, false, 1.0, {},
      object_B_radial_distribution);
  Maps maps_cube_B = sph_wedge_coordinate_maps<Frame::Inertial>(
      object_B_.outer_radius, sqrt(3.0) * 0.5 * length_inner_cube_, 1.0, 0.0,
      use_equiangular_map_, object_B_.x_coord, false);
  Maps maps_frustums = frustum_coordinate_maps<Frame::Inertial>(
      length_inner_cube_, length_outer_cube_, use_equiangular_map_,
      {{-translation_, 0.0, 0.0}}, projective_scale_factor_);

  if (outer_boundary_condition_ != nullptr) {
    for (size_t i = 0; i < maps_center_A.size(); ++i) {
      BcMap bcs{};
      if (object_A_.is_excised()) {
        bcs[Direction<3>::lower_zeta()] =
            (*object_A_.inner_boundary_condition)->get_clone();
      }
      boundary_conditions_all_blocks.push_back(std::move(bcs));
    }
  }
  std::move(maps_center_A.begin(), maps_center_A.end(),
            std::back_inserter(maps));
  if (outer_boundary_condition_ != nullptr) {
    for (size_t i = 0; i < maps_cube_A.size(); ++i) {
      boundary_conditions_all_blocks.emplace_back(BcMap{});
    }
  }
  std::move(maps_cube_A.begin(), maps_cube_A.end(), std::back_inserter(maps));
  if (outer_boundary_condition_ != nullptr) {
    for (size_t i = 0; i < maps_center_B.size(); ++i) {
      BcMap bcs{};
      if (object_B_.is_excised()) {
        bcs[Direction<3>::lower_zeta()] =
            (*object_B_.inner_boundary_condition)->get_clone();
      }
      boundary_conditions_all_blocks.push_back(std::move(bcs));
    }
  }
  std::move(maps_center_B.begin(), maps_center_B.end(),
            std::back_inserter(maps));
  if (outer_boundary_condition_ != nullptr) {
    for (size_t i = 0; i < maps_cube_B.size() + maps_frustums.size(); ++i) {
      boundary_conditions_all_blocks.emplace_back(BcMap{});
    }
  }
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

  const std::vector<domain::CoordinateMaps::Distribution>
      outer_spherical_shell_distribution{
          use_logarithmic_map_outer_spherical_shell_
              ? domain::CoordinateMaps::Distribution::Logarithmic
              : domain::CoordinateMaps::Distribution::Linear};

  Maps maps_first_outer_shell = sph_wedge_coordinate_maps<Frame::Inertial>(
      inner_radius_first_outer_shell, outer_radius_first_outer_shell, 0.0, 1.0,
      use_equiangular_map_, 0.0, true, 1.0, {},
      {domain::CoordinateMaps::Distribution::Linear}, ShellWedges::All);
  Maps maps_second_outer_shell = sph_wedge_coordinate_maps<Frame::Inertial>(
      outer_radius_first_outer_shell, radius_enveloping_sphere_, 1.0, 1.0,
      use_equiangular_map_, 0.0, true, 1.0, {},
      outer_spherical_shell_distribution, ShellWedges::All);
  if (outer_boundary_condition_ != nullptr) {
    // The outer 10 wedges all have to have the outer boundary condition
    // applied
    for (size_t i = 0; i < maps_first_outer_shell.size() +
                               maps_second_outer_shell.size() - 10;
         ++i) {
      boundary_conditions_all_blocks.emplace_back(BcMap{});
    }
    for (size_t i = 0; i < 10; ++i) {
      BcMap bcs{};
      bcs[Direction<3>::upper_zeta()] = outer_boundary_condition_->get_clone();
      boundary_conditions_all_blocks.push_back(std::move(bcs));
    }
  }
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
  if (not object_A_.is_excised()) {
    if (outer_boundary_condition_ != nullptr) {
      boundary_conditions_all_blocks.emplace_back(BcMap{});
    }

    auto shift_1d_A =
        Affine{-1.0, 1.0, -1.0 + object_A_.x_coord, 1.0 + object_A_.x_coord};
    const auto translation_A =
        CoordinateMaps::ProductOf2Maps<Affine, Identity2D>(shift_1d_A,
                                                           Identity2D{});

    const double scaled_r_inner_A = object_A_.inner_radius / sqrt(3.0);
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
  if (not object_B_.is_excised()) {
    if (outer_boundary_condition_ != nullptr) {
      boundary_conditions_all_blocks.emplace_back(BcMap{});
    }

    auto shift_1d_B =
        Affine{-1.0, 1.0, -1.0 + object_B_.x_coord, 1.0 + object_B_.x_coord};
    const auto translation_B =
        CoordinateMaps::ProductOf2Maps<Affine, Identity2D>(shift_1d_B,
                                                           Identity2D{});
    const double scaled_r_inner_B = object_B_.inner_radius / sqrt(3.0);
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
  Domain<3> domain{
      std::move(maps),
      corners_for_biradially_layered_domains(2, 3, not object_A_.is_excised(),
                                             not object_B_.is_excised()),
      {},
      std::move(boundary_conditions_all_blocks)};

  // Inject the hard-coded time-dependence
  if (enable_time_dependence_) {
    // Note on frames: Because the relevant maps will all be composed before
    // they are used, all maps here go from Frame::Grid (the frame after the
    // final time-independent map is applied) to Frame::Inertial
    // (the frame after the final time-dependent map is applied).
    using CubicScaleMap = domain::CoordinateMaps::TimeDependent::CubicScale<3>;
    using CubicScaleMapForComposition =
        domain::CoordinateMap<Frame::Grid, Frame::Inertial, CubicScaleMap>;

    using IdentityMap1D = domain::CoordinateMaps::Identity<1>;
    using RotationMap2D = domain::CoordinateMaps::TimeDependent::Rotation<2>;
    using RotationMap =
        domain::CoordinateMaps::TimeDependent::ProductOf2Maps<RotationMap2D,
                                                              IdentityMap1D>;
    using RotationMapForComposition =
        domain::CoordinateMap<Frame::Grid, Frame::Inertial, RotationMap>;

    using CubicScaleAndRotationMapForComposition =
        domain::CoordinateMap<Frame::Grid, Frame::Inertial, CubicScaleMap,
                              RotationMap>;

    using CompressionMap =
        domain::CoordinateMaps::TimeDependent::SphericalCompression<false>;
    using CompressionMapForComposition =
        domain::CoordinateMap<Frame::Grid, Frame::Inertial, CompressionMap>;

    using CompressionAndCubicScaleAndRotationMapForComposition =
        domain::CoordinateMap<Frame::Grid, Frame::Inertial, CompressionMap,
                              CubicScaleMap, RotationMap>;

    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
        block_maps{number_of_blocks_};

    // Some maps (e.g. expansion, rotation) are applied to all blocks,
    // while other maps (e.g. size) are only applied to some blocks. So
    // there are several different distinct combinations of time-dependent
    // maps that will be applied.
    // Here, set the time-dependent maps for each distinct combination in
    // a single block. Then, set the maps of the other blocks by cloning
    // the maps from the appropriate block.

    // All blocks except possibly blocks 0-5 and 12-17 get the same map, so
    // initialize the final block with the "base" map (here a composition of an
    // expansion and a rotation).
    block_maps[number_of_blocks_ - 1] =
        std::make_unique<CubicScaleAndRotationMapForComposition>(
            domain::push_back(
                CubicScaleMapForComposition{
                    CubicScaleMap{expansion_map_outer_boundary_,
                                  expansion_function_of_time_names_[0],
                                  expansion_function_of_time_names_[1]}},
                RotationMapForComposition{RotationMap{
                    RotationMap2D{rotation_about_z_axis_function_of_time_name_},
                    IdentityMap1D{}}}));

    // Initialize the first block of the layer 1 blocks for each object
    // (specifically, initialize block 0 and block 12). If excising interior
    // A or B, the block maps for the coresponding layer 1 blocks (blocks 0-5
    // for object A, blocks 12-17 for object B) should also include a size map.
    // If not excising interior A or B, the layer 1 blocks for that object
    // will have the same map as the final block.
    if (object_A_.is_excised()) {
      block_maps[0] = std::make_unique<
          CompressionAndCubicScaleAndRotationMapForComposition>(
          domain::push_back(
              CompressionMapForComposition{
                  CompressionMap{size_map_function_of_time_names_[0],
                                 object_A_.inner_radius,
                                 object_A_.outer_radius,
                                 {{object_A_.x_coord, 0.0, 0.0}}}},
              domain::push_back(
                  CubicScaleMapForComposition{
                      CubicScaleMap{expansion_map_outer_boundary_,
                                    expansion_function_of_time_names_[0],
                                    expansion_function_of_time_names_[1]}},
                  RotationMapForComposition{RotationMap{
                      RotationMap2D{
                          rotation_about_z_axis_function_of_time_name_},
                      IdentityMap1D{}}})));
    } else {
      block_maps[0] = block_maps[number_of_blocks_ - 1]->get_clone();
    }
    if (object_B_.is_excised()) {
      block_maps[12] = std::make_unique<
          CompressionAndCubicScaleAndRotationMapForComposition>(
          domain::push_back(
              CompressionMapForComposition{
                  CompressionMap{size_map_function_of_time_names_[1],
                                 object_B_.inner_radius,
                                 object_B_.outer_radius,
                                 {{object_B_.x_coord, 0.0, 0.0}}}},
              domain::push_back(
                  CubicScaleMapForComposition{
                      CubicScaleMap{expansion_map_outer_boundary_,
                                    expansion_function_of_time_names_[0],
                                    expansion_function_of_time_names_[1]}},
                  RotationMapForComposition{RotationMap{
                      RotationMap2D{
                          rotation_about_z_axis_function_of_time_name_},
                      IdentityMap1D{}}})));
    } else {
      block_maps[12] = block_maps[number_of_blocks_ - 1]->get_clone();
    }

    // Fill in the rest of the block maps by cloning the relevant maps
    for (size_t block = 1; block < number_of_blocks_ - 1; ++block) {
      if (block < 6) {
        block_maps[block] = block_maps[0]->get_clone();
      } else if (block == 12) {
        continue;  // block 12 already initialized
      } else if (block > 12 and block < 18) {
        block_maps[block] = block_maps[12]->get_clone();
      } else {
        block_maps[block] = block_maps[number_of_blocks_ - 1]->get_clone();
      }
    }

    // Finally, inject the time dependent maps into the corresponding blocks
    for (size_t block = 0; block < number_of_blocks_; ++block) {
      domain.inject_time_dependent_map_for_block(block,
                                                 std::move(block_maps[block]));
    }
  }

  return domain;
}

std::vector<std::array<size_t, 3>> BinaryCompactObject::initial_extents()
    const noexcept {
  return {number_of_blocks_, make_array<3>(initial_grid_points_per_dim_)};
}

std::vector<std::array<size_t, 3>>
BinaryCompactObject::initial_refinement_levels() const noexcept {
  std::vector<std::array<size_t, 3>> initial_levels{
      number_of_blocks_, make_array<3>(initial_refinement_)};
  // Increase the radial refinement level of the blocks corresponding to the
  // part of Layer 1 enveloping object A (block 0 through block 5, inclusive)
  if (object_A_.addition_to_radial_refinement_level > 0) {
    for (size_t block = 0; block < 6; ++block) {
      // Refine in the radial direction, which is direction 2
      // (i.e. the zeta direction)
      gsl::at(initial_levels[block], 2) +=
          object_A_.addition_to_radial_refinement_level;
    }
  }

  // Increase the radial refinement level of the blocks corresponding to the
  // part of Layer 1 enveloping object B (block 12 through block 17, inclusive).
  if (object_B_.addition_to_radial_refinement_level > 0) {
    for (size_t block = 12; block < 18; ++block) {
      // Refine in the radial direction, which is direction 2
      // (i.e. the zeta direction)
      gsl::at(initial_levels[block], 2) +=
          object_B_.addition_to_radial_refinement_level;
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
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};
  if (not enable_time_dependence_) {
    return result;
  }

  const double initial_expiration_time =
      initial_expiration_delta_t_ ? initial_time_ + *initial_expiration_delta_t_
                                  : std::numeric_limits<double>::infinity();

  // ExpansionMap FunctionOfTime for the function \f$a(t)\f$ in the
  // domain::CoordinateMaps::TimeDependent::CubicScale map
  result[expansion_function_of_time_names_[0]] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time_,
          std::array<DataVector, 3>{{{initial_expansion_[0]},
                                     {initial_expansion_velocity_[0]},
                                     {0.0}}},
          initial_expiration_time);

  // ExpansionMap FunctionOfTime for the function \f$b(t)\f$ in the
  // domain::CoordinateMaps::TimeDependent::CubicScale map
  result[expansion_function_of_time_names_[1]] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time_,
          std::array<DataVector, 3>{{{initial_expansion_[1]},
                                     {initial_expansion_velocity_[1]},
                                     {0.0}}},
          initial_expiration_time);

  // RotationAboutZAxisMap FunctionOfTime for the rotation angle about the z
  // axis \f$\phi\f$.
  result[rotation_about_z_axis_function_of_time_name_] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time_,
          std::array<DataVector, 4>{{{initial_rotation_angle_},
                                     {initial_angular_velocity_},
                                     {0.0},
                                     {0.0}}},
          initial_expiration_time);

  // CompressionMap FunctionOfTime for object A
  result[size_map_function_of_time_names_[0]] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time_,
          std::array<DataVector, 4>{{{initial_size_map_values_[0]},
                                     {initial_size_map_velocities_[0]},
                                     {initial_size_map_accelerations_[0]},
                                     {0.0}}},
          initial_expiration_time);
  // CompressionMap FunctionOfTime for object B
  result[size_map_function_of_time_names_[1]] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time_,
          std::array<DataVector, 4>{{{initial_size_map_values_[1]},
                                     {initial_size_map_velocities_[1]},
                                     {initial_size_map_accelerations_[1]},
                                     {0.0}}},
          initial_expiration_time);

  return result;
}
}  // namespace domain::creators
