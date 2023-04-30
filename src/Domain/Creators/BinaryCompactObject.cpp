// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/BinaryCompactObject.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/BinaryCompactObjectHelpers.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Creators/ExpandOverBlocks.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/Structure/ExcisionSphere.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct BlockLogical;
}  // namespace Frame

namespace domain::creators {

bool BinaryCompactObject::Object::is_excised() const {
  return inner_boundary_condition.has_value();
}

BinaryCompactObject::BinaryCompactObject(
    typename ObjectA::type object_A, typename ObjectB::type object_B,
    const double envelope_radius, const double outer_radius,
    const typename InitialRefinement::type& initial_refinement,
    const typename InitialGridPoints::type& initial_number_of_grid_points,
    const bool use_equiangular_map,
    const CoordinateMaps::Distribution radial_distribution_envelope,
    const CoordinateMaps::Distribution radial_distribution_outer_shell,
    const double opening_angle_in_degrees,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : object_A_(std::move(object_A)),
      object_B_(std::move(object_B)),
      envelope_radius_(envelope_radius),
      outer_radius_(outer_radius),
      use_equiangular_map_(use_equiangular_map),
      radial_distribution_envelope_(radial_distribution_envelope),
      radial_distribution_outer_shell_(radial_distribution_outer_shell),
      outer_boundary_condition_(std::move(outer_boundary_condition)),
      opening_angle_(M_PI * opening_angle_in_degrees / 180.0) {
  // Get useful information about the type of grid used around each compact
  // object
  x_coord_a_ =
      std::visit([](const auto& arg) { return arg.x_coord; }, object_A_);
  x_coord_b_ =
      std::visit([](const auto& arg) { return arg.x_coord; }, object_B_);
  is_excised_a_ =
      std::visit([](const auto& arg) { return arg.is_excised(); }, object_A_);
  is_excised_b_ =
      std::visit([](const auto& arg) { return arg.is_excised(); }, object_B_);
  use_single_block_a_ =
      std::holds_alternative<CartesianCubeAtXCoord>(object_A_);
  use_single_block_b_ =
      std::holds_alternative<CartesianCubeAtXCoord>(object_B_);

  // Determination of parameters for domain construction:
  const double tan_half_opening_angle = tan(0.5 * opening_angle_);
  translation_ = 0.5 * (x_coord_a_ + x_coord_b_);
  length_inner_cube_ = abs(x_coord_a_ - x_coord_b_);
  length_outer_cube_ =
      2.0 * envelope_radius_ / sqrt(2.0 + square(tan_half_opening_angle));

  // Calculate number of blocks
  // Object cubes and shells have 6 blocks each, for a total for 24 blocks.
  // The envelope and outer shell have another 10 blocks each.
  number_of_blocks_ = 44;
  // For each object whose interior is not excised, add 1 block
  if ((not use_single_block_a_) and (not is_excised_a_)) {
    number_of_blocks_++;
  }
  if ((not use_single_block_b_) and (not is_excised_b_)) {
    number_of_blocks_++;
  }

  // For each of the object replaced by a single block, remove (12-1)=11
  if (use_single_block_a_) {
    number_of_blocks_ -= 11;
  }
  if (use_single_block_b_) {
    number_of_blocks_ -= 11;
  }

  if (x_coord_a_ <= 0.0) {
    PARSE_ERROR(
        context,
        "The x-coordinate of ObjectA's center is expected to be positive.");
  }
  if (x_coord_b_ >= 0.0) {
    PARSE_ERROR(
        context,
        "The x-coordinate of ObjectB's center is expected to be negative.");
  }
  if (envelope_radius_ <= length_inner_cube_ * sqrt(3.0)) {
    const double suggested_value = 2.0 * length_inner_cube_ * sqrt(3.0);
    PARSE_ERROR(
        context,
        "The radius for the enveloping cube is too small! The Frustums will be "
        "malformed. A recommended radius is:\n"
            << suggested_value);
  }
  // The following options are irrelevant if the inner regions are covered
  // with simple blocks, so we only check them if object_A_ uses the first
  // variant type.
  if (not use_single_block_a_) {
    const auto& object_a = std::get<Object>(object_A_);
    if (object_a.outer_radius < object_a.inner_radius) {
      PARSE_ERROR(context,
                  "ObjectA's inner radius must be less than its outer radius.");
    }
    if (object_a.use_logarithmic_map and not object_a.is_excised()) {
      PARSE_ERROR(
          context,
          "Using a logarithmically spaced radial grid in the part "
          "of Layer 1 enveloping Object A requires excising the interior of "
          "Object A");
    }
    if (object_a.is_excised() and
        ((*object_a.inner_boundary_condition == nullptr) !=
         (outer_boundary_condition_ == nullptr))) {
      PARSE_ERROR(
          context,
          "Must specify either both inner and outer boundary conditions "
          "or neither.");
    }
  }
  if (not use_single_block_b_) {
    const auto& object_b = std::get<Object>(object_B_);
    if (object_b.outer_radius < object_b.inner_radius) {
      PARSE_ERROR(context,
                  "ObjectB's inner radius must be less than its outer radius.");
    }
    if (object_b.use_logarithmic_map and not object_b.is_excised()) {
      PARSE_ERROR(
          context,
          "Using a logarithmically spaced radial grid in the part "
          "of Layer 1 enveloping Object B requires excising the interior of "
          "Object B");
    }
    if (object_b.is_excised() and
        ((*object_b.inner_boundary_condition == nullptr) !=
         (outer_boundary_condition_ == nullptr))) {
      PARSE_ERROR(
          context,
          "Must specify either both inner and outer boundary conditions "
          "or neither.");
    }
  }
  if (envelope_radius_ >= outer_radius_) {
    PARSE_ERROR(context,
                "The outer radius must be larger than the envelope radius.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(outer_boundary_condition_) or
      (is_excised_a_ and
       is_periodic(*(std::get<Object>(object_A_).inner_boundary_condition))) or
      (is_excised_b_ and
       is_periodic(*(std::get<Object>(object_B_).inner_boundary_condition)))) {
    PARSE_ERROR(
        context,
        "Cannot have periodic boundary conditions with a binary domain");
  }

  // Create block names and groups
  static std::array<std::string, 6> wedge_directions{
      "UpperZ", "LowerZ", "UpperY", "LowerY", "UpperX", "LowerX"};
  const auto add_object_region = [this](const std::string& object_name,
                                        const std::string& region_name) {
    for (const std::string& wedge_direction : wedge_directions) {
      const std::string block_name =
          // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
          object_name + region_name + wedge_direction;
      block_names_.push_back(block_name);
      block_groups_[object_name + region_name].insert(block_name);
    }
  };
  const auto add_object_interior = [this](const std::string& object_name) {
    const std::string block_name = object_name + "Interior";
    block_names_.push_back(block_name);
  };
  const auto add_outer_region = [this](const std::string& region_name) {
    for (const std::string& wedge_direction : wedge_directions) {
      for (const std::string& leftright : {"Left"s, "Right"s}) {
        if ((wedge_direction == "UpperX" and leftright == "Left") or
            (wedge_direction == "LowerX" and leftright == "Right")) {
          // The outer regions are divided in half perpendicular to the
          // x-axis at x=0. Therefore, the left side only has a block in
          // negative x-direction, and the right side only has one in
          // positive x-direction.
          continue;
        }
        // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
        const std::string block_name =
            region_name + wedge_direction +
            (wedge_direction == "UpperX" or wedge_direction == "LowerX"
                 ? ""
                 : leftright);
        block_names_.push_back(block_name);
        block_groups_[region_name].insert(block_name);
      }
    }
  };
  if (use_single_block_a_) {
    block_names_.emplace_back("ObjectA");
  } else {
    add_object_region("ObjectA", "Shell");  // 6 blocks
    add_object_region("ObjectA", "Cube");   // 6 blocks
  }
  if (use_single_block_b_) {
    block_names_.emplace_back("ObjectB");
  } else {
    add_object_region("ObjectB", "Shell");  // 6 blocks
    add_object_region("ObjectB", "Cube");   // 6 blocks
  }
  add_outer_region("Envelope");    // 10 blocks
  add_outer_region("OuterShell");  // 10 blocks

  if ((not use_single_block_a_) and (not is_excised_a_)) {
    add_object_interior("ObjectA");  // 1 block
  }
  if ((not use_single_block_b_) and (not is_excised_b_)) {
    add_object_interior("ObjectB");  // 1 block
  }

  ASSERT(block_names_.size() == number_of_blocks_,
         "Number of block names (" << block_names_.size()
                                   << ") doesn't match number of blocks ("
                                   << number_of_blocks_ << ").");

  // Expand initial refinement and number of grid points over all blocks
  const ExpandOverBlocks<size_t, 3> expand_over_blocks{block_names_,
                                                       block_groups_};

  try {
    initial_refinement_ = std::visit(expand_over_blocks, initial_refinement);
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialRefinement': " << error.what());
  }
  try {
    initial_number_of_grid_points_ =
        std::visit(expand_over_blocks, initial_number_of_grid_points);
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialGridPoints': " << error.what());
  }
}

BinaryCompactObject::BinaryCompactObject(
    bco::TimeDependentMapOptions time_dependent_options,
    typename ObjectA::type object_A, typename ObjectB::type object_B,
    double envelope_radius, double outer_radius,
    const typename InitialRefinement::type& initial_refinement,
    const typename InitialGridPoints::type& initial_number_of_grid_points,
    const bool use_equiangular_map,
    const CoordinateMaps::Distribution radial_distribution_envelope,
    const CoordinateMaps::Distribution radial_distribution_outer_shell,
    const double opening_angle_in_degrees,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : BinaryCompactObject(
          std::move(object_A), std::move(object_B), envelope_radius,
          outer_radius, initial_refinement, initial_number_of_grid_points,
          use_equiangular_map, radial_distribution_envelope,
          radial_distribution_outer_shell, opening_angle_in_degrees,
          std::move(outer_boundary_condition), context) {
  time_dependent_options_ = std::move(time_dependent_options);

  const std::optional<double> inner_radius_A =
      use_single_block_a_
          ? std::nullopt
          : std::optional<double>{std::get<Object>(object_A_).inner_radius};
  const std::optional<double> inner_radius_B =
      use_single_block_b_
          ? std::nullopt
          : std::optional<double>{std::get<Object>(object_B_).inner_radius};
  const std::optional<double> outer_radius_A =
      use_single_block_a_
          ? std::nullopt
          : std::optional<double>{std::get<Object>(object_A_).outer_radius};
  const std::optional<double> outer_radius_B =
      use_single_block_b_
          ? std::nullopt
          : std::optional<double>{std::get<Object>(object_B_).outer_radius};

  time_dependent_options_->build_maps(
      std::array{std::array{x_coord_a_, 0.0, 0.0},
                 std::array{x_coord_b_, 0.0, 0.0}},
      std::array{inner_radius_A, inner_radius_B},
      std::array{outer_radius_A, outer_radius_B}, outer_radius_);
}

Domain<3> BinaryCompactObject::create_domain() const {
  const double inner_sphericity_A = is_excised_a_ ? 1.0 : 0.0;
  const double inner_sphericity_B = is_excised_b_ ? 1.0 : 0.0;

  using Maps = std::vector<std::unique_ptr<
      CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>;

  const std::vector<domain::CoordinateMaps::Distribution>
      object_A_radial_distribution{
          ((not use_single_block_a_) and
           std::get<Object>(object_A_).use_logarithmic_map)
              ? domain::CoordinateMaps::Distribution::Logarithmic
              : domain::CoordinateMaps::Distribution::Linear};

  const std::vector<domain::CoordinateMaps::Distribution>
      object_B_radial_distribution{
          ((not use_single_block_b_) and
           std::get<Object>(object_B_).use_logarithmic_map)
              ? domain::CoordinateMaps::Distribution::Logarithmic
              : domain::CoordinateMaps::Distribution::Linear};

  Maps maps{};

  // ObjectA/B is on the right/left, respectively.
  const Translation translation_A{
      Affine{-1.0, 1.0, -1.0 + x_coord_a_, 1.0 + x_coord_a_}, Identity2D{}};
  const Translation translation_B{
      Affine{-1.0, 1.0, -1.0 + x_coord_b_, 1.0 + x_coord_b_}, Identity2D{}};

  // Two blocks covering the compact objects and their immediate neighborhood
  if (use_single_block_a_) {
    maps.emplace_back(
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            Affine3D{Affine(-1.0, 1.0, -0.5 * length_inner_cube_ + x_coord_a_,
                            0.5 * length_inner_cube_ + x_coord_a_),
                     Affine(-1.0, 1.0, -0.5 * length_inner_cube_,
                            0.5 * length_inner_cube_),
                     Affine(-1.0, 1.0, -0.5 * length_inner_cube_,
                            0.5 * length_inner_cube_)}));
  } else {
    // --- Blocks enclosing each object (12 blocks per object) ---
    //
    // Each object is surrounded by 6 inner wedges that make a sphere, and
    // another 6 outer wedges that transition to a cube.
    const auto& object_a = std::get<Object>(object_A_);

    Maps maps_center_A =
        domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                                Frame::Inertial, 3>(
            sph_wedge_coordinate_maps(object_a.inner_radius,
                                      object_a.outer_radius, inner_sphericity_A,
                                      1.0, use_equiangular_map_, false, {},
                                      object_A_radial_distribution),
            translation_A);
    Maps maps_cube_A =
        domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                                Frame::Inertial, 3>(
            sph_wedge_coordinate_maps(object_a.outer_radius,
                                      sqrt(3.0) * 0.5 * length_inner_cube_, 1.0,
                                      0.0, use_equiangular_map_),
            translation_A);
    std::move(maps_center_A.begin(), maps_center_A.end(),
              std::back_inserter(maps));
    std::move(maps_cube_A.begin(), maps_cube_A.end(), std::back_inserter(maps));
  }
  if (use_single_block_b_) {
    maps.emplace_back(
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            Affine3D{Affine(-1.0, 1.0, -0.5 * length_inner_cube_ + x_coord_b_,
                            0.5 * length_inner_cube_ + x_coord_b_),
                     Affine(-1.0, 1.0, -0.5 * length_inner_cube_,
                            0.5 * length_inner_cube_),
                     Affine(-1.0, 1.0, -0.5 * length_inner_cube_,
                            0.5 * length_inner_cube_)}));
  } else {
    // --- Blocks enclosing each object (12 blocks per object) ---
    //
    // Each object is surrounded by 6 inner wedges that make a sphere, and
    // another 6 outer wedges that transition to a cube.
    const auto& object_b = std::get<Object>(object_B_);
    Maps maps_center_B =
        domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                                Frame::Inertial, 3>(
            sph_wedge_coordinate_maps(object_b.inner_radius,
                                      object_b.outer_radius, inner_sphericity_B,
                                      1.0, use_equiangular_map_, false, {},
                                      object_B_radial_distribution),
            translation_B);
    Maps maps_cube_B =
        domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                                Frame::Inertial, 3>(
            sph_wedge_coordinate_maps(object_b.outer_radius,
                                      sqrt(3.0) * 0.5 * length_inner_cube_, 1.0,
                                      0.0, use_equiangular_map_),
            translation_B);
    std::move(maps_center_B.begin(), maps_center_B.end(),
              std::back_inserter(maps));
    std::move(maps_cube_B.begin(), maps_cube_B.end(), std::back_inserter(maps));
  }

  // --- Frustums enclosing both objects (10 blocks) ---
  //
  // The two abutting cubes are enclosed by a layer of bulged frustums that form
  // a sphere enveloping both objects. While the two objects can be offset from
  // the origin to account for their center of mass, the enveloping frustums are
  // centered at the origin.
  Maps maps_frustums = domain::make_vector_coordinate_map_base<
      Frame::BlockLogical, Frame::Inertial, 3>(frustum_coordinate_maps(
      length_inner_cube_, length_outer_cube_, use_equiangular_map_,
      {{-translation_, 0.0, 0.0}}, radial_distribution_envelope_,
      radial_distribution_envelope_ ==
              domain::CoordinateMaps::Distribution::Projective
          ? std::optional<double>(length_inner_cube_ / length_outer_cube_)
          : std::nullopt,
      1.0, opening_angle_));
  std::move(maps_frustums.begin(), maps_frustums.end(),
            std::back_inserter(maps));

  // --- Outer spherical shell (10 blocks) ---
  Maps maps_outer_shell = domain::make_vector_coordinate_map_base<
      Frame::BlockLogical, Frame::Inertial, 3>(sph_wedge_coordinate_maps(
      envelope_radius_, outer_radius_, 1.0, 1.0, use_equiangular_map_, true, {},
      {radial_distribution_outer_shell_}, ShellWedges::All, opening_angle_));
  std::move(maps_outer_shell.begin(), maps_outer_shell.end(),
            std::back_inserter(maps));

  // --- (Optional) object centers (0 to 2 blocks) ---
  //
  // Each object can optionally be filled with a cube-shaped block, in which
  // case the enclosing wedges configured above transition from the cube to a
  // sphere.
  std::unordered_map<std::string, ExcisionSphere<3>> excision_spheres{};
  if (not use_single_block_a_) {
    if (not is_excised_a_) {
      const double scaled_r_inner_A =
          std::get<Object>(object_A_).inner_radius / sqrt(3.0);
      if (use_equiangular_map_) {
        maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                Equiangular3D{Equiangular(-1.0, 1.0, -1.0 * scaled_r_inner_A,
                                          scaled_r_inner_A),
                              Equiangular(-1.0, 1.0, -1.0 * scaled_r_inner_A,
                                          scaled_r_inner_A),
                              Equiangular(-1.0, 1.0, -1.0 * scaled_r_inner_A,
                                          scaled_r_inner_A)},
                translation_A));
      } else {
        maps.emplace_back(make_coordinate_map_base<Frame::BlockLogical,
                                                   Frame::Inertial>(
            Affine3D{
                Affine(-1.0, 1.0, -1.0 * scaled_r_inner_A, scaled_r_inner_A),
                Affine(-1.0, 1.0, -1.0 * scaled_r_inner_A, scaled_r_inner_A),
                Affine(-1.0, 1.0, -1.0 * scaled_r_inner_A, scaled_r_inner_A)},
            translation_A));
      }
    }
    // Excision spheres
    // - Block 0 through 5 enclose object A, and 12 through 17 enclose object B.
    // - The 3D wedge map is oriented such that the lower-zeta logical direction
    //   points radially inward.
    else {
      excision_spheres.emplace(
          "ExcisionSphereA",
          ExcisionSphere<3>{
              std::get<Object>(object_A_).inner_radius,
              tnsr::I<double, 3, Frame::Grid>{{x_coord_a_, 0.0, 0.0}},
              {{0, Direction<3>::lower_zeta()},
               {1, Direction<3>::lower_zeta()},
               {2, Direction<3>::lower_zeta()},
               {3, Direction<3>::lower_zeta()},
               {4, Direction<3>::lower_zeta()},
               {5, Direction<3>::lower_zeta()}}});
    }
  }
  if (not use_single_block_b_) {
    if (not is_excised_b_) {
      const double scaled_r_inner_B =
          std::get<Object>(object_B_).inner_radius / sqrt(3.0);
      if (use_equiangular_map_) {
        maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                Equiangular3D{Equiangular(-1.0, 1.0, -1.0 * scaled_r_inner_B,
                                          scaled_r_inner_B),
                              Equiangular(-1.0, 1.0, -1.0 * scaled_r_inner_B,
                                          scaled_r_inner_B),
                              Equiangular(-1.0, 1.0, -1.0 * scaled_r_inner_B,
                                          scaled_r_inner_B)},
                translation_B));
      } else {
        maps.emplace_back(make_coordinate_map_base<Frame::BlockLogical,
                                                   Frame::Inertial>(
            Affine3D{
                Affine(-1.0, 1.0, -1.0 * scaled_r_inner_B, scaled_r_inner_B),
                Affine(-1.0, 1.0, -1.0 * scaled_r_inner_B, scaled_r_inner_B),
                Affine(-1.0, 1.0, -1.0 * scaled_r_inner_B, scaled_r_inner_B)},
            translation_B));
      }
    }
    // Excision spheres
    // - Block 0 through 5 enclose object A, and 12 through 17 enclose object B.
    // - The 3D wedge map is oriented such that the lower-zeta logical direction
    //   points radially inward.
    else {
      excision_spheres.emplace(
          "ExcisionSphereB",
          ExcisionSphere<3>{
              std::get<Object>(object_B_).inner_radius,
              tnsr::I<double, 3, Frame::Grid>{{x_coord_b_, 0.0, 0.0}},
              {{12, Direction<3>::lower_zeta()},
               {13, Direction<3>::lower_zeta()},
               {14, Direction<3>::lower_zeta()},
               {15, Direction<3>::lower_zeta()},
               {16, Direction<3>::lower_zeta()},
               {17, Direction<3>::lower_zeta()}}});
    }
  }

  // Have corners determined automatically
  Domain<3> domain{std::move(maps), std::move(excision_spheres), block_names_,
                   block_groups_};

  // Inject the hard-coded time-dependence
  if (time_dependent_options_.has_value()) {
    // Default initialize everything to nullptr so that we only need to set the
    // appropriate block maps for the specific frames
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
        grid_to_inertial_block_maps{number_of_blocks_};
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, 3>>>
        grid_to_distorted_block_maps{number_of_blocks_};
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, 3>>>
        distorted_to_inertial_block_maps{number_of_blocks_};

    // Some maps (e.g. expansion, rotation) are applied to all blocks,
    // while other maps (e.g. size) are only applied to some blocks. Also, some
    // maps are applied from the Grid to the Inertial frame, while others are
    // applied to/from the Distorted frame. So there are several different
    // distinct combinations of time-dependent maps that will be applied. Here,
    // set the time-dependent maps for each distinct combination in a single
    // block. Then, set the maps of the other blocks by cloning the maps from
    // the appropriate block.

    // All blocks except possibly the first 6 blocks of each object get the same
    // map from the Grid to the Inertial frame, so initialize the final block
    // with the "base" map (here a composition of an expansion and a rotation).
    // When covering the inner regions with cubes, all blocks will use the same
    // time-dependent map instead.
    grid_to_inertial_block_maps[number_of_blocks_ - 1] =
        time_dependent_options_
            ->grid_to_inertial_map<domain::ObjectLabel::None>(false);

    // Initialize the first block of the layer 1 blocks for each object
    // If excising interior A or B, the block maps for the corrsponding layer 1
    // blocks (first 6 blocks) should also include a size map from the Grid to
    // the Distorted frame, and then the combination expansion + rotation from
    // the Distorted to Inertial frame. If not excising interior A or B, the
    // layer 1 blocks for that object will not have maps to the Distorted frame
    // (nullptr).
    grid_to_inertial_block_maps[0] =
        time_dependent_options_->grid_to_inertial_map<domain::ObjectLabel::A>(
            is_excised_a_);
    grid_to_distorted_block_maps[0] =
        time_dependent_options_->grid_to_distorted_map<domain::ObjectLabel::A>(
            is_excised_a_);
    distorted_to_inertial_block_maps[0] =
        time_dependent_options_->distorted_to_inertial_map(is_excised_a_);

    const size_t first_block_object_B = use_single_block_a_ ? 1 : 12;
    grid_to_inertial_block_maps[first_block_object_B] =
        time_dependent_options_->grid_to_inertial_map<domain::ObjectLabel::B>(
            is_excised_b_);
    grid_to_distorted_block_maps[first_block_object_B] =
        time_dependent_options_->grid_to_distorted_map<domain::ObjectLabel::B>(
            is_excised_b_);
    distorted_to_inertial_block_maps[first_block_object_B] =
        time_dependent_options_->distorted_to_inertial_map(is_excised_b_);

    // Fill in the rest of the block maps by cloning the relevant maps
    for (size_t block = 1; block < number_of_blocks_ - 1; ++block) {
      if ((not use_single_block_a_) and block < 6) {
        // We always have a grid to inertial map. We may or may not have maps to
        // the distorted frame.
        grid_to_inertial_block_maps[block] =
            grid_to_inertial_block_maps[0]->get_clone();
        if (grid_to_distorted_block_maps[0] != nullptr) {
          grid_to_distorted_block_maps[block] =
              grid_to_distorted_block_maps[0]->get_clone();
          distorted_to_inertial_block_maps[block] =
              distorted_to_inertial_block_maps[0]->get_clone();
        }
      } else if (block == first_block_object_B) {
        continue;  // already initialized
      } else if ((not use_single_block_b_) and block > first_block_object_B and
                 block < first_block_object_B + 6) {
        // We always have a grid to inertial map. We may or may not have maps to
        // the distorted frame.
        grid_to_inertial_block_maps[block] =
            grid_to_inertial_block_maps[first_block_object_B]->get_clone();
        if (grid_to_distorted_block_maps[first_block_object_B] != nullptr) {
          grid_to_distorted_block_maps[block] =
              grid_to_distorted_block_maps[first_block_object_B]->get_clone();
          distorted_to_inertial_block_maps[block] =
              distorted_to_inertial_block_maps[first_block_object_B]
                  ->get_clone();
        }
      } else {
        grid_to_inertial_block_maps[block] =
            grid_to_inertial_block_maps[number_of_blocks_ - 1]->get_clone();
      }
    }
    // Finally, inject the time dependent maps into the corresponding blocks
    for (size_t block = 0; block < number_of_blocks_; ++block) {
      domain.inject_time_dependent_map_for_block(
          block, std::move(grid_to_inertial_block_maps[block]),
          std::move(grid_to_distorted_block_maps[block]),
          std::move(distorted_to_inertial_block_maps[block]));
    }
  }
  return domain;
}

std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
BinaryCompactObject::external_boundary_conditions() const {
  if (outer_boundary_condition_ == nullptr) {
    return {};
  }
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{number_of_blocks_};
  // Excision surfaces
  for (size_t i = 0; i < 6; ++i) {
    // Block 0 - 5 wrap excision surface A
    if (is_excised_a_) {
      boundary_conditions[i][Direction<3>::lower_zeta()] =
          (*(std::get<Object>(object_A_).inner_boundary_condition))
              ->get_clone();
    }
    // Blocks 12 - 17 or 1 - 6 wrap excision surface B
    const size_t first_block_object_B = use_single_block_a_ ? 1 : 12;
    if (is_excised_b_) {
      boundary_conditions[i +
                          first_block_object_B][Direction<3>::lower_zeta()] =
          (*(std::get<Object>(object_B_).inner_boundary_condition))
              ->get_clone();
    }
  }
  // Outer boundary
  const size_t offset_outer_blocks =
      (use_single_block_a_ and use_single_block_b_)
          ? 12
          : ((use_single_block_a_ or use_single_block_b_) ? 23 : 34);
  for (size_t i = 0; i < 10; ++i) {
    boundary_conditions[i + offset_outer_blocks][Direction<3>::upper_zeta()] =
        outer_boundary_condition_->get_clone();
  }
  return boundary_conditions;
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
BinaryCompactObject::functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  return time_dependent_options_.has_value()
             ? time_dependent_options_->create_functions_of_time(
                   initial_expiration_times)
             : std::unordered_map<
                   std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{};
}
}  // namespace domain::creators
