// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/Sphere.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/Wedge.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/SettleToConstant.hpp"
#include "Domain/FunctionsOfTime/SettleToConstantQuaternion.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace domain::creators::sphere {

TimeDependentMapOptions::TimeDependentMapOptions(
    const double initial_time, ShapeMapOptionType shape_map_options,
    RotationMapOptionType rotation_map_options,
    ExpansionMapOptionType expansion_map_options,
    TranslationMapOptionType translation_map_options,
    const bool transition_rot_scale_trans)
    : initial_time_(initial_time),
      shape_map_options_(std::move(shape_map_options)),
      rotation_map_options_(std::move(rotation_map_options)),
      expansion_map_options_(std::move(expansion_map_options)),
      translation_map_options_(std::move(translation_map_options)),
      transition_rot_scale_trans_(transition_rot_scale_trans) {}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
TimeDependentMapOptions::create_functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};

  // Get existing function of time names that are used for the maps and assign
  // their initial expiration time to infinity (i.e. not expiring)
  std::unordered_map<std::string, double> expiration_times{
      {size_name, std::numeric_limits<double>::infinity()},
      {shape_name, std::numeric_limits<double>::infinity()},
      {rotation_name, std::numeric_limits<double>::infinity()},
      {expansion_name, std::numeric_limits<double>::infinity()},
      {translation_name, std::numeric_limits<double>::infinity()}};

  // If we have control systems, overwrite these expiration times with the ones
  // supplied by the control system
  for (const auto& [name, expr_time] : initial_expiration_times) {
    expiration_times[name] = expr_time;
  }

  if (shape_map_options_.has_value()) {
    // ShapeMap and Size FunctionOfTime (used in ShapeMap)
    auto shape_and_size = time_dependent_options::get_shape_and_size(
        shape_map_options_.value(), initial_time_,
        expiration_times.at(shape_name), expiration_times.at(size_name),
        deformed_radius_);
    result.merge(shape_and_size);
  }

  // ExpansionMap FunctionOfTime
  if (expansion_map_options_.has_value()) {
    auto expansion_functions_of_time = time_dependent_options::get_expansion(
        expansion_map_options_.value(), initial_time_,
        expiration_times.at(expansion_name));

    result.merge(expansion_functions_of_time);
  }

  // RotationMap FunctionOfTime
  if (rotation_map_options_.has_value()) {
    result[rotation_name] = time_dependent_options::get_rotation(
        rotation_map_options_.value(), initial_time_,
        expiration_times.at(rotation_name));
  }

  // Translation FunctionOfTime
  if (translation_map_options_.has_value()) {
    result[translation_name] = time_dependent_options::get_translation(
        translation_map_options_.value(), initial_time_,
        expiration_times.at(translation_name));
  }

  return result;
}

void TimeDependentMapOptions::build_maps(
    const std::array<double, 3>& center, const bool filled,
    const double inner_radius, const std::vector<double>& radial_partitions,
    const double outer_radius) {
  filled_ = filled;
  if (shape_map_options_.has_value()) {
    const size_t l_max = time_dependent_options::l_max_from_shape_options(
        shape_map_options_.value());
    std::unique_ptr<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                        ShapeMapTransitionFunction>
        transition_func;
    if (filled_) {
      if (transition_rot_scale_trans_ and radial_partitions.size() < 2) {
        ERROR(
            "Currently at least two radial partitions are required to "
            "transition the RotScaleTrans map to zero in the outer shell "
            "when a shape map is present and the interior is filled.");
      }
      using WedgeTransition =
          domain::CoordinateMaps::ShapeMapTransitionFunctions::Wedge;
      // Shape map transitions from 0 to 1 from the inner cube to this surface
      deformed_radius_ =
          radial_partitions.empty() ? outer_radius : radial_partitions.front();
      // Shape map transitions from 1 to 0 from the deformed surface to the next
      // radial partition or to the outer radius
      const bool has_shape_rolloff = not radial_partitions.empty();
      const double shape_outer_radius =
          radial_partitions.size() > 1 ? radial_partitions[1] : outer_radius;
      // These must match the order of orientations_for_sphere_wrappings() in
      // DomainHelpers.hpp. The values must match that of Wedge::Axis
      const std::array<int, 6> axes{3, -3, 2, -2, 1, -1};
      for (size_t j = 0; j < (has_shape_rolloff ? 12 : 6); ++j) {
        if (j < 6) {
          // Reverse the transition function so the shape map goes to zero at
          // the inner cube
          transition_func = std::make_unique<WedgeTransition>(
              center, inner_radius, /* inner_sphericity */ 0.0, center,
              deformed_radius_,
              /* outer_sphericity */ 1.0,
              static_cast<WedgeTransition::Axis>(gsl::at(axes, j)),
              /* reverse */ true);
        } else {
          transition_func = std::make_unique<WedgeTransition>(
              center, deformed_radius_, /* inner_sphericity */ 1.0, center,
              shape_outer_radius,
              /* outer_sphericity */ 1.0,
              static_cast<WedgeTransition::Axis>(gsl::at(axes, j % 6)));
        }
        gsl::at(shape_maps_, j) =
            ShapeMap{center,     l_max,    l_max, std::move(transition_func),
                     shape_name, size_name};
      }
    } else {
      // Shape map transitions from 1 to 0 from the inner radius to the first
      // radial partition or to the outer radius
      deformed_radius_ = inner_radius;
      const double shape_outer_radius =
          radial_partitions.empty() ? outer_radius : radial_partitions.front();
      transition_func =
          std::make_unique<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                               SphereTransition>(inner_radius,
                                                 shape_outer_radius);
      shape_maps_[0] =
          ShapeMap{center,     l_max,    l_max, std::move(transition_func),
                   shape_name, size_name};

      // Interior map
      transition_func =
          std::make_unique<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                               SphereTransition>(
              inner_radius, shape_outer_radius, false, true);
      shape_maps_[1] =
          ShapeMap{center,     l_max,    l_max, std::move(transition_func),
                   shape_name, size_name};
    }
  }

  const double outer_shell_inner_radius =
      radial_partitions.empty() ? inner_radius : radial_partitions.back();
  inner_rot_scale_trans_map_ = RotScaleTransMap{
      expansion_map_options_.has_value()
          ? std::optional<std::pair<
                std::string, std::string>>{{expansion_name,
                                            expansion_outer_boundary_name}}
          : std::nullopt,
      rotation_map_options_.has_value()
          ? std::optional<std::string>{rotation_name}
          : std::nullopt,
      translation_map_options_.has_value()
          ? std::optional<std::string>{translation_name}
          : std::nullopt,
      outer_shell_inner_radius,
      outer_radius,
      domain::CoordinateMaps::TimeDependent::RotScaleTrans<
          3>::BlockRegion::Inner};
  if (transition_rot_scale_trans_) {
    if (radial_partitions.empty()) {
      ERROR(
          "Currently at least one radial partition is required to transition "
          "the RotScaleTrans map to zero in the outer shell.");
    }
    transition_rot_scale_trans_map_ = RotScaleTransMap{
        expansion_map_options_.has_value()
            ? std::optional<std::pair<
                  std::string, std::string>>{{expansion_name,
                                              expansion_outer_boundary_name}}
            : std::nullopt,
        rotation_map_options_.has_value()
            ? std::optional<std::string>{rotation_name}
            : std::nullopt,
        translation_map_options_.has_value()
            ? std::optional<std::string>{translation_name}
            : std::nullopt,
        outer_shell_inner_radius,
        outer_radius,
        domain::CoordinateMaps::TimeDependent::RotScaleTrans<
            3>::BlockRegion::Transition};
  }
}

// If you edit any of the functions below, be sure to update the documentation
// in the Sphere domain creator as well as this class' documentation.
TimeDependentMapOptions::MapType<Frame::Distorted, Frame::Inertial>
TimeDependentMapOptions::distorted_to_inertial_map(
    const size_t block_number, const bool is_inner_cube,
    const size_t num_blocks_per_shell) const {
  const bool block_has_shape_map =
      shape_map_options_.has_value() and
      block_number <
          (filled_ ? 2 * num_blocks_per_shell : num_blocks_per_shell) and
      not is_inner_cube;
  if (block_has_shape_map) {
    return std::make_unique<DistortedToInertialComposition>(
        inner_rot_scale_trans_map_);
  } else {
    return nullptr;
  }
}

TimeDependentMapOptions::MapType<Frame::Grid, Frame::Distorted>
TimeDependentMapOptions::grid_to_distorted_map(
    const size_t block_number, const bool is_inner_cube,
    const size_t num_blocks_per_shell) const {
  const bool block_has_shape_map =
      shape_map_options_.has_value() and
      block_number <
          (filled_ ? 2 * num_blocks_per_shell : num_blocks_per_shell) and
      not is_inner_cube;
  if (block_has_shape_map) {
    // If the interior is not filled we use the SphereTransition function and
    // build only one shape map at index 0 (see `build_maps` above). Otherwise,
    // we use the Wedge transition function and build a shape map for each
    // direction, so we have to use the block number here to get the correct
    // shape map.
    return std::make_unique<GridToDistortedComposition>(
        gsl::at(shape_maps_, filled_ ? block_number : 0));
  } else {
    return nullptr;
  }
}

TimeDependentMapOptions::MapType<Frame::Grid, Frame::Inertial>
TimeDependentMapOptions::grid_to_inertial_map(
    const size_t block_number, const bool is_outer_shell,
    const bool is_central_region, const size_t num_blocks_per_shell) const {
  const bool block_has_shape_map =
      shape_map_options_.has_value() and
      block_number <
          (filled_ ? 2 * num_blocks_per_shell : num_blocks_per_shell) and
      not(is_central_region and filled_);
  if (block_has_shape_map) {
    // If the interior is not filled we use the SphereTransition function and
    // build only one shape map at index 0 (see `build_maps` above). Otherwise,
    // we use the Wedge transition function and build a shape map for each
    // direction, so we have to use the block number here to get the correct
    // shape map.
    return std::make_unique<GridToInertialComposition>(
        gsl::at(shape_maps_,
                filled_ ? block_number : (is_central_region ? 1 : 0)),
        inner_rot_scale_trans_map_);
  } else if (is_outer_shell and transition_rot_scale_trans_) {
    return std::make_unique<GridToInertialSimple>(
        transition_rot_scale_trans_map_);
  } else {
    return std::make_unique<GridToInertialSimple>(inner_rot_scale_trans_map_);
  }
}

bool TimeDependentMapOptions::using_distorted_frame() const {
  // We use shape map options and not the shape map just in case this is called
  // before `build_maps` is called.
  return shape_map_options_.has_value();
}
}  // namespace domain::creators::sphere
