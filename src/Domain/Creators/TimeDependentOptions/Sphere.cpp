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
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/SettleToConstant.hpp"
#include "Domain/FunctionsOfTime/SettleToConstantQuaternion.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace domain::creators::sphere {

TimeDependentMapOptions::TimeDependentMapOptions(
    const double initial_time, std::optional<ShapeMapOptions> shape_map_options,
    std::optional<RotationMapOptions> rotation_map_options,
    std::optional<ExpansionMapOptions> expansion_map_options,
    std::optional<TranslationMapOptions> translation_map_options)
    : initial_time_(initial_time),
      shape_map_options_(std::move(shape_map_options)),
      rotation_map_options_(std::move(rotation_map_options)),
      expansion_map_options_(std::move(expansion_map_options)),
      translation_map_options_(std::move(translation_map_options)) {}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
TimeDependentMapOptions::create_functions_of_time(
    const double inner_radius,
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
    auto [shape_funcs, size_funcs] =
        time_dependent_options::initial_shape_and_size_funcs(
            shape_map_options_.value(), inner_radius);

    // ShapeMap FunctionOfTime
    result[shape_name] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
            initial_time_, std::move(shape_funcs),
            expiration_times.at(shape_name));

    // Size FunctionOfTime (used in ShapeMap)
    result[size_name] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
            initial_time_, std::move(size_funcs),
            expiration_times.at(size_name));
  }

  // ExpansionMap FunctionOfTime
  if (expansion_map_options_.has_value()) {
    if (expansion_map_options_->decay_timescale.has_value()) {
      result[expansion_name] =
          std::make_unique<FunctionsOfTime::SettleToConstant>(
              expansion_map_options_->initial_values, initial_time_,
              expansion_map_options_->decay_timescale.value());
    } else {
      result[expansion_name] =
          std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
              initial_time_, expansion_map_options_->initial_values,
              expiration_times.at(expansion_name));
    }

    // ExpansionMap in the Outer regionFunctionOfTime
    if (expansion_map_options_->asymptotic_velocity_outer_boundary
            .has_value()) {
      result[expansion_outer_boundary_name] =
          std::make_unique<FunctionsOfTime::FixedSpeedCubic>(
              expansion_map_options_->initial_values[0][0], initial_time_,
              expansion_map_options_->asymptotic_velocity_outer_boundary
                  .value(),
              expansion_map_options_->decay_timescale_outer_boundary);
    } else {
      result[expansion_outer_boundary_name] =
          std::make_unique<FunctionsOfTime::SettleToConstant>(
              expansion_map_options_->initial_values_outer_boundary,
              initial_time_,
              expansion_map_options_->decay_timescale_outer_boundary);
    }
  }

  // RotationMap FunctionOfTime
  if (rotation_map_options_.has_value()) {
    if (rotation_map_options_->decay_timescale.has_value()) {
      result[rotation_name] =
          std::make_unique<FunctionsOfTime::QuaternionFunctionOfTime<2>>(
              initial_time_, std::array{rotation_map_options_->quaternions[0]},
              rotation_map_options_->angles,
              expiration_times.at(rotation_name));
    } else {
      result[rotation_name] =
          std::make_unique<FunctionsOfTime::SettleToConstantQuaternion>(
              rotation_map_options_->quaternions, initial_time_,
              rotation_map_options_->decay_timescale.value());
    }
  }

  // Translation FunctionOfTime
  if (translation_map_options_.has_value()) {
    result[translation_name] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
            initial_time_, translation_map_options_->initial_values,
            expiration_times.at(translation_name));
  }

  return result;
}

void TimeDependentMapOptions::build_maps(
    const std::array<double, 3>& center,
    std::pair<double, double> inner_shell_radii,
    std::pair<double, double> outer_shell_radii) {
  if (shape_map_options_.has_value()) {
    std::unique_ptr<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                        ShapeMapTransitionFunction>
        transition_func =
            std::make_unique<domain::CoordinateMaps::
                                 ShapeMapTransitionFunctions::SphereTransition>(
                inner_shell_radii.first, inner_shell_radii.second);
    shape_map_ = ShapeMap{center,
                          shape_map_options_->l_max,
                          shape_map_options_->l_max,
                          std::move(transition_func),
                          shape_name,
                          size_name};
  }

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
      outer_shell_radii.first,
      outer_shell_radii.second,
      domain::CoordinateMaps::TimeDependent::RotScaleTrans<
          3>::BlockRegion::Inner};

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
      outer_shell_radii.first,
      outer_shell_radii.second,
      domain::CoordinateMaps::TimeDependent::RotScaleTrans<
          3>::BlockRegion::Transition};
}

// If you edit any of the functions below, be sure to update the documentation
// in the Sphere domain creator as well as this class' documentation.
TimeDependentMapOptions::MapType<Frame::Distorted, Frame::Inertial>
TimeDependentMapOptions::distorted_to_inertial_map(
    const bool include_distorted_map) const {
  if (include_distorted_map) {
    return std::make_unique<DistortedToInertialComposition>(
        inner_rot_scale_trans_map_);
  } else {
    return nullptr;
  }
}

TimeDependentMapOptions::MapType<Frame::Grid, Frame::Distorted>
TimeDependentMapOptions::grid_to_distorted_map(
    const bool include_distorted_map) const {
  if (include_distorted_map) {
    if (not shape_map_.has_value()) {
      ERROR(
          "Requesting grid to distorted map with distorted frame but shape map "
          "options were not specified.");
    }
    return std::make_unique<GridToDistortedComposition>(shape_map_.value());
  } else {
    return nullptr;
  }
}

TimeDependentMapOptions::MapType<Frame::Grid, Frame::Inertial>
TimeDependentMapOptions::grid_to_inertial_map(const bool include_distorted_map,
                                              const bool use_rigid) const {
  if (include_distorted_map) {
    if (not shape_map_.has_value()) {
      ERROR(
          "Requesting grid to inertial map with distorted frame but shape map "
          "options were not specified.");
    }
    return std::make_unique<GridToInertialComposition>(
        shape_map_.value(), inner_rot_scale_trans_map_);
  } else if (use_rigid) {
    return std::make_unique<GridToInertialSimple>(inner_rot_scale_trans_map_);
  } else {
    return std::make_unique<GridToInertialSimple>(
        transition_rot_scale_trans_map_);
  }
}

bool TimeDependentMapOptions::using_distorted_frame() const {
  // We use shape map options and not the shape map just in case this is called
  // before `build_maps` is called.
  return shape_map_options_.has_value();
}
}  // namespace domain::creators::sphere
