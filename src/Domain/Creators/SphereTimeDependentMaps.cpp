// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/SphereTimeDependentMaps.hpp"

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
    const double initial_time, const ShapeMapOptions& shape_map_options,
    std::optional<RotationMapOptions> rotation_map_options,
    std::optional<ExpansionMapOptions> expansion_map_options,
    std::optional<TranslationMapOptions> translation_map_options)
    : initial_time_(initial_time),
      shape_map_options_(shape_map_options),
      rotation_map_options_(rotation_map_options),
      expansion_map_options_(expansion_map_options),
      translation_map_options_(translation_map_options) {}

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
      {translation_name, std::numeric_limits<double>::infinity()}};

  // If we have control systems, overwrite these expiration times with the ones
  // supplied by the control system
  for (const auto& [name, expr_time] : initial_expiration_times) {
    expiration_times[name] = expr_time;
  }

  DataVector shape_zeros{
      ylm::Spherepack::spectral_size(shape_map_options_.l_max,
                                     shape_map_options_.l_max),
      0.0};
  DataVector shape_func{};
  DataVector size_func{1, 0.0};

  if (shape_map_options_.initial_values.has_value()) {
    if (std::holds_alternative<KerrSchildFromBoyerLindquist>(
            shape_map_options_.initial_values.value())) {
      const ylm::Spherepack ylm{shape_map_options_.l_max,
                                shape_map_options_.l_max};
      const auto& mass_and_spin = std::get<KerrSchildFromBoyerLindquist>(
          shape_map_options_.initial_values.value());
      const DataVector radial_distortion =
          1.0 - get(gr::Solutions::kerr_schild_radius_from_boyer_lindquist(
                    inner_radius, ylm.theta_phi_points(), mass_and_spin.mass,
                    mass_and_spin.spin)) /
                    inner_radius;
      shape_func = ylm.phys_to_spec(radial_distortion);
      // Transform from SPHEREPACK to actual Ylm for size func
      size_func[0] = shape_func[0] * sqrt(0.5 * M_PI);
      // Set l=0 for shape map to 0 because size is going to be used
      shape_func[0] = 0.0;
    }
  } else {
    shape_func = shape_zeros;
    size_func[0] = 0.0;
  }

  // ShapeMap FunctionOfTime
  result[shape_name] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time_,
          std::array<DataVector, 3>{
              {std::move(shape_func), shape_zeros, shape_zeros}},
          expiration_times.at(shape_name));

  DataVector size_deriv{1, 0.0};
  DataVector size_2nd_deriv{1, 0.0};

  // Size FunctionOfTime (used in ShapeMap)
  result[size_name] = std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
      initial_time_,
      std::array<DataVector, 4>{{std::move(size_func),
                                 std::move(size_deriv),
                                 std::move(size_2nd_deriv),
                                 {0.0}}},
      expiration_times.at(size_name));

  // ExpansionMap FunctionOfTime
  if (expansion_map_options_.has_value()) {
    result[expansion_name] =
        std::make_unique<FunctionsOfTime::SettleToConstant>(
            std::array<DataVector, 3>{
                {{gsl::at(expansion_map_options_.value().initial_values, 0)},
                 {gsl::at(expansion_map_options_.value().initial_values, 1)},
                 {gsl::at(expansion_map_options_.value().initial_values, 2)}}},
            initial_time_, expansion_map_options_.value().decay_timescale);

    // ExpansionMap in the Outer regionFunctionOfTime
    result[expansion_outer_boundary_name] = std::make_unique<
        FunctionsOfTime::SettleToConstant>(
        std::array<DataVector, 3>{
            {{gsl::at(
                 expansion_map_options_.value().initial_values_outer_boundary,
                 0)},
             {gsl::at(
                 expansion_map_options_.value().initial_values_outer_boundary,
                 1)},
             {gsl::at(
                 expansion_map_options_.value().initial_values_outer_boundary,
                 2)}}},
        initial_time_,
        expansion_map_options_.value().decay_timescale_outer_boundary);
  }

  DataVector initial_quaternion_value{4, 0.0};
  DataVector initial_quaternion_first_derivative_value{4, 0.0};
  DataVector initial_quaternion_second_derivative_value{4, 0.0};

  // RotationMap FunctionOfTime
  if (rotation_map_options_.has_value()) {
    for (size_t i = 0; i < 4; i++) {
      initial_quaternion_value[i] =
          gsl::at(gsl::at(rotation_map_options_.value().initial_values, 0), i);
      initial_quaternion_first_derivative_value[i] =
          gsl::at(gsl::at(rotation_map_options_.value().initial_values, 1), i);
      initial_quaternion_second_derivative_value[i] =
          gsl::at(gsl::at(rotation_map_options_.value().initial_values, 2), i);
    }
    result[rotation_name] =
        std::make_unique<FunctionsOfTime::SettleToConstantQuaternion>(
            std::array<DataVector, 3>{
                std::move(initial_quaternion_value),
                std::move(initial_quaternion_first_derivative_value),
                std::move(initial_quaternion_second_derivative_value)},
            initial_time_, rotation_map_options_.value().decay_timescale);
  }

  DataVector initial_translation_center{3, 0.0};
  DataVector initial_translation_velocity{3, 0.0};
  DataVector initial_translation_acceleration{3, 0.0};

  // Translation FunctionOfTime
  if (translation_map_options_.has_value()) {
    for (size_t i = 0; i < 3; i++) {
      initial_translation_center[i] = gsl::at(
          gsl::at(translation_map_options_.value().initial_values, 0), i);
      initial_translation_velocity[i] = gsl::at(
          gsl::at(translation_map_options_.value().initial_values, 1), i);
      initial_translation_acceleration[i] = gsl::at(
          gsl::at(translation_map_options_.value().initial_values, 2), i);
    }
    result[translation_name] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
            initial_time_,
            std::array<DataVector, 3>{
                {std::move(initial_translation_center),
                 std::move(initial_translation_velocity),
                 std::move(initial_translation_acceleration)}},
            expiration_times.at(translation_name));
  }

  return result;
}

void TimeDependentMapOptions::build_maps(
    const std::array<double, 3>& center,
    std::pair<double, double> inner_shell_radii,
    std::pair<double, double> outer_shell_radii) {
  std::unique_ptr<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                      ShapeMapTransitionFunction>
      transition_func =
          std::make_unique<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                               SphereTransition>(inner_shell_radii.first,
                                                 inner_shell_radii.second);
  shape_map_ = ShapeMap{center,
                        shape_map_options_.l_max,
                        shape_map_options_.l_max,
                        std::move(transition_func),
                        shape_name,
                        size_name};

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
    return std::make_unique<GridToDistortedComposition>(shape_map_);
  } else {
    return nullptr;
  }
}

TimeDependentMapOptions::MapType<Frame::Grid, Frame::Inertial>
TimeDependentMapOptions::grid_to_inertial_map(const bool include_distorted_map,
                                              const bool use_rigid) const {
  if (include_distorted_map) {
    return std::make_unique<GridToInertialComposition>(
        shape_map_, inner_rot_scale_trans_map_);
  } else if (use_rigid) {
    return std::make_unique<GridToInertialSimple>(inner_rot_scale_trans_map_);
  } else {
    return std::make_unique<GridToInertialSimple>(
        transition_rot_scale_trans_map_);
  }
}
}  // namespace domain::creators::sphere
