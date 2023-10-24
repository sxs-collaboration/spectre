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
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace domain::creators::sphere {

TimeDependentMapOptions::TimeDependentMapOptions(
    const double initial_time, const ShapeMapOptions& shape_map_options,
    const std::array<double, 3>& initial_translation_velocity)
    : initial_time_(initial_time),
      initial_l_max_(shape_map_options.l_max),
      initial_shape_values_(shape_map_options.initial_values),
      initial_translation_velocity_(initial_translation_velocity) {}

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
      ylm::Spherepack::spectral_size(initial_l_max_, initial_l_max_), 0.0};
  DataVector shape_func{};
  DataVector size_func{1, 0.0};

  if (initial_shape_values_.has_value()) {
    if (std::holds_alternative<KerrSchildFromBoyerLindquist>(
            initial_shape_values_.value())) {
      const ylm::Spherepack ylm{initial_l_max_, initial_l_max_};
      const auto& mass_and_spin =
          std::get<KerrSchildFromBoyerLindquist>(initial_shape_values_.value());
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

  DataVector initial_translation_velocity_temp{3, 0.0};
  for (size_t i = 0; i < 3; i++) {
    initial_translation_velocity_temp[i] =
        gsl::at(initial_translation_velocity_, i);
  }

  // TranslationMap FunctionOfTime
  result[translation_name] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time_,
          std::array<DataVector, 3>{
              {{3, 0.0},
               std::move(initial_translation_velocity_temp),
               {3, 0.0}}},
          expiration_times.at(translation_name));

  return result;
}

void TimeDependentMapOptions::build_maps(
    const std::array<double, 3>& center, const double inner_radius,
    const double outer_radius,
    std::optional<std::pair<double, double>> translation_transition_radii) {
  std::unique_ptr<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                      ShapeMapTransitionFunction>
      transition_func =
          std::make_unique<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                               SphereTransition>(inner_radius, outer_radius);
  shape_map_ = ShapeMap{center,         initial_l_max_,
                        initial_l_max_, std::move(transition_func),
                        shape_name,     size_name};

  rigid_translation_map_ = TranslationMap{translation_name};

  if (translation_transition_radii.has_value()) {
    // Uniform translation
    translation_map_ = TranslationMap{
        translation_name, translation_transition_radii.value().first,
        translation_transition_radii.value().second};
  } else {
    // Translation with transition
    translation_map_ = TranslationMap{translation_name};
  }
}

// If you edit any of the functions below, be sure to update the documentation
// in the Sphere domain creator as well as this class' documentation.
TimeDependentMapOptions::MapType<Frame::Distorted, Frame::Inertial>
TimeDependentMapOptions::distorted_to_inertial_map(
    const bool include_distorted_map) const {
  if (include_distorted_map) {
    return std::make_unique<DistortedToInertialComposition>(
        rigid_translation_map_);
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
TimeDependentMapOptions::grid_to_inertial_map(
    const bool include_distorted_map, const bool use_rigid_translation) const {
  if (include_distorted_map) {
    return std::make_unique<GridToInertialComposition>(shape_map_,
                                                       rigid_translation_map_);
  } else if (use_rigid_translation) {
    return std::make_unique<GridToInertialSimple>(rigid_translation_map_);
  } else {
    return std::make_unique<GridToInertialSimple>(translation_map_);
  }
}
}  // namespace domain::creators::sphere
