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
    const double initial_time,
    const std::optional<std::array<double, 3>>& initial_size_values,
    const size_t initial_l_max,
    const typename ShapeMapInitialValues::type::value_type&
        initial_shape_values)
    : initial_time_(initial_time),
      initial_size_values_(initial_size_values),
      initial_l_max_(initial_l_max),
      initial_shape_values_(initial_shape_values) {}

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
      {shape_name, std::numeric_limits<double>::infinity()}};

  // If we have control systems, overwrite these expiration times with the ones
  // supplied by the control system
  for (const auto& [name, expr_time] : initial_expiration_times) {
    expiration_times[name] = expr_time;
  }

  DataVector shape_zeros{
      ylm::Spherepack::spectral_size(initial_l_max_, initial_l_max_), 0.0};
  DataVector shape_func{};
  double size_func_from_shape = std::numeric_limits<double>::signaling_NaN();

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
      size_func_from_shape = shape_func[0] * sqrt(0.5 * M_PI);
      // Set l=0 for shape map to 0 because size is going to be used
      shape_func[0] = 0.0;
    }
  } else {
    shape_func = shape_zeros;
    size_func_from_shape = 0.0;
  }

  ASSERT(not std::isnan(size_func_from_shape),
         "Size func value from shape coefficients is NaN.");

  // ShapeMap FunctionOfTime
  result[shape_name] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time_,
          std::array<DataVector, 3>{
              {std::move(shape_func), shape_zeros, shape_zeros}},
          expiration_times.at(shape_name));

  DataVector size_func{1, size_func_from_shape};
  DataVector size_deriv{1, 0.0};
  DataVector size_2nd_deriv{1, 0.0};

  if (initial_size_values_.has_value()) {
    size_func[0] = gsl::at(initial_size_values_.value(), 0);
    size_deriv[0] = gsl::at(initial_size_values_.value(), 1);
    size_2nd_deriv[0] = gsl::at(initial_size_values_.value(), 2);
  }

  // Size FunctionOfTime (used in ShapeMap)
  result[size_name] = std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
      initial_time_,
      std::array<DataVector, 4>{{std::move(size_func),
                                 std::move(size_deriv),
                                 std::move(size_2nd_deriv),
                                 {0.0}}},
      expiration_times.at(size_name));

  return result;
}

void TimeDependentMapOptions::build_maps(const std::array<double, 3>& center,
                                         const double inner_radius,
                                         const double outer_radius) {
  std::unique_ptr<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                      ShapeMapTransitionFunction>
      transition_func =
          std::make_unique<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                               SphereTransition>(inner_radius, outer_radius);
  shape_map_ = ShapeMap{center,         initial_l_max_,
                        initial_l_max_, std::move(transition_func),
                        shape_name,     size_name};
}

// If you edit any of the functions below, be sure to update the documentation
// in the Sphere domain creator as well as this class' documentation.
TimeDependentMapOptions::MapType<Frame::Distorted, Frame::Inertial>
TimeDependentMapOptions::distorted_to_inertial_map(
    const bool include_distorted_map) {
  if (include_distorted_map) {
    return std::make_unique<
        IdentityForComposition<Frame::Distorted, Frame::Inertial>>(
        IdentityMap{});
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
    const bool include_distorted_map) const {
  if (include_distorted_map) {
    return std::make_unique<GridToInertialComposition>(shape_map_);
  } else {
    return std::make_unique<
        IdentityForComposition<Frame::Grid, Frame::Inertial>>(IdentityMap{});
  }
}
}  // namespace domain::creators::sphere
