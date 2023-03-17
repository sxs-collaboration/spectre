// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/BinaryCompactObjectHelpers.hpp"

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Utilities/Gsl.hpp"

namespace domain::creators::bco {
TimeDependentMapOptions::TimeDependentMapOptions(
    double initial_time, ExpansionMapOptions expansion_map_options,
    std::array<double, 3> initial_angular_velocity,
    std::array<double, 3> initial_size_values_A,
    std::array<double, 3> initial_size_values_B)
    : initial_time_(initial_time),
      expansion_map_options_(expansion_map_options),
      initial_angular_velocity_(initial_angular_velocity),
      initial_size_values_(
          std::array{initial_size_values_A, initial_size_values_B}) {}

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
      {expansion_name, std::numeric_limits<double>::infinity()},
      {rotation_name, std::numeric_limits<double>::infinity()},
      {gsl::at(size_names, 0), std::numeric_limits<double>::infinity()},
      {gsl::at(size_names, 1), std::numeric_limits<double>::infinity()}};

  // If we have control systems, overwrite these expiration times with the ones
  // supplied by the control system
  for (const auto& [name, expr_time] : initial_expiration_times) {
    expiration_times[name] = expr_time;
  }

  // ExpansionMap FunctionOfTime for the function \f$a(t)\f$ in the
  // domain::CoordinateMaps::TimeDependent::CubicScale map
  result[expansion_name] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time_,
          std::array<DataVector, 3>{
              {{gsl::at(expansion_map_options_.initial_values, 0)},
               {gsl::at(expansion_map_options_.initial_values, 1)},
               {0.0}}},
          expiration_times.at(expansion_name));

  // ExpansionMap FunctionOfTime for the function \f$b(t)\f$ in the
  // domain::CoordinateMaps::TimeDependent::CubicScale map
  result[expansion_outer_boundary_name] =
      std::make_unique<FunctionsOfTime::FixedSpeedCubic>(
          1.0, initial_time_, expansion_map_options_.outer_boundary_velocity,
          expansion_map_options_.outer_boundary_decay_time);

  // RotationMap FunctionOfTime for the rotation angles about each
  // axis.  The initial rotation angles don't matter as we never
  // actually use the angles themselves. We only use their derivatives
  // (omega) to determine map parameters. In theory we could determine
  // each initial angle from the input axis-angle representation, but
  // we don't need to.
  result[rotation_name] =
      std::make_unique<FunctionsOfTime::QuaternionFunctionOfTime<3>>(
          initial_time_,
          std::array<DataVector, 1>{DataVector{1.0, 0.0, 0.0, 0.0}},
          std::array<DataVector, 4>{{{3, 0.0},
                                     {gsl::at(initial_angular_velocity_, 0),
                                      gsl::at(initial_angular_velocity_, 1),
                                      gsl::at(initial_angular_velocity_, 2)},
                                     {3, 0.0},
                                     {3, 0.0}}},
          expiration_times.at(rotation_name));

  // CompressionMap FunctionOfTime for objects A and B
  for (size_t i = 0; i < size_names.size(); i++) {
    result[gsl::at(size_names, i)] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
            initial_time_,
            std::array<DataVector, 4>{
                {{gsl::at(gsl::at(initial_size_values_, i), 0)},
                 {gsl::at(gsl::at(initial_size_values_, i), 1)},
                 {gsl::at(gsl::at(initial_size_values_, i), 2)},
                 {0.0}}},
            expiration_times.at(gsl::at(size_names, i)));
  }

  return result;
}
}  // namespace domain::creators::bco
