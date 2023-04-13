// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/SphereTimeDependentMaps.hpp"

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/YlmSpherepack.hpp"

namespace domain::creators::sphere {
TimeDependentMapOptions::TimeDependentMapOptions(
    const double initial_time, const std::array<double, 3> initial_size_values,
    const size_t initial_l_max)
    : initial_time_(initial_time),
      initial_size_values_(initial_size_values),
      initial_l_max_(initial_l_max) {}

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
      {shape_name, std::numeric_limits<double>::infinity()}};

  // If we have control systems, overwrite these expiration times with the ones
  // supplied by the control system
  for (const auto& [name, expr_time] : initial_expiration_times) {
    expiration_times[name] = expr_time;
  }

  // CompressionMap FunctionOfTime
  result[size_name] = std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
      initial_time_,
      std::array<DataVector, 4>{{{gsl::at(initial_size_values_, 0)},
                                 {gsl::at(initial_size_values_, 1)},
                                 {gsl::at(initial_size_values_, 2)},
                                 {0.0}}},
      expiration_times.at(size_name));

  const DataVector shape_zeros{
      YlmSpherepack::spectral_size(initial_l_max_, initial_l_max_), 0.0};

  // ShapeMap FunctionOfTime
  result[shape_name] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time_,
          std::array<DataVector, 3>{{shape_zeros, shape_zeros, shape_zeros}},
          expiration_times.at(shape_name));

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
