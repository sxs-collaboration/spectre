// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependence/SphericalCompression.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/MapInstantiationMacros.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain {
namespace creators::time_dependence {

SphericalCompression::SphericalCompression(
    const double initial_time, const double min_radius, const double max_radius,
    const std::array<double, 3> center, const double initial_value,
    const double initial_velocity, const double initial_acceleration,
    const Options::Context& context)
    : initial_time_(initial_time),
      min_radius_(min_radius),
      max_radius_(max_radius),
      center_(center),
      initial_value_(initial_value),
      initial_velocity_(initial_velocity),
      initial_acceleration_(initial_acceleration) {
  if (min_radius >= max_radius) {
    PARSE_ERROR(context,
                "Tried to create a SphericalCompression TimeDependence, but "
                "the minimum radius ("
                    << min_radius << ") is not less than the maximum radius ("
                    << max_radius << ")");
  }
}

std::unique_ptr<TimeDependence<3>> SphericalCompression::get_clone() const {
  return std::make_unique<SphericalCompression>(
      initial_time_, min_radius_, max_radius_, center_, initial_value_,
      initial_velocity_, initial_acceleration_);
}

std::vector<
    std::unique_ptr<domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
SphericalCompression::block_maps_grid_to_inertial(
    const size_t number_of_blocks) const {
  ASSERT(number_of_blocks > 0,
         "Must have at least one block on which to create a map.");
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
      result{number_of_blocks};
  result[0] = std::make_unique<GridToInertialMap>(grid_to_inertial_map());
  for (size_t i = 1; i < number_of_blocks; ++i) {
    result[i] = result[0]->get_clone();
  }
  return result;
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
SphericalCompression::functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};

  // Functions of time don't expire by default
  double expiration_time = std::numeric_limits<double>::infinity();

  // If we have control systems, overwrite the expiration time with the one
  // supplied by the control system
  if (initial_expiration_times.count(function_of_time_name_) == 1) {
    expiration_time = initial_expiration_times.at(function_of_time_name_);
  }

  result[function_of_time_name_] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time_,
          std::array<DataVector, 4>{{{initial_value_},
                                     {initial_velocity_},
                                     {initial_acceleration_},
                                     {0.0}}},
          expiration_time);
  return result;
}

auto SphericalCompression::grid_to_inertial_map() const -> GridToInertialMap {
  return GridToInertialMap{SphericalCompressionMap{
      function_of_time_name_, min_radius_, max_radius_, center_}};
}

bool operator==(const SphericalCompression& lhs,
                const SphericalCompression& rhs) {
  return lhs.initial_time_ == rhs.initial_time_ and
         lhs.min_radius_ == rhs.min_radius_ and
         lhs.max_radius_ == rhs.max_radius_ and lhs.center_ == rhs.center_ and
         lhs.initial_value_ == rhs.initial_value_ and
         lhs.initial_velocity_ == rhs.initial_velocity_ and
         lhs.initial_acceleration_ == rhs.initial_acceleration_;
}

bool operator!=(const SphericalCompression& lhs,
                const SphericalCompression& rhs) {
  return not(lhs == rhs);
}
}  // namespace creators::time_dependence

using SphericalCompressionMap3d =
    CoordinateMaps::TimeDependent::SphericalCompression<false>;

INSTANTIATE_MAPS_FUNCTIONS(((SphericalCompressionMap3d)), (Frame::Grid),
                           (Frame::Inertial), (double, DataVector))

}  // namespace domain
