// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "DataStructures/Tensor/Identity.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

namespace domain {
namespace CoordinateMap_detail {
// @{
/// Call the map passing in the time and FunctionsOfTime if the map is
/// time-dependent
template <typename T, size_t Dim, typename Map>
void apply_map(
    const gsl::not_null<std::array<T, Dim>*> t_map_point, const Map& the_map,
    const double /*t*/,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
    /*functions_of_time*/,
    const std::false_type /*is_time_independent*/) {
  if (LIKELY(not the_map.is_identity())) {
    *t_map_point = the_map(*t_map_point);
  }
}

template <typename T, size_t Dim, typename Map>
void apply_map(
    const gsl::not_null<std::array<T, Dim>*> t_map_point, const Map& the_map,
    const double t,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const std::true_type
    /*is_time_dependent*/) {
  ASSERT(not functions_of_time.empty(),
         "A function of time must be present if the maps are time-dependent.");
  ASSERT(
      [t]() noexcept {
        disable_floating_point_exceptions();
        const bool isnan = std::isnan(t);
        enable_floating_point_exceptions();
        return not isnan;
      }(),
      "The time must not be NaN for time-dependent maps.");
  *t_map_point = the_map(*t_map_point, t, functions_of_time);
}

template <typename T, size_t Dim, typename Map>
auto apply_map(
    const Map& the_map, const std::array<T, Dim>& source_points,
    const double /*t*/,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
    /*functions_of_time*/,
    const std::false_type /*is_time_independent*/) {
  if (LIKELY(not the_map.is_identity())) {
    return the_map(source_points);
  }
  std::decay_t<decltype(the_map(source_points))> result{};
  for (size_t i = 0; i < result.size(); ++i) {
    gsl::at(result, i) = gsl::at(source_points, i);
  }
  return result;
}

template <typename T, size_t Dim, typename Map>
auto apply_map(
    const Map& the_map, const std::array<T, Dim>& source_points, const double t,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const std::true_type
    /*is_time_dependent*/) {
  // Note: We don't forward to the return-by-not-null version to avoid
  // additional allocations of the target points array. That is, we would
  // allocate the target points array once here, and then again inside the call
  // to the coordinate map.
  ASSERT(not functions_of_time.empty(),
         "A function of time must be present if the maps are time-dependent.");
  ASSERT(
      [t]() noexcept {
        disable_floating_point_exceptions();
        const bool isnan = std::isnan(t);
        enable_floating_point_exceptions();
        return not isnan;
      }(),
      "The time must not be NaN for time-dependent maps.");
  return the_map(source_points, t, functions_of_time);
}
// @}

// @{
template <typename T, size_t Dim, typename Map>
auto apply_inverse_map(
    const Map& the_map, const std::array<T, Dim>& target_points,
    const double /*t*/,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
    /*functions_of_time*/,
    const std::false_type /*is_time_independent*/) {
  if (LIKELY(not the_map.is_identity())) {
    return the_map.inverse(target_points);
  }
  using UnwrappedT = tt::remove_cvref_wrap_t<T>;
  std::decay_t<decltype(the_map.inverse(target_points))> result{
      std::array<UnwrappedT, Dim>{}};
  for (size_t i = 0; i < target_points.size(); ++i) {
    gsl::at(*result, i) = gsl::at(target_points, i);
  }
  return result;
}

template <typename T, size_t Dim, typename Map>
auto apply_inverse_map(
    const Map& the_map, const std::array<T, Dim>& target_points, const double t,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const std::true_type
    /*is_time_dependent*/) {
  ASSERT(not functions_of_time.empty(),
         "A function of time must be present if the maps are time-dependent.");
  ASSERT(
      [t]() noexcept {
        disable_floating_point_exceptions();
        const bool isnan = std::isnan(t);
        enable_floating_point_exceptions();
        return not isnan;
      }(),
      "The time must not be NaN for time-dependent maps.");
  return the_map.inverse(target_points, t, functions_of_time);
}
// @}

// @{
/// Compute the frame velocity
template <typename T, size_t Dim, typename Map>
auto apply_frame_velocity(
    const Map& /*the_map*/, const std::array<T, Dim>& source_points,
    const double /*t*/,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
    /*functions_of_time*/,
    const std::false_type /*is_time_independent*/) {
  return make_array<Map::dim, tt::remove_cvref_wrap_t<T>>(
      make_with_value<tt::remove_cvref_wrap_t<T>>(
          dereference_wrapper(source_points[0]), 0.0));
}

template <typename T, size_t Dim, typename Map>
auto apply_frame_velocity(
    const Map& the_map, const std::array<T, Dim>& source_points, const double t,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const std::true_type
    /*is_time_dependent*/) {
  ASSERT(not functions_of_time.empty(),
         "A function of time must be present if the maps are time-dependent.");
  ASSERT(
      [t]() noexcept {
        disable_floating_point_exceptions();
        const bool isnan = std::isnan(t);
        enable_floating_point_exceptions();
        return not isnan;
      }(),
      "The time must not be NaN for time-dependent maps.");
  return the_map.frame_velocity(source_points, t, functions_of_time);
}
// @}

// @{
/// Compute the Jacobian
template <typename T, size_t Dim, typename Map>
auto apply_jacobian(
    const Map& the_map, const std::array<T, Dim>& source_points,
    const double /*t*/,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
    /*functions_of_time*/,
    const std::false_type /*is_time_independent*/) {
  if (LIKELY(not the_map.is_identity())) {
    return the_map.jacobian(source_points);
  }
  return identity<Dim>(dereference_wrapper(source_points[0]));
}

template <typename T, size_t Dim, typename Map>
auto apply_jacobian(
    const Map& the_map, const std::array<T, Dim>& source_points, const double t,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const std::true_type
    /*is_time_dependent*/) {
  ASSERT(not functions_of_time.empty(),
         "A function of time must be present if the maps are time-dependent.");
  ASSERT(
      [t]() noexcept {
        disable_floating_point_exceptions();
        const bool isnan = std::isnan(t);
        enable_floating_point_exceptions();
        return not isnan;
      }(),
      "The time must not be NaN for time-dependent maps.");
  if (LIKELY(not the_map.is_identity())) {
    return the_map.jacobian(source_points, t, functions_of_time);
  }
  return identity<Dim>(dereference_wrapper(source_points[0]));
}
// @}

// @{
/// Compute the Jacobian
template <typename T, size_t Dim, typename Map>
auto apply_inverse_jacobian(
    const Map& the_map, const std::array<T, Dim>& source_points,
    const double /*t*/,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
    /*functions_of_time*/,
    const std::false_type /*is_time_independent*/) {
  if (LIKELY(not the_map.is_identity())) {
    return the_map.inv_jacobian(source_points);
  }
  return identity<Dim>(dereference_wrapper(source_points[0]));
}

template <typename T, size_t Dim, typename Map>
auto apply_inverse_jacobian(
    const Map& the_map, const std::array<T, Dim>& source_points, const double t,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const std::true_type
    /*is_time_dependent*/) {
  ASSERT(not functions_of_time.empty(),
         "A function of time must be present if the maps are time-dependent.");
  ASSERT(
      [t]() noexcept {
        disable_floating_point_exceptions();
        const bool isnan = std::isnan(t);
        enable_floating_point_exceptions();
        return not isnan;
      }(),
      "The time must not be NaN for time-dependent maps.");
  if (LIKELY(not the_map.is_identity())) {
    return the_map.inv_jacobian(source_points, t, functions_of_time);
  }
  return identity<Dim>(dereference_wrapper(source_points[0]));
}
// @}
}  // namespace CoordinateMap_detail
}  // namespace domain
