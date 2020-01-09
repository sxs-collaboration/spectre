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
#include "Utilities/TypeTraits.hpp"

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
// @}
}  // namespace CoordinateMap_detail
}  // namespace domain
