// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"

namespace control_system {
/// \ingroup ControlSystemGroup
/// Updates a FunctionOfTime in the global cache. Intended to be used in
/// Parallel::mutate.
struct UpdateFunctionOfTime {
  static void apply(
      const gsl::not_null<std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
          f_of_t_list,
      const std::string& f_of_t_name, const double update_time,
      DataVector update_deriv, const double new_expiration_time) {
    (*f_of_t_list)
        .at(f_of_t_name)
        ->update(update_time, std::move(update_deriv), new_expiration_time);
  }
};

/// \ingroup ControlSystemGroup
/// Resets the expiration time of a FunctionOfTime in the global cache. Intended
/// to be used in Parallel::mutate.
struct ResetFunctionOfTimeExpirationTime {
  static void apply(
      const gsl::not_null<std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
          f_of_t_list,
      const std::string& f_of_t_name, const double new_expiration_time) {
    (*f_of_t_list).at(f_of_t_name)->reset_expiration_time(new_expiration_time);
  }
};
}  // namespace control_system
