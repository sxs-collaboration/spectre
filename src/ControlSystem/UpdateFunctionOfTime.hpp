// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"

namespace control_system {
/// \ingroup ControlSystemGroup
/// Updates a FunctionOfTime in the global cache. Intended to be used in
/// Parallel::mutate.
struct UpdateSingleFunctionOfTime {
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

/*!
 * \ingroup ControlSystemGroup
 * \brief Updates several FunctionOfTimes in the global cache at once. Intended
 * to be used in Parallel::mutate.
 *
 * \details All functions of time are updated at the same `update_time`. For the
 * `update_args`, the keys of the map are the names of the functions of time.
 * The value `std::pair<DataVector, double>` for each key is the updated
 * derivative for the function of time and the new expiration time,
 * respectively.
 */
struct UpdateMultipleFunctionsOfTime {
  static void apply(
      const gsl::not_null<std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
          f_of_t_list,
      const double update_time,
      const std::unordered_map<std::string, std::pair<DataVector, double>>&
          update_args) {
    for (auto& [f_of_t_name, update_deriv_and_expr_time] : update_args) {
      UpdateSingleFunctionOfTime::apply(f_of_t_list, f_of_t_name, update_time,
                                        update_deriv_and_expr_time.first,
                                        update_deriv_and_expr_time.second);
    }
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
