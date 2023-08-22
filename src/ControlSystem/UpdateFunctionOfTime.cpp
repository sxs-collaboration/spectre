// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/UpdateFunctionOfTime.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"

namespace control_system {
void UpdateSingleFunctionOfTime::apply(
    const gsl::not_null<std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
        f_of_t_list,
    const std::string& f_of_t_name, const double update_time,
    DataVector update_deriv, const double new_expiration_time) {
  (*f_of_t_list)
      .at(f_of_t_name)
      ->update(update_time, std::move(update_deriv), new_expiration_time);
}

void UpdateMultipleFunctionsOfTime::apply(
    const gsl::not_null<std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
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

void ResetFunctionOfTimeExpirationTime::apply(
    const gsl::not_null<std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
        f_of_t_list,
    const std::string& f_of_t_name, const double new_expiration_time) {
  (*f_of_t_list).at(f_of_t_name)->reset_expiration_time(new_expiration_time);
}
}  // namespace control_system
