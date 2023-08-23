// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/UpdateFunctionOfTime.hpp"

#include <limits>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

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

UpdateAggregator::UpdateAggregator(
    std::string combined_name,
    std::unordered_set<std::string> active_control_system_names)
    : active_names_(std::move(active_control_system_names)),
      combined_name_(std::move(combined_name)) {}

void UpdateAggregator::insert(const std::string& control_system_name,
                              const DataVector& new_measurement_timescale,
                              const double new_measurement_expiration_time,
                              DataVector control_signal,
                              const double new_fot_expiration_time) {
  ASSERT(expiration_times_.count(control_system_name) == 0,
         "Already received expiration time data for control system '"
             << control_system_name << "'.");
  ASSERT(active_names_.count(control_system_name) == 1,
         "Received expiration time data for a non-active control system '"
             << control_system_name << "'. Active control systems are "
             << active_names_);

  expiration_times_[control_system_name] = std::make_pair(
      std::make_pair(std::move(control_signal), new_fot_expiration_time),
      std::make_pair(min(new_measurement_timescale),
                     new_measurement_expiration_time));
}

bool UpdateAggregator::is_ready() const {
  // Short circuit if one name isn't ready
  for (const std::string& control_system_name : active_names_) {
    if (expiration_times_.count(control_system_name) != 1) {
      return false;
    }
  }
  return true;
}

const std::string& UpdateAggregator::combined_name() const {
  return combined_name_;
}

std::unordered_map<std::string, std::pair<DataVector, double>>
UpdateAggregator::combined_fot_expiration_times() const {
  ASSERT(is_ready(),
         "Trying to get combined expiration times, but have not received "
         "data from all control systems.");

  std::unordered_map<std::string, std::pair<DataVector, double>> result{};

  double min_expiration_time = std::numeric_limits<double>::infinity();
  for (const auto& control_system_name : active_names_) {
    min_expiration_time =
        std::min(min_expiration_time,
                 expiration_times_.at(control_system_name).first.second);
  }

  for (const auto& control_system_name : active_names_) {
    result[control_system_name] =
        expiration_times_.at(control_system_name).first;
    result[control_system_name].second = min_expiration_time;
  }

  return result;
}

std::pair<double, double>
UpdateAggregator::combined_measurement_expiration_time() {
  ASSERT(is_ready(),
         "Trying to get combined expiration times, but have not received "
         "data from all control systems.");

  double min_measurement_timescale = std::numeric_limits<double>::infinity();
  double min_expiration_time = std::numeric_limits<double>::infinity();

  for (const auto& control_system_name : active_names_) {
    min_measurement_timescale =
        std::min(min_measurement_timescale,
                 expiration_times_[control_system_name].second.first);
    min_expiration_time =
        std::min(min_expiration_time,
                 expiration_times_[control_system_name].second.second);
  }

  expiration_times_.clear();

  return std::make_pair(min_measurement_timescale, min_expiration_time);
}

void UpdateAggregator::pup(PUP::er& p) {
  p | expiration_times_;
  p | active_names_;
  p | combined_name_;
}
}  // namespace control_system
