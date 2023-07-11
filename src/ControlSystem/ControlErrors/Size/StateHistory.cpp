// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/Size/StateHistory.hpp"

#include <deque>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <unordered_map>
#include <utility>

#include "ControlSystem/ControlErrors/Size/AhSpeed.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaR.hpp"
#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/ControlErrors/Size/Initial.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"
#include "DataStructures/DataVector.hpp"

namespace control_system::size {
StateHistory::StateHistory() { initialize_stored_control_errors(); }

StateHistory::StateHistory(const size_t num_times_to_store)
    : num_times_to_store_(num_times_to_store) {
  initialize_stored_control_errors();
}

void StateHistory::initialize_stored_control_errors() {
  stored_control_errors_[States::Initial{}.number()];
  stored_control_errors_[States::DeltaR{}.number()];
  stored_control_errors_[States::AhSpeed{}.number()];
}

void StateHistory::store(double time, const Info& info,
                         const ControlErrorArgs& control_error_args) {
  const auto store_state = [this, &time, &info,
                            &control_error_args](auto state) {
    const double control_error = state.control_error(info, control_error_args);
    std::deque<std::pair<double, double>>& history =
        stored_control_errors_.at(state.number());
    history.emplace_back(time, control_error);
    while (history.size() > num_times_to_store_) {
      history.pop_front();
    }
  };

  store_state(States::Initial{});
  store_state(States::DeltaR{});
  store_state(States::AhSpeed{});
}

const std::deque<std::pair<double, double>>& StateHistory::state_history(
    const size_t state_number) const {
  return stored_control_errors_.at(state_number);
}

void StateHistory::pup(PUP::er& p) {
  p | num_times_to_store_;
  p | stored_control_errors_;
}
}  // namespace control_system::size
