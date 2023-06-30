// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <deque>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "ControlSystem/ControlErrors/Size/AhSpeed.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaR.hpp"
#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/ControlErrors/Size/Initial.hpp"
#include "ControlSystem/ControlErrors/Size/RegisterDerivedWithCharm.hpp"
#include "ControlSystem/ControlErrors/Size/StateHistory.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace control_system::size {
namespace {
void test_state_history(const size_t num_times_to_store) {
  CAPTURE(num_times_to_store);
  Info info{
      std::make_unique<States::Initial>(), 1.0, 1.0, 1.0, std::nullopt, false};
  ControlErrorArgs control_error_args{1.0, 1.0, 1.0, 1.0};

  StateHistory state_history{num_times_to_store};
  const std::vector<size_t> states{0, 1, 2};

  // Test that as we fill up the history, we have the expected number of stored
  // entries and that they are the correct values, for each state
  for (size_t i = 0; i < num_times_to_store; i++) {
    const double time = static_cast<double>(i);
    state_history.store(time, info, control_error_args);

    for (const size_t state : states) {
      CAPTURE(state);
      const auto history = state_history.state_history(state);
      CHECK(history.size() == i + 1);
      for (size_t j = 0; j < history.size(); j++) {
        double stored_time, control_error;
        std::tie(stored_time, control_error) = history[j];
        CHECK(static_cast<double>(j) == stored_time);
        // These are hand calculated from the above parameters Info and
        // ControlErrorArgs.
        switch (state) {
          case 0:
            CHECK(control_error == 0.0);
            break;
          case 1:
            CHECK(control_error == 0.0);
            break;
          case 2:
            CHECK(control_error == 1.0);
            break;
          default:
            ERROR("Unknown state: " << state);
        }
      }
    }
  }

  // Test that trying to store one more value causes us to pop off the initial
  // (first) value and add this new one to the end (last), while keeping the
  // total number of entries at num_times_to_store
  state_history.store(static_cast<double>(num_times_to_store), info,
                      control_error_args);
  for (const size_t state : states) {
    const auto history = state_history.state_history(state);
    CHECK(history.size() == num_times_to_store);
    CHECK(history.front().first == 1.0);
    CHECK(history.back().first == static_cast<double>(num_times_to_store));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.ControlErrors.StateHistory",
                  "[Domain][Unit]") {
  control_system::size::register_derived_with_charm();
  for (size_t num_times = 1; num_times < 5; num_times++) {
    test_state_history(num_times);
  }
}
}  // namespace control_system::size
