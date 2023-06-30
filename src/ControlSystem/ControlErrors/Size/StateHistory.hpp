// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <deque>
#include <pup.h>
#include <unordered_map>
#include <utility>

#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"

namespace control_system::size {
/*!
 * \brief A struct for holding a history of control errors for each state in the
 * `control_system::Systems::Size` control system.
 */
struct StateHistory {
  StateHistory();

  /// \brief Only keep `num_times_to_store` entries in the state_history
  StateHistory(size_t num_times_to_store);

  /*!
   * \brief Store the control errors for all `control_system::size::State`s.
   *
   * \param time Time to store control errors at
   * \param info `control_system::size::Info`
   * \param control_error_args `control_system::size::ControlErrorArgs`
   */
  void store(double time, const Info& info,
             const ControlErrorArgs& control_error_args);

  /*!
   * \brief Return a const reference to the stored control errors from all the
   * states.
   *
   * \param state_number `size_t` corresponding to the
   * `control_system::size::State::number()` of a state.
   * \return std::deque<std::pair<double, double>> The `std::pair` holds
   * the time and control error, respectively. The `std::deque` is ordered with
   * earlier times at the "front" and later times at the "back". This is to make
   * iteration over the deque easier as we typically want to start with earlier
   * times.
   */
  const std::deque<std::pair<double, double>>& state_history(
      size_t state_number) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  void initialize_stored_control_errors();

  size_t num_times_to_store_{};
  std::unordered_map<size_t, std::deque<std::pair<double, double>>>
      stored_control_errors_{};
};
}  // namespace control_system::size
