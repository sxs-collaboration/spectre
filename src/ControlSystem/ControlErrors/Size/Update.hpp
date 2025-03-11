// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <string>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/ControlErrors/Size.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/Gsl.hpp"

namespace control_system::size {
/*!
 * \brief Updates the Averager with information from the \link
 * control_system::ControlErrors::Size Size control error \endlink.
 *
 * \details After we calculate the control error, we check if a discontinuous
 * change has occurred (the internal `control_system::size::State` changed). If
 * no discontinuous change has occurred, then we do nothing. If one did occur,
 * we get a history of control errors from the \link
 * control_system::ControlErrors::Size Size control error \endlink and use that
 * to repopulate the Averager history.
 *
 * \note See `control_system::Tags::Averager` for why the Averager is templated
 * on `DerivOrder - 1`.
 */
template <size_t DerivOrder, ::domain::ObjectLabel Horizon,
          typename Metavariables>
void update_averager(
    const gsl::not_null<Averager<DerivOrder - 1>*> averager,
    const gsl::not_null<ControlErrors::Size<DerivOrder, Horizon>*>
        control_error,
    const Parallel::GlobalCache<Metavariables>& cache, const double time,
    const DataVector& current_timescale,
    const std::string& function_of_time_name, const int current_measurement) {
  if (control_error->discontinuous_change_has_occurred()) {
    control_error->reset();
    averager->clear();

    const std::deque<std::pair<double, double>> control_error_history =
        control_error->control_error_history();

    if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Verbose) {
      std::vector<double> history_times{};
      for (const auto& history : control_error_history) {
        history_times.push_back(history.first);
      }
      Parallel::printf(
          "%s, time = %.16f: current measurement = %d, Discontinuous "
          "change has occurred. Repopulating averager with control error "
          "history from times: %s\n",
          function_of_time_name, time, current_measurement, history_times);
    }

    for (const auto& [stored_time, stored_control_error] :
         control_error_history) {
      // Size 1 because it's size control
      averager->update(stored_time, DataVector{1, stored_control_error},
                       current_timescale);
    }
  }
}
}  // namespace control_system::size
