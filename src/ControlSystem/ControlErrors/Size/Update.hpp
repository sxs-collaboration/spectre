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
#include "Parallel/Printf.hpp"
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

/*!
 * \brief Updates the TimescaleTuner with information from the \link
 * control_system::ControlErrors::Size Size control error \endlink.
 *
 * \details We check for a suggested timescale from the \link
 * control_system::ControlErrors::Size Size control error \endlink. If one is
 * suggested and it is smaller than the current damping timescale, we set the
 * timescale in the TimescaleTuner to this suggested value. Otherwise, we let
 * the TimescaleTuner adjust the timescale. Regardless of whether a timescale
 * was suggested or not, we always reset the size control error.
 */
template <size_t DerivOrder, ::domain::ObjectLabel Horizon,
          typename Metavariables>
void update_tuner(const gsl::not_null<TimescaleTuner*> tuner,
                  const gsl::not_null<ControlErrors::Size<DerivOrder, Horizon>*>
                      control_error,
                  const Parallel::GlobalCache<Metavariables>& cache,
                  const double time, const std::string& function_of_time_name) {
  const std::optional<double>& suggested_timescale =
      control_error->get_suggested_timescale();
  const double old_timescale = min(tuner->current_timescale());

  if (suggested_timescale.value_or(std::numeric_limits<double>::infinity()) <
      old_timescale) {
    tuner->set_timescale_if_in_allowable_range(suggested_timescale.value());
  }

  if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Verbose) {
    using ::operator<<;
    Parallel::printf(
        "%s, time = %.16f:\n"
        " old_timescale = %.16f\n"
        " suggested_timescale = %s\n"
        " new_timescale = %.16f\n",
        function_of_time_name, time, old_timescale,
        MakeString{} << suggested_timescale, min(tuner->current_timescale()));
  }

  // This reset call is ok because if there was a discontinuous change
  // above after calculating the control error, the control error class was
  // already reset so this reset won't do anything. If there wasn't a
  // discontinuous change, then this will only affect the suggested
  // timescale, which we always want to reset.
  control_error->reset();
}
}  // namespace control_system::size
