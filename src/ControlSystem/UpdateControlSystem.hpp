// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/CalculateMeasurementTimescales.hpp"
#include "ControlSystem/Controller.hpp"
#include "ControlSystem/ExpirationTimes.hpp"
#include "ControlSystem/Tags/IsActive.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "ControlSystem/UpdateFunctionOfTime.hpp"
#include "ControlSystem/WriteData.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace domain::Tags {
struct FunctionsOfTime;
}  // namespace domain::Tags

namespace control_system {
/*!
 * \brief Functor for updating control systems when they are ready.
 *
 * \details The apply operator of this struct is meant to be called by the
 * UpdateMessageQueue action once an entire measurement has been made.
 *
 * Requires a few tags to be in the DataBox of the ControlComponent that this
 * is running on:
 * - \link Tags::Averager Averager \endlink
 * - \link Tags::Controller Controller \endlink
 * - \link Tags::TimescaleTuner TimescaleTuner \endlink
 * - \link Tags::ControlError ControlError \endlink
 * - \link Tags::WriteDataToDisk WriteDataToDisk \endlink
 * - \link Tags::CurrentNumberOfMeasurements CurrentNumberOfMeasurements
 *   \endlink
 *
 * And the \link control_system::Tags::MeasurementsPerUpdate \endlink must be in
 * the GlobalCache. If these tags are not present, a build error will occur.
 *
 * The algorithm to determine whether or not to update the functions of time is
 * as follows:
 * 1. Ensure this control system is active. If it isn't, end here and don't
 *    process the measurements.
 * 2. Increment the current measurement stored by
 *    `control_system::Tags::CurrentNumberOfMeasurements`. This keeps track of
 *    which measurement we are on out of
 *    `control_system::Tags::MeasurementsPerUpdate`.
 * 3. Calculate the control error and update the averager (store the current
 *    measurement). If the averager doesn't have enough information to determine
 *    the derivatives of the control error, exit now (this usually only happens
 *    for the first couple measurements of a simulation).
 * 4. If the `control_system::Tags::WriteDataToDisk` tag is set to `true`, write
 *    the time, function of time values, and the control error and its
 *    derivative to disk.
 * 5. Determine if we need to update. We only want to update when the current
 *    measurement is equal to the number of measurements per update. If it's not
 *    time to update, exit now. Once we determine that it is time to update, set
 *    the current measurement back to 0.
 * 6. Update the damping timescales in the TimescaleTuner and compute the
 *    control signal using the control error and its derivatives.
 * 7. Calculate the new measurement timescale based off the updated damping
 *    timescales and the number of measurements per update.
 * 8. Determine the new expiration times for the
 *    `::domain::Tags::FunctionsOfTime` and
 *    `control_system::Tags::MeasurementTimescales`. Update the MaxDeriv of the
 *    functions of time with the control signal and update the measurement
 *    timescales with the new measurement timescale (both of these are
 *    `Parallel::mutate` calls).
 */
template <typename ControlSystem>
struct UpdateControlSystem {
  static constexpr size_t deriv_order = ControlSystem::deriv_order;

  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename... TupleTags>
  static void apply(const gsl::not_null<db::DataBox<DbTags>*> box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, const double time,
                    tuples::TaggedTuple<TupleTags...> data) {
    // Begin step 1
    // If this control system isn't active, don't do anything
    if (not get<control_system::Tags::IsActive<ControlSystem>>(*box)) {
      return;
    }

    // Begin step 2
    const int measurements_per_update =
        get<control_system::Tags::MeasurementsPerUpdate>(cache);
    int& current_measurement = db::get_mutable_reference<
        control_system::Tags::CurrentNumberOfMeasurements>(box);

    ++current_measurement;

    const auto& functions_of_time =
        Parallel::get<::domain::Tags::FunctionsOfTime>(cache);
    const std::string& function_of_time_name = ControlSystem::name();
    const auto& function_of_time = functions_of_time.at(function_of_time_name);

    // Begin step 3
    // Get the averager, controller, tuner, and control error from the box
    auto& averager = db::get_mutable_reference<
        control_system::Tags::Averager<ControlSystem>>(box);
    auto& controller = db::get_mutable_reference<
        control_system::Tags::Controller<ControlSystem>>(box);
    auto& tuner = db::get_mutable_reference<
        control_system::Tags::TimescaleTuner<ControlSystem>>(box);
    auto& control_error = db::get_mutable_reference<
        control_system::Tags::ControlError<ControlSystem>>(box);
    const DataVector& current_timescale = tuner.current_timescale();

    // Compute control error
    const DataVector Q =
        control_error(cache, time, function_of_time_name, data);

    if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Verbose) {
      Parallel::printf(
          "%s, time = %.16f: current measurement = %d, control error = %s\n",
          function_of_time_name, time, current_measurement, Q);
    }

    // Update the averager. We do this for every measurement because we still
    // want the averager to be up-to-date even if we aren't updating at this
    // time
    averager.update(time, Q, current_timescale);

    // Get the averaged values of the control error and its derivatives
    const auto& opt_avg_values = averager(time);

    if (not opt_avg_values.has_value()) {
      if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Verbose) {
        Parallel::printf(
            "%s, time = %.16f: Averager does not have enough data.\n",
            function_of_time_name, time);
      }
      return;
    }

    // Begin step 4
    // Write data for every measurement
    std::array<DataVector, 2> q_and_dtq{{Q, {Q.size(), 0.0}}};
    q_and_dtq[0] = (*opt_avg_values)[0];
    if constexpr (deriv_order > 1) {
      q_and_dtq[1] = (*opt_avg_values)[1];
    }

    if (Parallel::get<control_system::Tags::WriteDataToDisk>(cache)) {
      // LCOV_EXCL_START
      write_components_to_disk<ControlSystem>(time, cache, function_of_time,
                                              q_and_dtq, current_timescale);
      // LCOV_EXCL_STOP
    }

    // Begin step 5
    // Check if it is time to update
    if (current_measurement != measurements_per_update) {
      if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Verbose) {
        Parallel::printf(
            "%s, time = %.16f: current measurement = %d is not at "
            "measurements_per_update = %d. Waiting for more data.\n",
            function_of_time_name, time, current_measurement,
            measurements_per_update);
      }
      return;
    }

    // Set current measurement back to 0 because we're updating now
    current_measurement = 0;

    // Begin step 6
    const double time_offset =
        averager.last_time_updated() - averager.average_time(time);
    const double time_offset_0th =
        averager.using_average_0th_deriv_of_q() ? time_offset : 0.0;

    tuner.update_timescale(q_and_dtq);

    // Calculate the control signal which will be used to update the highest
    // derivative of the FunctionOfTime
    const DataVector control_signal =
        controller(time, current_timescale, opt_avg_values.value(),
                   time_offset_0th, time_offset);

    // Begin step 7
    // Calculate new measurement timescales with updated damping timescales
    const DataVector new_measurement_timescale =
        calculate_measurement_timescales(controller, tuner,
                                         measurements_per_update);

    const auto& measurement_timescales =
        Parallel::get<Tags::MeasurementTimescales>(cache);
    const auto& measurement_timescale =
        measurement_timescales.at(function_of_time_name);
    const double current_fot_expiration_time =
        function_of_time->time_bounds()[1];
    const double current_measurement_expiration_time =
        measurement_timescale->time_bounds()[1];
    // This call is ok because the measurement timescales are still valid
    // because the measurement timescales expire half a measurement after this
    // time.
    const DataVector old_measurement_timescale =
        measurement_timescale->func(time)[0];

    // Begin step 8
    // Calculate the next expiration times for both the functions of time and
    // the measurement timescales based on the current time. Then, actually
    // update the functions of time and measurement timescales
    const double new_fot_expiration_time = function_of_time_expiration_time(
        time, old_measurement_timescale, new_measurement_timescale,
        measurements_per_update);

    const double new_measurement_expiration_time = measurement_expiration_time(
        time, old_measurement_timescale, new_measurement_timescale,
        measurements_per_update);

    if (Parallel::get<Tags::Verbosity>(cache) >= ::Verbosity::Verbose) {
      Parallel::printf("%s, time = %.16f: Control signal = %s\n",
                       function_of_time_name, time, control_signal);
      Parallel::printf(
          "%s, time = %.16f: Updating the functions of time.\n"
          " min(old_measure_timescale) = %.16f\n"
          " min(new_measure_timescale) = %.16f\n"
          " old_measure_expr_time = %.16f\n"
          " new_measure_expr_time = %.16f\n"
          " old_function_expr_time = %.16f\n"
          " new_function_expr_time = %.16f\n",
          function_of_time_name, time, min(old_measurement_timescale),
          min(new_measurement_timescale), current_measurement_expiration_time,
          new_measurement_expiration_time, current_fot_expiration_time,
          new_fot_expiration_time);
    }

    Parallel::mutate<::domain::Tags::FunctionsOfTime, UpdateFunctionOfTime>(
        cache, function_of_time_name, current_fot_expiration_time,
        control_signal, new_fot_expiration_time);

    Parallel::mutate<Tags::MeasurementTimescales, UpdateFunctionOfTime>(
        cache, function_of_time_name, current_measurement_expiration_time,
        new_measurement_timescale, new_measurement_expiration_time);
  }
};
}  // namespace control_system
