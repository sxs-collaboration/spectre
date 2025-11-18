// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ExpirationTimes.hpp"

#include "DataStructures/DataVector.hpp"

namespace control_system {
double function_of_time_expiration_time(
    const double time, const DataVector& old_measurement_timescales,
    const DataVector& new_measurement_timescales,
    const int measurements_per_update, const bool delay_update) {
  const double min_new_timescale = min(new_measurement_timescales);
  double expiration = time;
  if (delay_update) {
    expiration += min(old_measurement_timescales);
  }
  // Mathematically, this is just += measurements_per_update *
  // min_new_timescale, but that can differ by roundoff from repeated
  // addition.  Particularly with delay_update = false, it is critical
  // that the expiration time not be before the measurement time.  The
  // actual measurement times are calculated by adding the timescale
  // to the previous time, so we do that here to match.
  for (int i = 0; i < measurements_per_update; ++i) {
    expiration += min_new_timescale;
  }
  return expiration;
}

double function_of_time_initial_expiration_time(
    const double time, const DataVector& measurement_timescales,
    const int measurements_per_update, const bool delay_update) {
  return function_of_time_expiration_time(
      time, DataVector{}, measurement_timescales,
      delay_update ? measurements_per_update : measurements_per_update - 1,
      false);
}

double measurement_expiration_time(const double time,
                                   const DataVector& old_measurement_timescales,
                                   const DataVector& new_measurement_timescales,
                                   const int measurements_per_update,
                                   const bool delay_update) {
  return function_of_time_expiration_time(
             time, old_measurement_timescales, new_measurement_timescales,
             measurements_per_update, delay_update) -
         0.5 * min(new_measurement_timescales);
}

double measurement_initial_expiration_time(
    const double time, const DataVector& new_measurement_timescales,
    const int measurements_per_update, const bool delay_update) {
  return function_of_time_initial_expiration_time(
             time, new_measurement_timescales, measurements_per_update,
             delay_update) -
         0.5 * min(new_measurement_timescales);
}
}  // namespace control_system
