// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ExpirationTimes.hpp"

#include "DataStructures/DataVector.hpp"

namespace control_system {
double function_of_time_expiration_time(
    const double time, const DataVector& old_measurement_timescales,
    const DataVector& new_measurement_timescales,
    const int measurements_per_update, const bool delay_update) {
  const int new_intervals =
      delay_update ? measurements_per_update : measurements_per_update - 1;
  const double min_new_timescale = min(new_measurement_timescales);
  double expiration = time + min(old_measurement_timescales);
  // Mathematically, this is just += new_intervals *
  // min_new_timescale, but that can differ by roundoff from repeated
  // addition.  Particularly with delay_update = false, it is critical
  // that the expiration time not be before the measurement time.  The
  // actual measurement times are calculated by adding the timescale
  // to the previous time, so we do that here to match.
  for (int i = 0; i < new_intervals; ++i) {
    expiration += min_new_timescale;
  }
  return expiration;
}

double measurement_expiration_time(const double time,
                                   const DataVector& old_measurement_timescales,
                                   const DataVector& new_measurement_timescales,
                                   const int measurements_per_update) {
  return time + min(old_measurement_timescales) +
         (measurements_per_update - 0.5) * min(new_measurement_timescales);
}
}  // namespace control_system
