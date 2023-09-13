// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <deque>
#include <optional>
#include <pup.h>
#include <utility>

/// \cond
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
/// \endcond

namespace control_system {
/// Class for computing the upcoming measurement times for a control
/// system measurement.
///
/// At any time, the \ref control_system::Tags::MeasurementTimescales
/// "measurement timescales" can be queried to find the spacing
/// between control system measurements.  Most uses, however, require
/// the actual measurement times, and sometimes need to know times of
/// control-system update triggers.  This class calculates these
/// quantities from the timescales.
class FutureMeasurements {
 public:
  FutureMeasurements() = default;

  FutureMeasurements(size_t measurements_per_update,
                     double first_measurement_time);

  /// Next measurement time, if known.
  std::optional<double> next_measurement() const;
  /// Next measurement that triggers an update, if known.
  std::optional<double> next_update() const;

  /// Remove the earliest measurement form the list, generally because
  /// it has been performed.
  void pop_front();

  /// Calculate and store measurement times up through the expiration
  /// time of the argument.
  ///
  /// Given a measurement time \f$t_i\f$, the next measurement will
  /// occur at \f$t_{i+1} = t_i + \tau_m(t_i)\f$, where
  /// \f$\tau_m(t)\f$ is the measurement timescale at time \f$t\f$.
  void update(
      const domain::FunctionsOfTime::FunctionOfTime& measurement_timescale);

  void pup(PUP::er& p);

 private:
  // This stores the most recent (or current) measurement time,
  // followed by some future measurement times.  We need to keep one
  // old entry that isn't returned so that we can use it to calculate
  // a new first entry if we need to update an "empty" list.  The
  // bookkeeping is simpler if we keep it around even when we don't
  // need it.
  std::deque<double> measurements_{};
  size_t measurements_until_update_{};
  size_t measurements_per_update_{};
};
}  // namespace control_system
