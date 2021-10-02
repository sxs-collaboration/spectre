// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/DataVector.hpp"
#include "Options/Options.hpp"

/// \cond
namespace PUP {
class er;
}
/// \endcond

/*!
 * \ingroup ControlSystemGroup
 * \brief Manages control system timescales
 *
 * The TimescaleTuner adjusts the damping timescale, \f$\tau\f$, of the control
 * system.\n The damping timescale is restricted to
 * `min_timescale`\f$\le\tau\le\f$`max_timescale`
 *
 * The damping time is adjusted according to the following criteria:
 *
 * **Decrease** the timescale by a factor of `decrease_factor` if either \n
 * - the error is too large: \f$|Q| >\f$ `decrease_timescale_threshold`
 * OR
 * the error is changing quickly: \f$|\dot{Q}|\tau >\f$
 * `decrease_timescale_threshold`,\n
 * AND \n
 * - the error is growing: \f$\dot{Q}Q > 0\f$
 * OR
 * the expected change in \f$Q\f$ is less than half its current value:
 * \f$|\dot{Q}|\tau < |Q|/2\f$
 *
 * **Increase** the timescale by a factor of `increase_factor` if \n
 * - the error is sufficiently small: \f$|Q|<\f$ `increase_timescale_threshold`
 * \n
 * AND \n
 * - the expected change in \f$Q\f$ is less than the difference between the
 * current error and the threshold:
 * \f$|\dot{Q}|\tau < \f$ (`increase_timescale_threshold` \f$-|Q|\f$)
 */

class TimescaleTuner {
 public:
  static constexpr Options::String help{
      "TimescaleTuner: stores and dynamically updates the timescales for each "
      "component of a particular control system."};
  struct InitialTimescales {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "Initial timescales for each function of time"};
  };

  struct MinTimescale {
    using type = double;
    static constexpr Options::String help = {"Minimum timescale"};
  };

  struct MaxTimescale {
    using type = double;
    static constexpr Options::String help = {"Maximum timescale"};
  };

  struct DecreaseThreshold {
    using type = double;
    static constexpr Options::String help = {
        "Threshold for decrease of timescale"};
  };
  struct IncreaseThreshold {
    using type = double;
    static constexpr Options::String help = {
        "Threshold for increase of timescale"};
  };
  struct IncreaseFactor {
    using type = double;
    static constexpr Options::String help = {"Factor to increase timescale"};
  };
  struct DecreaseFactor {
    using type = double;
    static constexpr Options::String help = {"Factor to decrease timescale"};
  };

  using options = tmpl::list<InitialTimescales, MaxTimescale, MinTimescale,
                             DecreaseThreshold, IncreaseThreshold,
                             IncreaseFactor, DecreaseFactor>;

  TimescaleTuner(const std::vector<double>& initial_timescale,
                 double max_timescale, double min_timescale,
                 double decrease_timescale_threshold,
                 double increase_timescale_threshold, double increase_factor,
                 double decrease_factor);

  TimescaleTuner() = default;
  TimescaleTuner(TimescaleTuner&&) = default;
  TimescaleTuner& operator=(TimescaleTuner&&) = default;
  TimescaleTuner(const TimescaleTuner&) = delete;
  TimescaleTuner& operator=(const TimescaleTuner&) = delete;
  ~TimescaleTuner() = default;

  /// returns the current timescale for each component of a FunctionOfTime
  const DataVector& current_timescale() const { return timescale_; }
  /// manually sets all timescales to a specified value, unless the value is
  /// outside of the specified minimum and maximum timescale bounds, in which
  /// case it is set to the nearest bounded value
  void set_timescale_if_in_allowable_range(double suggested_timescale);
  /// the update function responsible for modifying the timescale based on
  /// the control system errors
  void update_timescale(const std::array<DataVector, 2>& q_and_dtq);

  void pup(PUP::er& p);

  friend bool operator==(const TimescaleTuner& lhs, const TimescaleTuner& rhs);

 private:
  DataVector timescale_;
  double max_timescale_;
  double min_timescale_;
  double decrease_timescale_threshold_;
  double increase_timescale_threshold_;
  double increase_factor_;
  double decrease_factor_;
};

bool operator!=(const TimescaleTuner& lhs, const TimescaleTuner& rhs);
