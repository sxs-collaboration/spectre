// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <optional>

/// \cond
struct TimeStepRequest;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// Combine TimeStepRequest objects to find a consensus step.  See
/// that class for details.
class TimeStepRequestProcessor {
 public:
  TimeStepRequestProcessor() = default;
  explicit TimeStepRequestProcessor(bool time_runs_forward);

  /// The new goal, if one was given.
  std::optional<double> new_step_size_goal() const;

  /// The new step.  Two versions are provided, as neither computing
  /// the step size from the end or the end from the size is
  /// guaranteed to be an exact operation on floating-point values,
  /// and which quantity is fundamental varies between consumers.  The
  /// \p step_size_goal is ignored if any requests have set a new
  /// goal.
  /// @{
  double step_size(double step_start, double step_size_goal) const;
  double step_end(double step_start, double step_size_goal) const;
  /// @}

  void process(const TimeStepRequest& request);

  /// Merge the results from another object.
  TimeStepRequestProcessor& operator+=(const TimeStepRequestProcessor& other);

  /// ERROR if \p size and \p end do not satisfy the hard limits.
  void error_on_hard_limit(double size, double end) const;

  void pup(PUP::er& p);

 private:
  friend bool operator==(const TimeStepRequestProcessor& a,
                         const TimeStepRequestProcessor& b);

  bool time_runs_forward_ = false;

  std::optional<double> step_size_goal_{};

  double step_size_request_ = std::numeric_limits<double>::signaling_NaN();
  double step_end_request_ = std::numeric_limits<double>::signaling_NaN();

  bool can_use_larger_limit_ = false;
  double next_step_size_request_ = std::numeric_limits<double>::signaling_NaN();
  double next_step_end_request_ = std::numeric_limits<double>::signaling_NaN();

  double step_size_hard_limit_ = std::numeric_limits<double>::signaling_NaN();
  double step_end_hard_limit_ = std::numeric_limits<double>::signaling_NaN();
};

bool operator!=(const TimeStepRequestProcessor& a,
                const TimeStepRequestProcessor& b);

/// Combine the limits from two TimeStepRequestProcessor objects.
TimeStepRequestProcessor operator+(const TimeStepRequestProcessor& a,
                                   const TimeStepRequestProcessor& b);
