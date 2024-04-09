// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <utility>
#include <vector>

#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"

/// \cond
class TimeStepper;
/// \endcond

namespace TimeStepperTestUtils {

void check_multistep_properties(const TimeStepper& stepper);

void check_substep_properties(const TimeStepper& stepper);

void integrate_test(const TimeStepper& stepper, size_t order,
                    size_t number_of_past_steps, double integration_time,
                    double epsilon);

void integrate_test_explicit_time_dependence(const TimeStepper& stepper,
                                             size_t order,
                                             size_t number_of_past_steps,
                                             double integration_time,
                                             double epsilon);

void integrate_variable_test(const TimeStepper& stepper, size_t order,
                             size_t number_of_past_steps, double epsilon);

void integrate_error_test(const TimeStepper& stepper, size_t order,
                          size_t number_of_past_steps, double integration_time,
                          double epsilon, size_t num_steps,
                          double error_factor);

template <typename F1, typename F2, typename EvolvedType>
void initialize_history(
    Time time,
    const gsl::not_null<TimeSteppers::History<EvolvedType>*> history,
    F1&& analytic, F2&& rhs, TimeDelta step_size,
    const size_t number_of_past_steps) {
  int64_t slab_number = -1;
  for (size_t j = 0; j < number_of_past_steps; ++j) {
    ASSERT(time.slab() == step_size.slab(), "Slab mismatch");
    if ((step_size.is_positive() and time.is_at_slab_start()) or
        (not step_size.is_positive() and time.is_at_slab_end())) {
      const Slab new_slab = time.slab().advance_towards(-step_size);
      time = time.with_slab(new_slab);
      step_size = step_size.with_slab(new_slab);
      --slab_number;
    }
    time -= step_size;
    history->insert_initial(
        TimeStepId(step_size.is_positive(), slab_number, time),
        analytic(time.value()), rhs(analytic(time.value()), time.value()));
  }
}

/// Check that the reported stable step is correct.  The \p phase
/// argument is the phase that is most unstable in the differential
/// equation \f$y' = (\exp(i \phi) - 1) y\f$.  For most common time
/// steppers this is \f$\pi\f$.
void stability_test(const TimeStepper& stepper, double phase = M_PI);

/// Check that integration converges as expected.
///
/// The \p step_range argument specifies the range of the number of
/// steps used to produce a fit, and should be cover a factor of a few
/// over which a log-log plot of the error is roughly linear.  An
/// appropriate value can be determined by passing `true` as the \p
/// output argument, which will produce a `convergence.dat` file.
void check_convergence_order(const TimeStepper& stepper,
                             const std::pair<int32_t, int32_t>& step_range,
                             bool output = false);

void check_dense_output(
    const TimeStepper& stepper, size_t history_integration_order,
    const std::pair<int32_t, int32_t>& convergence_step_range, int32_t stride,
    bool check_backward_continuity);

void check_strong_stability_preservation(const TimeStepper& stepper,
                                         double step_size);

template <typename F>
double convergence_rate(
    const std::pair<int32_t, int32_t>& number_of_steps_range,
    const int32_t stride, F&& error, const bool output = false) {
  // We do a least squares fit on a log-log error-vs-steps plot.  The
  // unequal points caused by the log scale will introduce some bias,
  // but the typical range this is used for is only a factor of a few,
  // so it shouldn't be too bad.

  // Make sure testing code is not left enabled.
  CHECK(not output);

  std::ofstream output_stream{};
  if (output) {
    output_stream.open("convergence.dat");
    output_stream.precision(18);
  }

  uint32_t num_tests = 0;
  std::vector<double> log_steps;
  std::vector<double> log_errors;
  for (auto steps = number_of_steps_range.first;
       steps <= number_of_steps_range.second;
       steps += stride) {
    const double this_error = abs(error(steps));
    if (output) {
      output_stream << steps << "\t" << this_error << std::endl;
    }
    log_steps.push_back(log(steps));
    log_errors.push_back(log(this_error));
    ++num_tests;
  }
  const double average_log_steps = alg::accumulate(log_steps, 0.0) / num_tests;
  const double average_log_errors =
      alg::accumulate(log_errors, 0.0) / num_tests;
  double numerator = 0.0;
  double denominator = 0.0;
  for (size_t i = 0; i < num_tests; ++i) {
    numerator += (log_steps[i] - average_log_steps) *
        (log_errors[i] - average_log_errors);
    denominator += square(log_steps[i] - average_log_steps);
  }
  return -numerator / denominator;
}
}  // namespace TimeStepperTestUtils
