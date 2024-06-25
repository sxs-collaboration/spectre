// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/AdamsBashforth.hpp"

#include <algorithm>
#include <boost/iterator/transform_iterator.hpp>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <pup.h>

#include "Time/ApproximateTime.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/History.hpp"
#include "Time/SelfStart.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsCoefficients.hpp"
#include "Time/TimeSteppers/AdamsLts.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace TimeSteppers {

// Don't include AdamsCoefficients.hpp in the header just to get one
// constant.
static_assert(adams_coefficients::maximum_order ==
              AdamsBashforth::maximum_order);

namespace {
template <typename T>
using OrderVector = adams_coefficients::OrderVector<T>;

template <typename Iter>
struct TimeFromRecord {
  Time operator()(typename std::iterator_traits<Iter>::reference record) const {
    return record.time_step_id.step_time();
  }
};

// This must be templated on the iterator type rather than the math
// wrapper type because of quirks in the template deduction rules.
template <typename Iter>
auto history_time_iterator(const Iter& it) {
  return boost::transform_iterator(it, TimeFromRecord<Iter>{});
}
}  // namespace

AdamsBashforth::AdamsBashforth(const size_t order) : order_(order) {
  if (order_ < 1 or order_ > maximum_order) {
    ERROR("The order for Adams-Bashforth Nth order must be 1 <= order <= "
          << maximum_order);
  }
}

size_t AdamsBashforth::order() const { return order_; }

size_t AdamsBashforth::error_estimate_order() const { return order_ - 1; }

uint64_t AdamsBashforth::number_of_substeps() const { return 1; }

uint64_t AdamsBashforth::number_of_substeps_for_error() const { return 1; }

size_t AdamsBashforth::number_of_past_steps() const { return order_ - 1; }

double AdamsBashforth::stable_step() const {
  if (order_ == 1) {
    return 1.;
  }

  // This is the condition that the characteristic polynomial of the
  // recurrence relation defined by the method has the correct sign at
  // -1.  It is not clear whether this is sufficient for all orders,
  // but it is for the ones we support.
  const auto& coefficients =
      adams_coefficients::constant_adams_bashforth_coefficients(order_);
  double invstep = 0.;
  for (const auto coef : coefficients) {
    invstep = coef - invstep;
  }
  return 1. / invstep;
}

bool AdamsBashforth::monotonic() const { return true; }

TimeStepId AdamsBashforth::next_time_id(const TimeStepId& current_id,
                                        const TimeDelta& time_step) const {
  ASSERT(current_id.substep() == 0, "Adams-Bashforth should not have substeps");
  return current_id.next_step(time_step);
}

TimeStepId AdamsBashforth::next_time_id_for_error(
    const TimeStepId& current_id, const TimeDelta& time_step) const {
  return next_time_id(current_id, time_step);
}

bool AdamsBashforth::neighbor_data_required(
    const TimeStepId& next_substep_id,
    const TimeStepId& neighbor_data_id) const {
  return neighbor_data_id < next_substep_id;
}

bool AdamsBashforth::neighbor_data_required(
    const double dense_output_time, const TimeStepId& neighbor_data_id) const {
  return evolution_less<double>{neighbor_data_id.time_runs_forward()}(
      neighbor_data_id.substep_time(), dense_output_time);
}

void AdamsBashforth::pup(PUP::er& p) {
  LtsTimeStepper::pup(p);
  p | order_;
}

template <typename T>
void AdamsBashforth::update_u_impl(const gsl::not_null<T*> u,
                                   const ConstUntypedHistory<T>& history,
                                   const TimeDelta& time_step) const {
  ASSERT(history.size() == history.integration_order(),
         "Incorrect data to take an order-" << history.integration_order()
         << " step.  Have " << history.size() << " times, need "
         << history.integration_order());
  *u = *history.back().value;
  update_u_common(u, history, time_step);
}

template <typename T>
bool AdamsBashforth::update_u_impl(const gsl::not_null<T*> u,
                                   const gsl::not_null<T*> u_error,
                                   const ConstUntypedHistory<T>& history,
                                   const TimeDelta& time_step) const {
  ASSERT(history.size() == history.integration_order(),
         "Incorrect data to take an order-" << history.integration_order()
         << " step.  Have " << history.size() << " times, need "
         << history.integration_order());
  *u = *history.back().value;
  update_u_common(u, history, time_step);
  step_error(u_error, history, time_step);
  return true;
}

template <typename T>
void AdamsBashforth::clean_history_impl(
    const MutableUntypedHistory<T>& history) const {
  while (history.size() >= history.integration_order()) {
    history.pop_front();
  }
  if (history.size() > 1) {
    history.discard_value(history[history.size() - 2].time_step_id);
  }
}

template <typename T>
bool AdamsBashforth::dense_update_u_impl(const gsl::not_null<T*> u,
                                         const ConstUntypedHistory<T>& history,
                                         const double time) const {
  const ApproximateTimeDelta time_step{
      time - history.back().time_step_id.step_time().value()};
  update_u_common(u, history, time_step);
  return true;
}

template <typename T, typename Delta>
void AdamsBashforth::update_u_common(const gsl::not_null<T*> u,
                                     const ConstUntypedHistory<T>& history,
                                     const Delta& time_step) const {
  ASSERT(
      history.size() > 0,
      "Cannot meaningfully update the evolved variables with an empty history");
  const auto order = history.integration_order();
  ASSERT(order <= order_,
         "Requested integration order higher than integrator order");

  const auto history_start =
      history.end() -
      static_cast<typename ConstUntypedHistory<T>::difference_type>(order);
  const auto coefficients = adams_coefficients::coefficients(
      history_time_iterator(history_start),
      history_time_iterator(history.end()),
      history.back().time_step_id.step_time(),
      history.back().time_step_id.step_time() + time_step);

  auto coefficient = coefficients.begin();
  for (auto history_entry = history_start;
       history_entry != history.end();
       ++history_entry, ++coefficient) {
    *u += *coefficient * history_entry->derivative;
  }
}

template <typename T>
void AdamsBashforth::step_error(const gsl::not_null<T*> u_error,
                                const ConstUntypedHistory<T>& history,
                                const TimeDelta& time_step) const {
  ASSERT(
      history.size() > 0,
      "Cannot meaningfully update the evolved variables with an empty history");
  const auto order = history.integration_order();
  ASSERT(order <= order_,
         "Requested integration order higher than integrator order");

  const auto history_start =
      history.end() -
      static_cast<typename ConstUntypedHistory<T>::difference_type>(order);
  auto coefficients = adams_coefficients::coefficients(
      history_time_iterator(history_start),
      history_time_iterator(history.end()),
      history.back().time_step_id.step_time(),
      history.back().time_step_id.step_time() + time_step);
  const auto lower_order_coefficients = adams_coefficients::coefficients(
      history_time_iterator(history_start + 1),
      history_time_iterator(history.end()),
      history.back().time_step_id.step_time(),
      history.back().time_step_id.step_time() + time_step);
  for (size_t i = 0; i < lower_order_coefficients.size(); ++i) {
    coefficients[i + 1] -= lower_order_coefficients[i];
  }

  auto coefficient = coefficients.begin();
  auto history_entry = history_start;
  *u_error = *coefficient * history_entry->derivative;
  for (;;) {
    ++history_entry;
    ++coefficient;
    if (history_entry == history.end()) {
      return;
    }
    *u_error += *coefficient * history_entry->derivative;
  }
}

template <typename T>
bool AdamsBashforth::can_change_step_size_impl(
    const TimeStepId& time_id, const ConstUntypedHistory<T>& history) const {
  // We need to prevent the next step from occurring at the same time
  // as one already in the history.  The self-start code ensures this
  // can't happen during self-start, and it clearly can't happen
  // during normal evolution where the steps are monotonic, but during
  // the transition between them we have to worry about a step being
  // placed on a self-start time.  The self-start algorithm guarantees
  // the final state is safe for constant-time-step evolution, so we
  // just force that until we've passed all the self-start times.
  const evolution_less_equal<Time> less_equal{time_id.time_runs_forward()};
  return not ::SelfStart::is_self_starting(time_id) and
         alg::all_of(history, [&](const auto& record) {
           return less_equal(record.time_step_id.step_time(),
                             time_id.step_time());
         });
}

template <typename T>
void AdamsBashforth::add_boundary_delta_impl(
    const gsl::not_null<T*> result,
    const TimeSteppers::ConstBoundaryHistoryTimes& local_times,
    const TimeSteppers::ConstBoundaryHistoryTimes& remote_times,
    const TimeSteppers::BoundaryHistoryEvaluator<T>& coupling,
    const TimeDelta& time_step) const {
  const auto step_start = local_times.back().step_time();
  const size_t integration_order =
      local_times.integration_order(local_times.size() - 1);

  const adams_lts::AdamsScheme scheme{adams_lts::SchemeType::Explicit,
                                      integration_order};
  const auto lts_coefficients = adams_lts::lts_coefficients(
      local_times, remote_times, step_start, step_start + time_step, scheme,
      scheme, scheme);
  adams_lts::apply_coefficients(result, lts_coefficients, coupling);
}

void AdamsBashforth::clean_boundary_history_impl(
    const TimeSteppers::MutableBoundaryHistoryTimes& local_times,
    const TimeSteppers::MutableBoundaryHistoryTimes& remote_times) const {
  const size_t integration_order =
      local_times.integration_order(local_times.size() - 1);

  while (local_times.size() >= integration_order) {
    local_times.pop_front();
  }
  // We're guaranteed to have a new local value inserted before the
  // next use, but not a new remote value, so we need to keep one more
  // of these.
  while (remote_times.size() > integration_order) {
    remote_times.pop_front();
  }
}

template <typename T>
void AdamsBashforth::boundary_dense_output_impl(
    const gsl::not_null<T*> result,
    const TimeSteppers::ConstBoundaryHistoryTimes& local_times,
    const TimeSteppers::ConstBoundaryHistoryTimes& remote_times,
    const TimeSteppers::BoundaryHistoryEvaluator<T>& coupling,
    const double time) const {
  if (local_times.back().step_time().value() == time) {
    // Nothing to do.  The requested time is the start of the step,
    // which is the input value of `result`.
    return;
  }
  const auto current_order =
      local_times.integration_order(local_times.size() - 1);
  const adams_lts::AdamsScheme scheme{adams_lts::SchemeType::Explicit,
                                      current_order};
  const auto small_step_start =
      std::max(local_times.back(), remote_times.back()).step_time();
  const auto lts_coefficients =
      adams_lts::lts_coefficients(local_times, remote_times,
                                  local_times.back().step_time(),
                                  small_step_start, scheme, scheme, scheme) +
      adams_lts::lts_coefficients(local_times, remote_times, small_step_start,
                                  ApproximateTime{time}, scheme, scheme,
                                  scheme);
  adams_lts::apply_coefficients(result, lts_coefficients, coupling);
}

bool operator==(const AdamsBashforth& lhs, const AdamsBashforth& rhs) {
  return lhs.order_ == rhs.order_;
}

bool operator!=(const AdamsBashforth& lhs, const AdamsBashforth& rhs) {
  return not(lhs == rhs);
}

TIME_STEPPER_DEFINE_OVERLOADS(AdamsBashforth)
LTS_TIME_STEPPER_DEFINE_OVERLOADS(AdamsBashforth)
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::AdamsBashforth::my_PUP_ID = 0;  // NOLINT
