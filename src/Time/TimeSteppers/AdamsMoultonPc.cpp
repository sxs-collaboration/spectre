// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/AdamsMoultonPc.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <pup.h>

#include "Time/ApproximateTime.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/History.hpp"
#include "Time/SelfStart.hpp"
#include "Time/TimeSteppers/AdamsCoefficients.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace TimeSteppers {

// Don't include AdamsCoefficients.hpp in the header just to get one
// constant.
static_assert(adams_coefficients::maximum_order ==
              AdamsMoultonPc::maximum_order);

namespace {
template <typename T>
void clean_history(const MutableUntypedHistory<T>& history) {
  ASSERT(history.integration_order() > 1, "Cannot run below second order.");
  const auto required_points = history.integration_order() - 1;
  ASSERT(history.size() >= required_points,
         "Insufficient data to take an order-" << history.integration_order()
         << " step.  Have " << history.size() << " times, need "
         << required_points);
  if (history.at_step_start()) {
    history.clear_substeps();
    while (history.size() > required_points) {
      history.pop_front();
    }
    if (history.size() > 1) {
      history.discard_value(history[history.size() - 2].time_step_id);
    }
  } else {
    history.discard_value(history.substeps().back().time_step_id);
  }
}

template <typename T, typename TimeType>
void update_u_common(const gsl::not_null<T*> u,
                     const ConstUntypedHistory<T>& history,
                     const TimeType& step_end, const size_t method_order,
                     const bool corrector) {
  ASSERT(history.size() >= method_order - 1, "Insufficient history");
  // Pass in whether to run the predictor or corrector even though we
  // can compute it as a sanity check.
  ASSERT(corrector != history.substeps().empty(),
         "Applying predictor or corrector when expecting the other.");
  ASSERT(corrector != history.at_step_start(), "Unexpected new data");

  const auto used_history_begin =
      history.end() -
      static_cast<typename ConstUntypedHistory<T>::difference_type>(
          method_order - 1);
  adams_coefficients::OrderVector<Time> control_times{};
  std::transform(used_history_begin, history.end(),
                 std::back_inserter(control_times),
                 [](const auto& r) { return r.time_step_id.step_time(); });
  if (corrector) {
    control_times.push_back(
        history.back().time_step_id.step_time() +
        history.substeps().front().time_step_id.step_size());
  }
  const auto coefficients = adams_coefficients::coefficients(
      control_times.begin(), control_times.end(),
      history.back().time_step_id.step_time(), step_end);

  *u = *history.back().value;
  auto coefficient = coefficients.begin();
  for (auto history_entry = used_history_begin;
       history_entry != history.end();
       ++history_entry, ++coefficient) {
    *u += *coefficient * history_entry->derivative;
  }
  if (corrector) {
    *u += coefficients.back() * history.substeps().front().derivative;
  }
}
}  // namespace

AdamsMoultonPc::AdamsMoultonPc(const size_t order) : order_(order) {
  ASSERT(order >= minimum_order and order <= maximum_order,
         "Invalid order: " << order);
}

size_t AdamsMoultonPc::order() const { return order_; }

size_t AdamsMoultonPc::error_estimate_order() const { return order_ - 1; }

uint64_t AdamsMoultonPc::number_of_substeps() const { return 2; }

uint64_t AdamsMoultonPc::number_of_substeps_for_error() const {
  return number_of_substeps();
}

size_t AdamsMoultonPc::number_of_past_steps() const { return order_ - 2; }

double AdamsMoultonPc::stable_step() const {
  switch (order_) {
    case 2:
      return 1.0;
    case 3:
      return 0.981297;
    case 4:
      return 0.794227;
    case 5:
      return 0.612340;
    case 6:
      return 0.464542;
    case 7:
      return 0.350596;
    case 8:
      return 0.264373;
    default:
      ERROR("Bad order");
  }
}

TimeStepId AdamsMoultonPc::next_time_id(const TimeStepId& current_id,
                                        const TimeDelta& time_step) const {
  switch (current_id.substep()) {
    case 0:
      return current_id.next_substep(time_step, 1.0);
    case 1:
      return current_id.next_step(time_step);
    default:
      ERROR("Bad id: " << current_id);
  }
}

TimeStepId AdamsMoultonPc::next_time_id_for_error(
    const TimeStepId& current_id, const TimeDelta& time_step) const {
  return next_time_id(current_id, time_step);
}

void AdamsMoultonPc::pup(PUP::er& p) {
  TimeStepper::pup(p);
  p | order_;
}

template <typename T>
void AdamsMoultonPc::update_u_impl(const gsl::not_null<T*> u,
                                   const MutableUntypedHistory<T>& history,
                                   const TimeDelta& time_step) const {
  clean_history(history);
  const Time next_time = history.back().time_step_id.step_time() + time_step;
  update_u_common(u, history, next_time, history.integration_order(),
                  not history.at_step_start());
}

template <typename T>
bool AdamsMoultonPc::update_u_impl(const gsl::not_null<T*> u,
                                   const gsl::not_null<T*> u_error,
                                   const MutableUntypedHistory<T>& history,
                                   const TimeDelta& time_step) const {
  clean_history(history);
  const bool predictor = history.at_step_start();
  const Time next_time = history.back().time_step_id.step_time() + time_step;
  update_u_common(u, history, next_time, history.integration_order(),
                  not predictor);
  if (predictor) {
    return false;
  }
  update_u_common(u_error, history, next_time, history.integration_order() - 1,
                  true);
  *u_error = *u - *u_error;
  return true;
}

template <typename T>
bool AdamsMoultonPc::dense_update_u_impl(const gsl::not_null<T*> u,
                                         const ConstUntypedHistory<T>& history,
                                         const double time) const {
  // Special case required to handle the initial time.
  if (time == history.back().time_step_id.step_time().value()) {
    *u = *history.back().value;
    return true;
  }
  if (history.at_step_start()) {
    return false;
  }
  update_u_common(u, history, ApproximateTime{time},
                  history.integration_order(), true);
  return true;
}

template <typename T>
bool AdamsMoultonPc::can_change_step_size_impl(
    const TimeStepId& time_id, const ConstUntypedHistory<T>& history) const {
  // We need to forbid local time-stepping before initialization is
  // complete.  We need to wait during the main evolution so we don't
  // increase the step size and step to a time still in use from
  // self-start, as that would cause the coefficients to diverge.
  const evolution_less<Time> less{time_id.time_runs_forward()};
  return not ::SelfStart::is_self_starting(time_id) and
         alg::all_of(history, [&](const auto& record) {
           return less(record.time_step_id.step_time(), time_id.step_time());
         });
}

bool operator==(const AdamsMoultonPc& lhs, const AdamsMoultonPc& rhs) {
  return lhs.order() == rhs.order();
}

bool operator!=(const AdamsMoultonPc& lhs, const AdamsMoultonPc& rhs) {
  return not(lhs == rhs);
}

TIME_STEPPER_DEFINE_OVERLOADS(AdamsMoultonPc)
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::AdamsMoultonPc::my_PUP_ID = 0;  // NOLINT
