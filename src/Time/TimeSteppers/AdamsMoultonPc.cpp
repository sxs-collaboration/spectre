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
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace TimeSteppers {

// Don't include AdamsCoefficients.hpp in the header just to get one
// constant.
static_assert(adams_coefficients::maximum_order ==
              AdamsMoultonPc<false>::maximum_order);

namespace {
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

template <bool Monotonic>
AdamsMoultonPc<Monotonic>::AdamsMoultonPc(const size_t order) : order_(order) {
  ASSERT(order >= minimum_order and order <= maximum_order,
         "Invalid order: " << order);
}

template <bool Monotonic>
size_t AdamsMoultonPc<Monotonic>::order() const {
  return order_;
}

template <bool Monotonic>
size_t AdamsMoultonPc<Monotonic>::error_estimate_order() const {
  return order_ - 1;
}

template <bool Monotonic>
uint64_t AdamsMoultonPc<Monotonic>::number_of_substeps() const {
  return 2;
}

template <bool Monotonic>
uint64_t AdamsMoultonPc<Monotonic>::number_of_substeps_for_error() const {
  return number_of_substeps();
}

template <bool Monotonic>
size_t AdamsMoultonPc<Monotonic>::number_of_past_steps() const {
  return order_ - 2;
}

template <bool Monotonic>
double AdamsMoultonPc<Monotonic>::stable_step() const {
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

template <bool Monotonic>
bool AdamsMoultonPc<Monotonic>::monotonic() const {
  return Monotonic;
}

template <bool Monotonic>
TimeStepId AdamsMoultonPc<Monotonic>::next_time_id(
    const TimeStepId& current_id, const TimeDelta& time_step) const {
  switch (current_id.substep()) {
    case 0:
      return current_id.next_substep(time_step, 1.0);
    case 1:
      return current_id.next_step(time_step);
    default:
      ERROR("Bad id: " << current_id);
  }
}

template <bool Monotonic>
TimeStepId AdamsMoultonPc<Monotonic>::next_time_id_for_error(
    const TimeStepId& current_id, const TimeDelta& time_step) const {
  return next_time_id(current_id, time_step);
}

template <bool Monotonic>
void AdamsMoultonPc<Monotonic>::pup(PUP::er& p) {
  TimeStepper::pup(p);
  p | order_;
}

template <bool Monotonic>
template <typename T>
void AdamsMoultonPc<Monotonic>::update_u_impl(
    const gsl::not_null<T*> u, const ConstUntypedHistory<T>& history,
    const TimeDelta& time_step) const {
  const Time next_time = history.back().time_step_id.step_time() + time_step;
  *u = *history.back().value;
  update_u_common(u, history, next_time, history.integration_order(),
                  not history.at_step_start());
}

template <bool Monotonic>
template <typename T>
bool AdamsMoultonPc<Monotonic>::update_u_impl(
    const gsl::not_null<T*> u, const gsl::not_null<T*> u_error,
    const ConstUntypedHistory<T>& history, const TimeDelta& time_step) const {
  const bool predictor = history.at_step_start();
  const Time next_time = history.back().time_step_id.step_time() + time_step;
  *u = *history.back().value;
  update_u_common(u, history, next_time, history.integration_order(),
                  not predictor);
  if (predictor) {
    return false;
  }
  *u_error = *history.back().value;
  update_u_common(u_error, history, next_time, history.integration_order() - 1,
                  true);
  *u_error = *u - *u_error;
  return true;
}

template <bool Monotonic>
template <typename T>
void AdamsMoultonPc<Monotonic>::clean_history_impl(
    const MutableUntypedHistory<T>& history) const {
  if (not history.at_step_start()) {
    ASSERT(history.integration_order() > 1, "Cannot run below second order.");
    const auto required_points = history.integration_order() - 2;
    ASSERT(history.size() >= required_points,
           "Insufficient data to take an order-" << history.integration_order()
           << " step.  Have " << history.size() << " times, need "
           << required_points);
    history.clear_substeps();
    while (history.size() > required_points) {
      history.pop_front();
    }
    if (not history.empty()) {
      history.discard_value(history.back().time_step_id);
    }
  }
}

template <bool Monotonic>
template <typename T>
bool AdamsMoultonPc<Monotonic>::dense_update_u_impl(
    const gsl::not_null<T*> u, const ConstUntypedHistory<T>& history,
    const double time) const {
  if constexpr (Monotonic) {
    if (not history.at_step_start()) {
      return false;
    }
    update_u_common(u, history, ApproximateTime{time},
                    history.integration_order(), false);
    return true;
  } else {
    if (history.at_step_start()) {
      return false;
    }
    update_u_common(u, history, ApproximateTime{time},
                    history.integration_order(), true);
    return true;
  }
}

template <bool Monotonic>
template <typename T>
bool AdamsMoultonPc<Monotonic>::can_change_step_size_impl(
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

template <bool Monotonic>
bool operator==(const AdamsMoultonPc<Monotonic>& lhs,
                const AdamsMoultonPc<Monotonic>& rhs) {
  return lhs.order() == rhs.order();
}

template <bool Monotonic>
bool operator!=(const AdamsMoultonPc<Monotonic>& lhs,
                const AdamsMoultonPc<Monotonic>& rhs) {
  return not(lhs == rhs);
}

TIME_STEPPER_DEFINE_OVERLOADS_TEMPLATED(AdamsMoultonPc<Monotonic>,
                                        bool Monotonic)

template <bool Monotonic>
PUP::able::PUP_ID AdamsMoultonPc<Monotonic>::my_PUP_ID = 0;  // NOLINT

#define MONOTONIC(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                            \
  template class AdamsMoultonPc<MONOTONIC(data)>;                       \
  template bool operator==(const AdamsMoultonPc<MONOTONIC(data)>& lhs,  \
                           const AdamsMoultonPc<MONOTONIC(data)>& rhs); \
  template bool operator!=(const AdamsMoultonPc<MONOTONIC(data)>& lhs,  \
                           const AdamsMoultonPc<MONOTONIC(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATE, (false, true))
#undef INSTANTIATE
#undef MONOTONIC
}  // namespace TimeSteppers
