// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/AdamsBashforth.hpp"

#include <algorithm>
#include <boost/container/small_vector.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <cstddef>
#include <iterator>
#include <limits>
#include <pup.h>
#include <utility>

#include "NumericalAlgorithms/Interpolation/LagrangePolynomial.hpp"
#include "Time/ApproximateTime.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/History.hpp"
#include "Time/SelfStart.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsCoefficients.hpp"
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

template <typename T>
void clean_history(const MutableUntypedHistory<T>& history) {
  ASSERT(history.size() >= history.integration_order(),
         "Insufficient data to take an order-" << history.integration_order()
         << " step.  Have " << history.size() << " times, need "
         << history.integration_order());
  while (history.size() > history.integration_order()) {
    history.pop_front();
  }
  if (history.size() > 1) {
    history.discard_value(history[history.size() - 2].time_step_id);
  }
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

TimeStepId AdamsBashforth::next_time_id(const TimeStepId& current_id,
                                        const TimeDelta& time_step) const {
  ASSERT(current_id.substep() == 0, "Adams-Bashforth should not have substeps");
  return current_id.next_step(time_step);
}

TimeStepId AdamsBashforth::next_time_id_for_error(
    const TimeStepId& current_id, const TimeDelta& time_step) const {
  return next_time_id(current_id, time_step);
}

void AdamsBashforth::pup(PUP::er& p) {
  LtsTimeStepper::pup(p);
  p | order_;
}

template <typename T>
void AdamsBashforth::update_u_impl(
    const gsl::not_null<T*> u, const MutableUntypedHistory<T>& history,
    const TimeDelta& time_step) const {
  clean_history(history);
  *u = *history.back().value;
  update_u_common(u, history, time_step, history.integration_order());
}

template <typename T>
bool AdamsBashforth::update_u_impl(
    const gsl::not_null<T*> u, const gsl::not_null<T*> u_error,
    const MutableUntypedHistory<T>& history, const TimeDelta& time_step) const {
  clean_history(history);
  *u = *history.back().value;
  update_u_common(u, history, time_step, history.integration_order());
  // the error estimate is only useful once the history has enough elements to
  // do more than one order of step
  *u_error = *history.back().value;
  update_u_common(u_error, history, time_step, history.integration_order() - 1);
  *u_error = *u - *u_error;
  return true;
}

template <typename T>
bool AdamsBashforth::dense_update_u_impl(const gsl::not_null<T*> u,
                                         const ConstUntypedHistory<T>& history,
                                         const double time) const {
  const ApproximateTimeDelta time_step{
      time - history.back().time_step_id.step_time().value()};
  update_u_common(u, history, time_step, history.integration_order());
  return true;
}

template <typename T, typename Delta>
void AdamsBashforth::update_u_common(const gsl::not_null<T*> u,
                                     const ConstUntypedHistory<T>& history,
                                     const Delta& time_step,
                                     const size_t order) const {
  ASSERT(
      history.size() > 0,
      "Cannot meaningfully update the evolved variables with an empty history");
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
    const TimeSteppers::MutableBoundaryHistoryTimes& local_times,
    const TimeSteppers::MutableBoundaryHistoryTimes& remote_times,
    const TimeSteppers::BoundaryHistoryEvaluator<T>& coupling,
    const TimeDelta& time_step) const {
  const size_t integration_order =
      local_times.integration_order(local_times.size() - 1);
  const auto signed_order =
      static_cast<typename decltype(local_times.end())::difference_type>(
          integration_order);

  ASSERT(local_times.size() >= integration_order,
         "Insufficient data to take an order-" << integration_order
         << " step.  Have " << local_times.size() << " times, need "
         << integration_order);
  while (local_times.size() > integration_order) {
    local_times.pop_front();
  }

  ASSERT(remote_times.size() >= integration_order,
         "Insufficient data to take an order-" << integration_order
         << " step.  Have " << remote_times.size() << " times, need "
         << integration_order);
  if (std::equal(local_times.begin(), local_times.end(),
                 remote_times.end() - signed_order)) {
    // GTS
    while (remote_times.size() > integration_order) {
      remote_times.pop_front();
    }
  } else {
    const auto remote_step_after_step_start = std::upper_bound(
        remote_times.begin(), remote_times.end(), local_times.back());
    ASSERT(remote_step_after_step_start - remote_times.begin() >= signed_order,
           "Insufficient data to take an order-" << integration_order
           << " step.  Have "
           << remote_step_after_step_start - remote_times.begin()
           << " times before the step, need " << integration_order);
    // The pop_front() calls invalidate remote_step_after_step_start.
    const TimeStepId first_needed =
        *(remote_step_after_step_start - signed_order);
    while (remote_times.front() != first_needed) {
      remote_times.pop_front();
    }
  }

  boundary_impl(result, local_times, remote_times, coupling,
                local_times.back().step_time() + time_step);
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
  return boundary_impl(result, local_times, remote_times, coupling,
                       ApproximateTime{time});
}

namespace {
template <typename T>
class SmallStepIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = Time;
  using pointer = const Time*;
  using reference = const Time&;
  using difference_type = std::ptrdiff_t;

  enum class Side { Local, Remote, Both };

  using history_iterator = typename ConstBoundaryHistoryTimes::const_iterator;

  SmallStepIterator() = default;

  SmallStepIterator(history_iterator local_begin, history_iterator remote_begin,
                    history_iterator local_end, history_iterator remote_end)
      : local_id_(std::move(local_begin)),
        remote_id_(std::move(remote_begin)),
        local_end_(std::move(local_end)),
        remote_end_(std::move(remote_end)) {}

  reference operator*() const {
    return std::max(*local_id_, *remote_id_).step_time();
  }
  pointer operator->() const { return &**this; }

  Side side() const {
    if (*local_id_ < *remote_id_) {
      return Side::Remote;
    } else if (*remote_id_ < *local_id_) {
      return Side::Local;
    } else {
      return Side::Both;
    }
  }

  // These are the m^s(n) in the paper.
  const history_iterator& local_iterator() const { return local_id_; }
  const history_iterator& remote_iterator() const { return remote_id_; }

  SmallStepIterator& operator++() {
    ASSERT(local_id_ != local_end_ and remote_id_ != remote_end_,
           "Overran iterator");
    auto local_candidate = std::next(local_id_);
    auto remote_candidate = std::next(remote_id_);

    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (local_candidate == local_end_ and remote_candidate == remote_end_) {
      local_id_ = std::move(local_candidate);
      remote_id_ = std::move(remote_candidate);
      // NOLINTNEXTLINE(bugprone-branch-clone)
    } else if (local_candidate == local_end_) {
      remote_id_ = std::move(remote_candidate);
      // NOLINTNEXTLINE(bugprone-branch-clone)
    } else if (remote_candidate == remote_end_) {
      local_id_ = std::move(local_candidate);
    } else if (*local_candidate < *remote_candidate) {
      local_id_ = std::move(local_candidate);
    } else if (*remote_candidate < *local_candidate) {
      remote_id_ = std::move(remote_candidate);
    } else {
      local_id_ = std::move(local_candidate);
      remote_id_ = std::move(remote_candidate);
    }

    return *this;
  }

  bool done() const {
    return local_id_ == local_end_ and remote_id_ == remote_end_;
  }

 private:
  history_iterator local_id_{};
  history_iterator remote_id_{};
  history_iterator local_end_{};
  history_iterator remote_end_{};
};

template <typename T>
bool operator==(const SmallStepIterator<T>& a, const SmallStepIterator<T>& b) {
  if (a.done() and b.done()) {
    return true;
  }
  if (a.done() or b.done()) {
    return false;
  }
  return *a == *b;
}

template <typename T>
bool operator!=(const SmallStepIterator<T>& a, const SmallStepIterator<T>& b) {
  return not(a == b);
}

template <typename T>
bool operator<(const SmallStepIterator<T>& a, const SmallStepIterator<T>& b) {
  return a.local_iterator() < b.local_iterator() or
         a.remote_iterator() < b.remote_iterator();
}

template <typename T>
bool operator>(const SmallStepIterator<T>& a, const SmallStepIterator<T>& b) {
  return b < a;
}

template <typename It>
It bounded_next_impl(std::input_iterator_tag /*meta*/, It it, const It& bound,
                     typename std::iterator_traits<It>::difference_type n) {
  ASSERT(n >= 0, "Can't advance an input iterator backwards.");
  for (typename std::iterator_traits<It>::difference_type i = 0;
       i < n and it != bound; ++i, ++it) {
  }
  return it;
}

template <typename It>
It bounded_next_impl(std::random_access_iterator_tag /*meta*/, const It& it,
                     typename std::iterator_traits<It>::difference_type n,
                     const It& bound) {
  if (bound - it < n) {
    return bound;
  } else {
    return it + n;
  }
}

template <typename It>
It bounded_next(const It& it, const It& bound, const size_t n) {
  return bounded_next_impl(
      typename std::iterator_traits<It>::iterator_category{}, it, bound,
      static_cast<typename std::iterator_traits<It>::difference_type>(n));
}
}  // namespace

template <typename T, typename TimeType>
void AdamsBashforth::boundary_impl(
    const gsl::not_null<T*> result,
    const ConstBoundaryHistoryTimes& local_times,
    const ConstBoundaryHistoryTimes& remote_times,
    const BoundaryHistoryEvaluator<T>& coupling,
    const TimeType& end_time) const {
  // Might be different from order_ during self-start.
  const auto current_order =
      local_times.integration_order(local_times.size() - 1);

  ASSERT(current_order <= order_,
         "Local history is too long for target order (" << current_order
         << " should not exceed " << order_ << ")");
  ASSERT(remote_times.size() >= current_order,
         "Remote history is too short (" << remote_times.size()
         << " should be at least " << current_order << ")");

  // Avoid billions of casts
  const auto order_s = static_cast<
      typename ConstBoundaryHistoryTimes::iterator::difference_type>(
      current_order);

  // Start and end of the step we are trying to take
  const Time start_time = local_times.back().step_time();
  const auto time_step = end_time - start_time;

  // We define the local_begin and remote_begin variables as the start
  // of the part of the history relevant to this calculation.
  // Boundary history cleanup happens immediately before the step, but
  // boundary dense output happens before that, so there may be data
  // left over that was needed for the previous step and has not been
  // cleaned out yet.
  const auto local_begin = local_times.end() - order_s;

  if (std::equal(local_begin, local_times.end(),
                 remote_times.end() - order_s)) {
    // No local time-stepping going on.
    OrderVector<Time> control_points{};
    std::transform(local_begin, local_times.end(),
                   std::back_inserter(control_points),
                   [](const TimeStepId& t) { return t.step_time(); });
    const auto coefficients = adams_coefficients::coefficients(
        control_points.begin(), control_points.end(), start_time, end_time);

    auto local_it = local_begin;
    auto remote_it = remote_times.end() - order_s;
    for (auto coefficients_it = coefficients.begin();
         coefficients_it != coefficients.end();
         ++coefficients_it, ++local_it, ++remote_it) {
      *result += *coefficients_it * *coupling(*local_it, *remote_it);
    }
    return;
  }

  ASSERT(current_order == order_,
         "Cannot perform local time-stepping while self-starting.");

  const evolution_less<> less{time_step.is_positive()};
  const auto remote_begin =
      std::upper_bound(remote_times.begin(), remote_times.end(),
                       local_times.back()) -
      order_s;

  ASSERT(std::is_sorted(local_begin, local_times.end()),
         "Local history not in order");
  ASSERT(std::is_sorted(remote_begin, remote_times.end()),
         "Remote history not in order");
  ASSERT(not less(start_time, (remote_begin + (order_s - 1))->step_time()),
         "Remote history does not extend far enough back");
  ASSERT(less((remote_times.end() - 1)->step_time(), end_time),
         "Please supply only older data: "
         << (remote_times.end() - 1)->step_time() << " is not before "
         << end_time);

  using difference_type = std::ptrdiff_t;

  SmallStepIterator<T> contributing_small_step(
      local_begin, remote_begin, local_times.end(), remote_times.end());
  SmallStepIterator<T> small_step_of_current_step = std::next(
      contributing_small_step, static_cast<difference_type>(current_order - 1));
  while (*small_step_of_current_step != start_time) {
    ++contributing_small_step;
    ++small_step_of_current_step;
  }

  // The size of this vector is the number of small steps in the
  // current step, which will almost always be 1 or 2, but could be
  // larger if two elements decide to do 4:1 stepping or something.
  boost::container::small_vector<OrderVector<double>, 2>
      small_step_coefficients{};
  {
    auto coefficient_eval_begin = contributing_small_step;
    auto coefficient_eval_end = small_step_of_current_step;
    while (coefficient_eval_end != SmallStepIterator<T>{}) {
      auto next_end = std::next(coefficient_eval_end);
      const double next_time = next_end == SmallStepIterator<T>{}
                                   ? end_time.value()
                                   : next_end->value();
      small_step_coefficients.push_back(adams_coefficients::coefficients(
          coefficient_eval_begin, next_end, *coefficient_eval_end,
          ApproximateTime{next_time}));
      ++coefficient_eval_begin;
      coefficient_eval_end = std::move(next_end);
    }
  }

  // Sum over the small steps that contribute to this step, doing the
  // appropriate interpolation for each.
  for (size_t contributing_step_index = 0;
       contributing_small_step != SmallStepIterator<T>{};
       ++contributing_small_step, ++contributing_step_index) {
    if (contributing_small_step.side() != SmallStepIterator<T>::Side::Local) {
      double overall_prefactor = 0.0;
      auto small_step_within_current_step = static_cast<size_t>(
          std::max(static_cast<difference_type>(contributing_step_index + 1 -
                                                current_order),
                   static_cast<difference_type>(0)));
      const size_t small_step_within_current_step_end =
          std::min(small_step_coefficients.size(), contributing_step_index + 1);
      for (;
           small_step_within_current_step < small_step_within_current_step_end;
           ++small_step_within_current_step) {
        overall_prefactor +=
            small_step_coefficients[small_step_within_current_step]
                                   [contributing_step_index -
                                    small_step_within_current_step];
      }
      if (contributing_small_step.side() == SmallStepIterator<T>::Side::Both) {
        *result += overall_prefactor *
                   *coupling(*contributing_small_step.local_iterator(),
                             *contributing_small_step.remote_iterator());
      } else {
        // Side::Remote
        OrderVector<double> past_steps(current_order);
        std::transform(
            local_times.end() - static_cast<difference_type>(current_order),
            local_times.end(), past_steps.begin(),
            [](const TimeStepId& t) { return t.step_time().value(); });
        size_t interpolation_index = 0;
        for (auto interpolation_time =
                 local_times.end() -
                 static_cast<difference_type>(current_order);
             interpolation_time != local_times.end();
             ++interpolation_time, ++interpolation_index) {
          const double coefficient =
              overall_prefactor *
              lagrange_polynomial(interpolation_index,
                                  contributing_small_step->value(),
                                  past_steps.begin(), past_steps.end());
          *result += coefficient *
                     *coupling(*interpolation_time,
                               *contributing_small_step.remote_iterator());
        }
      }
    } else {
      // Side::Local
      auto interpolation_time =
          std::max(small_step_of_current_step.remote_iterator(),
                   contributing_small_step.remote_iterator()) -
          static_cast<difference_type>(current_order - 1);
      const auto interpolation_time_end =
          bounded_next(contributing_small_step, SmallStepIterator<T>{},
                       current_order)
              .remote_iterator();
      for (; interpolation_time != interpolation_time_end;
           ++interpolation_time) {
        double coefficient = 0.0;
        size_t small_step_within_current_step_index = 0;
        auto small_step_within_current_step = small_step_of_current_step;
        while (small_step_within_current_step.remote_iterator() <
               interpolation_time) {
          ++small_step_within_current_step;
          ++small_step_within_current_step_index;
        }
        auto small_step_within_current_step_end = contributing_small_step;
        const auto bound_from_interpolation_time =
            bounded_next(interpolation_time, remote_times.end(), current_order);
        for (size_t i = 0;
             i < current_order and
             small_step_within_current_step_end != SmallStepIterator<T>{} and
             small_step_within_current_step_end.remote_iterator() <
                 bound_from_interpolation_time;
             ++i) {
          ++small_step_within_current_step_end;
        }
        auto remote_steps_end = remote_times.begin();
        double lagrange_factor = std::numeric_limits<double>::signaling_NaN();
        for (; small_step_within_current_step !=
               small_step_within_current_step_end;
             ++small_step_within_current_step,
             ++small_step_within_current_step_index) {
          if (remote_steps_end !=
              small_step_within_current_step.remote_iterator() + 1) {
            remote_steps_end =
                small_step_within_current_step.remote_iterator() + 1;
            OrderVector<double> past_steps(current_order);
            std::transform(
                remote_steps_end - static_cast<difference_type>(current_order),
                remote_steps_end, past_steps.begin(),
                [](const TimeStepId& t) { return t.step_time().value(); });
            lagrange_factor = lagrange_polynomial(
                current_order -
                    static_cast<size_t>(remote_steps_end - interpolation_time),
                contributing_small_step->value(), past_steps.begin(),
                past_steps.end());
          }
          coefficient +=
              lagrange_factor *
              small_step_coefficients[small_step_within_current_step_index]
                                     [contributing_step_index -
                                      small_step_within_current_step_index];
        }
        *result +=
            coefficient * *coupling(*contributing_small_step.local_iterator(),
                                    *interpolation_time);
      }
    }
  }
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
