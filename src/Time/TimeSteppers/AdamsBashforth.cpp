// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/AdamsBashforth.hpp"

#include <algorithm>
#include <boost/container/static_vector.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <iterator>
#include <limits>
#include <map>
#include <ostream>
#include <pup.h>
#include <tuple>

#include "NumericalAlgorithms/Interpolation/LagrangePolynomial.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/History.hpp"
#include "Time/SelfStart.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/CachedFunction.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/Overloader.hpp"

namespace TimeSteppers {

namespace {
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

// TimeDelta-like interface to a double used for dense output
struct ApproximateTimeDelta {
  double delta = std::numeric_limits<double>::signaling_NaN();
  double value() const { return delta; }
  bool is_positive() const { return delta > 0.; }

  // Only the operators that are actually used are defined.
  friend bool operator<(const ApproximateTimeDelta& a,
                        const ApproximateTimeDelta& b) {
    return a.value() < b.value();
  }
};

// Time-like interface to a double used for dense output
struct ApproximateTime {
  double time = std::numeric_limits<double>::signaling_NaN();
  double value() const { return time; }

  // Only the operators that are actually used are defined.
  friend ApproximateTimeDelta operator-(const ApproximateTime& a,
                                        const Time& b) {
    return {a.value() - b.value()};
  }

  friend bool operator<(const Time& a, const ApproximateTime& b) {
    return a.value() < b.value();
  }

  friend bool operator<(const ApproximateTime& a, const Time& b) {
    return a.value() < b.value();
  }

  friend std::ostream& operator<<(std::ostream& s, const ApproximateTime& t) {
    return s << t.value();
  }
};

// A vector holding one entry per order of integration.
template <typename T>
using OrderVector =
    boost::container::static_vector<T, AdamsBashforth::maximum_order>;

OrderVector<double> constant_coefficients(const size_t order) {
  switch (order) {
    case 0: return {};
    case 1: return {1.};
    case 2: return {1.5, -0.5};
    case 3: return {23.0 / 12.0, -4.0 / 3.0, 5.0 / 12.0};
    case 4: return {55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -3.0 / 8.0};
    case 5: return {1901.0 / 720.0, -1387.0 / 360.0, 109.0 / 30.0,
          -637.0 / 360.0, 251.0 / 720.0};
    case 6: return {4277.0 / 1440.0, -2641.0 / 480.0, 4991.0 / 720.0,
          -3649.0 / 720.0, 959.0 / 480.0, -95.0 / 288.0};
    case 7: return {198721.0 / 60480.0, -18637.0 / 2520.0, 235183.0 / 20160.0,
          -10754.0 / 945.0, 135713.0 / 20160.0, -5603.0 / 2520.0,
          19087.0 / 60480.0};
    case 8: return {16083.0 / 4480.0, -1152169.0 / 120960.0, 242653.0 / 13440.0,
          -296053.0 / 13440.0, 2102243.0 / 120960.0, -115747.0 / 13440.0,
          32863.0 / 13440.0, -5257.0 / 17280.0};
    default:
      ERROR("Bad order: " << order);
  }
}

// Only T=double is used, but this can be used with T=Rational to
// generate coefficient tables.
template <typename T>
OrderVector<T> variable_coefficients(const OrderVector<T>& control_times) {
  // The argument vector contains the control times for a step from
  // the last time in the list to t=0.

  // The coefficients are, for each j,
  // 1/step \int_0^{step} dt ell_j(t; -control_times),
  // where the step size is step=-control_times.back().

  const size_t order = control_times.size();
  OrderVector<T> result;
  for (size_t j = 0; j < order; ++j) {
    // Calculate coefficients of the Lagrange interpolating polynomials,
    // in the standard a_0 + a_1 t + a_2 t^2 + ... form.
    OrderVector<T> poly(order, 0);

    poly[0] = 1;

    for (size_t m = 0; m < order; ++m) {
      if (m == j) {
        continue;
      }
      const T denom =
          1 / (control_times[order - m - 1] - control_times[order - j - 1]);
      for (size_t i = m < j ? m + 1 : m; i > 0; --i) {
        poly[i] =
            (poly[i - 1] + poly[i] * control_times[order - m - 1]) * denom;
      }
      poly[0] *= control_times[order - m - 1] * denom;
    }

    // Integrate p(t), term by term.  We choose the constant of
    // integration so the indefinite integral is zero at t=0.  We do
    // not adjust the indexing, so the t^n term in the integral is in
    // the (n-1)th entry of the vector (as opposed to the nth entry
    // before integrating).  This is convenient because we want to
    // divide by the step size in the end, which is equivalent to this
    // shift.
    for (size_t m = 0; m < order; ++m) {
      poly[m] /= m + 1;
    }
    result.push_back(evaluate_polynomial(poly, -control_times.back()));
  }
  return result;
}

// Get coefficients for a time step.  Arguments are an iterator
// pair to past times, oldest to newest, and the time step to take.
template <typename Iterator, typename Delta>
OrderVector<double> get_coefficients(const Iterator& times_begin,
                                     const Iterator& times_end,
                                     const Delta& step) {
  bool constant_step_size = true;
  OrderVector<double> control_times;
  for (auto t = times_begin; t != times_end; ++t) {
    // Ideally we would also include the slab size in the scale of the
    // roundoff comparison, but there's no good way to get it here,
    // and it should only matter for slabs near time zero.
    if (constant_step_size and not control_times.empty() and
        not equal_within_roundoff(
            t->value() - control_times.back(), step.value(),
            100.0 * std::numeric_limits<double>::epsilon(), abs(t->value()))) {
      constant_step_size = false;
    }
    control_times.push_back(t->value());
  }
  if (constant_step_size) {
    return constant_coefficients(control_times.size());
  }

  const double goal_time = control_times.back() + step.value();
  for (auto& t : control_times) {
    t -= goal_time;
  }

  return variable_coefficients(control_times);
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
  history.discard_value(history.back().time_step_id);
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

size_t AdamsBashforth::number_of_past_steps() const { return order_ - 1; }

double AdamsBashforth::stable_step() const {
  if (order_ == 1) {
    return 1.;
  }

  // This is the condition that the characteristic polynomial of the
  // recurrence relation defined by the method has the correct sign at
  // -1.  It is not clear whether this is actually sufficient.
  const auto& coefficients = constant_coefficients(order_);
  double invstep = 0.;
  double sign = 1.;
  for (const auto coef : coefficients) {
    invstep += sign * coef;
    sign = -sign;
  }
  return 1. / invstep;
}

TimeStepId AdamsBashforth::next_time_id(const TimeStepId& current_id,
                                        const TimeDelta& time_step) const {
  ASSERT(current_id.substep() == 0, "Adams-Bashforth should not have substeps");
  return current_id.next_step(time_step);
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
  update_u_common(u, history, time_step, history.integration_order());
}

template <typename T>
bool AdamsBashforth::update_u_impl(
    const gsl::not_null<T*> u, const gsl::not_null<T*> u_error,
    const MutableUntypedHistory<T>& history, const TimeDelta& time_step) const {
  clean_history(history);
  *u_error = *u;
  update_u_common(u, history, time_step, history.integration_order());
  // the error estimate is only useful once the history has enough elements to
  // do more than one order of step
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
  const auto coefficients =
      get_coefficients(history_time_iterator(history_start),
                       history_time_iterator(history.end()), time_step);

  auto coefficient = coefficients.rbegin();
  for (auto history_entry = history_start;
       history_entry != history.end();
       ++history_entry, ++coefficient) {
    *u += time_step.value() * *coefficient * history_entry->derivative;
  }
}

template <typename T>
bool AdamsBashforth::can_change_step_size_impl(
    const TimeStepId& time_id, const ConstUntypedHistory<T>& history) const {
  // We need to forbid local time-stepping before initialization is
  // complete.  The self-start procedure itself should never consider
  // changing the step size, but we need to wait during the main
  // evolution until the self-start history has been replaced with
  // "real" values.
  const evolution_less<Time> less{time_id.time_runs_forward()};
  return not ::SelfStart::is_self_starting(time_id) and
         (history.size() == 0 or
          (less(history.back().time_step_id.step_time(),
                time_id.step_time()) and
           std::is_sorted(history_time_iterator(history.begin()),
                          history_time_iterator(history.end()), less)));
}

template <typename T>
void AdamsBashforth::add_boundary_delta_impl(
    const gsl::not_null<T*> result,
    const TimeSteppers::BoundaryHistoryEvaluator<T>& coupling,
    const TimeSteppers::BoundaryHistoryCleaner& cleaner,
    const TimeDelta& time_step) const {
  const auto signed_order =
      static_cast<typename decltype(cleaner.local_end())::difference_type>(
          cleaner.integration_order());

  ASSERT(cleaner.local_size() >= cleaner.integration_order(),
         "Insufficient data to take an order-" << cleaner.integration_order()
         << " step.  Have " << cleaner.local_size() << " times, need "
         << cleaner.integration_order());
  cleaner.local_mark_unneeded(cleaner.local_end() - signed_order);

  if (std::equal(cleaner.local_begin(), cleaner.local_end(),
                 cleaner.remote_end() - signed_order)) {
    // GTS
    ASSERT(cleaner.remote_size() >= cleaner.integration_order(),
           "Insufficient data to take an order-" << cleaner.integration_order()
           << " step.  Have " << cleaner.remote_size() << " times, need "
           << cleaner.integration_order());
    cleaner.remote_mark_unneeded(cleaner.remote_end() - signed_order);
  } else {
    const auto remote_step_for_step_start =
        std::upper_bound(cleaner.remote_begin(), cleaner.remote_end(),
                         *(cleaner.local_end() - 1),
                         evolution_less<Time>{time_step.is_positive()});
    ASSERT(remote_step_for_step_start - cleaner.remote_begin() >= signed_order,
           "Insufficient data to take an order-" << cleaner.integration_order()
           << " step.  Have "
           << remote_step_for_step_start - cleaner.remote_begin()
           << " times before the step, need " << cleaner.integration_order());
    cleaner.remote_mark_unneeded(remote_step_for_step_start - signed_order);
  }

  boundary_impl(result, coupling, *(cleaner.local_end() - 1) + time_step);
}

template <typename T>
void AdamsBashforth::boundary_dense_output_impl(
    const gsl::not_null<T*> result,
    const TimeSteppers::BoundaryHistoryEvaluator<T>& coupling,
    const double time) const {
  return boundary_impl(result, coupling, ApproximateTime{time});
}

template <typename T, typename TimeType>
void AdamsBashforth::boundary_impl(const gsl::not_null<T*> result,
                                   const BoundaryHistoryEvaluator<T>& coupling,
                                   const TimeType& end_time) const {
  // Might be different from order_ during self-start.
  const auto current_order = coupling.integration_order();

  ASSERT(current_order <= order_,
         "Local history is too long for target order (" << current_order
         << " should not exceed " << order_ << ")");
  ASSERT(coupling.remote_size() >= current_order,
         "Remote history is too short (" << coupling.remote_size()
         << " should be at least " << current_order << ")");

  // Avoid billions of casts
  const auto order_s = static_cast<
      typename BoundaryHistoryEvaluator<T>::iterator::difference_type>(
      current_order);

  // Start and end of the step we are trying to take
  const Time start_time = *(coupling.local_end() - 1);
  const auto time_step = end_time - start_time;

  // We define the local_begin and remote_begin variables as the start
  // of the part of the history relevant to this calculation.
  // Boundary history cleanup happens immediately before the step, but
  // boundary dense output happens before that, so there may be data
  // left over that was needed for the previous step and has not been
  // cleaned out yet.
  const auto local_begin = coupling.local_end() - order_s;

  if (std::equal(local_begin, coupling.local_end(),
                 coupling.remote_end() - order_s)) {
    // No local time-stepping going on.
    const auto coefficients =
        get_coefficients(local_begin, coupling.local_end(), time_step);

    auto local_it = local_begin;
    auto remote_it = coupling.remote_end() - order_s;
    for (auto coefficients_it = coefficients.rbegin();
         coefficients_it != coefficients.rend();
         ++coefficients_it, ++local_it, ++remote_it) {
      *result +=
          time_step.value() * *coefficients_it * *coupling(local_it, remote_it);
    }
    return;
  }

  ASSERT(current_order == order_,
         "Cannot perform local time-stepping while self-starting.");

  const evolution_less<> less{time_step.is_positive()};
  const auto remote_begin =
      std::upper_bound(coupling.remote_begin(), coupling.remote_end(),
                       start_time, less) -
      order_s;

  ASSERT(std::is_sorted(local_begin, coupling.local_end(), less),
         "Local history not in order");
  ASSERT(std::is_sorted(remote_begin, coupling.remote_end(), less),
         "Remote history not in order");
  ASSERT(not less(start_time, *(remote_begin + (order_s - 1))),
         "Remote history does not extend far enough back");
  ASSERT(less(*(coupling.remote_end() - 1), end_time),
         "Please supply only older data: " << *(coupling.remote_end() - 1)
         << " is not before " << end_time);

  // Union of times of all step boundaries on any side.
  const auto union_times = [&coupling, &local_begin, &remote_begin, &less]() {
    std::vector<Time> ret;
    ret.reserve(coupling.local_size() + coupling.remote_size());
    std::set_union(local_begin, coupling.local_end(), remote_begin,
                   coupling.remote_end(), std::back_inserter(ret), less);
    return ret;
  }();

  using UnionIter = typename decltype(union_times)::const_iterator;

  // Find the union times iterator for a given time.
  const auto union_step = [&union_times, &less](const Time& t) {
    return std::lower_bound(union_times.cbegin(), union_times.cend(), t, less);
  };

  // The union time index for the step start.
  const auto union_step_start = union_step(start_time);

  // min(union_times.end(), it + order_s) except being careful not
  // to create out-of-range iterators.
  const auto advance_within_step = [order_s,
                                    &union_times](const UnionIter& it) {
    return union_times.end() - it >
                   static_cast<typename decltype(union_times)::difference_type>(
                       order_s)
               ? it + static_cast<typename decltype(
                          union_times)::difference_type>(order_s)
               : union_times.end();
  };

  // Calculating the Adams-Bashforth coefficients is somewhat
  // expensive, so we cache them.  ab_coefs(it, step) returns the
  // coefficients used to step from *it to *it + step.
  auto ab_coefs = Overloader{
      make_cached_function<std::tuple<UnionIter, TimeDelta>, std::map>(
          [order_s](const std::tuple<UnionIter, TimeDelta>& args) {
            return get_coefficients(
                std::get<0>(args) -
                    static_cast<typename UnionIter::difference_type>(order_s -
                                                                     1),
                std::get<0>(args) + 1, std::get<1>(args));
          }),
      make_cached_function<std::tuple<UnionIter, ApproximateTimeDelta>,
                           std::map>(
          [order_s](const std::tuple<UnionIter, ApproximateTimeDelta>& args) {
            return get_coefficients(
                std::get<0>(args) -
                    static_cast<typename UnionIter::difference_type>(order_s -
                                                                     1),
                std::get<0>(args) + 1, std::get<1>(args));
          })};

  // The value of the coefficient of `evaluation_step` when doing
  // a standard Adams-Bashforth integration over the union times
  // from `step` to `step + 1`.
  const auto base_summand = [&ab_coefs, &end_time, &union_times](
                                const UnionIter& step,
                                const UnionIter& evaluation_step) {
    if (step + 1 != union_times.end()) {
      const TimeDelta step_size = *(step + 1) - *step;
      return step_size.value() *
             ab_coefs(std::make_tuple(
                 step, step_size))[static_cast<size_t>(step - evaluation_step)];
    } else {
      const auto step_size = end_time - *step;
      return step_size.value() *
             ab_coefs(std::make_tuple(
                 step, step_size))[static_cast<size_t>(step - evaluation_step)];
    }
  };

  for (auto local_evaluation_step = local_begin;
       local_evaluation_step != coupling.local_end();
       ++local_evaluation_step) {
    const auto union_local_evaluation_step = union_step(*local_evaluation_step);
    for (auto remote_evaluation_step = remote_begin;
         remote_evaluation_step != coupling.remote_end();
         ++remote_evaluation_step) {
      double deriv_coef = 0.;

      if (*local_evaluation_step == *remote_evaluation_step) {
        // The two elements stepped at the same time.  This gives a
        // standard Adams-Bashforth contribution to each segment
        // making up the current step.
        const auto union_step_upper_bound =
            advance_within_step(union_local_evaluation_step);
        for (auto step = union_step_start;
             step < union_step_upper_bound;
             ++step) {
          deriv_coef += base_summand(step, union_local_evaluation_step);
        }
      } else {
        // In this block we consider a coupling evaluation that is not
        // performed at equal times on the two sides of the mortar.

        // Makes an iterator with a map to give time as a double.
        const auto make_lagrange_iterator = [](const auto& it) {
          return boost::make_transform_iterator(
              it, [](const Time& t) { return t.value(); });
        };

        const auto union_remote_evaluation_step =
            union_step(*remote_evaluation_step);
        const auto union_step_lower_bound =
            std::max(union_step_start, union_remote_evaluation_step);

        // Compute the contribution to an interpolation over the local
        // times to `remote_evaluation_step->value()`, which we will
        // use as the coupling value for that time.  If there is an
        // actual evaluation at that time then skip this because the
        // Lagrange polynomial will be zero.
        if (not std::binary_search(local_begin, coupling.local_end(),
                                   *remote_evaluation_step, less)) {
          const auto union_step_upper_bound =
              advance_within_step(union_remote_evaluation_step);
          for (auto step = union_step_lower_bound;
               step < union_step_upper_bound;
               ++step) {
            deriv_coef += base_summand(step, union_remote_evaluation_step);
          }
          deriv_coef *=
              lagrange_polynomial(make_lagrange_iterator(local_evaluation_step),
                                  remote_evaluation_step->value(),
                                  make_lagrange_iterator(local_begin),
                                  make_lagrange_iterator(coupling.local_end()));
        }

        // Same qualitative calculation as the previous block, but
        // interpolating over the remote times.  This case is somewhat
        // more complicated because the latest remote time that can be
        // used varies for the different segments making up the step.
        if (not std::binary_search(remote_begin, coupling.remote_end(),
                                   *local_evaluation_step, less)) {
          auto union_step_upper_bound =
              advance_within_step(union_local_evaluation_step);
          if (coupling.remote_end() - remote_evaluation_step > order_s) {
            union_step_upper_bound = std::min(
                union_step_upper_bound,
                union_step(*(remote_evaluation_step + order_s)));
          }

          auto control_points = make_lagrange_iterator(
              remote_evaluation_step - remote_begin >= order_s
                  ? remote_evaluation_step - (order_s - 1)
                  : remote_begin);
          for (auto step = union_step_lower_bound;
               step < union_step_upper_bound;
               ++step, ++control_points) {
            deriv_coef +=
                base_summand(step, union_local_evaluation_step) *
                lagrange_polynomial(
                    make_lagrange_iterator(remote_evaluation_step),
                    local_evaluation_step->value(), control_points,
                    control_points +
                        static_cast<typename decltype(
                            control_points)::difference_type>(order_s));
          }
        }
      }

      if (deriv_coef != 0.) {
        // Skip the (potentially expensive) coupling calculation if
        // the coefficient is zero.
        *result += deriv_coef *
                   *coupling(local_evaluation_step, remote_evaluation_step);
      }
    }  // for remote_evaluation_step
  }  // for local_evaluation_step
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
