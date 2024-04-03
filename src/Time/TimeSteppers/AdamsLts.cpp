// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/AdamsLts.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

#include "DataStructures/MathWrapper.hpp"
#include "NumericalAlgorithms/Interpolation/LagrangePolynomial.hpp"
#include "Time/ApproximateTime.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsCoefficients.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace TimeSteppers::adams_lts {
Time exact_substep_time(const TimeStepId& id) {
  const Time result =
      id.substep() == 0 ? id.step_time() : id.step_time() + id.step_size();
  ASSERT(result.value() == id.substep_time(), "Substep not at expected time.");
  return result;
}

namespace {
template <typename T>
using OrderVector = adams_coefficients::OrderVector<T>;

template <typename Op>
LtsCoefficients& add_assign_impl(LtsCoefficients& a, const LtsCoefficients& b,
                                 const Op& op) {
  const auto key_equal = [](const LtsCoefficients::value_type& l,
                            const LtsCoefficients::value_type& r) {
    return get<0>(l) == get<0>(r) and get<1>(l) == get<1>(r);
  };
  const auto key_less = [](const LtsCoefficients::value_type& l,
                           const LtsCoefficients::value_type& r) {
    return get<0>(l) < get<0>(r) or
           (get<0>(l) == get<0>(r) and get<1>(l) < get<1>(r));
  };

  // Two passes: first the common entries, and then the new entries.
  size_t common_entries = 0;
  {
    auto a_it = a.begin();
    auto b_it = b.begin();
    while (a_it != a.end() and b_it != b.end()) {
      if (key_less(*a_it, *b_it)) {
        ++a_it;
      } else if (key_less(*b_it, *a_it)) {
        ++b_it;
      } else {
        ++common_entries;
        get<2>(*a_it) += op(get<2>(*b_it));
        ++a_it;
        ++b_it;
      }
    }
  }
  a.resize(a.size() + b.size() - common_entries);
  {
    auto write = a.rbegin();
    auto a_it = write + static_cast<LtsCoefficients::difference_type>(
                            b.size() - common_entries);
    auto b_it = b.rbegin();
    while (b_it != b.rend()) {
      if (a_it == a.rend() or key_less(*a_it, *b_it)) {
        *write = *b_it;
        get<2>(*write) = op(get<2>(*write));
        ++b_it;
      } else {
        if (key_equal(*a_it, *b_it)) {
          ++b_it;
        }
        *write = *a_it;
        ++a_it;
      }
      ++write;
    }
  }
  return a;
}
}  // namespace

LtsCoefficients& operator+=(LtsCoefficients& a, const LtsCoefficients& b) {
  return add_assign_impl(a, b, [](const double x) { return x; });
}
LtsCoefficients& operator-=(LtsCoefficients& a, const LtsCoefficients& b) {
  return add_assign_impl(a, b, [](const double x) { return -x; });
}

LtsCoefficients operator+(LtsCoefficients&& a, LtsCoefficients&& b) {
  return std::move(a += b);
}
LtsCoefficients operator+(LtsCoefficients&& a, const LtsCoefficients& b) {
  return std::move(a += b);
}
LtsCoefficients operator+(const LtsCoefficients& a, LtsCoefficients&& b) {
  return std::move(b += a);
}
LtsCoefficients operator+(const LtsCoefficients& a, const LtsCoefficients& b) {
  auto a2 = a;
  return std::move(a2 += b);
}

LtsCoefficients operator-(LtsCoefficients&& a, LtsCoefficients&& b) {
  return std::move(a -= b);
}
LtsCoefficients operator-(LtsCoefficients&& a, const LtsCoefficients& b) {
  return std::move(a -= b);
}
LtsCoefficients operator-(const LtsCoefficients& a, const LtsCoefficients& b) {
  auto a2 = a;
  return std::move(a2 -= b);
}
LtsCoefficients operator-(const LtsCoefficients& a, LtsCoefficients&& b) {
  alg::for_each(b, [](auto& coef) { get<2>(coef) *= -1.0; });
  return std::move(b += a);
}

template <typename T>
void apply_coefficients(const gsl::not_null<T*> result,
                        const LtsCoefficients& coefficients,
                        const BoundaryHistoryEvaluator<T>& coupling) {
  for (const auto& term : coefficients) {
    *result += get<2>(term) * *coupling(get<0>(term), get<1>(term));
  }
}

bool operator==(const AdamsScheme& a, const AdamsScheme& b) {
  return a.type == b.type and a.order == b.order;
}
bool operator!=(const AdamsScheme& a, const AdamsScheme& b) {
  return not(a == b);
}

namespace {
// Collect the ids used for interpolating during a step to `end_time`
// from `times`.
//
// For implicit schemes, the step containing (or ending at) `end_time`
// must have predictor data available.
template <typename TimeType>
OrderVector<TimeStepId> find_relevant_ids(
    const ConstBoundaryHistoryTimes& times, const TimeType& end_time,
    const AdamsScheme& scheme) {
  OrderVector<TimeStepId> ids{};
  using difference_type = std::iterator_traits<
      ConstBoundaryHistoryTimes::const_iterator>::difference_type;
  const evolution_less<> less{times.front().time_runs_forward()};
  // Can't do a binary search because times are not sorted during self-start.
  auto used_range_end = times.end();
  while (used_range_end != times.begin()) {
    if (less((used_range_end - 1)->step_time(), end_time)) {
      break;
    }
    --used_range_end;
  }
  const auto number_of_past_steps =
      static_cast<difference_type>(scheme.order) -
      (scheme.type == SchemeType::Implicit ? 1 : 0);
  ASSERT(used_range_end - times.begin() >= number_of_past_steps,
         "Insufficient past data.");
  std::copy(used_range_end - number_of_past_steps, used_range_end,
            std::back_inserter(ids));
  if (scheme.type == SchemeType::Implicit) {
    const auto last_step =
        static_cast<size_t>(used_range_end - times.begin() - 1);
    ASSERT(times.number_of_substeps(last_step) == 2,
           "Must have substep data for implicit stepping.");
    ids.push_back(times[{last_step, 1}]);
  }
  return ids;
}

// Choose the relevant times from `local` and `remote` for defining
// small steps for `small_step_scheme`, using the specified schemes
// for interpolation on the local and remote sides.
//
// This is the most recent values from the union of the `local` and
// `remote` times with any values that should only be used for
// interpolation removed.
OrderVector<Time> merge_to_small_steps(const OrderVector<Time>& local,
                                       const OrderVector<Time>& remote,
                                       const evolution_less<Time>& less,
                                       const AdamsScheme& local_scheme,
                                       const AdamsScheme& remote_scheme,
                                       const AdamsScheme& small_step_scheme) {
  OrderVector<Time> small_steps(small_step_scheme.order);
  auto local_it = local.rbegin();
  auto remote_it = remote.rbegin();

  ASSERT(not(local_scheme.type == SchemeType::Implicit and
             remote_scheme.type == SchemeType::Explicit and
             less(*local_it, *remote_it)),
         "Explicit time " << *remote_it << " after implicit " << *local_it);
  ASSERT(not(remote_scheme.type == SchemeType::Implicit and
             local_scheme.type == SchemeType::Explicit and
             less(*remote_it, *local_it)),
         "Explicit time " << *local_it << " after implicit " << *remote_it);

  if (small_step_scheme.type == SchemeType::Explicit) {
    // Don't use implicit interpolation points for an explicit step
    if (local_scheme.type == SchemeType::Implicit) {
      ++local_it;
    }
    if (remote_scheme.type == SchemeType::Implicit) {
      ++remote_it;
    }
  } else {
    if (local_scheme.type == SchemeType::Implicit and
        remote_scheme.type == SchemeType::Implicit) {
      // If both the interpolation schemes are implicit, we will get
      // two times after the small step we are working on, one from
      // each.  One of them (if they are different) belongs to a later
      // small step, and we should ignore it.  If they are the same,
      // ignoring one of them is harmless.
      if (less(*local_it, *remote_it)) {
        ++remote_it;
      } else {
        ++local_it;
      }
    }
  }

  for (auto out = small_steps.rbegin(); out != small_steps.rend(); ++out) {
    if (local_it == local.rend()) {
      ASSERT(remote_it != remote.rend(), "Ran out of data");
      *out = *remote_it;
      ++remote_it;
    } else if (remote_it == remote.rend()) {
      *out = *local_it;
      ++local_it;
    } else {
      *out = std::max(*local_it, *remote_it, less);
      if (*local_it == *out) {
        ++local_it;
      }
      if (*remote_it == *out) {
        ++remote_it;
      }
    }
  }
  return small_steps;
}

// Evaluate the Lagrange interpolating polynomials with the given
// `control_times` at `time`.  The returned vector contains the values
// of all the Lagrange polynomials.
template <typename TimeType>
OrderVector<double> interpolation_coefficients(
    const OrderVector<Time>& control_times, const TimeType& time) {
  if constexpr (std::is_same_v<TimeType, Time>) {
    // Skip the Lagrange polynomial calculations if we are evaluating
    // at a control time.  This should be common.
    for (size_t i = 0; i < control_times.size(); ++i) {
      if (control_times[i] == time) {
        OrderVector<double> coefficients(control_times.size(), 0.0);
        coefficients[i] = 1.0;
        return coefficients;
      }
    }
  }

  OrderVector<double> control_times_fp(control_times.size());
  alg::transform(control_times, control_times_fp.begin(),
                 [](const Time& t) { return t.value(); });

  OrderVector<double> coefficients{};
  for (size_t i = 0; i < control_times.size(); ++i) {
    coefficients.push_back(lagrange_polynomial(
        i, time.value(), control_times_fp.begin(), control_times_fp.end()));
  }
  return coefficients;
}

template <typename TimeType>
LtsCoefficients lts_coefficients_for_gts(
    const OrderVector<TimeStepId>& control_ids, const Time& start_time,
    const TimeType& end_time) {
  // The sides are stepping at the same rate, so no LTS is happening
  // at this boundary.
  OrderVector<Time> control_times(control_ids.size());
  alg::transform(control_ids, control_times.begin(), exact_substep_time);

  const OrderVector<double> gts_coefficients = adams_coefficients::coefficients(
      control_times.begin(), control_times.end(), start_time, end_time);
  LtsCoefficients lts_coefficients{};
  for (size_t step = 0; step < gts_coefficients.size(); ++step) {
    lts_coefficients.emplace_back(control_ids[step], control_ids[step],
                                  gts_coefficients[step]);
  }
  return lts_coefficients;
}
}  // namespace

template <typename TimeType>
LtsCoefficients lts_coefficients(const ConstBoundaryHistoryTimes& local_times,
                                 const ConstBoundaryHistoryTimes& remote_times,
                                 const Time& start_time,
                                 const TimeType& end_time,
                                 const AdamsScheme& local_scheme,
                                 const AdamsScheme& remote_scheme,
                                 const AdamsScheme& small_step_scheme) {
  if (start_time == end_time) {
    return {};
  }
  const evolution_less<Time> time_less{local_times.front().time_runs_forward()};

  LtsCoefficients step_coefficients{};

  TimeType small_step_end = end_time;
  for (;;) {
    const OrderVector<TimeStepId> local_ids =
        find_relevant_ids(local_times, small_step_end, local_scheme);
    const OrderVector<TimeStepId> remote_ids =
        find_relevant_ids(remote_times, small_step_end, remote_scheme);

    // Check is the there is actually local time-stepping happening at
    // this boundary.  Only check for the latest small step, before we
    // have generated any coefficients.
    if (step_coefficients.empty() and small_step_scheme == local_scheme and
        small_step_scheme == remote_scheme and local_ids == remote_ids) {
      return lts_coefficients_for_gts(local_ids, start_time, end_time);
    }

    OrderVector<Time> local_control_times(local_scheme.order);
    alg::transform(local_ids, local_control_times.begin(), exact_substep_time);
    OrderVector<Time> remote_control_times(remote_scheme.order);
    alg::transform(remote_ids, remote_control_times.begin(),
                   exact_substep_time);

    const OrderVector<Time> small_step_times = merge_to_small_steps(
        local_control_times, remote_control_times, time_less, local_scheme,
        remote_scheme, small_step_scheme);
    const Time current_small_step =
        small_step_times[small_step_times.size() -
                         (small_step_scheme.type == SchemeType::Implicit ? 2
                                                                         : 1)];
    ASSERT(not time_less(current_small_step, start_time),
           "Reached time " << current_small_step
           << " without hitting start time " << start_time
           << " while iterating over small steps.  Most likely the supplied "
              "start time was not a step boundary.");

    const OrderVector<double> small_step_coefficients =
        adams_coefficients::coefficients(small_step_times.begin(),
                                         small_step_times.end(),
                                         current_small_step, small_step_end);

    for (size_t contributing_small_step = 0;
         contributing_small_step < small_step_times.size();
         ++contributing_small_step) {
      const OrderVector<double> local_interpolation_coefficients =
          interpolation_coefficients(local_control_times,
                                     small_step_times[contributing_small_step]);
      const OrderVector<double> remote_interpolation_coefficients =
          interpolation_coefficients(remote_control_times,
                                     small_step_times[contributing_small_step]);
      for (size_t local_step_index = 0;
           local_step_index < local_interpolation_coefficients.size();
           ++local_step_index) {
        if (local_interpolation_coefficients[local_step_index] == 0.0) {
          continue;
        }
        for (size_t remote_step_index = 0;
             remote_step_index < remote_interpolation_coefficients.size();
             ++remote_step_index) {
          if (remote_interpolation_coefficients[remote_step_index] == 0.0) {
            continue;
          }
          step_coefficients.emplace_back(
              local_ids[local_step_index], remote_ids[remote_step_index],
              small_step_coefficients[contributing_small_step] *
                  local_interpolation_coefficients[local_step_index] *
                  remote_interpolation_coefficients[remote_step_index]);
        }
      }
    }
    if (current_small_step == start_time) {
      break;
    }

    if constexpr (std::is_same_v<TimeType, Time>) {
      // We're iterating backwards, so the next step temporally is the
      // one we just did.
      small_step_end = current_small_step;
    } else {
      ERROR(
          "Multiple-small-step dense output is not supported.  Calculate "
          "coefficients as the sum of the exact update for the complete small "
          "steps and a dense update for a single partial small step.");
    }
  }

  // Combine duplicate entries
  ASSERT(not step_coefficients.empty(),
         "Generated no coefficients.  This an algorithmic bug.  Please file "
         "an issue.");
  alg::sort(step_coefficients);
  auto unique_entry = step_coefficients.begin();
  for (auto generated_entry = std::next(step_coefficients.begin());
       generated_entry != step_coefficients.end();
       ++generated_entry) {
    if (get<0>(*generated_entry) == get<0>(*unique_entry) and
        get<1>(*generated_entry) == get<1>(*unique_entry)) {
      get<2>(*unique_entry) += get<2>(*generated_entry);
    } else {
      *++unique_entry = *generated_entry;
    }
  }
  step_coefficients.erase(std::next(unique_entry), step_coefficients.end());
  return step_coefficients;
}

#define MATH_WRAPPER_TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                          \
  template void apply_coefficients(                   \
      gsl::not_null<MATH_WRAPPER_TYPE(data)*> result, \
      const LtsCoefficients& coefficients,            \
      const BoundaryHistoryEvaluator<MATH_WRAPPER_TYPE(data)>& coupling);

GENERATE_INSTANTIATIONS(INSTANTIATE, (MATH_WRAPPER_TYPES))
#undef INSTANTIATE
#undef MATH_WRAPPER_TYPE

#define TIME_TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template LtsCoefficients lts_coefficients(                                 \
      const ConstBoundaryHistoryTimes& local_times,                          \
      const ConstBoundaryHistoryTimes& remote_times, const Time& start_time, \
      const TIME_TYPE(data) & end_time, const AdamsScheme& local_scheme,     \
      const AdamsScheme& remote_scheme, const AdamsScheme& small_step_scheme);

GENERATE_INSTANTIATIONS(INSTANTIATE, (Time, ApproximateTime))
#undef INSTANTIATE
#undef TIME_TYPE
}  // namespace TimeSteppers::adams_lts
