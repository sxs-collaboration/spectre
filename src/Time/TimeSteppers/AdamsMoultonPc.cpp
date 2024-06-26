// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/AdamsMoultonPc.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <optional>
#include <pup.h>

#include "Time/ApproximateTime.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/History.hpp"
#include "Time/LargestStepperError.hpp"
#include "Time/SelfStart.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/TimeSteppers/AdamsCoefficients.hpp"
#include "Time/TimeSteppers/AdamsLts.hpp"
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
                     const TimeType& step_end, const bool corrector) {
  const auto method_order = history.integration_order();
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

template <typename T>
void step_error(const gsl::not_null<T*> u_error,
                const ConstUntypedHistory<T>& history, const Time& step_end) {
  const auto method_order = history.integration_order();
  ASSERT(history.size() >= method_order - 1, "Insufficient history");
  ASSERT(not history.substeps().empty(),
         "step_error called without substep data.");
  ASSERT(not history.at_step_start(), "Unexpected new data");

  const auto used_history_begin =
      history.end() -
      static_cast<typename ConstUntypedHistory<T>::difference_type>(
          method_order - 1);
  adams_coefficients::OrderVector<Time> control_times{};
  std::transform(used_history_begin, history.end(),
                 std::back_inserter(control_times),
                 [](const auto& r) { return r.time_step_id.step_time(); });
  control_times.push_back(history.back().time_step_id.step_time() +
                          history.substeps().front().time_step_id.step_size());
  auto coefficients = adams_coefficients::coefficients(
      control_times.begin(), control_times.end(),
      history.back().time_step_id.step_time(), step_end);
  control_times.erase(control_times.begin());
  const auto lower_order_coefficients = adams_coefficients::coefficients(
      control_times.begin(), control_times.end(),
      history.back().time_step_id.step_time(), step_end);
  for (size_t i = 0; i < lower_order_coefficients.size(); ++i) {
    coefficients[i + 1] -= lower_order_coefficients[i];
  }

  *u_error = coefficients.back() * history.substeps().front().derivative;
  auto coefficient = coefficients.begin();
  for (auto history_entry = used_history_begin;
       history_entry != history.end();
       ++history_entry, ++coefficient) {
    *u_error += *coefficient * history_entry->derivative;
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
bool AdamsMoultonPc<Monotonic>::neighbor_data_required(
    const TimeStepId& next_substep_id,
    const TimeStepId& neighbor_data_id) const {
  // Because of self-start, step times may not be monotonic across
  // slabs, so check that first.
  if (neighbor_data_id.slab_number() != next_substep_id.slab_number()) {
    return neighbor_data_id.slab_number() < next_substep_id.slab_number();
  }

  const evolution_less<Time> before{neighbor_data_id.time_runs_forward()};

  if constexpr (Monotonic) {
    const auto next_time = adams_lts::exact_substep_time(next_substep_id);
    const auto neighbor_time = adams_lts::exact_substep_time(neighbor_data_id);
    return before(neighbor_time, next_time) or
           (neighbor_time == next_time and neighbor_data_id.substep() == 1 and
            next_substep_id.substep() == 0);
  } else {
    if (next_substep_id.substep() == 1) {
      // predictor
      return before(neighbor_data_id.step_time(),
                    next_substep_id.step_time()) or
             (neighbor_data_id.step_time() == next_substep_id.step_time() and
              neighbor_data_id.substep() == 0);
    } else {
      // corrector
      return before(neighbor_data_id.step_time(), next_substep_id.step_time());
    }
  }
}

template <bool Monotonic>
bool AdamsMoultonPc<Monotonic>::neighbor_data_required(
    const double dense_output_time, const TimeStepId& neighbor_data_id) const {
  const evolution_less<double> before{neighbor_data_id.time_runs_forward()};
  if constexpr (Monotonic) {
    return not before(dense_output_time,
                      adams_lts::exact_substep_time(neighbor_data_id).value());
  } else {
    return before(neighbor_data_id.step_time().value(), dense_output_time);
  }
}

template <bool Monotonic>
void AdamsMoultonPc<Monotonic>::pup(PUP::er& p) {
  LtsTimeStepper::pup(p);
  p | order_;
}

template <bool Monotonic>
template <typename T>
void AdamsMoultonPc<Monotonic>::update_u_impl(
    const gsl::not_null<T*> u, const ConstUntypedHistory<T>& history,
    const TimeDelta& time_step) const {
  const Time next_time = history.back().time_step_id.step_time() + time_step;
  *u = *history.back().value;
  update_u_common(u, history, next_time, not history.at_step_start());
}

template <bool Monotonic>
template <typename T>
std::optional<StepperErrorEstimate> AdamsMoultonPc<Monotonic>::update_u_impl(
    gsl::not_null<T*> u, const ConstUntypedHistory<T>& history,
    const TimeDelta& time_step,
    const std::optional<StepperErrorTolerances>& tolerances) const {
  const bool corrector = not history.at_step_start();
  const Time next_time = history.back().time_step_id.step_time() + time_step;
  std::optional<StepperErrorEstimate> error{};
  if (corrector and tolerances.has_value()) {
    step_error(u, history, next_time);
    error.emplace(StepperErrorEstimate{
        history.back().time_step_id.step_time(), time_step,
        history.integration_order() - 1,
        largest_stepper_error(*history.back().value, *u, *tolerances)});
  }
  *u = *history.back().value;
  update_u_common(u, history, next_time, corrector);
  return error;
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
    update_u_common(u, history, ApproximateTime{time}, false);
    return true;
  } else {
    if (history.at_step_start()) {
      return false;
    }
    update_u_common(u, history, ApproximateTime{time}, true);
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
template <typename T>
void AdamsMoultonPc<Monotonic>::add_boundary_delta_impl(
    const gsl::not_null<T*> result,
    const TimeSteppers::ConstBoundaryHistoryTimes& local_times,
    const TimeSteppers::ConstBoundaryHistoryTimes& remote_times,
    const TimeSteppers::BoundaryHistoryEvaluator<T>& coupling,
    const TimeDelta& time_step) const {
  ASSERT(not local_times.empty(), "No local data provided.");
  ASSERT(not remote_times.empty(), "No remote data provided.");
  const auto current_order =
      local_times.integration_order(local_times.size() - 1);
  if constexpr (Monotonic) {
    const adams_lts::AdamsScheme predictor_scheme{
        adams_lts::SchemeType::Explicit, current_order - 1};
    const adams_lts::AdamsScheme corrector_scheme{
        adams_lts::SchemeType::Implicit, current_order};
    const auto step_start = local_times.back().step_time();
    const auto step_end = step_start + time_step;
    const auto small_step_start =
        std::max(local_times.back(), remote_times.back()).step_time();
    const auto synchronization_time =
        std::min(local_times.back(), remote_times.back()).step_time();
    const auto is_synchronization_time = [&](const TimeStepId& id) {
      return id.step_time() == synchronization_time;
    };
    ASSERT(alg::any_of(local_times, is_synchronization_time) and
               alg::any_of(remote_times, is_synchronization_time),
           "Only nested step patterns (N:1) are supported.");

    if (local_times.number_of_substeps(local_times.size() - 1) == 1) {
      // Predictor
      auto lts_coefficients = adams_lts::lts_coefficients(
          local_times, remote_times, small_step_start, step_end,
          predictor_scheme, predictor_scheme, predictor_scheme);
      if (not is_synchronization_time(remote_times.back())) {
        lts_coefficients += adams_lts::lts_coefficients(
            local_times, remote_times, synchronization_time, small_step_start,
            predictor_scheme, corrector_scheme, corrector_scheme);
      }
      adams_lts::apply_coefficients(result, lts_coefficients, coupling);
    } else {
      // Corrector
      if (remote_times.number_of_substeps(remote_times.size() - 1) == 2) {
        // Aligned corrector
        ASSERT(adams_lts::exact_substep_time(
                   remote_times[{remote_times.size() - 1, 1}]) == step_end,
               "Have remote substep data, but it isn't aligned with the local "
               "data.");
        const auto lts_coefficients =
            adams_lts::lts_coefficients(
                local_times, remote_times, synchronization_time, step_end,
                corrector_scheme, corrector_scheme, corrector_scheme) -
            adams_lts::lts_coefficients(
                local_times, remote_times, synchronization_time, step_start,
                corrector_scheme, predictor_scheme, corrector_scheme);
        adams_lts::apply_coefficients(result, lts_coefficients, coupling);
      } else {
        // Unaligned corrector
        ASSERT(step_start == small_step_start,
               "Trying to take unaligned step, but remote side is smaller.");
        const auto lts_coefficients = adams_lts::lts_coefficients(
            local_times, remote_times, step_start, step_end, corrector_scheme,
            predictor_scheme, corrector_scheme);
        adams_lts::apply_coefficients(result, lts_coefficients, coupling);
      }
    }
  } else {
    adams_lts::AdamsScheme scheme{adams_lts::SchemeType::Implicit,
                                  current_order};
    auto remote_scheme = scheme;

    if (local_times.number_of_substeps(local_times.size() - 1) == 1) {
      // Predictor
      scheme = {adams_lts::SchemeType::Explicit, current_order - 1};
      ASSERT(remote_times.back() <= local_times.back(),
             "Unexpected remote values available.");
      // If the sides are not aligned, we use the predictor data
      // available from the neighbor.  If they are, that data has not
      // been received.
      remote_scheme = {remote_times.back() == local_times.back()
                           ? adams_lts::SchemeType::Explicit
                           : adams_lts::SchemeType::Implicit,
                       current_order - 1};
    }

    const auto lts_coefficients = adams_lts::lts_coefficients(
        local_times, remote_times, local_times.back().step_time(),
        local_times.back().step_time() + time_step, scheme, remote_scheme,
        scheme);
    adams_lts::apply_coefficients(result, lts_coefficients, coupling);
  }
}

template <bool Monotonic>
void AdamsMoultonPc<Monotonic>::clean_boundary_history_impl(
    const TimeSteppers::MutableBoundaryHistoryTimes& local_times,
    const TimeSteppers::MutableBoundaryHistoryTimes& remote_times) const {
  if (local_times.empty() or
      local_times.number_of_substeps(local_times.size() - 1) != 2) {
    return;
  }
  ASSERT(not remote_times.empty(), "No remote data available.");

  const bool synchronized =
      remote_times.number_of_substeps(remote_times.size() - 1) == 2 and
      adams_lts::exact_substep_time(local_times[{local_times.size() - 1, 1}]) ==
          adams_lts::exact_substep_time(
              remote_times[{remote_times.size() - 1, 1}]);

  if (Monotonic and not synchronized) {
    return;
  }

  const auto required_points =
      local_times.integration_order(local_times.size() - 1) - 2;

  while (local_times.size() > required_points) {
    local_times.pop_front();
  }
  for (size_t i = 0; i < local_times.size(); ++i) {
    local_times.clear_substeps(i);
  }

  // If the sides are not aligned, then we are in the middle of the
  // remote step, so still need its data.
  if (synchronized) {
    while (remote_times.size() > required_points) {
      remote_times.pop_front();
    }
    for (size_t i = 0; i < remote_times.size(); ++i) {
      remote_times.clear_substeps(i);
    }
  }
}

template <bool Monotonic>
template <typename T>
void AdamsMoultonPc<Monotonic>::boundary_dense_output_impl(
    const gsl::not_null<T*> result,
    const TimeSteppers::ConstBoundaryHistoryTimes& local_times,
    const TimeSteppers::ConstBoundaryHistoryTimes& remote_times,
    const TimeSteppers::BoundaryHistoryEvaluator<T>& coupling,
    const double time) const {
  if constexpr (Monotonic) {
    ASSERT(local_times.number_of_substeps(local_times.size() - 1) == 1,
           "Dense output must be done before predictor evaluation.");

    const auto current_order =
        local_times.integration_order(local_times.size() - 1);
    const adams_lts::AdamsScheme predictor_scheme{
        adams_lts::SchemeType::Explicit, current_order - 1};
    const adams_lts::AdamsScheme corrector_scheme{
        adams_lts::SchemeType::Implicit, current_order};
    const auto small_step_start =
        std::max(local_times.back(), remote_times.back()).step_time();
    const auto synchronization_time =
        std::min(local_times.back(), remote_times.back()).step_time();
    const auto is_synchronization_time = [&](const TimeStepId& id) {
      return id.step_time() == synchronization_time;
    };
    ASSERT(alg::any_of(local_times, is_synchronization_time) and
               alg::any_of(remote_times, is_synchronization_time),
           "Only nested step patterns (N:1) are supported.");

    auto lts_coefficients = adams_lts::lts_coefficients(
        local_times, remote_times, small_step_start, ApproximateTime{time},
        predictor_scheme, predictor_scheme, predictor_scheme);
    if (not is_synchronization_time(remote_times.back())) {
      lts_coefficients += adams_lts::lts_coefficients(
          local_times, remote_times, synchronization_time, small_step_start,
          predictor_scheme, corrector_scheme, corrector_scheme);
    }
    adams_lts::apply_coefficients(result, lts_coefficients, coupling);
  } else {
    if (local_times.back().step_time().value() == time) {
      // Nothing to do.  The requested time is the start of the step,
      // which is the input value of `result`.
      return;
    }
    ASSERT(local_times.number_of_substeps(local_times.size() - 1) == 2,
           "Dense output must be done after predictor evaluation.");

    const auto current_order =
        local_times.integration_order(local_times.size() - 1);
    const adams_lts::AdamsScheme scheme{adams_lts::SchemeType::Implicit,
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
LTS_TIME_STEPPER_DEFINE_OVERLOADS_TEMPLATED(AdamsMoultonPc<Monotonic>,
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
