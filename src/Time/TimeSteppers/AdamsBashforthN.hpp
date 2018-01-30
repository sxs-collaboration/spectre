// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class AdamsBashforthN

#pragma once

#include <algorithm>
#include <boost/iterator/transform_iterator.hpp>
#include <deque>
#include <iterator>
#include <map>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/GeneralIndexIterator.hpp"
#include "DataStructures/MakeWithValue.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/Interpolation/LagrangePolynomial.hpp"
#include "Options/Options.hpp"
#include "Time/History.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/CachedFunction.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"

struct TimeId;

namespace TimeSteppers {

/// \ingroup TimeSteppersGroup
///
/// An Nth Adams-Bashforth time stepper.
class AdamsBashforthN : public TimeStepper::Inherit {
 public:
  static constexpr const size_t maximum_order = 8;

  struct TargetOrder {
    using type = size_t;
    static constexpr OptionString help = {
        "Target order of Adams-Bashforth method."};
    static type lower_bound() { return 1; }
    static type upper_bound() { return maximum_order; }
  };
  struct SelfStart {
    using type = bool;
    static constexpr OptionString help = {
        "Start at first order and increase."};
    static type default_value() { return false; }
  };
  using options = tmpl::list<TargetOrder, SelfStart>;
  static constexpr OptionString help = {
      "An Adams-Bashforth Nth order time-stepper. The target order is the\n"
      "order of the method. If a self-starting approach is chosen then the\n"
      "method starts at first order and increases the step-size until the\n"
      "desired order is reached."};

  AdamsBashforthN() = default;
  AdamsBashforthN(size_t target_order, bool self_start,
                  const OptionContext& context = {});
  AdamsBashforthN(const AdamsBashforthN&) noexcept = default;
  AdamsBashforthN& operator=(const AdamsBashforthN&) noexcept = default;
  AdamsBashforthN(AdamsBashforthN&&) noexcept = default;
  AdamsBashforthN& operator=(AdamsBashforthN&&) noexcept = default;
  ~AdamsBashforthN() noexcept override = default;

  template <typename Vars, typename DerivVars>
  void update_u(gsl::not_null<Vars*> u,
                gsl::not_null<History<Vars, DerivVars>*> history,
                const TimeDelta& time_step) const noexcept;

  /*!
   * An explanation of the computation being performed by this
   * function:
   *
   * We number the sides \f$1\f$ through \f$S\f$, and let the
   * evaluation times on side \f$s\f$ be \f$\ldots, t^s_{-1}, t^s_0,
   * t^s_1, \ldots\f$, with the starting location of the numbering
   * arbitrary.  The union of all these sets of times produces the
   * sequence \f$\ldots, \tilde{t}_{-1}, \tilde{t}_0, \tilde{t}_1,
   * \ldots\f$, again numbered arbitrarily.  For example, one possible
   * sequence of times for a two-sided boundary is:
   * \f{equation}
   *   \begin{aligned}
   *     \text{Side 1:} \\ \text{Union times:} \\ \text{Side 2:}
   *   \end{aligned}
   *   \cdots
   *   \begin{gathered}
   *     t^1_4 \\ \tilde{t}_8 \\ t^2_1
   *   \end{gathered}
   *   \leftarrow \Delta \tilde{t}_8 \rightarrow
   *   \begin{gathered}
   *    \, \\ \tilde{t}_9 \\ t^2_2
   *   \end{gathered}
   *   \leftarrow \Delta \tilde{t}_9 \rightarrow
   *   \begin{gathered}
   *     t^1_5 \\ \tilde{t}_{10} \\ \,
   *   \end{gathered}
   *   \cdots
   * \f}
   *
   * We call the indices of the start and end times in the union time
   * set \f$n_S\f$ and \f$n_E\f$, respectively, so for the above
   * example, if we wish to compute the step on side 1 from
   * \f$t^1_4\f$ to \f$t^1_5\f$, we would have \f$n_S = 8\f$ and
   * \f$n_E = 10\f$.  The indices of steps on each side containing or
   * starting with union time number \f$n\f$ are denoted
   * \f$\vec{m}_n\f$, so in the above example \f$\vec{m}_9 = (4,
   * 2)\f$.  The union indices corresponding to the start of a set of
   * side step indices \f$\vec{m}\f$ are denoted
   * \f$\vec{n}_{\vec{m}}\f$, so in the example \f$\vec{n}_{(4, 2)} =
   * (8, 9)\f$.  In the below equations, addition, subtraction, min,
   * and max operations applied to vectors operate on each component
   * individually, i.e., \f$\vec{q} + k = (q^1 + k, \ldots, q^S +
   * k)\f$.
   *
   * Let \f$k\f$ be the order of the integrator.  For a sequence of
   * times \f$t^s_{j_s - (k-1)}, \ldots, t^s_{j_s}\f$ on each side
   * \f$s\f$, we can write the cardinal functions on those times as
   * \f{equation}{
   *   \newcommand\of[1]{\!\!\left(#1\right)}
   *   C_{a_1, \ldots, a_S}(t_1, \ldots, t_S; j_1, \ldots, j_S) =
   *   \prod_{s=1}^S \ell_{a_s}\of{t_s; t^s_{j_s - (k-1)}, \ldots, t^s_{j_s}},
   * \f}
   * with \f$\ell_a(t; x_1, \ldots, x_k)\f$ the Lagrange interpolating
   * polynomials.  (Note that we take the indices \f$a\f$ to run from
   * \f$1\f$ to \f$k\f$ here, but from \f$0\f$ to \f$k-1\f$ in the
   * code.)  The cardinal function satisfies
   * \f$ C_{a_1, \ldots, a_S}(t^1_{i_1}, \ldots, t^S_{i_S}; j_1, \ldots, j_S) =
   *   \prod_{s=1}^S \delta_{i_s, j_s + a_s - k} \f$.
   * We write the integral of a cardinal function using an
   * Adams-Bashforth step based on the union times along the diagonal
   * \f$t_1 = t_2 = \cdots = t_S\f$ from \f$\tilde{t}_n\f$ to
   * \f$\tilde{t}_{n+1}\f$ as
   * \f{equation}{
   *   \newcommand\of[1]{\!\!\left(#1\right)}
   *   I_{n,\vec{q}} =
   *   \Delta \tilde{t}_n
   *   \sum_{j = 0}^{k-1}
   *   \tilde{\alpha}_{nj}
   *   C_{\vec{q}-\vec{m}_n+k}
   *     \of{\tilde{t}_{n - j}, \ldots, \tilde{t}_{n - j}; \vec{m}_n},
   * \f}
   * where \f$\tilde{\alpha}_{nj}\f$ are the Adams-Bashforth
   * coefficients derived from the union times and \f$\vec{q}\f$ is a
   * vector specifying the cardinal function indices.  The boundary
   * delta for the step is then
   * \f{equation}{
   *   \newcommand\mat{\mathbf}
   *   \mat{F}_{n_S, n_E} =
   *   \mspace{-7mu}
   *   \sum_{\vec{q} = \vec{m}_{n_s}-(k-1)}^{\vec{m}_{n_E-1}}
   *   \mspace{-4mu}
   *   \mat{D}_{\vec{q}}
   *   \mspace{-4mu}
   *   \sum_{n = \max\left\{n_S, \vec{n}_{\vec{q}} \right\}}
   *       ^{\min\left\{n_E, \vec{n}_{\vec{q} + k}\right\} - 1}
   *   \mspace{-7mu}
   *   I_{n,\vec{q}},
   * \f}
   * where \f$\mat{D}_{\vec{q}}\f$ is the coupling function evaluated
   * at the side values at side step indices \f$\vec{q}\f$.
   */
  template <typename BoundaryVars, typename FluxVars, typename Coupling>
  BoundaryVars compute_boundary_delta(
      const Coupling& coupling,
      gsl::not_null<std::vector<std::deque<std::tuple<
          Time, BoundaryVars, FluxVars>>>*>
          history,
      const TimeDelta& time_step) const noexcept;

  size_t number_of_substeps() const noexcept override;

  size_t number_of_past_steps() const noexcept override;

  bool is_self_starting() const noexcept override;

  double stable_step() const noexcept override;

  TimeId next_time_id(const TimeId& current_id,
                      const TimeDelta& time_step) const noexcept override;

  WRAPPED_PUPable_decl_template(AdamsBashforthN);  // NOLINT

  explicit AdamsBashforthN(CkMigrateMessage* /*unused*/) noexcept {}

  // clang-tidy: do not pass by non-const reference
  void pup(PUP::er& p) noexcept override;  // NOLINT

 private:
  friend bool operator==(const AdamsBashforthN& lhs,
                         const AdamsBashforthN& rhs) noexcept;

  /// Get coefficients for a time step.  Arguments are an iterator
  /// pair to past times, oldest to newest, and the time step to take.
  template <typename Iterator>
  static std::vector<double> get_coefficients(const Iterator& times_begin,
                                              const Iterator& times_end,
                                              const TimeDelta& step) noexcept;

  static std::vector<double> get_coefficients_impl(
      const std::vector<double>& steps) noexcept;

  static std::vector<double> variable_coefficients(
      const std::vector<double>& steps) noexcept;

  static std::vector<double> constant_coefficients(size_t order) noexcept;

  /// Comparator for ordering by "simulation time"
  class SimulationLess {
   public:
    explicit SimulationLess(const bool forward_in_time) noexcept
        : forward_in_time_(forward_in_time) {}

    bool operator()(const Time& a, const Time& b) const noexcept {
      return forward_in_time_ ? a < b : b < a;
    }

   private:
    bool forward_in_time_;
  };

  /// Convert an iterator over history entries to an iterator over
  /// their times.
  template <typename Iterator>
  static auto time_iter(const Iterator& it) noexcept {
    // Cast the overloaded get function to a function pointer taking a
    // history entry to a time reference.  This is needed to prevent the
    // boost::make_transform_iterator calls from being ambiguous.
    const Time& (*const get_time)(const typename Iterator::value_type&) =
        std::get<0>;
    return boost::make_transform_iterator(it, get_time);
  }

  /// Find the step on a given side containing the given time.  Steps
  /// contain their start time but not their end time.
  template <typename SideHistory, typename Compare>
  static typename SideHistory::const_iterator step_containing_time(
      const SideHistory& side_hist,
      const Time& time,
      const Compare& comparator) noexcept {
    return std::upper_bound(time_iter(side_hist.begin()),
                            time_iter(side_hist.end()),
                            time, comparator)
               .base() - 1;
  }

  size_t target_order_ = 3;
  bool is_self_starting_ = true;
};

bool operator!=(const AdamsBashforthN& lhs,
                const AdamsBashforthN& rhs) noexcept;

template <typename Vars, typename DerivVars>
void AdamsBashforthN::update_u(
    const gsl::not_null<Vars*> u,
    const gsl::not_null<History<Vars, DerivVars>*> history,
    const TimeDelta& time_step) const noexcept {
  ASSERT(is_self_starting_ or target_order_ == history->size(),
         "Length of history should be the order, so "
         << target_order_ << ", but is: " << history->size());
  ASSERT(history->size() <= target_order_,
         "Length of history (" << history->size() << ") "
         << "should not exceed target order (" << target_order_ << ")");

  const auto& coefficients =
      get_coefficients(history->begin(), history->end(), time_step);

  const auto do_update =
      [u, &time_step, &coefficients, &history](auto order) noexcept {
    *u += time_step.value() * constexpr_sum<order>(
        [order, &coefficients, &history](auto i) noexcept {
          return coefficients[order - 1 - i] *
              (history->begin() + static_cast<ssize_t>(i)).derivative();
        });
  };

  switch (history->size()) {
    case 1:
      do_update(std::integral_constant<size_t, 1>{});
      break;
    case 2:
      do_update(std::integral_constant<size_t, 2>{});
      break;
    case 3:
      do_update(std::integral_constant<size_t, 3>{});
      break;
    case 4:
      do_update(std::integral_constant<size_t, 4>{});
      break;
    case 5:
      do_update(std::integral_constant<size_t, 5>{});
      break;
    case 6:
      do_update(std::integral_constant<size_t, 6>{});
      break;
    case 7:
      do_update(std::integral_constant<size_t, 7>{});
      break;
    case 8:
      do_update(std::integral_constant<size_t, 8>{});
      break;
    default:
      ERROR("Bad amount of history data: " << history->size());
  }

  // Clean up old history
  if (history->size() >= target_order_) {
    history->mark_unneeded(
        history->end() - static_cast<ssize_t>(target_order_ - 1));
  }
}

template <typename BoundaryVars, typename FluxVars, typename Coupling>
BoundaryVars AdamsBashforthN::compute_boundary_delta(
    const Coupling& coupling,
    const gsl::not_null<std::vector<std::deque<std::tuple<
        Time, BoundaryVars, FluxVars>>>*>
        history,
    const TimeDelta& time_step) const noexcept {
  ASSERT(not is_self_starting_, "Unimplemented");

  using HistoryEntry = decltype(cpp17::as_const(*history)[0][0]);
  using HistoryIterator = decltype((*history)[0].cbegin());

  // Avoid billions of casts
  const auto target_order_s = static_cast<ssize_t>(target_order_);

  const SimulationLess simulation_less(time_step.is_positive());

  // Cast the overloaded get function to a function pointer taking a
  // history entry to a time reference.  This is needed to prevent the
  // boost::make_transform_iterator calls from being ambiguous.
  const Time& (*const get_time)(const HistoryEntry&) = std::get<0>;

  // Time::value is fairly slow, so cache values.
  auto time_value =
      make_cached_function<Time, std::map, Time::StructuralCompare>(
          [](const Time& t) noexcept { return t.value(); });

  const size_t num_sides = history->size();

  // Evaluate the cardinal function for a grid of points at a point on
  // the diagonal.
  const auto grid_cardinal_function =
      [target_order_s, &time_value, &get_time](
          const double evaluation_time,
          const auto& function_index,
          const auto& grid_times) noexcept {
    double result = 1.;
    for (size_t side = 0; side < grid_times.size(); ++side) {
      // Stop calculating things if we're just going to multiply
      // them by zero.
      if (result == 0.) { break; }

      // Makes an iterator with a map to give time as a double.
      const auto make_lagrange_iterator =
          [&time_value, &get_time](const HistoryIterator& it) noexcept {
        return boost::make_transform_iterator(
            it, [&time_value, &get_time](const HistoryEntry& h) noexcept {
              return time_value(get_time(h));
            });
      };
      result *= lagrange_polynomial(
          make_lagrange_iterator(function_index[side]),
          evaluation_time,
          make_lagrange_iterator(grid_times[side] - (target_order_s - 1)),
          make_lagrange_iterator(grid_times[side] + 1));
    }
    return result;
  };

  for (const auto& side_hist : *history) {
    ASSERT(not side_hist.empty(), "No data");
    ASSERT(std::is_sorted(time_iter(side_hist.begin()),
                          time_iter(side_hist.end()),
                          simulation_less),
           "History not in order");
  }

  // Start and end of the step we are trying to take
  const Time start_time = get_time((*history)[0].back());
  const Time end_time = start_time + time_step;

  // Union of times of all step boundaries on any side.
  const auto union_times =
      [&end_time, &get_time, &history, &simulation_less]() noexcept {
    std::set<Time, decltype(simulation_less)> ret({end_time}, simulation_less);
    for (const auto& side_hist : *history) {
      ret.insert(time_iter(side_hist.cbegin()), time_iter(side_hist.cend()));
      ASSERT(simulation_less(get_time(side_hist.back()), end_time),
             "Please supply only older data: " << get_time(side_hist.back())
             << " is not before " << end_time);
    }
    return ret;
  }();

  // Calculating the Adams-Bashforth coefficients is somewhat
  // expensive, so we cache them.  ab_coefs(it) returns the
  // coefficients used to step from *it to *(it + 1).
  auto ab_coefs = [target_order_s]() noexcept {
    using Iter = decltype(union_times.cbegin());
    auto compare = [](const Iter& a, const Iter& b) noexcept {
      return Time::StructuralCompare{}(*a, *b);
    };
    return make_cached_function<Iter, std::map, decltype(compare)>(
        [target_order = target_order_s](const Iter& last_time) noexcept {
          const auto times_end = std::next(last_time);
          const auto times_begin = std::prev(last_time, target_order - 1);
          return get_coefficients(times_begin, times_end,
                                  *times_end - *last_time);
        },
        std::move(compare));
  }();

  // Result variable.
  auto accumulated_change =
      make_with_value<BoundaryVars>(std::get<1>((*history)[0].back()), 0.);

  // Ranges of values where we evaluate the coupling function.
  std::vector<std::pair<HistoryIterator, HistoryIterator>>
      coupling_evaluation_ranges;
  for (size_t side = 0; side < num_sides; ++side) {
    const auto& side_hist = (*history)[side];
    const auto current_step_on_side =
        step_containing_time(side_hist, start_time, simulation_less);
    ASSERT(std::distance(side_hist.begin(), current_step_on_side) >=
           target_order_s - 1,
           "Not enough data on side " << side << ": "
           << std::distance(side_hist.begin(), current_step_on_side)
           << " (" << get_time(side_hist.front()) << " to "
           << get_time(*current_step_on_side) << ")");
    coupling_evaluation_ranges.emplace_back(
        current_step_on_side - (target_order_s - 1), side_hist.end());
  }
  for (auto coupling_evaluation =
           make_general_index_iterator(std::move(coupling_evaluation_ranges));
       coupling_evaluation;
       ++coupling_evaluation) {
    double deriv_coef = 0.;

    // Prepare to loop over the union times relevant for the current
    // coupling evaluation.

    // Latest of start of current step (otherwise was computed in a
    // previous call) and the times of the side data being coupled
    // (since segments can't depend on later data).
    Time coupling_start_time = start_time;
    // Earliest of the end of the current step (since the step can't
    // depend on later data), and target_order steps after the
    // evaluation on each side (since AB only uses data for that
    // long).
    Time coupling_end_time = end_time;
    for (size_t side = 0; side < num_sides; ++side) {
      const auto& evaluation_side = coupling_evaluation[side];
      if (simulation_less(coupling_start_time, get_time(*evaluation_side))) {
        coupling_start_time = get_time(*evaluation_side);
      }
      if ((*history)[side].end() - evaluation_side > target_order_s and
          simulation_less(get_time(*(evaluation_side + target_order_s)),
                          coupling_end_time)) {
        coupling_end_time = get_time(*(evaluation_side + target_order_s));
      }
    }
    const auto union_times_for_coupling_begin =
        std::lower_bound(union_times.cbegin(), union_times.cend(),
                         coupling_start_time, simulation_less);
    const auto union_times_for_coupling_end =
        std::lower_bound(union_times_for_coupling_begin, union_times.cend(),
                         coupling_end_time, simulation_less);

    // Iterators to the time on each side containing the union time we are
    // currently considering.
    std::vector<HistoryIterator> side_indices;
    for (const auto& side_hist : *history) {
      side_indices.push_back(step_containing_time(
          side_hist, *union_times_for_coupling_begin, simulation_less));
    }

    // This loop computes the coefficient of the coupling evaluation,
    // i.e., the sum over the I_{n,q}, as deriv_coef.
    for (auto union_time = union_times_for_coupling_begin;
         union_time != union_times_for_coupling_end;
         ++union_time) {
      // Advance the step of any side where we've moved into the next
      // step.
      for (size_t side = 0; side < num_sides; ++side) {
        const auto next_index = side_indices[side] + 1;
        if (next_index != (*history)[side].end() and
            not simulation_less(*union_time, get_time(*next_index))) {
          side_indices[side] = next_index;
        }
      }

      // Perform a single step Adams-Bashforth integration of a grid
      // cardinal function.  This computes the quantity called I_{n,q}
      // in the documentation as integrated_cardinal_function.
      const std::vector<double>& method_coefs = ab_coefs(union_time);
      // This produces a reversed iterator pointing to the same entry
      // as union_time.
      auto recent_time = std::make_reverse_iterator(std::next(union_time));
      auto method_it = method_coefs.begin();
      double integrated_cardinal_function = 0.;
      for (; method_it != method_coefs.end(); ++method_it, ++recent_time) {
        integrated_cardinal_function +=
            *method_it * grid_cardinal_function(time_value(*recent_time),
                                                coupling_evaluation,
                                                side_indices);
      }
      integrated_cardinal_function *=
          (*std::next(union_time) - *union_time).value();

      deriv_coef += integrated_cardinal_function;
    }  // for union_time

    if (deriv_coef != 0.) {
      // We sometimes get exact zeros from the Lagrange polynomials.
      // Skip the (potentially expensive) coupling calculation in that
      // case.

      std::vector<std::reference_wrapper<const FluxVars>> args;
      for (size_t side = 0; side < num_sides; ++side) {
        args.push_back(std::cref(std::get<2>(*coupling_evaluation[side])));
      }

      accumulated_change += deriv_coef * coupling(args);
    }
  }  // for coupling_evaluation

  // Clean up old history

  // We know that the local side will step at next_start_time, so the
  // step containing that time will be the next step, which is not
  // currently in the history.
  (*history)[0].erase(
      (*history)[0].begin(),
      (*history)[0].end() - static_cast<ssize_t>(target_order_ - 1));
  // We don't know whether other sides will step at next_start_time,
  // so we have to be conservative and assume they will not.
  for (size_t side = 1; side < history->size(); ++side) {
    (*history)[side].erase(
        (*history)[side].begin(),
        step_containing_time((*history)[side], end_time, simulation_less) -
            static_cast<ssize_t>(target_order_ - 1));
  }

  return accumulated_change;
}

template <typename Iterator>
std::vector<double> AdamsBashforthN::get_coefficients(
    const Iterator& times_begin, const Iterator& times_end,
    const TimeDelta& step) noexcept {
  std::vector<double> steps;
  // This may be slightly more space than we need, but we can't get
  // the exact amount without iterating through the iterators, which
  // is not necessarily cheap depending on the iterator type.
  steps.reserve(maximum_order);
  for (auto t = times_begin; std::next(t) != times_end; ++t) {
    steps.push_back((*std::next(t) - *t) / step);
  }
  steps.push_back(1.);
  return get_coefficients_impl(steps);
}

}  // namespace TimeSteppers
