// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class AdamsBashforthN

#pragma once

#include <algorithm>
#include <boost/iterator/transform_iterator.hpp>
#include <cstddef>
#include <iterator>
#include <map>
#include <ostream>
#include <pup.h>
#include <set>
#include <type_traits>
#include <vector>

#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/Interpolation/LagrangePolynomial.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/BoundaryHistory.hpp"  // IWYU pragma: keep
#include "Time/History.hpp"          // IWYU pragma: keep
#include "Time/Time.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"  // IWYU pragma: keep
#include "Utilities/CachedFunction.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
struct TimeId;
/// \endcond

// IWYU pragma: no_include <sys/types.h>

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
   * \f$\newcommand\tL{t^L}\newcommand\tR{t^R}\newcommand\tU{\tilde{t}}
   * \newcommand\mat{\mathbf}\newcommand\of[1]{\!\!\left(#1\right)}\f$
   *
   * Suppose the local and remote sides of the interface are evaluated
   * at times \f$\ldots, \tL_{-1}, \tL_0, \tL_1, \ldots\f$ and
   * \f$\ldots, \tR_{-1}, \tR_0, \tR_1, \ldots\f$, respectively, with
   * the starting location of the numbering arbitrary in each case.
   * Let the step we wish to calculate the effect of be the step from
   * \f$\tL_{m_S}\f$ to \f$\tL_{m_S+1}\f$.
   *
   * We call the sequence produced from the union of the local and
   * remote time sequences \f$\ldots, \tU_{-1}, \tU_0, \tU_1,
   * \ldots\f$, where for convenience we choose the numbering such
   * that \f$\tU_n = \tR_n\f$ for times between \f$\tL_{m_S}\f$ and
   * \f$\tL_{m_S+1}\f$.  (If there are no such times, give the union
   * time matching \f$\tL_{m_S}\f$ the number of the last remote time
   * not later than \f$\tL_{m_S}\f$).  For example, one possible
   * sequence of times is:
   * \f{equation}
   *   \begin{aligned}
   *     \text{Local side:} \\ \text{Union times:} \\ \text{Remote side:}
   *   \end{aligned}
   *   \cdots
   *   \begin{gathered}
   *     \, \\ \tU_1 \\ \tR_2
   *   \end{gathered}
   *   \leftarrow \Delta \tU_1 \rightarrow
   *   \begin{gathered}
   *     \tL_4 \\ \tU_2 \\ \,
   *   \end{gathered}
   *   \leftarrow \Delta \tU_2 \rightarrow
   *   \begin{gathered}
   *     \, \\ \tU_3 \\ \tR_3
   *   \end{gathered}
   *   \leftarrow \Delta \tU_3 \rightarrow
   *   \begin{gathered}
   *    \, \\ \tU_4 \\ \tR_4
   *   \end{gathered}
   *   \leftarrow \Delta \tU_4 \rightarrow
   *   \begin{gathered}
   *     \tL_5 \\ \tU_5 \\ \,
   *   \end{gathered}
   *   \cdots
   * \f}
   * We call the indices of the step's start and end times in the
   * union time sequence \f$n_S\f$ and \f$n_E\f$, respectively, so for
   * the above example, if we wish to compute the step from
   * \f$\tL_4\f$ to \f$\tL_5\f$, we would have \f$m_S = 4\f$, \f$n_S =
   * 2\f$, and \f$n_E = 5\f$.
   *
   * Let \f$k\f$ be the order of the integrator.  For the sequences of
   * evaluation times ending with \f$\tL_{j_L}\f$ and \f$\tR_{j_R}\f$,
   * we can write the cardinal functions on those times as
   * \f{equation}{
   *   C_{a_L, a_R}(t_L, t_R; j_L, j_R) =
   *   \ell_{a_L}\of{t_L; \tL_{j_L - (k-1)}, \ldots, \tL_{j_L}}
   *   \ell_{a_R}\of{t_R; \tR_{j_R - (k-1)}, \ldots, \tR_{j_R}},
   * \f}
   * with \f$\ell_a(t; x_1, \ldots, x_k)\f$ the Lagrange interpolating
   * polynomials with \f$a\f$ running from \f$1\f$ to \f$k\f$.  The
   * cardinal function satisfies \f$ C_{a_L, a_R}(\tL_{i_L},
   * \tR_{i_R}; j_L, j_R) = \delta_{i_L, j_L + a_L - k} \delta_{i_R,
   * j_R + a_R - k} \f$.  We write the integral of a cardinal function
   * using an Adams-Bashforth step based on the union times along the
   * diagonal \f$\tL = \tR\f$ from \f$\tU_n\f$ to \f$\tU_{n+1}\f$ as
   * \f{equation}{
   *   I_{n,q^L,q^R} =
   *   \Delta \tU_n
   *   \sum_{j = 0}^{k-1}
   *   \tilde{\alpha}_{nj}
   *   C_{k-(m_S-q^L), k-(n-q^R)}\of{\tU_{n-j}, \tU_{n-j}; m_S, n},
   * \f}
   * where \f$\tilde{\alpha}_{nj}\f$ are the Adams-Bashforth
   * coefficients derived from the union times.  The boundary delta
   * for the step is then
   * \f{equation}
   *   \mat{F}_{m_S} =
   *   \mspace{-7mu}
   *   \sum_{q^L = m_S-(k-1)}^{m_S}
   *   \sum_{q^R = n_S-(k-1)}^{n_E-1}
   *   \mspace{-4mu}
   *   \mat{D}_{q^L,q^R}
   *   \sum_{n = \max\left\{n_S, q^R \right\}
   *   }^{\min\left\{n_E, q^R + k \right\} - 1}
   *   I_{n,q^L,q^R}
   * \f}
   * where \f$\mat{D}_{q^L,q^R}\f$ is the coupling function evaluated
   * at the indicated side values.
   */
  template <typename LocalVars, typename RemoteVars, typename Coupling>
  std::result_of_t<const Coupling&(LocalVars, RemoteVars)>
  compute_boundary_delta(
      const Coupling& coupling,
      gsl::not_null<BoundaryHistory<
          LocalVars, RemoteVars,
          std::result_of_t<const Coupling&(LocalVars, RemoteVars)>>*>
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

template <typename LocalVars, typename RemoteVars, typename Coupling>
std::result_of_t<const Coupling&(LocalVars, RemoteVars)>
AdamsBashforthN::compute_boundary_delta(
    const Coupling& coupling,
    const gsl::not_null<BoundaryHistory<
        LocalVars, RemoteVars,
        std::result_of_t<const Coupling&(LocalVars, RemoteVars)>>*>
        history,
    const TimeDelta& time_step) const noexcept {
  ASSERT(not is_self_starting_, "Unimplemented");

  // Avoid billions of casts
  const auto target_order_s = static_cast<ssize_t>(target_order_);

  const SimulationLess simulation_less(time_step.is_positive());

  // Evaluate the cardinal function for a grid of points at a point on
  // the diagonal.
  const auto grid_cardinal_function =
      [target_order_s, &history](const double evaluation_time,
                                 const auto& local_index,
                                 const auto& remote_index,
                                 const auto& remote_times_start) noexcept {
    // Makes an iterator with a map to give time as a double.
    const auto make_lagrange_iterator = [](const auto& it) noexcept {
      return boost::make_transform_iterator(
          it, [](const Time& t) noexcept {
            return t.value();
          });
    };

    return lagrange_polynomial(
        make_lagrange_iterator(local_index),
        evaluation_time,
        make_lagrange_iterator(history->local_begin()),
        make_lagrange_iterator(history->local_end())) *
    lagrange_polynomial(
        make_lagrange_iterator(remote_index),
        evaluation_time,
        make_lagrange_iterator(remote_times_start),
        make_lagrange_iterator(remote_times_start + target_order_s));
  };

  ASSERT(history->local_size() == target_order_,
         "Local history has wrong length (" << history->local_size()
         << " should be " << target_order_s << ")");
  ASSERT(std::is_sorted(history->local_begin(), history->local_end(),
                        simulation_less),
         "Local history not in order");
  ASSERT(std::is_sorted(history->remote_begin(), history->remote_end(),
                        simulation_less),
         "Remote history not in order");

  // Start and end of the step we are trying to take
  const Time start_time = *(history->local_end() - 1);
  const Time end_time = start_time + time_step;

  ASSERT(simulation_less(*(history->remote_end() - 1), end_time),
         "Please supply only older data: " << *(history->remote_end() - 1)
         << " is not before " << end_time);

  // Union of times of all step boundaries on any side.
  const auto union_times = [&end_time, &history, &simulation_less]() noexcept {
    std::set<Time, SimulationLess> ret({end_time}, simulation_less);
    ret.insert(history->local_begin(), history->local_end());
    ret.insert(history->remote_begin(), history->remote_end());
    return ret;
  }();

  // Calculating the Adams-Bashforth coefficients is somewhat
  // expensive, so we cache them.  ab_coefs(it) returns the
  // coefficients used to step from *(it - 1) to *it.
  auto ab_coefs = [target_order_s]() noexcept {
    using Iter = decltype(union_times.cbegin());
    auto compare = [](const Iter& a, const Iter& b) noexcept {
      return Time::StructuralCompare{}(*a, *b);
    };
    return make_cached_function<Iter, std::map, decltype(compare)>(
        [target_order_s](const Iter& times_end) noexcept {
          return get_coefficients(std::prev(times_end, target_order_s),
                                  times_end,
                                  *times_end - *std::prev(times_end));
        },
        std::move(compare));
  }();

  // Result variable.  We evaluate the coupling only for the
  // structure.  This evaluation may be expensive, but by choosing the
  // most recent times on both sides we should guarantee that it is a
  // result we need later, so this will serve to get it into the
  // coupling cache so we don't have to compute it when we actually use it.
  auto accumulated_change =
      make_with_value<std::result_of_t<const Coupling&(LocalVars, RemoteVars)>>(
          history->coupling(
              coupling, history->local_end() - 1, history->remote_end() - 1),
          0.);

  const auto remote_start_step =
      std::upper_bound(history->remote_begin(), history->remote_end(),
                       start_time, simulation_less) - target_order_s;
  for (auto local_evaluation_step = history->local_begin();
       local_evaluation_step != history->local_end();
       ++local_evaluation_step) {
    for (auto remote_evaluation_step = remote_start_step;
         remote_evaluation_step != history->remote_end();
         ++remote_evaluation_step) {
      double deriv_coef = 0.;

      // Prepare to loop over the union times relevant for the current
      // coupling evaluation.

      // Latest of the time at the start of current step (otherwise
      // was computed in a previous call) and the times of the side
      // data being coupled (since segments can't depend on later
      // data).  It can be shown that it is only necessary to check
      // the remote side.
      const Time coupling_start_time =
          std::max(start_time, *remote_evaluation_step, simulation_less);
      // Earliest of the time at the end of the current step (since
      // the step can't depend on later data), and the time
      // target_order steps after the evaluation on each side (since
      // AB only uses data for that long).  It can be shown that it is
      // only necessary to check the remote side.
      const Time coupling_end_time =
          history->remote_end() - remote_evaluation_step > target_order_s
              ? *(remote_evaluation_step + target_order_s)
              : end_time;
      const auto union_times_for_coupling_begin =
          std::upper_bound(union_times.cbegin(), union_times.cend(),
                           coupling_start_time, simulation_less);
      const auto union_times_for_coupling_end =
          std::upper_bound(union_times_for_coupling_begin, union_times.cend(),
                           coupling_end_time, simulation_less);

      // Iterator to the first time on the remote side relevant for the
      // current union step
      auto remote_interpolation_start =
          std::max(remote_evaluation_step - (target_order_s - 1),
                   remote_start_step);

      // This loop computes the coefficient of the coupling evaluation,
      // i.e., the sum over the I_{n,q}, as deriv_coef.  union_time
      // points to the end of the current union step.
      for (auto union_time = union_times_for_coupling_begin;
           union_time != union_times_for_coupling_end;
           ++union_time, ++remote_interpolation_start) {
        // Perform a single step Adams-Bashforth integration of a grid
        // cardinal function.  This computes the quantity called I_{n,q}
        // in the documentation as integrated_cardinal_function.
        const std::vector<double>& method_coefs = ab_coefs(union_time);
        // This produces a reversed iterator pointing to the start of
        // the current union step.
        auto recent_time = std::make_reverse_iterator(union_time);
        auto method_it = method_coefs.begin();
        double integrated_cardinal_function = 0.;
        for (; method_it != method_coefs.end(); ++method_it, ++recent_time) {
          integrated_cardinal_function +=
              *method_it * grid_cardinal_function(recent_time->value(),
                                                  local_evaluation_step,
                                                  remote_evaluation_step,
                                                  remote_interpolation_start);
        }
        integrated_cardinal_function *=
            union_time->value() - std::prev(union_time)->value();

        deriv_coef += integrated_cardinal_function;
      }  // for union_time

      if (deriv_coef != 0.) {
        // We sometimes get exact zeros from the Lagrange polynomials.
        // Skip the (potentially expensive) coupling calculation in that
        // case.

        accumulated_change +=
            deriv_coef * history->coupling(coupling, local_evaluation_step,
                                           remote_evaluation_step);
      }
    }  // for remote_evaluation_step
  }  // for local_evaluation_step

  // Clean up old history

  // We know that the local side will step at end_time, so the
  // step containing that time will be the next step, which is not
  // currently in the history.
  history->local_mark_unneeded(history->local_end() - (target_order_s - 1));
  // We don't know whether other sides will step at end_time,
  // so we have to be conservative and assume they will not.
  history->remote_mark_unneeded(history->remote_end() - target_order_s);

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
