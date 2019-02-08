// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class AdamsBashforthN

#pragma once

#include <algorithm>
#include <boost/iterator/transform_iterator.hpp>
#include <cstddef>
#include <iosfwd>
#include <iterator>
#include <limits>
#include <map>
#include <pup.h>
#include <tuple>
#include <type_traits>
#include <vector>

#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/Interpolation/LagrangePolynomial.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"  // IWYU pragma: keep
#include "Utilities/CachedFunction.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace TimeSteppers {
template <typename LocalVars, typename RemoteVars, typename CouplingResult>
class BoundaryHistory;  // IWYU pragma: keep
template <typename Vars, typename DerivVars>
class History;
}  // namespace TimeSteppers
/// \endcond

namespace TimeSteppers {

/// \ingroup TimeSteppersGroup
///
/// An Nth Adams-Bashforth time stepper.
class AdamsBashforthN : public LtsTimeStepper::Inherit {
 public:
  static constexpr const size_t maximum_order = 8;

  struct Order {
    using type = size_t;
    static constexpr OptionString help = {"Convergence order"};
    static type lower_bound() noexcept { return 1; }
    static type upper_bound() noexcept { return maximum_order; }
  };
  using options = tmpl::list<Order>;
  static constexpr OptionString help = {
      "An Adams-Bashforth Nth order time-stepper."};

  AdamsBashforthN() = default;
  explicit AdamsBashforthN(size_t order) noexcept;
  AdamsBashforthN(const AdamsBashforthN&) noexcept = default;
  AdamsBashforthN& operator=(const AdamsBashforthN&) noexcept = default;
  AdamsBashforthN(AdamsBashforthN&&) noexcept = default;
  AdamsBashforthN& operator=(AdamsBashforthN&&) noexcept = default;
  ~AdamsBashforthN() noexcept override = default;

  template <typename Vars, typename DerivVars>
  void update_u(gsl::not_null<Vars*> u,
                gsl::not_null<History<Vars, DerivVars>*> history,
                const TimeDelta& time_step) const noexcept;

  template <typename Vars, typename DerivVars>
  void dense_update_u(gsl::not_null<Vars*> u,
                      const History<Vars, DerivVars>& history,
                      double time) const noexcept;

  // This is defined as a separate type alias to keep the doxygen page
  // width somewhat under control.
  template <typename LocalVars, typename RemoteVars, typename Coupling>
  using BoundaryHistoryType =
      BoundaryHistory<LocalVars, RemoteVars,
                      std::result_of_t<const Coupling&(LocalVars, RemoteVars)>>;

  /*!
   * An explanation of the computation being performed by this
   * function:
   * \f$\newcommand\tL{t^L}\newcommand\tR{t^R}\newcommand\tU{\tilde{t}\!}
   * \newcommand\mat{\mathbf}\f$
   *
   * Suppose the local and remote sides of the interface are evaluated
   * at times \f$\ldots, \tL_{-1}, \tL_0, \tL_1, \ldots\f$ and
   * \f$\ldots, \tR_{-1}, \tR_0, \tR_1, \ldots\f$, respectively, with
   * the starting location of the numbering arbitrary in each case.
   * Let the step we wish to calculate the effect of be the step from
   * \f$\tL_{m_S}\f$ to \f$\tL_{m_S+1}\f$.  We call the sequence
   * produced from the union of the local and remote time sequences
   * \f$\ldots, \tU_{-1}, \tU_0, \tU_1, \ldots\f$.  For example, one
   * possible sequence of times is:
   * \f{equation}
   *   \begin{aligned}
   *     \text{Local side:} \\ \text{Union times:} \\ \text{Remote side:}
   *   \end{aligned}
   *   \cdots
   *   \begin{gathered}
   *     \, \\ \tU_1 \\ \tR_5
   *   \end{gathered}
   *   \leftarrow \Delta \tU_1 \rightarrow
   *   \begin{gathered}
   *     \tL_4 \\ \tU_2 \\ \,
   *   \end{gathered}
   *   \leftarrow \Delta \tU_2 \rightarrow
   *   \begin{gathered}
   *     \, \\ \tU_3 \\ \tR_6
   *   \end{gathered}
   *   \leftarrow \Delta \tU_3 \rightarrow
   *   \begin{gathered}
   *    \, \\ \tU_4 \\ \tR_7
   *   \end{gathered}
   *   \leftarrow \Delta \tU_4 \rightarrow
   *   \begin{gathered}
   *     \tL_5 \\ \tU_5 \\ \,
   *   \end{gathered}
   *   \cdots
   * \f}
   * We call the indices of the step's start and end times in the
   * union time sequence \f$n_S\f$ and \f$n_E\f$, respectively.  We
   * define \f$n^L_m\f$ to be the union-time index corresponding to
   * \f$\tL_m\f$ and \f$m^L_n\f$ to be the index of the last local
   * time not later than \f$\tU_n\f$ and similarly for the remote
   * side.  So for the above example, \f$n^L_4 = 2\f$ and \f$m^R_2 =
   * 5\f$, and if we wish to compute the step from \f$\tL_4\f$ to
   * \f$\tL_5\f$ we would have \f$m_S = 4\f$, \f$n_S = 2\f$, and
   * \f$n_E = 5\f$.
   *
   * If we wish to evaluate the change over this step to \f$k\f$th
   * order, we can write the change in the value as a linear
   * combination of the values of the coupling between the elements at
   * unequal times:
   * \f{equation}
   *   \mat{F}_{m_S} =
   *   \mspace{-10mu}
   *   \sum_{q^L = m_S-(k-1)}^{m_S}
   *   \,
   *   \sum_{q^R = m^R_{n_S}-(k-1)}^{m^R_{n_E-1}}
   *   \mspace{-10mu}
   *   \mat{D}_{q^Lq^R}
   *   I_{q^Lq^R},
   * \f}
   * where \f$\mat{D}_{q^Lq^R}\f$ is the coupling function evaluated
   * between data from \f$\tL_{q^L}\f$ and \f$\tR_{q^R}\f$.  The
   * coefficients can be written as the sum of three terms,
   * \f{equation}
   *   I_{q^Lq^R} = I^E_{q^Lq^R} + I^R_{q^Lq^R} + I^L_{q^Lq^R},
   * \f}
   * which can be interpreted as a contribution from equal-time
   * evaluations and contributions related to the remote and local
   * evaluation times.  These are given by
   * \f{align}
   *   I^E_{q^Lq^R} &=
   *   \mspace{-10mu}
   *   \sum_{n=n_S}^{\min\left\{n_E, n^L+k\right\}-1}
   *   \mspace{-10mu}
   *   \tilde{\alpha}_{n,n-n^L} \Delta \tU_n
   *   &&\text{if $\tL_{q^L} = \tR_{q^R}$, otherwise 0}
   *   \\
   *   I^R_{q^Lq^R} &=
   *   \ell_{q^L - m_S + k}\!\left(
   *     \tU_{n^R}; \tL_{m_S - (k-1)}, \ldots, \tL_{m_S}\right)
   *   \mspace{-10mu}
   *   \sum_{n=\max\left\{n_S, n^R\right\}}
   *       ^{\min\left\{n_E, n^R+k\right\}-1}
   *   \mspace{-10mu}
   *   \tilde{\alpha}_{n,n-n^R} \Delta \tU_n
   *   &&\text{if $\tR_{q^R}$ is not in $\{\tL_{\vphantom{|}\cdots}\}$,
   *     otherwise 0}
   *   \\
   *   I^L_{q^Lq^R} &=
   *   \mspace{-10mu}
   *   \sum_{n=\max\left\{n_S, n^R\right\}}
   *       ^{\min\left\{n_E, n^L+k, n^R_{q^R+k}\right\}-1}
   *   \mspace{-10mu}
   *   \ell_{q^R - m^R_n + k}\!\left(\tU_{n^L};
   *     \tR_{m^R_n - (k-1)}, \ldots, \tR_{m^R_n}\right)
   *   \tilde{\alpha}_{n,n-n^L} \Delta \tU_n
   *   &&\text{if $\tL_{q^L}$ is not in $\{\tR_{\vphantom{|}\cdots}\}$,
   *     otherwise 0,}
   * \f}
   * where for brevity we write \f$n^L = n^L_{q^L}\f$ and \f$n^R =
   * n^R_{q^R}\f$, and where \f$\ell_a(t; x_1, \ldots, x_k)\f$ a
   * Lagrange interpolating polynomial and \f$\tilde{\alpha}_{nj}\f$
   * is the \f$j\f$th coefficient for an Adams-Bashforth step over the
   * union times from step \f$n\f$ to step \f$n+1\f$.
   */
  template <typename LocalVars, typename RemoteVars, typename Coupling>
  std::result_of_t<const Coupling&(LocalVars, RemoteVars)>
  compute_boundary_delta(
      const Coupling& coupling,
      gsl::not_null<BoundaryHistoryType<LocalVars, RemoteVars, Coupling>*>
          history,
      const TimeDelta& time_step) const noexcept;

  template <typename LocalVars, typename RemoteVars, typename Coupling>
  std::result_of_t<const Coupling&(LocalVars, RemoteVars)>
  boundary_dense_output(
      const Coupling& coupling,
      const BoundaryHistoryType<LocalVars, RemoteVars, Coupling>& history,
      double time) const noexcept;

  size_t number_of_past_steps() const noexcept override;

  double stable_step() const noexcept override;

  TimeId next_time_id(const TimeId& current_id,
                      const TimeDelta& time_step) const noexcept override;

  template <typename Vars, typename DerivVars>
  bool can_change_step_size(
      const TimeId& time_id,
      const TimeSteppers::History<Vars, DerivVars>& history) const noexcept;

  WRAPPED_PUPable_decl_template(AdamsBashforthN);  // NOLINT

  explicit AdamsBashforthN(CkMigrateMessage* /*unused*/) noexcept {}

  // clang-tidy: do not pass by non-const reference
  void pup(PUP::er& p) noexcept override;  // NOLINT

 private:
  friend bool operator==(const AdamsBashforthN& lhs,
                         const AdamsBashforthN& rhs) noexcept;

  // Some of the private methods take a parameter of type "Delta" or
  // "TimeType".  Delta is expected to be a TimeDelta or an
  // ApproximateTimeDelta, and TimeType is expected to be a Time or an
  // ApproximateTime.  The former cases will detect and optimize the
  // constant-time-step case, while the latter are necessary for dense
  // output.

  template <typename Vars, typename DerivVars, typename Delta>
  void update_u_impl(gsl::not_null<Vars*> u,
                     const History<Vars, DerivVars>& history,
                     const Delta& time_step) const noexcept;

  template <typename LocalVars, typename RemoteVars, typename Coupling,
            typename TimeType>
  std::result_of_t<const Coupling&(LocalVars, RemoteVars)> boundary_impl(
      const Coupling& coupling,
      const BoundaryHistoryType<LocalVars, RemoteVars, Coupling>& history,
      const TimeType& end_time) const noexcept;

  /// Get coefficients for a time step.  Arguments are an iterator
  /// pair to past times, oldest to newest, and the time step to take.
  template <typename Iterator, typename Delta>
  static std::vector<double> get_coefficients(const Iterator& times_begin,
                                              const Iterator& times_end,
                                              const Delta& step) noexcept;

  static std::vector<double> get_coefficients_impl(
      const std::vector<double>& steps) noexcept;

  static std::vector<double> variable_coefficients(
      const std::vector<double>& steps) noexcept;

  static std::vector<double> constant_coefficients(size_t order) noexcept;

  struct ApproximateTimeDelta;

  // Time-like interface to a double used for dense output
  struct ApproximateTime {
    double time = std::numeric_limits<double>::signaling_NaN();
    double value() const noexcept { return time; }

    // Only the operators that are actually used are defined.
    friend ApproximateTimeDelta operator-(const ApproximateTime& a,
                                          const Time& b) noexcept {
      return {a.value() - b.value()};
    }

    friend bool operator<(const Time& a, const ApproximateTime& b) noexcept {
      return a.value() < b.value();
    }

    friend bool operator<(const ApproximateTime& a, const Time& b) noexcept {
      return a.value() < b.value();
    }

    friend std::ostream& operator<<(std::ostream& s,
                                    const ApproximateTime& t) noexcept {
      return s << t.value();
    }
  };

  // TimeDelta-like interface to a double used for dense output
  struct ApproximateTimeDelta {
    double delta = std::numeric_limits<double>::signaling_NaN();
    double value() const noexcept { return delta; }
    bool is_positive() const noexcept { return delta > 0.; }

    // Only the operators that are actually used are defined.
    friend bool operator<(const ApproximateTimeDelta& a,
                          const ApproximateTimeDelta& b) noexcept {
      return a.value() < b.value();
    }

    friend double operator/(
        const TimeDelta& a,
        const AdamsBashforthN::ApproximateTimeDelta& b) noexcept {
      return a.value() / b.value();
    }
  };

  size_t order_ = 3;
};

bool operator!=(const AdamsBashforthN& lhs,
                const AdamsBashforthN& rhs) noexcept;

template <typename Vars, typename DerivVars>
void AdamsBashforthN::update_u(
    const gsl::not_null<Vars*> u,
    const gsl::not_null<History<Vars, DerivVars>*> history,
    const TimeDelta& time_step) const noexcept {
  update_u_impl(u, *history, time_step);
  history->mark_unneeded(history->begin() + 1);
}

template <typename Vars, typename DerivVars>
void AdamsBashforthN::dense_update_u(const gsl::not_null<Vars*> u,
                                     const History<Vars, DerivVars>& history,
                                     const double time) const noexcept {
  const ApproximateTimeDelta time_step{
      time - history[history.size() - 1].value()};
  update_u_impl(u, history, time_step);
}

template <typename Vars, typename DerivVars, typename Delta>
void AdamsBashforthN::update_u_impl(const gsl::not_null<Vars*> u,
                                    const History<Vars, DerivVars>& history,
                                    const Delta& time_step) const noexcept {
  ASSERT(history.size() <= order_,
         "Length of history (" << history.size() << ") "
         << "should not exceed target order (" << order_ << ")");

  const auto& coefficients =
      get_coefficients(history.begin(), history.end(), time_step);

  const auto do_update =
      [u, &time_step, &coefficients, &history](auto order) noexcept {
    *u += time_step.value() * constexpr_sum<order>(
        [order, &coefficients, &history](auto i) noexcept {
          return coefficients[order - 1 - i] *
              (history.begin() +
               static_cast<
                   typename History<Vars, DerivVars>::difference_type>(i))
                  .derivative();
        });
  };

  switch (history.size()) {
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
      ERROR("Bad amount of history data: " << history.size());
  }
}

template <typename LocalVars, typename RemoteVars, typename Coupling>
std::result_of_t<const Coupling&(LocalVars, RemoteVars)>
AdamsBashforthN::compute_boundary_delta(
    const Coupling& coupling,
    const gsl::not_null<BoundaryHistoryType<LocalVars, RemoteVars, Coupling>*>
        history,
    const TimeDelta& time_step) const noexcept {
  auto result = boundary_impl(coupling, *history,
                              *(history->local_end() - 1) + time_step);

  // We know that the local side will step at end_time, so the step
  // containing that time will be the next step, which is not
  // currently in the history.  We therefore know we won't need the
  // oldest value for the next step.
  history->local_mark_unneeded(history->local_begin() + 1);
  // We don't know whether the remote side will step at end_time, so
  // we have to be conservative and assume it will not.  If it does we
  // will ignore the first value in the next call to this function.
  history->remote_mark_unneeded(
      history->remote_end() -
      static_cast<typename decltype(history->remote_begin())::difference_type>(
          history->local_size() + 1));

  return result;
}

template <typename LocalVars, typename RemoteVars, typename Coupling>
std::result_of_t<const Coupling&(LocalVars, RemoteVars)>
AdamsBashforthN::boundary_dense_output(
    const Coupling& coupling,
    const BoundaryHistoryType<LocalVars, RemoteVars, Coupling>& history,
    const double time) const noexcept {
  return boundary_impl(coupling, history, ApproximateTime{time});
}

template <typename LocalVars, typename RemoteVars, typename Coupling,
          typename TimeType>
std::result_of_t<const Coupling&(LocalVars, RemoteVars)>
AdamsBashforthN::boundary_impl(
    const Coupling& coupling,
    const BoundaryHistoryType<LocalVars, RemoteVars, Coupling>& history,
    const TimeType& end_time) const noexcept {
  // Might be different from order_ during self-start.
  const auto current_order = history.local_size();

  ASSERT(current_order <= order_,
         "Local history is too long for target order (" << current_order
         << " should not exceed " << order_ << ")");
  ASSERT(history.remote_size() >= current_order,
         "Remote history is too short (" << history.remote_size()
         << " should be at least " << current_order << ")");

  // Start and end of the step we are trying to take
  const Time start_time = *(history.local_end() - 1);
  const auto time_step = end_time - start_time;

  // If a remote evaluation is done at the start of the step then that
  // is part of the history for the first union step.  When we did
  // history cleanup at the end of the previous step we didn't know we
  // were going to get this point so we kept an extra remote history
  // value.
  const bool remote_aligned_at_step_start =
      history.remote_size() > current_order and
      *(history.remote_begin() +
        static_cast<typename decltype(history.remote_begin())::difference_type>(
            current_order)) == start_time;
  const auto remote_begin =
      history.remote_begin() + (remote_aligned_at_step_start ? 1 : 0);

  // Result variable.  We evaluate the coupling only for the
  // structure.  This evaluation may be expensive, but by choosing the
  // most recent times on both sides we should guarantee that it is a
  // result we need later, so this will serve to get it into the
  // coupling cache so we don't have to compute it when we actually use it.
  auto accumulated_change =
      make_with_value<std::result_of_t<const Coupling&(LocalVars, RemoteVars)>>(
          history.coupling(coupling, history.local_end() - 1,
                           history.remote_end() - 1),
          0.);

  if (history.local_size() ==
          static_cast<size_t>(history.remote_end() - remote_begin) and
      std::equal(history.local_begin(), history.local_end(), remote_begin)) {
    // No local time-stepping going on.
    const auto coefficients =
        get_coefficients(history.local_begin(), history.local_end(), time_step);

    auto local_it = history.local_begin();
    auto remote_it = remote_begin;
    for (auto coefficients_it = coefficients.rbegin();
         coefficients_it != coefficients.rend();
         ++coefficients_it, ++local_it, ++remote_it) {
      accumulated_change +=
          *coefficients_it * history.coupling(coupling, local_it, remote_it);
    }
    accumulated_change *= time_step.value();

    return accumulated_change;
  }

  ASSERT(current_order == order_,
         "Cannot perform local time-stepping while self-starting.");

  // Avoid billions of casts
  const auto order_s = static_cast<typename BoundaryHistoryType<
      LocalVars, RemoteVars, Coupling>::remote_iterator::difference_type>(
      order_);

  const evolution_less<> less{time_step.is_positive()};

  ASSERT(std::is_sorted(history.local_begin(), history.local_end(), less),
         "Local history not in order");
  ASSERT(std::is_sorted(remote_begin, history.remote_end(), less),
         "Remote history not in order");
  ASSERT(not less(start_time, *(remote_begin + (order_s - 1))),
         "Remote history does not extend far enough back");
  ASSERT(less(*(history.remote_end() - 1), end_time),
         "Please supply only older data: " << *(history.remote_end() - 1)
         << " is not before " << end_time);

  // Union of times of all step boundaries on any side.
  const auto union_times = [&history, &remote_begin, &less]() noexcept {
    std::vector<Time> ret;
    ret.reserve(history.local_size() + history.remote_size());
    std::set_union(history.local_begin(), history.local_end(), remote_begin,
                   history.remote_end(), std::back_inserter(ret), less);
    return ret;
  }();

  using UnionIter = typename decltype(union_times)::const_iterator;

  // Find the union times iterator for a given time.
  const auto union_step = [&union_times, &less](const Time& t) noexcept {
    return std::lower_bound(union_times.cbegin(), union_times.cend(), t, less);
  };

  // The union time index for the step start.
  const auto union_step_start = union_step(start_time);

  // min(union_times.end(), it + order_s) except being careful not
  // to create out-of-range iterators.
  const auto advance_within_step =
      [order_s, &union_times](const UnionIter& it) noexcept {
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
  auto ab_coefs = make_overloader(
      make_cached_function<std::tuple<UnionIter, TimeDelta>,
                           std::map>([order_s](
          const std::tuple<UnionIter, TimeDelta>& args) noexcept {
        return get_coefficients(
            std::get<0>(args) -
                static_cast<typename UnionIter::difference_type>(order_s - 1),
            std::get<0>(args) + 1, std::get<1>(args));
      }),
      make_cached_function<std::tuple<UnionIter, ApproximateTimeDelta>,
                           std::map>([order_s](
          const std::tuple<UnionIter, ApproximateTimeDelta>& args) noexcept {
        return get_coefficients(
            std::get<0>(args) -
                static_cast<typename UnionIter::difference_type>(order_s - 1),
            std::get<0>(args) + 1, std::get<1>(args));
      }));

  // The value of the coefficient of `evaluation_step` when doing
  // a standard Adams-Bashforth integration over the union times
  // from `step` to `step + 1`.
  const auto base_summand = [&ab_coefs, &end_time, &union_times](
      const UnionIter& step, const UnionIter& evaluation_step) noexcept {
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

  for (auto local_evaluation_step = history.local_begin();
       local_evaluation_step != history.local_end();
       ++local_evaluation_step) {
    const auto union_local_evaluation_step = union_step(*local_evaluation_step);
    for (auto remote_evaluation_step = remote_begin;
         remote_evaluation_step != history.remote_end();
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
        const auto make_lagrange_iterator = [](const auto& it) noexcept {
          return boost::make_transform_iterator(
              it, [](const Time& t) noexcept { return t.value(); });
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
        if (not std::binary_search(history.local_begin(), history.local_end(),
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
                                  make_lagrange_iterator(history.local_begin()),
                                  make_lagrange_iterator(history.local_end()));
        }

        // Same qualitative calculation as the previous block, but
        // interpolating over the remote times.  This case is somewhat
        // more complicated because the latest remote time that can be
        // used varies for the different segments making up the step.
        if (not std::binary_search(remote_begin, history.remote_end(),
                                   *local_evaluation_step, less)) {
          auto union_step_upper_bound =
              advance_within_step(union_local_evaluation_step);
          if (history.remote_end() - remote_evaluation_step > order_s) {
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
        accumulated_change +=
            deriv_coef * history.coupling(coupling, local_evaluation_step,
                                          remote_evaluation_step);
      }
    }  // for remote_evaluation_step
  }  // for local_evaluation_step

  return accumulated_change;
}

template <typename Vars, typename DerivVars>
bool AdamsBashforthN::can_change_step_size(
    const TimeId& time_id,
    const TimeSteppers::History<Vars, DerivVars>& history) const noexcept {
  // We need to forbid local time-stepping before initialization is
  // complete.  The self-start procedure itself should never consider
  // changing the step size, but we need to wait during the main
  // evolution until the self-start history has been replaced with
  // "real" values.
  const evolution_less<Time> less{time_id.time_runs_forward()};
  return history.size() == 0 or
         (less(history.back(), time_id.time()) and
          std::is_sorted(history.begin(), history.end(), less));
}

template <typename Iterator, typename Delta>
std::vector<double> AdamsBashforthN::get_coefficients(
    const Iterator& times_begin, const Iterator& times_end,
    const Delta& step) noexcept {
  ASSERT(times_begin != times_end, "No history provided");
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
