// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "Options/String.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class TimeDelta;
namespace PUP {
class er;
}  // namespace PUP
namespace TimeSteppers {
class BoundaryHistoryCleaner;
template <typename T>
class BoundaryHistoryEvaluator;
template <typename T>
class ConstUntypedHistory;
template <typename T>
class MutableUntypedHistory;
}  // namespace TimeSteppers
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace TimeSteppers {

/*!
 * \ingroup TimeSteppersGroup
 *
 * An Nth order Adams-Bashforth time stepper.
 *
 * The stable step size factors for different orders are given by:
 *
 * <table class="doxtable">
 *  <tr>
 *    <th> %Order </th>
 *    <th> CFL Factor </th>
 *  </tr>
 *  <tr>
 *    <td> 1 </td>
 *    <td> 1 </td>
 *  </tr>
 *  <tr>
 *    <td> 2 </td>
 *    <td> 1 / 2 </td>
 *  </tr>
 *  <tr>
 *    <td> 3 </td>
 *    <td> 3 / 11 </td>
 *  </tr>
 *  <tr>
 *    <td> 4 </td>
 *    <td> 3 / 20 </td>
 *  </tr>
 *  <tr>
 *    <td> 5 </td>
 *    <td> 45 / 551 </td>
 *  </tr>
 *  <tr>
 *    <td> 6 </td>
 *    <td> 5 / 114 </td>
 *  </tr>
 *  <tr>
 *    <td> 7 </td>
 *    <td> 945 / 40663 </td>
 *  </tr>
 *  <tr>
 *    <td> 8 </td>
 *    <td> 945 / 77432 </td>
 *  </tr>
 * </table>
 *
 * \section lts Local time stepping calculation
 *
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
class AdamsBashforth : public LtsTimeStepper {
 public:
  static constexpr const size_t maximum_order = 8;

  struct Order {
    using type = size_t;
    static constexpr Options::String help = {"Convergence order"};
    static type lower_bound() { return 1; }
    static type upper_bound() { return maximum_order; }
  };
  using options = tmpl::list<Order>;
  static constexpr Options::String help = {
      "An Adams-Bashforth Nth order time-stepper."};

  AdamsBashforth() = default;
  explicit AdamsBashforth(size_t order);
  AdamsBashforth(const AdamsBashforth&) = default;
  AdamsBashforth& operator=(const AdamsBashforth&) = default;
  AdamsBashforth(AdamsBashforth&&) = default;
  AdamsBashforth& operator=(AdamsBashforth&&) = default;
  ~AdamsBashforth() override = default;

  size_t order() const override;

  size_t error_estimate_order() const override;

  size_t number_of_past_steps() const override;

  double stable_step() const override;

  TimeStepId next_time_id(const TimeStepId& current_id,
                          const TimeDelta& time_step) const override;

  WRAPPED_PUPable_decl_template(AdamsBashforth);  // NOLINT

  explicit AdamsBashforth(CkMigrateMessage* /*unused*/) {}

  // clang-tidy: do not pass by non-const reference
  void pup(PUP::er& p) override;  // NOLINT

 private:
  friend bool operator==(const AdamsBashforth& lhs, const AdamsBashforth& rhs);

  // Some of the private methods take a parameter of type "Delta" or
  // "TimeType".  Delta is expected to be a TimeDelta or an
  // ApproximateTimeDelta, and TimeType is expected to be a Time or an
  // ApproximateTime.  The former cases will detect and optimize the
  // constant-time-step case, while the latter are necessary for dense
  // output.
  template <typename T>
  void update_u_impl(gsl::not_null<T*> u,
                     const MutableUntypedHistory<T>& history,
                     const TimeDelta& time_step) const;

  template <typename T>
  bool update_u_impl(gsl::not_null<T*> u, gsl::not_null<T*> u_error,
                     const MutableUntypedHistory<T>& history,
                     const TimeDelta& time_step) const;

  template <typename T>
  bool dense_update_u_impl(gsl::not_null<T*> u,
                           const ConstUntypedHistory<T>& history,
                           double time) const;

  template <typename T, typename Delta>
  void update_u_common(gsl::not_null<T*> u,
                       const ConstUntypedHistory<T>& history,
                       const Delta& time_step, size_t order) const;

  template <typename T>
  bool can_change_step_size_impl(const TimeStepId& time_id,
                                 const ConstUntypedHistory<T>& history) const;

  template <typename T>
  void add_boundary_delta_impl(
      gsl::not_null<T*> result,
      const TimeSteppers::BoundaryHistoryEvaluator<T>& coupling,
      const TimeSteppers::BoundaryHistoryCleaner& cleaner,
      const TimeDelta& time_step) const;

  template <typename T>
  void boundary_dense_output_impl(
      gsl::not_null<T*> result,
      const TimeSteppers::BoundaryHistoryEvaluator<T>& coupling,
      const double time) const;

  template <typename T, typename TimeType>
  void boundary_impl(gsl::not_null<T*> result,
                     const BoundaryHistoryEvaluator<T>& coupling,
                     const TimeType& end_time) const;

  TIME_STEPPER_DECLARE_OVERLOADS
  LTS_TIME_STEPPER_DECLARE_OVERLOADS

  size_t order_ = 3;
};

bool operator!=(const AdamsBashforth& lhs, const AdamsBashforth& rhs);
}  // namespace TimeSteppers
