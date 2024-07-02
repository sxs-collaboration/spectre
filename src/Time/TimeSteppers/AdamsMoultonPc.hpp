// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <pup.h>
#include <string>

#include "Options/String.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
struct StepperErrorTolerances;
class TimeDelta;
class TimeStepId;
namespace PUP {
class er;
}  // namespace PUP
namespace TimeSteppers {
template <typename T>
class BoundaryHistoryEvaluator;
class ConstBoundaryHistoryTimes;
template <typename T>
class ConstUntypedHistory;
class MutableBoundaryHistoryTimes;
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
 * An \f$N\$th order Adams-Moulton predictor-corrector method using an
 * \$(N - 1)\$th order Adams-Bashforth predictor.
 *
 * If \p Monotonic is true, dense output is performed using the
 * predictor stage, otherwise the corrector is used.  The corrector
 * results are more accurate (but still formally the same order), but
 * require a RHS evaluation at the end of the step before dense output
 * can be performed.
 *
 * The stable step size factors for different orders are (to
 * approximately 4-5 digits):
 *
 * <table class="doxtable">
 *  <tr>
 *    <th> %Order </th>
 *    <th> CFL Factor </th>
 *  </tr>
 *  <tr>
 *    <td> 2 </td>
 *    <td> 1 </td>
 *  </tr>
 *  <tr>
 *    <td> 3 </td>
 *    <td> 0.981297 </td>
 *  </tr>
 *  <tr>
 *    <td> 4 </td>
 *    <td> 0.794227 </td>
 *  </tr>
 *  <tr>
 *    <td> 5 </td>
 *    <td> 0.612340 </td>
 *  </tr>
 *  <tr>
 *    <td> 6 </td>
 *    <td> 0.464542 </td>
 *  </tr>
 *  <tr>
 *    <td> 7 </td>
 *    <td> 0.350596 </td>
 *  </tr>
 *  <tr>
 *    <td> 8 </td>
 *    <td> 0.264373 </td>
 *  </tr>
 * </table>
 */
template <bool Monotonic>
class AdamsMoultonPc : public LtsTimeStepper {
 public:
  static std::string name() {
    return Monotonic ? "AdamsMoultonPcMonotonic" : "AdamsMoultonPc";
  }

  static constexpr size_t minimum_order = 2;
  static constexpr size_t maximum_order = 8;

  struct Order {
    using type = size_t;
    static constexpr Options::String help = {"Convergence order"};
    static type lower_bound() { return minimum_order; }
    static type upper_bound() { return maximum_order; }
  };
  using options = tmpl::list<Order>;
  static constexpr Options::String help =
      Monotonic
          ? "An Adams-Moulton predictor-corrector time-stepper with monotonic "
            "dense output."
          : "An Adams-Moulton predictor-corrector time-stepper.";

  AdamsMoultonPc() = default;
  explicit AdamsMoultonPc(size_t order);
  AdamsMoultonPc(const AdamsMoultonPc&) = default;
  AdamsMoultonPc& operator=(const AdamsMoultonPc&) = default;
  AdamsMoultonPc(AdamsMoultonPc&&) = default;
  AdamsMoultonPc& operator=(AdamsMoultonPc&&) = default;
  ~AdamsMoultonPc() override = default;

  size_t order() const override;

  uint64_t number_of_substeps() const override;

  uint64_t number_of_substeps_for_error() const override;

  size_t number_of_past_steps() const override;

  double stable_step() const override;

  bool monotonic() const override;

  TimeStepId next_time_id(const TimeStepId& current_id,
                          const TimeDelta& time_step) const override;

  TimeStepId next_time_id_for_error(const TimeStepId& current_id,
                                    const TimeDelta& time_step) const override;

  bool neighbor_data_required(
      const TimeStepId& next_substep_id,
      const TimeStepId& neighbor_data_id) const override;

  bool neighbor_data_required(
      double dense_output_time,
      const TimeStepId& neighbor_data_id) const override;

  WRAPPED_PUPable_decl_template(AdamsMoultonPc);  // NOLINT

  explicit AdamsMoultonPc(CkMigrateMessage* /*unused*/) {}

  void pup(PUP::er& p) override;

 private:
  template <typename T>
  void update_u_impl(gsl::not_null<T*> u, const ConstUntypedHistory<T>& history,
                     const TimeDelta& time_step) const;

  template <typename T>
  std::optional<StepperErrorEstimate> update_u_impl(
      gsl::not_null<T*> u, const ConstUntypedHistory<T>& history,
      const TimeDelta& time_step,
      const std::optional<StepperErrorTolerances>& tolerances) const;

  template <typename T>
  void clean_history_impl(const MutableUntypedHistory<T>& history) const;

  template <typename T>
  bool dense_update_u_impl(gsl::not_null<T*> u,
                           const ConstUntypedHistory<T>& history,
                           double time) const;

  template <typename T>
  bool can_change_step_size_impl(const TimeStepId& time_id,
                                 const ConstUntypedHistory<T>& history) const;

  template <typename T>
  void add_boundary_delta_impl(
      gsl::not_null<T*> result,
      const TimeSteppers::ConstBoundaryHistoryTimes& local_times,
      const TimeSteppers::ConstBoundaryHistoryTimes& remote_times,
      const TimeSteppers::BoundaryHistoryEvaluator<T>& coupling,
      const TimeDelta& time_step) const;

  void clean_boundary_history_impl(
      const TimeSteppers::MutableBoundaryHistoryTimes& local_times,
      const TimeSteppers::MutableBoundaryHistoryTimes& remote_times)
      const override;

  template <typename T>
  void boundary_dense_output_impl(
      gsl::not_null<T*> result,
      const TimeSteppers::ConstBoundaryHistoryTimes& local_times,
      const TimeSteppers::ConstBoundaryHistoryTimes& remote_times,
      const TimeSteppers::BoundaryHistoryEvaluator<T>& coupling,
      double time) const;

  TIME_STEPPER_DECLARE_OVERLOADS
  LTS_TIME_STEPPER_DECLARE_OVERLOADS

  size_t order_{};
};

template <bool Monotonic>
bool operator==(const AdamsMoultonPc<Monotonic>& lhs,
                const AdamsMoultonPc<Monotonic>& rhs);
template <bool Monotonic>
bool operator!=(const AdamsMoultonPc<Monotonic>& lhs,
                const AdamsMoultonPc<Monotonic>& rhs);
}  // namespace TimeSteppers
