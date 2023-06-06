// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>

#include "Options/String.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class TimeDelta;
class TimeStepId;
namespace PUP {
class er;
}  // namespace PUP
namespace TimeSteppers {
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
 * An \f$N\$th order Adams-Moulton predictor-corrector method using an
 * \$(N - 1)\$th order Adams-Bashforth predictor.
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
class AdamsMoultonPc : public TimeStepper {
 public:
  static constexpr size_t minimum_order = 2;
  static constexpr size_t maximum_order = 8;

  struct Order {
    using type = size_t;
    static constexpr Options::String help = {"Convergence order"};
    static type lower_bound() { return minimum_order; }
    static type upper_bound() { return maximum_order; }
  };
  using options = tmpl::list<Order>;
  static constexpr Options::String help = {
      "An Adams-Moulton predictor-corrector time-stepper."};

  AdamsMoultonPc() = default;
  explicit AdamsMoultonPc(size_t order);
  AdamsMoultonPc(const AdamsMoultonPc&) = default;
  AdamsMoultonPc& operator=(const AdamsMoultonPc&) = default;
  AdamsMoultonPc(AdamsMoultonPc&&) = default;
  AdamsMoultonPc& operator=(AdamsMoultonPc&&) = default;
  ~AdamsMoultonPc() override = default;

  size_t order() const override;

  size_t error_estimate_order() const override;

  uint64_t number_of_substeps() const override;

  uint64_t number_of_substeps_for_error() const override;

  size_t number_of_past_steps() const override;

  double stable_step() const override;

  TimeStepId next_time_id(const TimeStepId& current_id,
                          const TimeDelta& time_step) const override;

  TimeStepId next_time_id_for_error(const TimeStepId& current_id,
                                    const TimeDelta& time_step) const override;

  WRAPPED_PUPable_decl_template(AdamsMoultonPc);  // NOLINT

  explicit AdamsMoultonPc(CkMigrateMessage* /*unused*/) {}

  void pup(PUP::er& p) override;

 private:
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

  template <typename T>
  bool can_change_step_size_impl(const TimeStepId& time_id,
                                 const ConstUntypedHistory<T>& history) const;

  TIME_STEPPER_DECLARE_OVERLOADS

  size_t order_{};
};

bool operator==(const AdamsMoultonPc& lhs, const AdamsMoultonPc& rhs);
bool operator!=(const AdamsMoultonPc& lhs, const AdamsMoultonPc& rhs);
}  // namespace TimeSteppers
