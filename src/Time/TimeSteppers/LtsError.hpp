// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "DataStructures/TaggedVariant.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"

/// \cond
class StepperErrorTolerances;
class TimeDelta;
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
 * Class used to allow compiling LTS code with GTS time steppers.  All
 * methods error at runtime.
 */
class LtsError : public LtsTimeStepper {
 public:
  LtsError() = default;
  LtsError(const LtsError&) = default;
  LtsError& operator=(const LtsError&) = default;
  LtsError(LtsError&&) = default;
  LtsError& operator=(LtsError&&) = default;
  ~LtsError() override = default;

  [[noreturn]] variants::TaggedVariant<Tags::FixedOrder, Tags::VariableOrder>
  order() const override;

  [[noreturn]] uint64_t number_of_substeps() const override;

  [[noreturn]] uint64_t number_of_substeps_for_error() const override;

  [[noreturn]] size_t number_of_past_steps() const override;

  [[noreturn]] double stable_step() const override;

  [[noreturn]] bool monotonic() const override;

  [[noreturn]] TimeStepId next_time_id(
      const TimeStepId& current_id, const TimeDelta& time_step) const override;

  [[noreturn]] TimeStepId next_time_id_for_error(
      const TimeStepId& current_id, const TimeDelta& time_step) const override;

  [[noreturn]] bool neighbor_data_required(
      const TimeStepId& next_substep_id,
      const TimeStepId& neighbor_data_id) const override;

  [[noreturn]] bool neighbor_data_required(
      double dense_output_time,
      const TimeStepId& neighbor_data_id) const override;

#if defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-noreturn"
#endif
  WRAPPED_PUPable_decl_template(LtsError);  // NOLINT
#if defined(__clang__)
#pragma GCC diagnostic pop
#endif

  explicit LtsError(CkMigrateMessage* /*unused*/) {}

  [[noreturn]] void pup(PUP::er& p) override;

 private:
  template <typename T>
  [[noreturn]] void update_u_impl(gsl::not_null<T*> u,
                                  const ConstUntypedHistory<T>& history,
                                  const TimeDelta& time_step) const;

  template <typename T>
  [[noreturn]] std::optional<StepperErrorEstimate> update_u_impl(
      gsl::not_null<T*> u, const ConstUntypedHistory<T>& history,
      const TimeDelta& time_step,
      const StepperErrorTolerances& tolerances) const;

  template <typename T>
  [[noreturn]] void clean_history_impl(
      const MutableUntypedHistory<T>& history) const;

  template <typename T>
  [[noreturn]] bool dense_update_u_impl(gsl::not_null<T*> u,
                                        const ConstUntypedHistory<T>& history,
                                        double time) const;

  template <typename T>
  [[noreturn]] bool can_change_step_size_impl(
      const TimeStepId& time_id, const ConstUntypedHistory<T>& history) const;

  template <typename T>
  [[noreturn]] void add_boundary_delta_impl(
      gsl::not_null<T*> result,
      const TimeSteppers::ConstBoundaryHistoryTimes& local_times,
      const TimeSteppers::ConstBoundaryHistoryTimes& remote_times,
      const TimeSteppers::BoundaryHistoryEvaluator<T>& coupling,
      const TimeDelta& time_step) const;

  [[noreturn]] void clean_boundary_history_impl(
      const TimeSteppers::MutableBoundaryHistoryTimes& local_times,
      const TimeSteppers::MutableBoundaryHistoryTimes& remote_times)
      const override;

  template <typename T>
  [[noreturn]] void boundary_dense_output_impl(
      gsl::not_null<T*> result,
      const TimeSteppers::ConstBoundaryHistoryTimes& local_times,
      const TimeSteppers::ConstBoundaryHistoryTimes& remote_times,
      const TimeSteppers::BoundaryHistoryEvaluator<T>& coupling,
      double time) const;

#if defined(__GNUC__) and not defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif
  TIME_STEPPER_DECLARE_OVERLOADS
  LTS_TIME_STEPPER_DECLARE_OVERLOADS
#if defined(__GNUC__) and not defined(__clang__)
#pragma GCC diagnostic pop
#endif
};
}  // namespace TimeSteppers
