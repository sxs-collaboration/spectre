// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/LtsError.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>

#include "DataStructures/TaggedVariant.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

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

namespace TimeSteppers {
namespace {
[[noreturn]] void gts_error() {
  ERROR("Attempted to use LtsTimeStepper in a GTS evolution.  This is a bug.");
}
}  // namespace

[[noreturn]] variants::TaggedVariant<Tags::FixedOrder, Tags::VariableOrder>
LtsError::order() const {
  gts_error();
}

[[noreturn]] uint64_t LtsError::number_of_substeps() const { gts_error(); }

[[noreturn]] uint64_t LtsError::number_of_substeps_for_error() const {
  gts_error();
}

[[noreturn]] size_t LtsError::number_of_past_steps() const { gts_error(); }

[[noreturn]] double LtsError::stable_step() const { gts_error(); }

[[noreturn]] bool LtsError::monotonic() const { gts_error(); }

[[noreturn]] TimeStepId LtsError::next_time_id(
    const TimeStepId& /*current_id*/, const TimeDelta& /*time_step*/) const {
  gts_error();
}

[[noreturn]] TimeStepId LtsError::next_time_id_for_error(
    const TimeStepId& /*current_id*/, const TimeDelta& /*time_step*/) const {
  gts_error();
}

[[noreturn]] bool LtsError::neighbor_data_required(
    const TimeStepId& /*next_substep_id*/,
    const TimeStepId& /*neighbor_data_id*/) const {
  gts_error();
}

[[noreturn]] bool LtsError::neighbor_data_required(
    const double /*dense_output_time*/,
    const TimeStepId& /*neighbor_data_id*/) const {
  gts_error();
}

[[noreturn]] void LtsError::pup(PUP::er& /*p*/) {
  ERROR("Attempting to serialize LtsError class.");
}

template <typename T>
[[noreturn]] void LtsError::update_u_impl(
    const gsl::not_null<T*> /*u*/, const ConstUntypedHistory<T>& /*history*/,
    const TimeDelta& /*time_step*/) const {
  gts_error();
}

template <typename T>
[[noreturn]] std::optional<StepperErrorEstimate> LtsError::update_u_impl(
    const gsl::not_null<T*> /*u*/, const ConstUntypedHistory<T>& /*history*/,
    const TimeDelta& /*time_step*/,
    const StepperErrorTolerances& /*tolerances*/) const {
  gts_error();
}

template <typename T>
[[noreturn]] void LtsError::clean_history_impl(
    const MutableUntypedHistory<T>& /*history*/) const {
  gts_error();
}

template <typename T>
[[noreturn]] bool LtsError::dense_update_u_impl(
    const gsl::not_null<T*> /*u*/, const ConstUntypedHistory<T>& /*history*/,
    const double /*time*/) const {
  gts_error();
}

template <typename T>
[[noreturn]] bool LtsError::can_change_step_size_impl(
    const TimeStepId& /*time_id*/,
    const ConstUntypedHistory<T>& /*history*/) const {
  gts_error();
}

template <typename T>
[[noreturn]] void LtsError::add_boundary_delta_impl(
    const gsl::not_null<T*> /*result*/,
    const TimeSteppers::ConstBoundaryHistoryTimes& /*local_times*/,
    const TimeSteppers::ConstBoundaryHistoryTimes& /*remote_times*/,
    const TimeSteppers::BoundaryHistoryEvaluator<T>& /*coupling*/,
    const TimeDelta& /*time_step*/) const {
  gts_error();
}

[[noreturn]] void LtsError::clean_boundary_history_impl(
    const TimeSteppers::MutableBoundaryHistoryTimes& /*local_times*/,
    const TimeSteppers::MutableBoundaryHistoryTimes& /*remote_times*/) const {
  gts_error();
}

template <typename T>
[[noreturn]] void LtsError::boundary_dense_output_impl(
    const gsl::not_null<T*> /*result*/,
    const TimeSteppers::ConstBoundaryHistoryTimes& /*local_times*/,
    const TimeSteppers::ConstBoundaryHistoryTimes& /*remote_times*/,
    const TimeSteppers::BoundaryHistoryEvaluator<T>& /*coupling*/,
    const double /*time*/) const {
  gts_error();
}

#if defined(__GNUC__) and not defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif
TIME_STEPPER_DEFINE_OVERLOADS(LtsError)
LTS_TIME_STEPPER_DEFINE_OVERLOADS(LtsError)
#if defined(__GNUC__) and not defined(__clang__)
#pragma GCC diagnostic pop
#endif
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::LtsError::my_PUP_ID = 0;  // NOLINT
