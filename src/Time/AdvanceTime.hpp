// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <type_traits>

#include "Utilities/TMPL.hpp"

/// \cond
class AdaptiveSteppingDiagnostics;
class TimeDelta;
class TimeStepId;
class TimeStepper;
namespace Tags {
struct AdaptiveSteppingDiagnostics;
template <typename Tag>
struct Next;
struct StepperErrorEstimatesEnabled;
struct StepNumberWithinSlab;
struct Time;
struct TimeStep;
struct TimeStepId;
template <typename StepperInterface>
struct TimeStepper;
}  // namespace Tags
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace AdvanceTime_detail {
void apply(gsl::not_null<TimeStepId*> time_id,
           gsl::not_null<TimeStepId*> next_time_id,
           gsl::not_null<TimeDelta*> time_step, gsl::not_null<double*> time,
           gsl::not_null<uint64_t*> step_number_within_slab,
           gsl::not_null<AdaptiveSteppingDiagnostics*> diags,
           const TimeStepper& time_stepper, bool using_error_control);
}  // namespace AdvanceTime_detail

/// \ingroup TimeGroup
/// \brief Advance time one substep
///
/// Replaces the time state with the `Tags::Next` values, advances the
/// `Tags::Next` values, and sets `Tags::Time` to the new substep time.
template <template <typename> typename CacheTagPrefix = std::type_identity_t>
struct AdvanceTime {
  using return_tags =
      tmpl::list<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
                 Tags::Time, Tags::StepNumberWithinSlab,
                 Tags::AdaptiveSteppingDiagnostics>;
  using argument_tags =
      tmpl::list<CacheTagPrefix<Tags::TimeStepper<TimeStepper>>,
                 Tags::StepperErrorEstimatesEnabled>;

  static constexpr auto apply = &AdvanceTime_detail::apply;
};
